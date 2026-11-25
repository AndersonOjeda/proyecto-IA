"""Experiment 5 Runner: Outdoor-Expanded Residue Detector (DetectorS8 from scratch).
Trains on 5,150 real outdoor images (3,645 base + 1,505 verified SALSA & TACO unofficial).
Strict adherence to Exp 3C loss formulation, grid32 encoding, and random weight initialization.
Evaluates strictly on 670 original validation images with zero test set leakage.
"""
from pathlib import Path
import sys
import time
import json
import random
from copy import deepcopy
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from residuos import CLASSES
from residuos.data import Store, read_json, write_json
from scripts.experiment3_runner import (
    DetectorS8, CachedDataset32, encode_grid32, decode_grid32,
    GRID, SLOTS, seed_all, device_for
)
from scripts.experiment3c_runner import loss_fn_3c, CLASS_WEIGHTS_3C

EXP05_DIR = ROOT / 'runs/detector_exp05'
EXP05_DIR.mkdir(parents=True, exist_ok=True)

class ModelEMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model, decay=0.996):
        self.ema = deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(v * d + msd[k] * (1.0 - d))

class DatasetExp05(Dataset):
    """Integrated Cached Dataset for Experiment 5 combining 3,645 base and 1,505 external images."""
    def __init__(self, orig_rows, orig_npy_path, ext_rows, ext_npy_path, augment=True):
        self.orig_rows = orig_rows
        self.ext_rows = ext_rows
        self.orig_pixels = np.load(orig_npy_path, mmap_mode='r')
        self.ext_pixels = np.load(ext_npy_path, mmap_mode='r')
        self.augment = augment

        assert len(self.orig_rows) == len(self.orig_pixels), "Mismatch in base train cache length"
        assert len(self.ext_rows) == len(self.ext_pixels), "Mismatch in ext train cache length"

        self.samples = [('orig', i) for i in range(len(self.orig_rows))] + [('ext', j) for j in range(len(self.ext_rows))]
        print(f"DatasetExp05: Total muestras = {len(self.samples)} ({len(self.orig_rows)} base + {len(self.ext_rows)} externas)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        kind, real_idx = self.samples[idx]
        if kind == 'orig':
            r = self.orig_rows[real_idx]
            frame = np.array(self.orig_pixels[real_idx], copy=True)
        else:
            r = self.ext_rows[real_idx]
            frame = np.array(self.ext_pixels[real_idx], copy=True)

        w_orig, h_orig = r['width'], r['height']
        boxes = [dict(b, xyxy=list(b['xyxy'])) for b in r['boxes']]

        if self.augment:
            if random.random() < 0.5:
                frame = np.ascontiguousarray(frame[:, ::-1])
                for b in boxes:
                    x1, y1, x2, y2 = b['xyxy']
                    b['xyxy'] = [w_orig - x2, y1, w_orig - x1, y2]
            if random.random() < 0.5:
                factor = random.uniform(0.85, 1.15)
                frame = np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        x_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        target_tensor = encode_grid32(boxes, w_orig, h_orig)
        return x_tensor, target_tensor

def evaluate_on_val_map(model, val_rows, val_npy, device):
    """Fast validation mAP50 evaluation."""
    model.eval()
    valset = CachedDataset32(val_rows, val_npy, augment=False)
    loader = DataLoader(valset, batch_size=16, shuffle=False)

    outputs = []
    with torch.no_grad():
        for x, _ in loader:
            outputs.extend(model(x.to(device)).cpu())

    class_scores = {c: [] for c in CLASSES}
    class_hits = {c: [] for c in CLASSES}
    gt_counts = Counter()
    for r in val_rows:
        for b in r['boxes']:
            gt_counts[b['class']] += 1

    for i, r in enumerate(val_rows):
        out = outputs[i]
        w, h = r['width'], r['height']
        raw_preds = decode_grid32(out, w, h, threshold=0.01, nms=0.45)
        matched_gt = set()
        for p in raw_preds:
            cls = p['class']
            class_scores[cls].append(p['score'])
            best_iou = 0.0
            best_idx = -1
            for gi, gb in enumerate(r['boxes']):
                if gb['class'] == cls and gi not in matched_gt:
                    gx1, gy1, gx2, gy2 = gb['xyxy']
                    px1, py1, px2, py2 = p['xyxy']
                    ix1 = max(gx1, px1); iy1 = max(gy1, py1)
                    ix2 = min(gx2, px2); iy2 = min(gy2, py2)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    union = (gx2 - gx1) * (gy2 - gy1) + (px2 - px1) * (py2 - py1) - inter
                    iou = inter / max(union, 1e-6)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = gi
            if best_iou >= 0.5:
                class_hits[cls].append(1)
                matched_gt.add(best_idx)
            else:
                class_hits[cls].append(0)

    aps = {}
    for c in CLASSES:
        sc = class_scores[c]
        ht = class_hits[c]
        n_gt = gt_counts[c]
        if n_gt == 0 or len(sc) == 0:
            aps[c] = 0.0
            continue
        order = np.argsort(-np.array(sc))
        sorted_hits = np.array(ht)[order]
        tp_cum = np.cumsum(sorted_hits)
        fp_cum = np.cumsum(1 - sorted_hits)
        recall_curve = tp_cum / n_gt
        prec_curve = tp_cum / (tp_cum + fp_cum)
        ap = 0.0
        for t in np.linspace(0.0, 1.0, 101):
            prec_at_rec = prec_curve[recall_curve >= t]
            p_val = np.max(prec_at_rec) if len(prec_at_rec) > 0 else 0.0
            ap += p_val / 101.0
        aps[c] = float(ap)

    mAP50 = float(np.mean(list(aps.values())))
    return mAP50, aps

def evaluate_exp05_full(ckpt_path, out_json_path):
    """Full 101-point VOC AP50 and operating threshold sweep evaluation."""
    device = device_for('auto')
    print(f"\n================ EVALUANDO EXPERIMENTO 5 ({Path(ckpt_path).name}) SOBRE 670 IMAGENES DE VAL ==================", flush=True)

    saved = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = DetectorS8().to(device)
    model.load_state_dict(saved['state'])
    model.eval()

    store = Store(ROOT / 'data/preparado/deteccion')
    val_rows = store.records('val')
    val_npy = ROOT / 'runs/detector_exp01/val.npy'
    valset = CachedDataset32(val_rows, val_npy, augment=False)
    loader = DataLoader(valset, batch_size=16, shuffle=False)

    outputs = []
    with torch.no_grad():
        for x, _ in loader:
            outputs.extend(model(x.to(device)).cpu())

    class_scores = {c: [] for c in CLASSES}
    class_hits = {c: [] for c in CLASSES}
    gt_counts = Counter()
    for r in val_rows:
        for b in r['boxes']:
            gt_counts[b['class']] += 1

    stages = {c: dict(objects=0, raw_any_class=0, raw_correct_class=0) for c in CLASSES}
    source_stats = defaultdict(lambda: defaultdict(lambda: {'objects': 0, 'scores': [], 'hits': []}))

    for i, r in enumerate(val_rows):
        out = outputs[i]
        w, h = r['width'], r['height']
        src = r.get('source', 'unknown')
        for b in r['boxes']:
            stages[b['class']]['objects'] += 1
            source_stats[src][b['class']]['objects'] += 1

        raw_preds = decode_grid32(out, w, h, threshold=0.01, nms=0.45)
        matched_gt = set()
        for p in raw_preds:
            cls = p['class']
            class_scores[cls].append(p['score'])
            source_stats[src][cls]['scores'].append(p['score'])

            best_iou = 0.0
            best_idx = -1
            for gi, gb in enumerate(r['boxes']):
                if gb['class'] == cls and gi not in matched_gt:
                    gx1, gy1, gx2, gy2 = gb['xyxy']
                    px1, py1, px2, py2 = p['xyxy']
                    ix1 = max(gx1, px1); iy1 = max(gy1, py1)
                    ix2 = min(gx2, px2); iy2 = min(gy2, py2)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    union = (gx2 - gx1) * (gy2 - gy1) + (px2 - px1) * (py2 - py1) - inter
                    iou = inter / max(union, 1e-6)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = gi
            if best_iou >= 0.5:
                class_hits[cls].append(1)
                source_stats[src][cls]['hits'].append(1)
                matched_gt.add(best_idx)
            else:
                class_hits[cls].append(0)
                source_stats[src][cls]['hits'].append(0)

    # Calculate AP50
    ap50 = {}
    for c in CLASSES:
        sc = class_scores[c]
        ht = class_hits[c]
        n_gt = gt_counts[c]
        if n_gt == 0 or len(sc) == 0:
            ap50[c] = 0.0
            continue
        order = np.argsort(-np.array(sc))
        sorted_hits = np.array(ht)[order]
        tp_cum = np.cumsum(sorted_hits)
        fp_cum = np.cumsum(1 - sorted_hits)
        rec = tp_cum / n_gt
        prec = tp_cum / (tp_cum + fp_cum)
        ap = 0.0
        for t in np.linspace(0.0, 1.0, 101):
            p_val = np.max(prec[rec >= t]) if np.any(rec >= t) else 0.0
            ap += p_val / 101.0
        ap50[c] = float(ap)

    mAP50 = float(np.mean(list(ap50.values())))

    # Threshold sweep 0.01 to 0.80
    thresholds = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80]
    sweep = {}
    best_f1 = -1.0
    best_op = {}

    for th in thresholds:
        tp_tot = 0; fp_tot = 0; fn_tot = 0
        cls_res = {}
        for c in CLASSES:
            tp_c = 0; fp_c = 0
            for i, r in enumerate(val_rows):
                w, h = r['width'], r['height']
                p_th = decode_grid32(outputs[i], w, h, threshold=th, nms=0.45)
                p_c = [p for p in p_th if p['class'] == c]
                g_c = [g for g in r['boxes'] if g['class'] == c]
                matched = set()
                for p in p_c:
                    best_iou = 0.0
                    best_gi = -1
                    for gi, g in enumerate(g_c):
                        if gi not in matched:
                            gx1, gy1, gx2, gy2 = g['xyxy']
                            px1, py1, px2, py2 = p['xyxy']
                            ix1 = max(gx1, px1); iy1 = max(gy1, py1)
                            ix2 = min(gx2, px2); iy2 = min(gy2, py2)
                            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                            union = (gx2 - gx1) * (gy2 - gy1) + (px2 - px1) * (py2 - py1) - inter
                            iou = inter / max(union, 1e-6)
                            if iou > best_iou:
                                best_iou = iou
                                best_gi = gi
                    if best_iou >= 0.5:
                        tp_c += 1
                        matched.add(best_gi)
                    else:
                        fp_c += 1
            fn_c = max(0, gt_counts[c] - tp_c)
            p_prec = tp_c / max(tp_c + fp_c, 1)
            p_rec = tp_c / max(gt_counts[c], 1)
            p_f1 = 2 * p_prec * p_rec / max(p_prec + p_rec, 1e-6)
            cls_res[c] = {
                'tp': tp_c, 'fp': fp_c, 'fn': fn_c,
                'precision': float(p_prec), 'recall': float(p_rec), 'f1': float(p_f1)
            }
            tp_tot += tp_c; fp_tot += fp_c; fn_tot += fn_c

        micro_p = tp_tot / max(tp_tot + fp_tot, 1)
        micro_r = tp_tot / max(tp_tot + fn_tot, 1)
        micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-6)

        th_str = f"{th:.2f}"
        sweep[th_str] = {
            'threshold': float(th),
            'micro_f1': float(micro_f1),
            'precision': float(micro_p),
            'recall': float(micro_r),
            'tp': tp_tot, 'fp': fp_tot, 'fn': fn_tot,
            'classes': cls_res
        }
        if micro_f1 > best_f1:
            best_f1 = micro_f1
            best_op = sweep[th_str]

    # Size breakdown at best_op
    op_th = best_op['threshold']
    size_bins = {
        'small_<32px': {'objects': 0, 'tp': 0},
        'medium_32-96px': {'objects': 0, 'tp': 0},
        'large_>=96px': {'objects': 0, 'tp': 0}
    }
    for i, r in enumerate(val_rows):
        w, h = r['width'], r['height']
        p_th = decode_grid32(outputs[i], w, h, threshold=op_th, nms=0.45)
        for g in r['boxes']:
            gx1, gy1, gx2, gy2 = g['xyxy']
            sz = max(gx2 - gx1, gy2 - gy1) * 256 / max(w, h)
            bname = 'small_<32px' if sz < 32 else ('medium_32-96px' if sz < 96 else 'large_>=96px')
            size_bins[bname]['objects'] += 1
            for p in p_th:
                if p['class'] == g['class']:
                    px1, py1, px2, py2 = p['xyxy']
                    ix1 = max(gx1, px1); iy1 = max(gy1, py1)
                    ix2 = min(gx2, px2); iy2 = min(gy2, py2)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    union = (gx2 - gx1) * (gy2 - gy1) + (px2 - px1) * (py2 - py1) - inter
                    if (inter / max(union, 1e-6)) >= 0.5:
                        size_bins[bname]['tp'] += 1
                        break
    for k in size_bins:
        size_bins[k]['recall'] = float(size_bins[k]['tp'] / max(size_bins[k]['objects'], 1))

    # Confusion matrix at op_th
    conf_mat = {c: {c2: 0 for c2 in CLASSES} for c in CLASSES}
    for c in conf_mat:
        conf_mat[c]['fn_missed'] = 0
    bg_fp = {c: 0 for c in CLASSES}

    for i, r in enumerate(val_rows):
        w, h = r['width'], r['height']
        p_th = decode_grid32(outputs[i], w, h, threshold=op_th, nms=0.45)
        g_boxes = list(r['boxes'])
        matched_g = set()
        for p in p_th:
            best_iou = 0.0
            best_gi = -1
            for gi, g in enumerate(g_boxes):
                if gi not in matched_g:
                    gx1, gy1, gx2, gy2 = g['xyxy']
                    px1, py1, px2, py2 = p['xyxy']
                    ix1 = max(gx1, px1); iy1 = max(gy1, py1)
                    ix2 = min(gx2, px2); iy2 = min(gy2, py2)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    union = (gx2 - gx1) * (gy2 - gy1) + (px2 - px1) * (py2 - py1) - inter
                    iou = inter / max(union, 1e-6)
                    if iou > best_iou:
                        best_iou = iou
                        best_gi = gi
            if best_iou >= 0.5:
                true_cls = g_boxes[best_gi]['class']
                pred_cls = p['class']
                conf_mat[true_cls][pred_cls] += 1
                matched_g.add(best_gi)
            else:
                bg_fp[p['class']] += 1

        for gi, g in enumerate(g_boxes):
            if gi not in matched_g:
                conf_mat[g['class']]['fn_missed'] += 1

    conf_mat['background_fp'] = bg_fp

    eval_summary = {
        'variant': 'Exp05_OutdoorExpanded',
        'checkpoint': str(ckpt_path),
        'best_epoch': saved.get('epoch'),
        'weights_type': saved.get('weights_type', 'RAW'),
        'val_loss': saved.get('val_loss'),
        'true_unthresholded_map50': mAP50,
        'true_unthresholded_ap50': ap50,
        'best_operating_point': best_op,
        'metrics_at_th_035': sweep.get('0.35', {}),
        'metrics_at_th_040': sweep.get('0.40', {}),
        'threshold_sweep': sweep,
        'recall_by_size': size_bins,
        'confusion_matrix_at_op_th': conf_mat
    }

    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(eval_summary, f, indent=2, ensure_ascii=False)

    print(f"\n[Exp05 Evaluacion Final ({Path(ckpt_path).name})]:")
    print(f"  True Unthresholded mAP50: {mAP50*100:.2f}%")
    for c in CLASSES:
        print(f"    {c:10s}: AP50 = {ap50[c]*100:.2f}%")
    print(f"  Best Operating Point: th={best_op['threshold']:.2f} -> Micro-F1 = {best_op['micro_f1']*100:.2f}% (TP={best_op['tp']}, FP={best_op['fp']}, FN={best_op['fn']})")
    for c in CLASSES:
        cm = best_op['classes'][c]
        print(f"    {c:10s}: TP={cm['tp']:3d} | FP={cm['fp']:4d} | FN={cm['fn']:3d} | Prec={cm['precision']*100:5.2f}% | Rec={cm['recall']*100:5.2f}% | F1={cm['f1']*100:5.2f}%")

    return eval_summary

def train_exp05(epochs=40, batch_size=16, lr=0.001, seed=42):
    """Full 40-epoch training from scratch with random weights."""
    seed_all(seed)
    device = device_for('auto')
    print(f"\n================ STARTING EXPERIMENT 5 TRAINING ON {device} ================", flush=True)

    store = Store(ROOT / 'data/preparado/deteccion')
    orig_train = store.records('train')
    orig_val = store.records('val')
    with open(EXP05_DIR / 'manifiesto_descargas.json', encoding='utf-8') as f:
        ext_manifest = json.load(f)
    ext_rows = ext_manifest['accepted_images']

    trainset = DatasetExp05(
        orig_rows=orig_train,
        orig_npy_path=ROOT / 'runs/detector_exp01/train.npy',
        ext_rows=ext_rows,
        ext_npy_path=EXP05_DIR / 'ext_train.npy',
        augment=True
    )
    val_npy = ROOT / 'runs/detector_exp01/val.npy'
    valset = CachedDataset32(orig_val, val_npy, augment=False)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = DetectorS8().to(device)
    ema = ModelEMA(model, decay=0.996)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    config = {
        'variant': 'Exp05_OutdoorExpanded',
        'architecture': 'DetectorS8',
        'weights_init': 'random_from_scratch',
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'seed': seed,
        'device': str(device),
        'ema_decay': 0.996,
        'total_train_images': len(trainset),
        'base_train_images': len(orig_train),
        'external_verified_images': len(ext_rows),
        'val_images': len(orig_val)
    }
    with open(EXP05_DIR / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    history = []
    best_val_loss = float('inf')
    best_map50 = -1.0
    start_time = time.time()

    for epoch in range(epochs):
        ep_start = time.time()
        model.train()
        train_loss = 0.0
        train_batches = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn_3c(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ema.update(model)

            train_loss += loss.item()
            train_batches += 1

        train_loss /= max(train_batches, 1)

        # Validation loss
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                loss = loss_fn_3c(pred, y)
                val_loss += loss.item()
                val_batches += 1
        val_loss /= max(val_batches, 1)

        scheduler.step()
        ep_sec = time.time() - ep_start
        total_elapsed = time.time() - start_time

        row = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': opt.param_groups[0]['lr'],
            'epoch_seconds': ep_sec,
            'total_seconds': total_elapsed
        }

        # Check best val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'origin': 'residuos-detector-exp05-outdoor-from-scratch',
                'epoch': epoch + 1,
                'val_loss': best_val_loss,
                'state': model.state_dict(),
                'config': config
            }, EXP05_DIR / 'best_val_loss.pt')
            star_loss = " ★ [BestLoss]"
        else:
            star_loss = ""

        # Validation mAP50 evaluation
        if (epoch + 1) >= 10:
            raw_map, raw_ap = evaluate_on_val_map(model, orig_val, val_npy, device)
            ema_map, ema_ap = evaluate_on_val_map(ema.ema, orig_val, val_npy, device)
            row['val_map50_raw'] = raw_map
            row['val_map50_ema'] = ema_map
            row['val_ap50_raw'] = raw_ap
            row['val_ap50_ema'] = ema_ap

            cand_map, cand_state, cand_tag = (
                (ema_map, ema.ema.state_dict(), 'EMA') if ema_map >= raw_map
                else (raw_map, model.state_dict(), 'RAW')
            )

            if cand_map > best_map50:
                best_map50 = cand_map
                torch.save({
                    'origin': 'residuos-detector-exp05-outdoor-from-scratch',
                    'epoch': epoch + 1,
                    'weights_type': cand_tag,
                    'val_loss': val_loss,
                    'val_map50': best_map50,
                    'val_ap50': ema_ap if cand_tag == 'EMA' else raw_ap,
                    'state': cand_state,
                    'config': config
                }, EXP05_DIR / 'best.pt')
                star_map = f" ★ [BEST mAP50 ({cand_tag}): {best_map50*100:.2f}%]"
            else:
                star_map = f" (mAP raw:{raw_map*100:.2f}% | ema:{ema_map*100:.2f}%)"
        else:
            star_map = ""

        torch.save({
            'origin': 'residuos-detector-exp05-outdoor-from-scratch',
            'epoch': epoch + 1,
            'val_loss': val_loss,
            'state': model.state_dict(),
            'ema_state': ema.ema.state_dict(),
            'config': config
        }, EXP05_DIR / 'last.pt')

        history.append(row)
        with open(EXP05_DIR / 'history.json', 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)

        print(f"[Exp05] Epoca {epoch+1:02d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | {ep_sec:.1f}s{star_loss}{star_map}", flush=True)

    print(f"\n>>> ENTRENAMIENTO EXPERIMENTO 5 COMPLETADO: best_val_loss = {best_val_loss:.4f} | best_val_mAP50 = {best_map50*100:.2f}% <<<", flush=True)

    # Run full evaluations on checkpoints
    evaluate_exp05_full(EXP05_DIR / 'best.pt', EXP05_DIR / 'evaluation.json')
    if (EXP05_DIR / 'best_val_loss.pt').exists():
        evaluate_exp05_full(EXP05_DIR / 'best_val_loss.pt', EXP05_DIR / 'evaluation_best_val_loss.json')

    return history

if __name__ == '__main__':
    train_exp05(epochs=40, batch_size=16, lr=0.001, seed=42)
