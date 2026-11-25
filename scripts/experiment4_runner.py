"""Experiment 4 Runner: Dataset Expansion, Balanced Sampling, Preflight Validation,
Training from Scratch, and Homogeneous Evaluation on the 670 Validation Images.
Supports:
- Variant 4A (Broad 1,600 external expansion + multi-scale zoom-out)
- Variant 4B (Controlled Domain-Aligned 400 natural-background TACO-scale external expansion + exact 3C loss)
"""
from pathlib import Path
import sys
import time
import json
import random
import hashlib
import argparse
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from residuos import CLASSES
from residuos.data import Store, read_json, write_json
from residuos.detector import iou
from scripts.experiment3_runner import (
    DetectorS8, CachedDataset32, encode_grid32, decode_grid32,
    GRID, SLOTS, SIZE, seed_all, device_for
)
from scripts.audit_evaluation import ap50_voc101, overlaps, nms_indices, extract_candidates_generic

EXP04_DIR = ROOT / 'runs/detector_exp04'
CLASS_WEIGHTS_4A = torch.tensor([0.7653, 1.1834, 1.1442, 1.1517, 1.0511, 0.7043], dtype=torch.float32)
CLASS_WEIGHTS_4B = torch.tensor([0.53, 1.39, 1.27, 1.35, 0.95, 0.53], dtype=torch.float32)

def loss_fn_4(pred, target, cls_weights=CLASS_WEIGHTS_4B, gamma=2.0, alpha=0.75, obj_scale=1.5, box_weight=10.0):
    """Experiment 4 loss: same Focal Loss + Smooth L1 + weighted CE structure as 3C."""
    positive = target[..., 0] == 1
    n_pos = max(float(positive.sum()), 1.0)

    # 1. Focal Loss for presence normalized by N_pos
    p = torch.sigmoid(pred[..., 0])
    pt = p * target[..., 0] + (1.0 - p) * (1.0 - target[..., 0])
    focal_weight = (1.0 - pt) ** gamma
    alpha_weight = alpha * target[..., 0] + (1.0 - alpha) * (1.0 - target[..., 0])
    bce = nn.functional.binary_cross_entropy_with_logits(pred[..., 0], target[..., 0], reduction='none')
    fl = alpha_weight * focal_weight * bce
    obj = (fl.sum() / n_pos) * obj_scale

    # 2. Smooth L1 box loss on positive cells
    zero = pred.sum() * 0
    box = nn.functional.smooth_l1_loss(pred[..., 1:5][positive].sigmoid(), target[..., 1:5][positive]) if positive.any() else zero

    # 3. Balanced Cross-Entropy class loss on positive cells
    if positive.any():
        w = cls_weights.to(pred.device) if cls_weights is not None else None
        cls = nn.functional.cross_entropy(pred[..., 5:][positive], target[..., 5][positive].long(), weight=w)
    else:
        cls = zero

    return obj + box_weight * box + cls


def select_domain_aligned_4b_indices(ext_rows, ext_pixels, quotas=None):
    """Selects external images with natural textured (non-white-studio) backgrounds
    and TACO-aligned object scales (0.015 <= area_ratio <= 0.35).
    """
    if quotas is None:
        quotas = {'vidrio': 115, 'papel': 105, 'carton': 100, 'metal': 80}
    cands_by_cls = defaultdict(list)
    for j, r in enumerate(ext_rows):
        c = r['class_primary']
        if c not in quotas:
            continue
        img = ext_pixels[j]
        border = np.concatenate([img[0, :], img[-1, :], img[:, 0], img[:, -1]], axis=0).astype(np.float32)
        bg_std = float(border.std())
        bg_mean = float(border.mean())
        ar = float(r['mean_area_ratio'])
        # Tier 0: natural non-white background + TACO object scale
        is_natural_bg = (bg_std >= 22.0 and 35.0 <= bg_mean <= 210.0)
        is_taco_scale = (0.015 <= ar <= 0.32)
        if is_natural_bg and is_taco_scale:
            tier = 0
        elif is_natural_bg and ar <= 0.45:
            tier = 1
        elif bg_std >= 16.0 and bg_mean <= 225.0 and ar <= 0.40:
            tier = 2
        else:
            tier = 3
        cands_by_cls[c].append((tier, -bg_std, abs(ar - 0.08), r['id'], j))

    chosen_indices = []
    summary_by_cls = Counter()
    summary_by_src = defaultdict(Counter)
    boxes_by_cls = Counter()
    for c, q in quotas.items():
        sorted_c = sorted(cands_by_cls[c])[:q]
        for _, _, _, _, j in sorted_c:
            chosen_indices.append(j)
            summary_by_cls[c] += 1
            summary_by_src[ext_rows[j]['source']][c] += 1
            boxes_by_cls[c] += len(ext_rows[j]['boxes'])
    chosen_indices.sort()
    return chosen_indices, dict(summary_by_cls), dict(boxes_by_cls), {k: dict(v) for k, v in summary_by_src.items()}


class DatasetExp04(Dataset):
    """Combined Cached Dataset for Experiment 4."""
    def __init__(self, orig_rows, orig_npy_path, ext_rows, ext_npy_path,
                 augment=True, anchor_taco=False, ext_subset_indices=None, zoom_paste=False):
        self.orig_rows = orig_rows
        self.ext_rows = ext_rows
        self.orig_pixels = np.load(orig_npy_path, mmap_mode='r')
        self.ext_pixels = np.load(ext_npy_path, mmap_mode='r')
        self.augment = augment
        self.zoom_paste = zoom_paste

        assert len(self.orig_rows) == len(self.orig_pixels), "Mismatch in orig train cache length"
        assert len(self.ext_rows) == len(self.ext_pixels), "Mismatch in ext train cache length"

        self.samples = [('orig', i) for i in range(len(self.orig_rows))]
        if ext_subset_indices is not None:
            self.samples.extend([('ext', j) for j in ext_subset_indices])
        else:
            self.samples.extend([('ext', j) for j in range(len(self.ext_rows))])

        if anchor_taco:
            for i, r in enumerate(self.orig_rows):
                if r['source'] == 'TACO oficial' and any(b['class'] in ('vidrio', 'papel', 'carton', 'metal') for b in r['boxes']):
                    self.samples.append(('orig', i))

        for i, r in enumerate(self.orig_rows):
            encode_grid32(r['boxes'], r['width'], r['height'])
        for j, r in enumerate(self.ext_rows):
            encode_grid32(r['boxes'], r['width'], r['height'])

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
            frame = np.clip(frame.astype(np.float32) * random.uniform(0.85, 1.15), 0, 255).astype(np.uint8)

        x_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        target_tensor = encode_grid32(boxes, w_orig, h_orig)
        return x_tensor, target_tensor


def train_exp04_4b(epochs=40, batch_size=16, lr=0.001, seed=42):
    seed_all(seed)
    device = device_for('auto')
    print(f"\n================ STARTING EXPERIMENT 4B (DOMAIN-ALIGNED) FROM SCRATCH ON {device} ================", flush=True)

    EXP04_DIR.mkdir(parents=True, exist_ok=True)
    store = Store(ROOT / 'data/preparado/deteccion')
    orig_train = store.records('train')
    orig_val = store.records('val')
    ext_rows = read_json(EXP04_DIR / 'manifest_exp04_ext.json')
    ext_pixels = np.load(EXP04_DIR / 'ext_train.npy', mmap_mode='r')

    ext_indices_4b, imgs_4b, boxes_4b, src_4b = select_domain_aligned_4b_indices(ext_rows, ext_pixels)
    print(f"Selected {len(ext_indices_4b)} domain-aligned natural-background external images: {imgs_4b} (boxes: {boxes_4b})", flush=True)

    trainset = DatasetExp04(
        orig_train, ROOT / 'runs/detector_exp01/train.npy',
        ext_rows, EXP04_DIR / 'ext_train.npy',
        augment=True, anchor_taco=False, ext_subset_indices=ext_indices_4b, zoom_paste=False
    )
    valset = CachedDataset32(orig_val, ROOT / 'runs/detector_exp01/val.npy', augment=False)

    loaders = {
        'train': DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True),
        'val': DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    }

    config = {
        'variant': 'Exp04B_DomainAligned',
        'description': 'DetectorS8 trained 100% from scratch on 3645 orig + 400 natural-background TACO-scale external minority images with controlled 3C Focal Loss',
        'pretrained_weights_used': False,
        'initialized_from_3c': False,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'seed': seed,
        'device': str(device),
        'grid': GRID,
        'slots': SLOTS,
        'class_weights': [float(x) for x in CLASS_WEIGHTS_4B],
        'focal_gamma': 2.0,
        'focal_alpha': 0.75,
        'obj_scale': 1.5,
        'box_weight': 10.0,
        'classes': list(CLASSES),
        'orig_train_images': len(orig_train),
        'ext_train_images': len(ext_indices_4b),
        'ext_images_by_class': imgs_4b,
        'ext_boxes_by_class': boxes_4b,
        'ext_images_by_source': src_4b,
        'total_unique_train_images': len(trainset),
        'val_images': len(valset),
        'test_images_touched': 0
    }
    write_json(EXP04_DIR / 'config.json', config)

    model = DetectorS8(num_classes=len(CLASSES), slots=SLOTS).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    best_val_loss = float('inf')
    best_map50 = -1.0
    history = []
    start_time = time.perf_counter()
    gt_cnt = Counter(b['class'] for r in orig_val for b in r['boxes'])

    for epoch in range(epochs):
        ep_start = time.perf_counter()
        row = {'epoch': epoch + 1}

        for split, loader in loaders.items():
            is_train = (split == 'train')
            model.train(is_train)
            total_loss = 0.0
            total_samples = 0

            for x, y in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with torch.set_grad_enabled(is_train):
                    out = model(x)
                    loss = loss_fn_4(out, y, cls_weights=CLASS_WEIGHTS_4B)
                    if not torch.isfinite(loss):
                        raise ValueError(f"Non-finite loss at epoch {epoch+1}, split {split}")

                    if is_train:
                        opt.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        opt.step()

                total_loss += loss.item() * len(x)
                total_samples += len(x)

            row[f'{split}_loss'] = total_loss / total_samples

        scheduler.step()
        row['lr'] = opt.param_groups[0]['lr']
        row['seconds_elapsed'] = time.perf_counter() - start_time
        ep_duration = time.perf_counter() - ep_start

        val_loss = row['val_loss']
        train_loss = row['train_loss']
        is_best_loss = val_loss < best_val_loss
        if is_best_loss:
            best_val_loss = val_loss
            torch.save({
                'origin': 'residuos-detector-exp04b-from-scratch',
                'epoch': epoch + 1,
                'val_loss': best_val_loss,
                'state': model.state_dict(),
                'config': config
            }, EXP04_DIR / 'best_val_loss.pt')

        # Evaluate validation mAP50 on every epoch >= 12 (or when best val_loss)
        if is_best_loss or (epoch + 1) >= 12:
            model.eval()
            c_scores = {c: [] for c in CLASSES}
            c_hits = {c: [] for c in CLASSES}
            val_outs = []
            with torch.no_grad():
                for vx, _ in loaders['val']:
                    val_outs.extend(model(vx.to(device)).cpu())
            for i_v, r_v in enumerate(orig_val):
                xy, cls_v, sc_v, _, _ = extract_candidates_generic(val_outs[i_v], r_v['width'], r_v['height'], GRID)
                kept = nms_indices(xy, cls_v, sc_v, threshold=0.45, max_detections=100)
                n_xy, n_cls, n_sc = xy[kept], cls_v[kept], sc_v[kept]
                gt_b = [b['xyxy'] for b in r_v['boxes']]
                gt_c = [CLASSES.index(b['class']) for b in r_v['boxes']]
                used_g = set()
                if len(gt_b) > 0 and len(n_xy) > 0:
                    ov = overlaps(n_xy, gt_b)
                    for p_i in np.argsort(-n_sc, kind='stable'):
                        cp = n_cls[p_i]
                        cn = CLASSES[cp]
                        elig = [g_i for g_i in range(len(gt_b)) if gt_c[g_i] == cp and g_i not in used_g]
                        vo, bg = max([(ov[p_i, g_i], g_i) for g_i in elig], default=(0.0, -1))
                        if vo >= 0.5:
                            used_g.add(bg)
                            c_scores[cn].append(float(n_sc[p_i]))
                            c_hits[cn].append(True)
                        else:
                            c_scores[cn].append(float(n_sc[p_i]))
                            c_hits[cn].append(False)
                elif len(n_xy) > 0:
                    for p_i in range(len(n_xy)):
                        cn = CLASSES[n_cls[p_i]]
                        c_scores[cn].append(float(n_sc[p_i]))
                        c_hits[cn].append(False)
            ap_ep = {c: ap50_voc101(c_scores[c], c_hits[c], gt_cnt[c]) for c in CLASSES}
            map_ep = float(np.mean(list(ap_ep.values())))
            row['val_map50'] = map_ep
            row['val_ap50'] = ap_ep
            if map_ep > best_map50:
                best_map50 = map_ep
                torch.save({
                    'origin': 'residuos-detector-exp04b-from-scratch',
                    'epoch': epoch + 1,
                    'val_loss': val_loss,
                    'val_map50': best_map50,
                    'val_ap50': ap_ep,
                    'state': model.state_dict(),
                    'config': config
                }, EXP04_DIR / 'best.pt')
                star_map = f" ★ (BEST mAP50: {best_map50*100:.2f}%)"
            else:
                star_map = f" (mAP50: {map_ep*100:.2f}%)"
        else:
            star_map = ""

        torch.save({
            'origin': 'residuos-detector-exp04b-from-scratch',
            'epoch': epoch + 1,
            'val_loss': val_loss,
            'state': model.state_dict(),
            'config': config
        }, EXP04_DIR / 'last.pt')

        history.append(row)
        write_json(EXP04_DIR / 'history.json', history)
        star_loss = " [BestLoss]" if is_best_loss else ""
        print(f"[Exp04B] Época {epoch+1:02d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | {ep_duration:.1f}s{star_loss}{star_map}", flush=True)

    print(f"\n>>> ENTRENAMIENTO EXP 04B COMPLETADO: best_val_loss = {best_val_loss:.4f} | best_val_mAP50 = {best_map50*100:.2f}% <<<", flush=True)
    return history


def evaluate_exp04(ckpt_name='best.pt', out_json_name='evaluation.json'):
    device = device_for('auto')
    ckpt_path = EXP04_DIR / ckpt_name
    print(f"\n================ EVALUATING EXPERIMENT 4 ({ckpt_name}) ON 670 VAL IMAGES ================", flush=True)

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
    total_gt = sum(gt_counts.values())

    source_scores = defaultdict(lambda: defaultdict(list))
    source_hits = defaultdict(lambda: defaultdict(list))
    source_gt_counts = defaultdict(Counter)

    size_gt_counts = Counter()
    size_class_gt_counts = defaultdict(Counter)

    stages = {c: dict(objects=0, raw_any_class=0, raw_correct_class=0) for c in CLASSES}
    pos_p_obj = {c: [] for c in CLASSES}
    pos_p_cls = {c: [] for c in CLASSES}
    pos_scores = {c: [] for c in CLASSES}

    per_image_nms_preds = []

    for i, r in enumerate(val_rows):
        out = outputs[i]
        w, h = r['width'], r['height']
        src = r['source']
        xy, cls, scores, p_obj, p_cls = extract_candidates_generic(out, w, h, GRID)

        gt_boxes = [b['xyxy'] for b in r['boxes']]
        gt_classes = [CLASSES.index(b['class']) for b in r['boxes']]
        gt_sizes = []
        for b in r['boxes']:
            c_name = b['class']
            source_gt_counts[src][c_name] += 1
            w_norm = (b['xyxy'][2] - b['xyxy'][0]) * 256.0 / w
            h_norm = (b['xyxy'][3] - b['xyxy'][1]) * 256.0 / h
            area = w_norm * h_norm
            s_bin = 'small_<32px' if area < 32**2 else 'medium_32-96px' if area < 96**2 else 'large_>=96px'
            gt_sizes.append(s_bin)
            size_gt_counts[s_bin] += 1
            size_class_gt_counts[c_name][s_bin] += 1

        kept_idx = nms_indices(xy, cls, scores, threshold=0.45, max_detections=100)
        nms_xy = xy[kept_idx]
        nms_cls = cls[kept_idx]
        nms_scores = scores[kept_idx]
        per_image_nms_preds.append((nms_xy, nms_cls, nms_scores, gt_boxes, gt_classes, gt_sizes, src, r['id']))

        used_gt = set()
        if len(gt_boxes) > 0 and len(nms_xy) > 0:
            ov = overlaps(nms_xy, gt_boxes)
            for p_i in np.argsort(-nms_scores, kind='stable'):
                c_pred = nms_cls[p_i]
                c_name = CLASSES[c_pred]
                eligible = [g_i for g_i in range(len(gt_boxes)) if gt_classes[g_i] == c_pred and g_i not in used_gt]
                val_ov, best_g = max([(ov[p_i, g_i], g_i) for g_i in eligible], default=(0.0, -1))
                if val_ov >= 0.5:
                    used_gt.add(best_g)
                    class_scores[c_name].append(float(nms_scores[p_i]))
                    class_hits[c_name].append(True)
                    source_scores[src][c_name].append(float(nms_scores[p_i]))
                    source_hits[src][c_name].append(True)
                else:
                    class_scores[c_name].append(float(nms_scores[p_i]))
                    class_hits[c_name].append(False)
                    source_scores[src][c_name].append(float(nms_scores[p_i]))
                    source_hits[src][c_name].append(False)
        elif len(nms_xy) > 0:
            for p_i in range(len(nms_xy)):
                c_name = CLASSES[nms_cls[p_i]]
                class_scores[c_name].append(float(nms_scores[p_i]))
                class_hits[c_name].append(False)
                source_scores[src][c_name].append(float(nms_scores[p_i]))
                source_hits[src][c_name].append(False)

        if len(gt_boxes) > 0:
            raw_ov = overlaps(xy, gt_boxes)
            for g_i, b in enumerate(r['boxes']):
                c_idx = CLASSES.index(b['class'])
                c_name = b['class']
                stages[c_name]['objects'] += 1
                has_geom = bool((raw_ov[:, g_i] >= 0.5).any())
                has_correct = bool(((raw_ov[:, g_i] >= 0.5) & (cls == c_idx)).any())
                if has_geom: stages[c_name]['raw_any_class'] += 1
                if has_correct: stages[c_name]['raw_correct_class'] += 1

                cx = (b['xyxy'][0] + b['xyxy'][2]) / 2.0 / w
                cy = (b['xyxy'][1] + b['xyxy'][3]) / 2.0 / h
                gx = min(int(cx * GRID), GRID - 1)
                gy = min(int(cy * GRID), GRID - 1)
                cell_z = out[gy, gx]
                cell_p_obj = torch.sigmoid(cell_z[:, 0]).numpy()
                cell_probs = torch.softmax(cell_z[:, 5:], dim=-1)[:, c_idx].numpy()
                cell_sc = cell_p_obj * cell_probs
                bs = np.argmax(cell_sc)
                pos_p_obj[c_name].append(float(cell_p_obj[bs]))
                pos_p_cls[c_name].append(float(cell_probs[bs]))
                pos_scores[c_name].append(float(cell_sc[bs]))

    ap50_results = {c: ap50_voc101(class_scores[c], class_hits[c], gt_counts[c]) for c in CLASSES}
    map50 = float(np.mean(list(ap50_results.values())))

    th_sweep = {}
    best_op = None
    best_f1 = -1.0
    for th in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.65, 0.80]:
        tot_tp = 0
        tot_fp = 0
        cls_breakdown = {}
        for c in CLASSES:
            sc = np.array(class_scores[c])
            hi = np.array(class_hits[c], dtype=bool)
            mask = sc >= th
            c_tp = int(np.sum(hi[mask]))
            c_fp = int(np.sum(~hi[mask]))
            c_fn = gt_counts[c] - c_tp
            c_p = c_tp / max(c_tp + c_fp, 1)
            c_r = c_tp / gt_counts[c] if gt_counts[c] > 0 else 0.0
            c_f1 = (2 * c_p * c_r / (c_p + c_r)) if (c_p + c_r) > 0 else 0.0
            cls_breakdown[c] = {'tp': c_tp, 'fp': c_fp, 'fn': c_fn, 'precision': c_p, 'recall': c_r, 'f1': c_f1}
            tot_tp += c_tp
            tot_fp += c_fp
        tot_fn = total_gt - tot_tp
        prec = tot_tp / max(tot_tp + tot_fp, 1)
        rec = tot_tp / total_gt
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        th_key = f"{th:.2f}"
        th_sweep[th_key] = {
            'tp': tot_tp, 'fp': tot_fp, 'fn': tot_fn,
            'precision': float(prec), 'recall': float(rec), 'micro_f1': float(f1),
            'classes': cls_breakdown
        }
        if f1 > best_f1:
            best_f1 = f1
            best_op = (th_key, th_sweep[th_key])

    op_th = float(best_op[0])

    conf_matrix = {c: {c2: 0 for c2 in list(CLASSES) + ['fn_missed']} for c in list(CLASSES) + ['background_fp']}
    recall_by_size = {s: {'objects': size_gt_counts[s], 'tp_at_op_th': 0, 'tp_at_025': 0} for s in ('small_<32px', 'medium_32-96px', 'large_>=96px')}
    recall_by_class_size = defaultdict(lambda: {'objects': 0, 'tp_at_op_th': 0, 'tp_at_025': 0})
    for c in CLASSES:
        for s in ('small_<32px', 'medium_32-96px', 'large_>=96px'):
            if size_class_gt_counts[c][s] > 0:
                recall_by_class_size[f"{c}_{s}"]['objects'] = size_class_gt_counts[c][s]

    provenance_metrics = {}
    for src in sorted(source_gt_counts.keys()):
        src_ap = {}
        for c in CLASSES:
            if source_gt_counts[src][c] > 0:
                src_ap[c] = {
                    'objects': source_gt_counts[src][c],
                    'ap50': ap50_voc101(source_scores[src][c], source_hits[src][c], source_gt_counts[src][c]),
                    'tp_at_op_th': int(np.sum(np.array(source_hits[src][c], dtype=bool)[np.array(source_scores[src][c]) >= op_th])) if source_scores[src][c] else 0,
                    'fp_at_op_th': int(np.sum(~np.array(source_hits[src][c], dtype=bool)[np.array(source_scores[src][c]) >= op_th])) if source_scores[src][c] else 0
                }
        provenance_metrics[src] = src_ap

    for nms_xy, nms_cls, nms_scores, gt_boxes, gt_classes, gt_sizes, src, img_id in per_image_nms_preds:
        if len(gt_boxes) > 0 and len(nms_xy) > 0:
            ov = overlaps(nms_xy, gt_boxes)
            used_25 = set()
            for p_i in np.argsort(-nms_scores, kind='stable'):
                if nms_scores[p_i] < 0.25: continue
                cp = nms_cls[p_i]
                elig = [g_i for g_i in range(len(gt_boxes)) if gt_classes[g_i] == cp and g_i not in used_25]
                vo, bg = max([(ov[p_i, g_i], g_i) for g_i in elig], default=(0.0, -1))
                if vo >= 0.5:
                    used_25.add(bg)
                    sb = gt_sizes[bg]
                    cn = CLASSES[gt_classes[bg]]
                    recall_by_size[sb]['tp_at_025'] += 1
                    recall_by_class_size[f"{cn}_{sb}"]['tp_at_025'] += 1

        mask_op = nms_scores >= op_th
        p_xy = nms_xy[mask_op]
        p_cls = nms_cls[mask_op]
        p_sc = nms_scores[mask_op]

        if len(gt_boxes) == 0:
            for cp in p_cls:
                conf_matrix['background_fp'][CLASSES[cp]] += 1
            continue

        if len(p_xy) == 0:
            for gc in gt_classes:
                conf_matrix[CLASSES[gc]]['fn_missed'] += 1
            continue

        ov = overlaps(p_xy, gt_boxes)
        matched_gt = set()
        matched_pred = set()

        for p_i in np.argsort(-p_sc, kind='stable'):
            cp = p_cls[p_i]
            elig = [g_i for g_i in range(len(gt_boxes)) if gt_classes[g_i] == cp and g_i not in matched_gt]
            vo, bg = max([(ov[p_i, g_i], g_i) for g_i in elig], default=(0.0, -1))
            if vo >= 0.5:
                matched_gt.add(bg)
                matched_pred.add(p_i)
                cn = CLASSES[cp]
                conf_matrix[cn][cn] += 1
                sb = gt_sizes[bg]
                recall_by_size[sb]['tp_at_op_th'] += 1
                recall_by_class_size[f"{cn}_{sb}"]['tp_at_op_th'] += 1

        for p_i in np.argsort(-p_sc, kind='stable'):
            if p_i in matched_pred: continue
            cp = p_cls[p_i]
            elig = [g_i for g_i in range(len(gt_boxes)) if g_i not in matched_gt]
            vo, bg = max([(ov[p_i, g_i], g_i) for g_i in elig], default=(0.0, -1))
            if vo >= 0.5:
                matched_gt.add(bg)
                matched_pred.add(p_i)
                true_cn = CLASSES[gt_classes[bg]]
                pred_cn = CLASSES[cp]
                conf_matrix[true_cn][pred_cn] += 1
            else:
                conf_matrix['background_fp'][CLASSES[cp]] += 1

        for g_i, gc in enumerate(gt_classes):
            if g_i not in matched_gt:
                conf_matrix[CLASSES[gc]]['fn_missed'] += 1

    for s in recall_by_size:
        n_s = max(recall_by_size[s]['objects'], 1)
        recall_by_size[s]['recall_at_op_th'] = recall_by_size[s]['tp_at_op_th'] / n_s
        recall_by_size[s]['recall_at_025'] = recall_by_size[s]['tp_at_025'] / n_s

    for k in recall_by_class_size:
        n_s = max(recall_by_class_size[k]['objects'], 1)
        recall_by_class_size[k]['recall_at_op_th'] = recall_by_class_size[k]['tp_at_op_th'] / n_s
        recall_by_class_size[k]['recall_at_025'] = recall_by_class_size[k]['tp_at_025'] / n_s

    eval_summary = {
        'variant': saved.get('config', {}).get('variant', 'Exp04'),
        'checkpoint': str(ckpt_path),
        'best_epoch': saved.get('epoch', 0),
        'val_loss': saved.get('val_loss', 0.0),
        'true_unthresholded_ap50': ap50_results,
        'true_unthresholded_map50': map50,
        'best_operating_point': {
            'threshold': best_op[0],
            'micro_f1': best_op[1]['micro_f1'],
            'precision': best_op[1]['precision'],
            'recall': best_op[1]['recall'],
            'tp': best_op[1]['tp'],
            'fp': best_op[1]['fp'],
            'fn': best_op[1]['fn'],
            'classes': best_op[1]['classes']
        },
        'metrics_at_th_035': th_sweep['0.35'],
        'threshold_sweep': th_sweep,
        'stages': stages,
        'recall_by_size': recall_by_size,
        'recall_by_class_and_size': dict(recall_by_class_size),
        'confusion_matrix_at_op_th': conf_matrix,
        'results_by_provenance': provenance_metrics,
        'positive_cell_distribution': {
            c: {
                'mean_p_obj': float(np.mean(pos_p_obj[c])),
                'mean_p_cls': float(np.mean(pos_p_cls[c])),
                'mean_score': float(np.mean(pos_scores[c]))
            } for c in CLASSES
        }
    }

    write_json(EXP04_DIR / out_json_name, eval_summary)
    print(f"\n[Exp04 Final Evaluation ({ckpt_name}, epoch {saved.get('epoch')})]:", flush=True)
    print(f"  True Unthresholded mAP50: {map50*100:.2f}%", flush=True)
    for c in CLASSES:
        print(f"    {c:10s}: AP50 = {ap50_results[c]*100:.2f}%", flush=True)
    print(f"  Best Operating Point: th={best_op[0]} -> Micro-F1 = {best_op[1]['micro_f1']*100:.2f}% (TP={best_op[1]['tp']}, FP={best_op[1]['fp']}, FN={best_op[1]['fn']})", flush=True)
    for c in CLASSES:
        cb = best_op[1]['classes'][c]
        print(f"    {c:10s}: TP={cb['tp']:3d} | FP={cb['fp']:4d} | FN={cb['fn']:3d} | Prec={cb['precision']*100:5.2f}% | Rec={cb['recall']*100:5.2f}% | F1={cb['f1']*100:5.2f}%", flush=True)
    return eval_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['train_4b', 'evaluate', 'eval_4a_bestloss', 'all_4b'], default='all_4b')
    parser.add_argument('--epochs', type=int, default=40)
    args = parser.parse_args()

    if args.action == 'eval_4a_bestloss':
        evaluate_exp04('variant_4a_broad/best_val_loss.pt', 'variant_4a_broad/evaluation_best_val_loss.json')
    elif args.action == 'train_4b':
        train_exp04_4b(epochs=args.epochs, batch_size=16, lr=0.001, seed=42)
    elif args.action == 'evaluate':
        evaluate_exp04('best.pt', 'evaluation.json')
    elif args.action == 'all_4b':
        evaluate_exp04('variant_4a_broad/best_val_loss.pt', 'variant_4a_broad/evaluation_best_val_loss.json')
        train_exp04_4b(epochs=args.epochs, batch_size=16, lr=0.001, seed=42)
        evaluate_exp04('best.pt', 'evaluation.json')
        evaluate_exp04('best_val_loss.pt', 'evaluation_best_val_loss.json')
