"""Experiment 6 Runner: DetectorS8 entrenado 100% desde cero incorporando las 52 fotos manuales.
Combina:
1. Base Train original: 3.645 imágenes
2. Ampliación Exterior SALSA + TACO: 1.505 imágenes
3. Lote Manual Humano revisado: 52 imágenes (vidrio, cartón, metal, papel)
Total entrenamiento: 5.202 imágenes.
Evaluación: sobre las 670 imágenes canónicas de validación bajo protocolo idéntico.
CERO modelos o pesos preentrenados (100% random weights).
"""
import os
import sys
import time
import json
import random
from pathlib import Path
from copy import deepcopy
from collections import Counter, defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from residuos import CLASSES
from residuos.data import Store
from scripts.experiment3_runner import (
    DetectorS8, CachedDataset32, encode_grid32, decode_grid32,
    GRID, SLOTS, seed_all, device_for
)
from scripts.experiment3c_runner import loss_fn_3c, CLASS_WEIGHTS_3C

EXP06_DIR = ROOT / 'runs/detector_exp06'
EXP06_DIR.mkdir(parents=True, exist_ok=True)

class ModelEMA:
    """Exponential Moving Average de los parámetros del modelo."""
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

class DatasetExp06(Dataset):
    """Dataset integrado para Exp06: 3.645 base + 1.505 externas + 52 manuales humanas."""
    def __init__(self, orig_rows, orig_npy_path, ext_rows, ext_npy_path, manual_rows, data_root, augment=True):
        self.orig_rows = orig_rows
        self.ext_rows = ext_rows
        self.manual_rows = manual_rows
        self.orig_pixels = np.load(orig_npy_path, mmap_mode='r')
        self.ext_pixels = np.load(ext_npy_path, mmap_mode='r')
        self.augment = augment

        # Pre-cargar en memoria RAM las 52 imágenes manuales
        print(f"Pre-cargando {len(manual_rows)} imágenes manuales humanas en RAM...")
        manual_list = []
        for r in manual_rows:
            p = data_root / r['path']
            img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"No se pudo leer la imagen: {p}")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_res = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
            manual_list.append(img_res)
        self.manual_pixels = np.stack(manual_list, axis=0) # [N, 256, 256, 3]

        self.samples = (
            [('orig', i) for i in range(len(self.orig_rows))] +
            [('ext', j) for j in range(len(self.ext_rows))] +
            [('manual', k) for k in range(len(self.manual_rows))]
        )
        print(f"DatasetExp06: Total muestras = {len(self.samples)} ({len(orig_rows)} base + {len(ext_rows)} ext + {len(manual_rows)} manuales humanas)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        kind, real_idx = self.samples[idx]
        if kind == 'orig':
            r = self.orig_rows[real_idx]
            frame = np.array(self.orig_pixels[real_idx], copy=True)
        elif kind == 'ext':
            r = self.ext_rows[real_idx]
            frame = np.array(self.ext_pixels[real_idx], copy=True)
        else:
            r = self.manual_rows[real_idx]
            frame = np.array(self.manual_pixels[real_idx], copy=True)

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
    """Evaluación rápida de mAP50 durante entrenamiento."""
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

    return float(np.mean(list(ap50.values()))), ap50

def evaluate_on_val_full(ckpt_path, val_rows, val_npy, device, out_json_path):
    """Evaluación completa exhaustiva en validación con barrido de umbrales y calibración."""
    print(f"\nEjecutando evaluación completa en validación para {ckpt_path}...")
    saved = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = saved['model'] if 'model' in saved else saved
    model = DetectorS8()
    model.load_state_dict(state)
    model.to(device)
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

    thresholds = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
    sweep = {}
    best_f1 = -1.0
    best_op = {}

    for th in thresholds:
        th_key = f"{th:.2f}"
        metrics = {c: {'tp': 0, 'fp': 0, 'fn': 0} for c in CLASSES}
        for i, r in enumerate(val_rows):
            w, h = r['width'], r['height']
            preds = decode_grid32(outputs[i], w, h, threshold=th, nms=0.45)
            gt_boxes = list(r['boxes'])
            matched = set()
            for p in preds:
                best_iou = 0.0
                best_gi = -1
                for gi, g in enumerate(gt_boxes):
                    if g['class'] == p['class'] and gi not in matched:
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
                    metrics[p['class']]['tp'] += 1
                    matched.add(best_gi)
                else:
                    metrics[p['class']]['fp'] += 1
            for gi, g in enumerate(gt_boxes):
                if gi not in matched:
                    metrics[g['class']]['fn'] += 1

        tot_tp = sum(m['tp'] for m in metrics.values())
        tot_fp = sum(m['fp'] for m in metrics.values())
        tot_fn = sum(m['fn'] for m in metrics.values())
        prec = tot_tp / max(tot_tp + tot_fp, 1)
        rec = tot_tp / max(tot_tp + tot_fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-6)

        class_details = {}
        for c in CLASSES:
            c_tp = metrics[c]['tp']
            c_fp = metrics[c]['fp']
            c_fn = metrics[c]['fn']
            c_prec = c_tp / max(c_tp + c_fp, 1)
            c_rec = c_tp / max(c_tp + c_fn, 1)
            c_f1 = 2 * c_prec * c_rec / max(c_prec + c_rec, 1e-6)
            class_details[c] = {
                'tp': c_tp, 'fp': c_fp, 'fn': c_fn,
                'precision': c_prec, 'recall': c_rec, 'f1': c_f1
            }

        sweep[th_key] = {
            'threshold': th, 'tp': tot_tp, 'fp': tot_fp, 'fn': tot_fn,
            'precision': prec, 'recall': rec, 'micro_f1': f1,
            'classes': class_details
        }

        if f1 > best_f1:
            best_f1 = f1
            best_op = sweep[th_key]

    eval_summary = {
        'variant': 'Exp06_ManualBatchIntegrated',
        'checkpoint': str(ckpt_path),
        'best_epoch': saved.get('epoch'),
        'val_loss': saved.get('val_loss'),
        'true_unthresholded_map50': mAP50,
        'true_unthresholded_ap50': ap50,
        'best_operating_point': best_op,
        'metrics_at_th_035': sweep.get('0.35', {}),
        'metrics_at_th_040': sweep.get('0.40', {}),
        'threshold_sweep': sweep
    }

    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(eval_summary, f, indent=2, ensure_ascii=False)

    print(f"\n[Exp06 Evaluación]:")
    print(f"  mAP50: {mAP50*100:.2f}%")
    for c in CLASSES:
        print(f"    {c:10s}: AP50 = {ap50[c]*100:.2f}%")
    print(f"  Mejor Umbral: th={best_op['threshold']:.2f} -> Micro-F1 = {best_op['micro_f1']*100:.2f}% (TP={best_op['tp']}, FP={best_op['fp']})")
    return eval_summary

def train_exp06(epochs=25, batch_size=16, lr=0.001, seed=42):
    """Entrenamiento completo de Exp06 desde cero con pesos aleatorios."""
    seed_all(seed)
    device = device_for('auto')
    print(f"\n================ INICIANDO ENTRENAMIENTO EXPERIMENTO 6 EN {device} ================", flush=True)

    data_root = ROOT / 'data/preparado/deteccion'
    store = Store(data_root)
    
    # Cargar los 670 registros canónicos de validación que corresponden a val.npy
    with open(ROOT / 'runs/detector_exp01/snapshot/manifest.json', encoding='utf-8') as f:
        m01 = json.load(f)
    orig_val = [r for r in m01['images'] if r.get('split') == 'val' and r.get('reviewed', True)]
    assert len(orig_val) == 670, f"Esperado 670 orig val, recibido {len(orig_val)}"
    
    # Cargar las particiones exactas de base (3.645) y ext (1.505) de Exp05
    with open(ROOT / 'runs/detector_exp05/manifest_exp05.json', encoding='utf-8') as f:
        m05 = json.load(f)
    orig_train = m05['orig_train_rows']
    ext_rows = m05['ext_train_rows']
    assert len(orig_train) == 3645, f"Esperado 3645 base train, recibido {len(orig_train)}"
    assert len(ext_rows) == 1505, f"Esperado 1505 ext train, recibido {len(ext_rows)}"

    # Cargar las 52 fotos manuales de entrenamiento
    with open(data_root / 'manifest.json', encoding='utf-8') as f:
        full_m = json.load(f)
    manual_train = [r for r in full_m['images'] if r.get('manual_batch') and r.get('reviewed') and r.get('split') == 'train']
    print(f"Base train: {len(orig_train)} | Ext train: {len(ext_rows)} | Manuales revisadas: {len(manual_train)}")
    print(f"Total imágenes en entrenamiento Exp06: {len(orig_train) + len(ext_rows) + len(manual_train)}")

    trainset = DatasetExp06(
        orig_rows=orig_train,
        orig_npy_path=ROOT / 'runs/detector_exp01/train.npy',
        ext_rows=ext_rows,
        ext_npy_path=ROOT / 'runs/detector_exp05/ext_train.npy',
        manual_rows=manual_train,
        data_root=data_root,
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
        'variant': 'Exp06_ManualBatchIntegrated',
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
        'manual_train_images': len(manual_train),
        'val_images': len(orig_val)
    }
    with open(EXP06_DIR / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    history = []
    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        model.train()
        train_loss = 0.0
        train_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
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

        # Pérdida de validación con EMA
        ema.ema.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = ema.ema(x)
                loss = loss_fn_3c(pred, y)
                val_loss += loss.item()
                val_batches += 1

        val_loss /= max(val_batches, 1)
        scheduler.step()

        ep_time = time.time() - ep_start
        tot_time = time.time() - start_time
        curr_lr = opt.param_groups[0]['lr']

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': curr_lr,
            'epoch_seconds': ep_time,
            'total_seconds': tot_time
        })

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save({
                'model': ema.ema.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'config': config,
                'origin': 'residuos-from-scratch-v1'
            }, EXP06_DIR / 'best.pt')

        torch.save({
            'model': ema.ema.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
            'config': config,
            'origin': 'residuos-from-scratch-v1'
        }, EXP06_DIR / 'last.pt')

        print(f"Ep {epoch:02d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} {'[BEST]' if is_best else '      '} | LR: {curr_lr:.6f} | {ep_time:.1f}s", flush=True)

    with open(EXP06_DIR / 'history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    # Evaluación completa final
    print("\nEvaluando modelo Exp06 en validación canónica...")
    eval_res = evaluate_on_val_full(
        EXP06_DIR / 'best.pt',
        orig_val,
        val_npy,
        device,
        EXP06_DIR / 'evaluation.json'
    )

    # Generar threshold.json de Exp06
    import hashlib
    with open(EXP06_DIR / 'best.pt', 'rb') as f:
        ckpt_sha = hashlib.sha256(f.read()).hexdigest()

    th_data = {
        "checkpoint": str(EXP06_DIR / 'best.pt'),
        "checkpoint_sha256": ckpt_sha,
        "epoch": eval_res.get('best_epoch'),
        "threshold": eval_res['best_operating_point']['threshold'],
        "micro_f1": eval_res['best_operating_point']['micro_f1'],
        "modes": {
            "cobertura": {
                "threshold": 0.35,
                "min_box_size": 0,
                "description": "Modo máxima cobertura"
            },
            "precision": {
                "threshold": 0.40,
                "min_box_size": 32,
                "description": "Modo alta precisión"
            }
        }
    }
    with open(EXP06_DIR / 'threshold.json', 'w', encoding='utf-8') as f:
        json.dump(th_data, f, indent=2)

    print("\n¡Entrenamiento y evaluación de Experimento 6 completados con éxito!")
    return eval_res

if __name__ == '__main__':
    train_exp06(epochs=25, batch_size=16, lr=0.001, seed=42)
