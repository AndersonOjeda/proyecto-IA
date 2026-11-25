"""Entrenamiento del detector multiobjeto desde cero integrando GARBAGE CLASSIFICATION 3 (Roboflow).
Total: 12.442 imágenes de entrenamiento (3.645 base + 1.504 exterior SALSA/TACO + 7.293 GC3).
CERO modelos ni pesos preentrenados.
Evaluación rigurosa sobre 670 imágenes reales de validación con IoU >= 0.5.
"""
import copy
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from residuos import CLASSES
from residuos.data import digest, write_json
from residuos.detector import iou
from scripts.experiment3_runner import (
    DetectorS8, CachedDataset32, encode_grid32, decode_grid32,
    GRID, SLOTS, seed_all, device_for
)
from scripts.experiment3c_runner import loss_fn_3c

OUT_DIR = ROOT / 'runs/detector_gc3_integration'
OUT_DIR.mkdir(parents=True, exist_ok=True)

class ModelEMA:
    """Exponential Moving Average de pesos del modelo."""
    def __init__(self, model, decay=0.996):
        self.ema = copy.deepcopy(model).eval()
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

class UnifiedDataset(Dataset):
    """Dataset unificado multifuente en memoria mapeada (12.442 imágenes)."""
    def __init__(self, base_rows, base_npy, ext_rows, ext_npy, gc3_rows, gc3_npy, augment=True):
        self.base_rows = base_rows
        self.ext_rows = ext_rows
        self.gc3_rows = gc3_rows
        self.base_pixels = np.load(base_npy, mmap_mode='r')
        self.ext_pixels = np.load(ext_npy, mmap_mode='r')
        self.gc3_pixels = np.load(gc3_npy, mmap_mode='r')
        self.augment = augment
        
        self.samples = (
            [('base', i) for i in range(len(base_rows))] +
            [('ext', j) for j in range(len(ext_rows))] +
            [('gc3', k) for k in range(len(gc3_rows))]
        )
        print(f"UnifiedDataset: {len(self.samples)} imágenes en total ({len(base_rows)} base + {len(ext_rows)} exterior + {len(gc3_rows)} GC3)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src, real_idx = self.samples[idx]
        if src == 'base':
            r = self.base_rows[real_idx]
            frame = np.array(self.base_pixels[real_idx], copy=True)
        elif src == 'ext':
            r = self.ext_rows[real_idx]
            frame = np.array(self.ext_pixels[real_idx], copy=True)
        else:
            r = self.gc3_rows[real_idx]
            frame = np.array(self.gc3_pixels[real_idx], copy=True)
            
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
        target = encode_grid32(boxes, w_orig, h_orig)
        return x_tensor, target

def compute_voc_ap(recalls, precisions):
    """Calcula AP VOC de 101 puntos."""
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 101):
        prec_t = [p for r, p in zip(recalls, precisions) if r >= t]
        p_val = max(prec_t) if prec_t else 0.0
        ap += p_val / 101.0
    return ap

def evaluate_val_full(model, val_loader, val_rows, device):
    """Evalúa predicciones y métricas en validación sin y con umbrales."""
    model.eval()
    all_raw_preds = []
    
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(val_loader):
            x = x.to(device)
            out = model(x)
            for i in range(len(x)):
                row_idx = batch_idx * val_loader.batch_size + i
                w = val_rows[row_idx]['width']
                h = val_rows[row_idx]['height']
                boxes = decode_grid32(out[i], w, h, threshold=0.01, nms=0.45)
                all_raw_preds.append((boxes, val_rows[row_idx]['boxes']))
                
    # Barrido de umbrales
    thresholds = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]
    sweep_results = {}
    
    for th in thresholds:
        tp_total, fp_total, fn_total = 0, 0, 0
        cls_stats = {c: {'tp': 0, 'fp': 0, 'fn': 0} for c in CLASSES}
        
        for preds, truths in all_raw_preds:
            filt_preds = [b for b in preds if b['score'] >= th]
            matched_gt = set()
            
            # Ordenar por score descendente
            for pb in sorted(filt_preds, key=lambda b: b['score'], reverse=True):
                best_iou = 0.0
                best_gt_idx = -1
                for gt_idx, gt in enumerate(truths):
                    if gt_idx in matched_gt or gt['class'] != pb['class']:
                        continue
                    ov = iou(pb['xyxy'], gt['xyxy'])
                    if ov > best_iou:
                        best_iou = ov
                        best_gt_idx = gt_idx
                        
                if best_iou >= 0.5:
                    matched_gt.add(best_gt_idx)
                    cls_stats[pb['class']]['tp'] += 1
                    tp_total += 1
                else:
                    cls_stats[pb['class']]['fp'] += 1
                    fp_total += 1
                    
            for gt_idx, gt in enumerate(truths):
                if gt_idx not in matched_gt:
                    cls_stats[gt['class']]['fn'] += 1
                    fn_total += 1
                    
        p_glob = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        r_glob = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        f1_glob = 2 * p_glob * r_glob / (p_glob + r_glob) if (p_glob + r_glob) > 0 else 0.0
        
        per_class = {}
        for c, st in cls_stats.items():
            p_c = st['tp'] / (st['tp'] + st['fp']) if (st['tp'] + st['fp']) > 0 else 0.0
            r_c = st['tp'] / (st['tp'] + st['fn']) if (st['tp'] + st['fn']) > 0 else 0.0
            f1_c = 2 * p_c * r_c / (p_c + r_c) if (p_c + r_c) > 0 else 0.0
            per_class[c] = {'tp': st['tp'], 'fp': st['fp'], 'fn': st['fn'], 'precision': p_c, 'recall': r_c, 'f1': f1_c}
            
        sweep_results[f"{th:.2f}"] = {
            'threshold': th,
            'precision': p_glob,
            'recall': r_glob,
            'micro_f1': f1_glob,
            'tp': tp_total,
            'fp': fp_total,
            'fn': fn_total,
            'classes': per_class
        }
        
    # Calcular AP50 VOC 101 pts
    class_aps = {}
    for c in CLASSES:
        pts = [(sweep_results[f"{th:.2f}"]['classes'][c]['recall'], sweep_results[f"{th:.2f}"]['classes'][c]['precision']) for th in thresholds]
        pts = sorted(pts, key=lambda kv: kv[0])
        recalls = [pt[0] for pt in pts]
        precisions = [pt[1] for pt in pts]
        class_aps[c] = compute_voc_ap(recalls, precisions)
        
    map50 = sum(class_aps.values()) / len(class_aps)
    return map50, class_aps, sweep_results

def main():
    seed_all(42)
    device = device_for('auto')
    print("="*75, flush=True)
    print("  ENTRENAMIENTO MASIVO UNIFICADO DESDE CERO CON ROBFLOW GC3", flush=True)
    print("="*75, flush=True)
    print(f"Dispositivo activo: {device}", flush=True)
    
    # 1. Cargar metadatos
    orig_train_rows = json.loads((ROOT / 'data/preparado/deteccion/manifest.json').read_text(encoding='utf-8'))['images']
    orig_train_rows = [r for r in orig_train_rows if r['split'] == 'train' and not r.get('id', '').startswith(('salsa', 'taco_ext', 'gc3'))][:3645]
    from residuos.data import Store
    store = Store(ROOT / 'data/preparado/deteccion')
    val_rows = store.records('val')
    
    ext_rows = json.loads((ROOT / 'runs/detector_exp05/manifiesto_descargas.json').read_text(encoding='utf-8'))['accepted_images']
    gc3_rows = json.loads((OUT_DIR / 'gc3_clean_train.json').read_text(encoding='utf-8'))
    
    base_npy = ROOT / 'runs/detector_exp01/train.npy'
    ext_npy = ROOT / 'runs/detector_exp05/ext_train.npy'
    gc3_npy = OUT_DIR / 'gc3_train.npy'
    val_npy = ROOT / 'runs/detector_exp01/val.npy'
    
    trainset = UnifiedDataset(orig_train_rows, base_npy, ext_rows, ext_npy, gc3_rows, gc3_npy, augment=True)
    valset = CachedDataset32(val_rows, val_npy, augment=False)
    
    train_loader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=32, shuffle=False, num_workers=0)
    
    model = DetectorS8().to(device)
    ema = ModelEMA(model, decay=0.996)
    
    # Pesos de clase balanceados
    cls_weights = torch.tensor([0.70, 1.20, 1.20, 1.10, 1.00, 0.60], dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    epochs = 20
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    print(f"\nIniciando entrenamiento por {epochs} épocas...", flush=True)
    best_loss = float('inf')
    history = []
    t_start = time.perf_counter()
    
    for epoch in range(epochs):
        t_ep = time.perf_counter()
        model.train()
        train_loss = 0.0
        
        for b_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn_3c(pred, y, cls_weights=cls_weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            ema.update(model)
            train_loss += loss.item() * len(x)
            
            if (b_idx + 1) % 100 == 0 or (b_idx + 1) == len(train_loader):
                print(f"  [Época {epoch+1:02d}/{epochs:02d}] Lote {b_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}", flush=True)
                
        train_loss /= len(trainset)
        scheduler.step()
        
        # Validación
        ema.ema.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = ema.ema(x)
                loss = loss_fn_3c(pred, y, cls_weights=cls_weights)
                val_loss += loss.item() * len(x)
        val_loss /= len(valset)
        
        dt_ep = time.perf_counter() - t_ep
        row = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': float(scheduler.get_last_lr()[0]),
            'epoch_seconds': dt_ep
        }
        history.append(row)
        write_json(OUT_DIR / 'history.json', history)
        
        print(f"--> Época {epoch+1:02d} completada en {dt_ep:.1f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}", flush=True)
        
        if val_loss < best_loss:
            best_loss = val_loss
            payload = {
                'origin': 'residuos-detector-exp05-outdoor-from-scratch',
                'epoch': epoch + 1,
                'val_loss': best_loss,
                'classes': list(CLASSES),
                'state': ema.ema.state_dict(),
                'config': {
                    'architecture': 'DetectorS8',
                    'total_train_images': len(trainset),
                    'epochs': epochs,
                    'batch_size': 32
                }
            }
            torch.save(payload, OUT_DIR / 'best.pt')
            print(f"    [*] Nuevo mejor checkpoint guardado en {OUT_DIR / 'best.pt'} (val_loss: {best_loss:.4f})", flush=True)
            
    total_time = time.perf_counter() - t_start
    print(f"\nEntrenamiento completado en {total_time/60:.2f} minutos. Mejor val_loss: {best_loss:.4f}", flush=True)
    
    # 2. Evaluación detallada y calibración de umbrales
    print("\nEjecutando calibración y cálculo de AP50 VOC sobre las 670 imágenes de validación...", flush=True)
    best_model = DetectorS8().to(device)
    best_cp = torch.load(OUT_DIR / 'best.pt', map_location=device, weights_only=False)
    best_model.load_state_dict(best_cp['state'])
    
    map50, class_aps, sweep = evaluate_val_full(best_model, val_loader, val_rows, device)
    
    best_f1_th = max(sweep.values(), key=lambda r: r['micro_f1'])
    
    eval_summary = {
        'total_train_images': len(trainset),
        'best_epoch': best_cp['epoch'],
        'val_loss': best_loss,
        'true_unthresholded_map50': map50,
        'true_unthresholded_ap50': class_aps,
        'best_operating_point': best_f1_th,
        'metrics_at_th_035': sweep.get('0.35'),
        'metrics_at_th_040': sweep.get('0.40'),
        'threshold_sweep': sweep
    }
    write_json(OUT_DIR / 'evaluation.json', eval_summary)
    
    threshold_payload = {
        'threshold': best_f1_th['threshold'],
        'checkpoint': str(OUT_DIR / 'best.pt'),
        'checkpoint_sha256': digest(OUT_DIR / 'best.pt'),
        'selected_operating_point': best_f1_th,
        'origin': 'residuos-detector-exp05-outdoor-from-scratch'
    }
    write_json(OUT_DIR / 'threshold.json', threshold_payload)
    
    print("\n" + "="*75, flush=True)
    print("RESUMEN DE EVALUACIÓN FINAL TRAS INTEGRAR GARBAGE CLASSIFICATION 3:", flush=True)
    print("="*75, flush=True)
    print(f"mAP50 Global: {map50*100:.2f}%", flush=True)
    for c, ap in class_aps.items():
        print(f"  - {c:<12}: {ap*100:6.2f}%", flush=True)
    print(f"\nMejor Umbral Calibrado: {best_f1_th['threshold']:.2f}", flush=True)
    print(f"  Micro-F1 : {best_f1_th['micro_f1']*100:.2f}%", flush=True)
    print(f"  Precision: {best_f1_th['precision']*100:.2f}%", flush=True)
    print(f"  Recall   : {best_f1_th['recall']*100:.2f}% (TP={best_f1_th['tp']}, FP={best_f1_th['fp']}, FN={best_f1_th['fn']})", flush=True)
    print("="*75, flush=True)

if __name__ == '__main__':
    main()
