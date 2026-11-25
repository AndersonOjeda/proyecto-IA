"""Experiment 3C Runner: Corrected presence loss (Focal Loss) and calibrated class weighting.
Ensures:
- Exact reproducibility with seed 42.
- Clean weights initialized from scratch (no pre-trained models).
- Test set strictly held out (only train and val are used).
- Full 101-point VOC unthresholded AP50 and calibrated threshold evaluation.
"""
from pathlib import Path
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from residuos import CLASSES
from residuos.data import Store, read_json, write_json, digest
from residuos.detector import iou
from scripts.experiment3_runner import (
    DetectorS8, CachedDataset32, encode_grid32, decode_grid32,
    GRID, SLOTS, seed_all, device_for
)

CLASS_WEIGHTS_3C = torch.tensor([0.53, 1.39, 1.27, 1.32, 0.95, 0.53], dtype=torch.float32)

def loss_fn_3c(pred, target, cls_weights=CLASS_WEIGHTS_3C, gamma=2.0, alpha=0.75, obj_scale=1.5, box_weight=10.0):
    """Experiment 3C loss: Focal Loss for objectness, Smooth L1 for box, balanced CE for class."""
    positive = target[..., 0] == 1
    n_pos = max(float(positive.sum()), 1.0)
    
    # 1. Focal Loss for presence
    p = torch.sigmoid(pred[..., 0])
    pt = p * target[..., 0] + (1.0 - p) * (1.0 - target[..., 0])
    focal_weight = (1.0 - pt) ** gamma
    alpha_weight = alpha * target[..., 0] + (1.0 - alpha) * (1.0 - target[..., 0])
    bce = nn.functional.binary_cross_entropy_with_logits(pred[..., 0], target[..., 0], reduction='none')
    fl = alpha_weight * focal_weight * bce
    obj = (fl.sum() / n_pos) * obj_scale
    
    # 2. Box loss on positive cells
    zero = pred.sum() * 0
    box = nn.functional.smooth_l1_loss(pred[..., 1:5][positive].sigmoid(), target[..., 1:5][positive]) if positive.any() else zero
    
    # 3. Class loss on positive cells with soft bounded weights
    if positive.any():
        w = cls_weights.to(pred.device) if cls_weights is not None else None
        cls = nn.functional.cross_entropy(pred[..., 5:][positive], target[..., 5][positive].long(), weight=w)
    else:
        cls = zero
        
    return obj + box_weight * box + cls

def train_3c(epochs=40, batch_size=16, lr=0.001, seed=42):
    seed_all(seed)
    device = device_for('auto')
    print(f"\n================ STARTING EXPERIMENT 3C TRAINING ON {device} ================", flush=True)
    
    out_dir = ROOT / 'runs/detector_exp03/corrected'
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = out_dir / 'snapshot'
    snapshot_dir.mkdir(exist_ok=True)
    
    store = Store(ROOT / 'data/preparado/deteccion')
    rows = {s: store.records(s) for s in ('train', 'val')}
    
    # Snapshot manifest and eligible ids
    write_json(snapshot_dir / 'manifest.json', store.db)
    write_json(snapshot_dir / 'eligible_ids.json', [r['id'] for r in rows['train'] + rows['val']])
    
    npy_paths = {
        'train': ROOT / 'runs/detector_exp01/train.npy',
        'val': ROOT / 'runs/detector_exp01/val.npy'
    }
    
    trainset = CachedDataset32(rows['train'], npy_paths['train'], augment=True)
    valset = CachedDataset32(rows['val'], npy_paths['val'], augment=False)
    
    loaders = {
        'train': DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True),
        'val': DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    }
    
    config = {
        'variant': 'Exp3C',
        'description': 'DetectorS8 with Focal Loss presence (gamma=2, alpha=0.75) and soft bounded class weights',
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'seed': seed,
        'device': str(device),
        'grid': GRID,
        'slots': SLOTS,
        'class_weights': [float(x) for x in CLASS_WEIGHTS_3C],
        'focal_gamma': 2.0,
        'focal_alpha': 0.75,
        'obj_scale': 1.5,
        'box_weight': 10.0,
        'classes': list(CLASSES),
        'train_samples': len(trainset),
        'val_samples': len(valset)
    }
    write_json(out_dir / 'config.json', config)
    
    model = DetectorS8(num_classes=len(CLASSES), slots=SLOTS).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    
    best_loss = float('inf')
    history = []
    start_time = time.perf_counter()
    
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
                    loss = loss_fn_3c(out, y)
                    if not torch.isfinite(loss):
                        raise ValueError(f"Pérdida no finita en época {epoch+1}, split {split}")
                    
                    if is_train:
                        opt.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        opt.step()
                        
                total_loss += loss.item() * len(x)
                total_samples += len(x)
                
            row[f'{split}_loss'] = total_loss / total_samples
            
        if scheduler:
            scheduler.step()
            
        row['lr'] = opt.param_groups[0]['lr']
        row['seconds_elapsed'] = time.perf_counter() - start_time
        ep_duration = time.perf_counter() - ep_start
        history.append(row)
        write_json(out_dir / 'history.json', history)
        
        val_loss = row['val_loss']
        train_loss = row['train_loss']
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            torch.save({
                'origin': 'residuos-detector-exp03c',
                'epoch': epoch + 1,
                'val_loss': best_loss,
                'state': model.state_dict(),
                'config': config
            }, out_dir / 'best.pt')
            
        torch.save({
            'epoch': epoch + 1,
            'state': model.state_dict(),
            'config': config
        }, out_dir / 'last.pt')
        
        star = " ★ (MEJOR)" if is_best else ""
        print(f"[Exp3C] Época {epoch+1:02d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | {ep_duration:.2f}s{star}", flush=True)
        
    print(f"\n>>> ENTRENAMIENTO 3C COMPLETADO: mejor val_loss = {best_loss:.4f} <<<", flush=True)
    return history

def evaluate_3c():
    from scripts.audit_evaluation import ap50_voc101, overlaps, nms_indices, extract_candidates_generic
    device = device_for('auto')
    out_dir = ROOT / 'runs/detector_exp03/corrected'
    ckpt_path = out_dir / 'best.pt'
    print(f"\n================ EVALUATING EXPERIMENT 3C: {ckpt_path.name} ================", flush=True)
    
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
            
    # Collect predictions post-NMS
    class_scores = {c: [] for c in CLASSES}
    class_hits = {c: [] for c in CLASSES}
    gt_counts = Counter()
    for r in val_rows:
        for b in r['boxes']:
            gt_counts[b['class']] += 1
    total_gt = sum(gt_counts.values())
    
    pos_p_obj = {c: [] for c in CLASSES}
    pos_p_cls = {c: [] for c in CLASSES}
    pos_scores = {c: [] for c in CLASSES}
    neg_p_obj = []
    
    stages = {c: dict(objects=0, raw_any_class=0, raw_correct_class=0, final_correct=0) for c in CLASSES}
    size_breakdown = defaultdict(lambda: Counter())
    
    for i, r in enumerate(val_rows):
        out = outputs[i] # (32, 32, 3, 11)
        w, h = r['width'], r['height']
        xy, cls, scores, p_obj, p_cls = extract_candidates_generic(out, w, h, GRID)
        
        gt_boxes = [b['xyxy'] for b in r['boxes']]
        gt_classes = [CLASSES.index(b['class']) for b in r['boxes']]
        
        # NMS per image
        kept_idx = nms_indices(xy, cls, scores, threshold=0.45, max_detections=100)
        nms_xy = xy[kept_idx]
        nms_cls = cls[kept_idx]
        nms_scores = scores[kept_idx]
        
        used_gt = set()
        if len(gt_boxes) > 0 and len(nms_xy) > 0:
            ov = overlaps(nms_xy, gt_boxes)
            sort_order = np.argsort(-nms_scores, kind='stable')
            for p_i in sort_order:
                c_pred = nms_cls[p_i]
                c_name = CLASSES[c_pred]
                eligible = [g_i for g_i in range(len(gt_boxes)) if gt_classes[g_i] == c_pred and g_i not in used_gt]
                val_ov, best_g = max([(ov[p_i, g_i], g_i) for g_i in eligible], default=(0.0, -1))
                if val_ov >= 0.5:
                    used_gt.add(best_g)
                    class_scores[c_name].append(float(nms_scores[p_i]))
                    class_hits[c_name].append(True)
                else:
                    class_scores[c_name].append(float(nms_scores[p_i]))
                    class_hits[c_name].append(False)
        elif len(nms_xy) > 0:
            for p_i in range(len(nms_xy)):
                c_name = CLASSES[nms_cls[p_i]]
                class_scores[c_name].append(float(nms_scores[p_i]))
                class_hits[c_name].append(False)
                
        # Candidate stage auditing
        if len(gt_boxes) > 0:
            raw_ov = overlaps(xy, gt_boxes)
            for g_i, b in enumerate(r['boxes']):
                c_idx = CLASSES.index(b['class'])
                c_name = b['class']
                stages[c_name]['objects'] += 1
                has_geom = (raw_ov[:, g_i] >= 0.5).any()
                has_correct = ((raw_ov[:, g_i] >= 0.5) & (cls == c_idx)).any()
                if has_geom: stages[c_name]['raw_any_class'] += 1
                if has_correct: stages[c_name]['raw_correct_class'] += 1
                
                # Size classification
                w_norm = (b['xyxy'][2] - b['xyxy'][0]) * 256.0 / w
                h_norm = (b['xyxy'][3] - b['xyxy'][1]) * 256.0 / h
                area = w_norm * h_norm
                s_bin = 'small_<32px' if area < 32**2 else 'medium_32-96px' if area < 96**2 else 'large_>=96px'
                size_breakdown[(c_name, s_bin)]['objects'] += 1
                if has_geom: size_breakdown[(c_name, s_bin)]['geom'] += 1
                if has_correct: size_breakdown[(c_name, s_bin)]['correct'] += 1
                
                # Positive cell probability sampling
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

    # True Unthresholded AP50
    ap50_results = {}
    for c in CLASSES:
        ap50_results[c] = ap50_voc101(class_scores[c], class_hits[c], gt_counts[c])
    map50 = float(np.mean(list(ap50_results.values())))
    
    # Threshold sweep
    th_sweep = {}
    best_op = None
    best_f1 = -1.0
    for th in [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.65, 0.80]:
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
            c_r = c_tp / gt_counts[c]
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
            
    eval_summary = {
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
        'threshold_sweep': th_sweep,
        'stages': stages,
        'size_breakdown': {f"{k[0]}_{k[1]}": dict(v) for k, v in size_breakdown.items()},
        'positive_cell_distribution': {
            c: {
                'mean_p_obj': float(np.mean(pos_p_obj[c])),
                'mean_p_cls': float(np.mean(pos_p_cls[c])),
                'mean_score': float(np.mean(pos_scores[c]))
            } for c in CLASSES
        }
    }
    
    write_json(out_dir / 'evaluation.json', eval_summary)
    print(f"\n[Exp3C Final Evaluation]:", flush=True)
    print(f"  True Unthresholded mAP50: {map50*100:.2f}%", flush=True)
    for c in CLASSES:
        print(f"    {c:10s}: AP50 = {ap50_results[c]*100:.2f}%", flush=True)
    print(f"  Best Operating Point: th={best_op[0]} -> Micro-F1 = {best_op[1]['micro_f1']*100:.2f}% (TP={best_op[1]['tp']}, FP={best_op[1]['fp']})", flush=True)
    for c in CLASSES:
        cb = best_op[1]['classes'][c]
        print(f"    {c:10s}: TP={cb['tp']} | FP={cb['fp']} | FN={cb['fn']} | F1={cb['f1']*100:.2f}%", flush=True)
    print("Evaluation saved to runs/detector_exp03/corrected/evaluation.json", flush=True)
    return eval_summary

if __name__ == '__main__':
    train_3c(epochs=40, batch_size=16, lr=0.001, seed=42)
    evaluate_3c()
