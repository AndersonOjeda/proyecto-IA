"""Comprehensive audit of evaluation criteria, loss formulations, and score distributions
for Baseline Exp 1, Exp 3A, and Exp 3B on the validation set.
"""
from pathlib import Path
import sys
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from residuos import CLASSES
from residuos.data import Store, read_json
from residuos.detector import Detector, SIZE as SIZE16, GRID as GRID16
from scripts.experiment3_runner import DetectorS8, CachedDataset32, GRID as GRID32

def overlaps(a, b):
    a = np.asarray(a).reshape(-1, 4)
    b = np.asarray(b).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.maximum(hi - lo, 0).prod(2)
    areas = lambda x: np.maximum(x[:, 2:] - x[:, :2], 0).prod(1)
    return inter / np.maximum(areas(a)[:, None] + areas(b)[None, :] - inter, 1e-12)

def nms_indices(xy, classes, scores, threshold=0.45, max_detections=100):
    order = np.argsort(-scores, kind='stable')
    kept = []
    while len(order):
        k = order[0]
        kept.append(int(k))
        if len(kept) == max_detections:
            break
        order = order[1:]
        if not len(order):
            break
        suppress = (classes[order] == classes[k]) & (overlaps(xy[k:k+1], xy[order])[0] > threshold)
        order = order[~suppress]
    return np.array(kept, dtype=int)

def ap50_voc101(scores, hits, n_gt):
    if n_gt == 0 or len(scores) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores), kind='stable')
    tp = np.cumsum(np.asarray(hits, dtype=int)[order])
    fp = np.arange(1, len(order) + 1) - tp
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    interp_r = np.linspace(0, 1, 101)
    ap = float(np.mean([envelope[np.searchsorted(recall, r)] if np.searchsorted(recall, r) < len(envelope) else 0.0 for r in interp_r]))
    return ap

def extract_candidates_generic(z, width, height, grid):
    # z: (grid, grid, slots, 11)
    slots = z.shape[2]
    wh = torch.sigmoid(z[..., 1:5]).numpy().reshape(-1, 4)
    gy, gx, slot = np.indices((grid, grid, slots))
    gx, gy = gx.ravel(), gy.ravel()
    cx = (gx + wh[:, 0]) / grid
    cy = (gy + wh[:, 1]) / grid
    xy = np.stack([
        np.clip(cx - wh[:, 2] / 2, 0, 1) * width,
        np.clip(cy - wh[:, 3] / 2, 0, 1) * height,
        np.clip(cx + wh[:, 2] / 2, 0, 1) * width,
        np.clip(cy + wh[:, 3] / 2, 0, 1) * height
    ], axis=1)
    probs, cls = torch.softmax(z[..., 5:], dim=-1).max(dim=-1)
    p_obj = torch.sigmoid(z[..., 0]).numpy().ravel()
    p_cls = probs.numpy().ravel()
    scores = p_obj * p_cls
    return xy, cls.numpy().ravel(), scores, p_obj, p_cls

def audit_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Audit using device: {device}", flush=True)

    store = Store(ROOT / 'data/preparado/deteccion')
    val_rows = store.records('val')
    print(f"Validation images: {len(val_rows)}", flush=True)

    models_info = [
        {
            "name": "Baseline Exp 1",
            "path": ROOT / 'runs/detector_exp01/best.pt',
            "arch": "s16",
            "grid": 16,
            "cls_key": "state"
        },
        {
            "name": "Exp 3A (Stride 8)",
            "path": ROOT / 'runs/detector_exp03/architecture/best.pt',
            "arch": "s8",
            "grid": 32,
            "cls_key": "state"
        },
        {
            "name": "Exp 3B (Balance)",
            "path": ROOT / 'runs/detector_exp03/balance/best.pt',
            "arch": "s8",
            "grid": 32,
            "cls_key": "state"
        }
    ]

    results = {}

    for info in models_info:
        mname = info['name']
        print(f"\n================ AUDITING {mname} ================", flush=True)
        grid = info['grid']
        if info['arch'] == 's16':
            model = Detector().to(device)
            ckpt = torch.load(info['path'], map_location=device, weights_only=False)
            model.load_state_dict(ckpt.get('state', ckpt.get('state_dict')))
        else:
            model = DetectorS8().to(device)
            ckpt = torch.load(info['path'], map_location=device, weights_only=False)
            model.load_state_dict(ckpt.get('state', ckpt.get('state_dict')))
        model.eval()

        # We will collect:
        # 1. Unthresholded candidate predictions post-NMS (per-image top 100) -> for true standard AP50
        # 2. Score distributions on GT positive cells vs negative cells
        class_scores = {c: [] for c in CLASSES}
        class_hits = {c: [] for c in CLASSES}

        # Stats on positive cells
        pos_p_obj = {c: [] for c in CLASSES}
        pos_p_cls = {c: [] for c in CLASSES}
        pos_scores = {c: [] for c in CLASSES}
        neg_p_obj = []

        # GT objects count
        gt_objects_count = Counter()
        for r in val_rows:
            for b in r['boxes']:
                gt_objects_count[b['class']] += 1

        # Process each image
        total_geom_hits = 0
        total_geom_correct_hits = 0
        total_gt = sum(gt_objects_count.values())

        with torch.no_grad():
            for i, r in enumerate(val_rows):
                img_path = store.root / r['path']
                frame = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                h, w = r['height'], r['width']
                resized = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)
                rgb = torch.from_numpy(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                out = model(rgb.to(device)).cpu()[0] # (grid, grid, slots, 11)

                xy, cls, scores, p_obj, p_cls = extract_candidates_generic(out, w, h, grid)

                # Collect negative p_obj sample (downsampled to 50 random cells per image)
                # First mark cells near GT
                gt_boxes = [b['xyxy'] for b in r['boxes']]
                gt_classes = [CLASSES.index(b['class']) for b in r['boxes']]

                # NMS per image across all classes (or per-class NMS like diagnose_detector)
                # Notice: diagnose_detector uses class-aware NMS!
                kept_idx = nms_indices(xy, cls, scores, threshold=0.45, max_detections=100)
                nms_xy = xy[kept_idx]
                nms_cls = cls[kept_idx]
                nms_scores = scores[kept_idx]

                # Match NMS predictions to GT for AP50
                used_gt = set()
                if len(gt_boxes) > 0 and len(nms_xy) > 0:
                    ov = overlaps(nms_xy, gt_boxes)
                    # Sort predictions descending by score
                    sort_order = np.argsort(-nms_scores, kind='stable')
                    for p_i in sort_order:
                        c_pred = nms_cls[p_i]
                        c_name = CLASSES[c_pred]
                        # Find eligible GT
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

                # Positive cell statistics (examine the cell corresponding to each GT center)
                for b in r['boxes']:
                    c_idx = CLASSES.index(b['class'])
                    c_name = b['class']
                    cx = (b['xyxy'][0] + b['xyxy'][2]) / 2.0 / w
                    cy = (b['xyxy'][1] + b['xyxy'][3]) / 2.0 / h
                    gx = min(int(cx * grid), grid - 1)
                    gy = min(int(cy * grid), grid - 1)
                    # Look at all slots at this cell
                    cell_z = out[gy, gx] # (slots, 11)
                    cell_p_obj = torch.sigmoid(cell_z[:, 0]).numpy()
                    cell_probs = torch.softmax(cell_z[:, 5:], dim=-1)[:, c_idx].numpy()
                    cell_scores = cell_p_obj * cell_probs
                    best_slot = np.argmax(cell_scores)
                    pos_p_obj[c_name].append(float(cell_p_obj[best_slot]))
                    pos_p_cls[c_name].append(float(cell_probs[best_slot]))
                    pos_scores[c_name].append(float(cell_scores[best_slot]))

                # Check geometric coverage before NMS
                if len(gt_boxes) > 0:
                    raw_ov = overlaps(xy, gt_boxes)
                    for g_i, b in enumerate(r['boxes']):
                        c_idx = CLASSES.index(b['class'])
                        has_geom = (raw_ov[:, g_i] >= 0.5).any()
                        has_geom_correct = ((raw_ov[:, g_i] >= 0.5) & (cls == c_idx)).any()
                        if has_geom: total_geom_hits += 1
                        if has_geom_correct: total_geom_correct_hits += 1

                # Sample negative p_obj (cells far from GT)
                all_p_obj = p_obj.reshape(grid, grid, -1)
                # pick corners or center of image without GT
                sample_neg = [all_p_obj[gy, gx, 0] for gy in range(0, grid, 4) for gx in range(0, grid, 4)]
                neg_p_obj.extend(sample_neg)

        # 3. Compute True Standard AP50 (unthresholded)
        true_ap50 = {}
        for c in CLASSES:
            true_ap50[c] = ap50_voc101(class_scores[c], class_hits[c], gt_objects_count[c])
        true_map50 = float(np.mean(list(true_ap50.values())))

        # 4. Sweep thresholds to evaluate micro-F1 and thresholded AP50
        th_sweep = {}
        for th in [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.65, 0.80]:
            tot_tp = 0
            tot_fp = 0
            for c in CLASSES:
                sc = np.array(class_scores[c])
                hi = np.array(class_hits[c], dtype=bool)
                mask = sc >= th
                tot_tp += int(np.sum(hi[mask]))
                tot_fp += int(np.sum(~hi[mask]))
            tot_fn = total_gt - tot_tp
            prec = tot_tp / max(tot_tp + tot_fp, 1)
            rec = tot_tp / total_gt
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            th_sweep[f"{th:.2f}"] = {
                "tp": tot_tp,
                "fp": tot_fp,
                "fn": tot_fn,
                "precision": float(prec),
                "recall": float(rec),
                "micro_f1": float(f1)
            }

        # 5. Summarize score distributions on GT positive cells
        pos_dist = {}
        for c in CLASSES:
            pos_dist[c] = {
                "n": len(pos_scores[c]),
                "mean_p_obj": float(np.mean(pos_p_obj[c])) if pos_p_obj[c] else 0.0,
                "median_p_obj": float(np.median(pos_p_obj[c])) if pos_p_obj[c] else 0.0,
                "mean_p_cls": float(np.mean(pos_p_cls[c])) if pos_p_cls[c] else 0.0,
                "median_p_cls": float(np.median(pos_p_cls[c])) if pos_p_cls[c] else 0.0,
                "mean_score": float(np.mean(pos_scores[c])) if pos_scores[c] else 0.0,
                "median_score": float(np.median(pos_scores[c])) if pos_scores[c] else 0.0,
                "pct_score_ge_010": float(np.mean(np.array(pos_scores[c]) >= 0.10)) if pos_scores[c] else 0.0,
                "pct_score_ge_025": float(np.mean(np.array(pos_scores[c]) >= 0.25)) if pos_scores[c] else 0.0,
                "pct_score_ge_050": float(np.mean(np.array(pos_scores[c]) >= 0.50)) if pos_scores[c] else 0.0,
            }

        neg_summary = {
            "mean_p_obj": float(np.mean(neg_p_obj)),
            "median_p_obj": float(np.median(neg_p_obj)),
            "max_p_obj": float(np.max(neg_p_obj)),
            "pct_p_obj_ge_010": float(np.mean(np.array(neg_p_obj) >= 0.10)),
            "pct_p_obj_ge_025": float(np.mean(np.array(neg_p_obj) >= 0.25))
        }

        results[mname] = {
            "true_unthresholded_ap50": true_ap50,
            "true_unthresholded_map50": true_map50,
            "geom_coverage": f"{total_geom_hits} / {total_gt} ({total_geom_hits/total_gt*100:.2f}%)",
            "geom_correct_class_coverage": f"{total_geom_correct_hits} / {total_gt} ({total_geom_correct_hits/total_gt*100:.2f}%)",
            "threshold_sweep": th_sweep,
            "positive_cell_distribution": pos_dist,
            "negative_cell_distribution": neg_summary,
            "total_candidates_evaluated": sum(len(class_scores[c]) for c in CLASSES)
        }

        print(f"Results for {mname}:", flush=True)
        print(f"  True Unthresholded mAP50: {true_map50*100:.2f}%", flush=True)
        for c in CLASSES:
            print(f"    {c}: AP50 = {true_ap50[c]*100:.2f}% | mean p_obj: {pos_dist[c]['mean_p_obj']:.3f} | mean p_cls: {pos_dist[c]['mean_p_cls']:.3f} | mean score: {pos_dist[c]['mean_score']:.3f}", flush=True)
        best_f1_th = max(th_sweep.items(), key=lambda kv: kv[1]['micro_f1'])
        print(f"  Best Operating Point: th={best_f1_th[0]} -> Micro-F1={best_f1_th[1]['micro_f1']*100:.2f}% (TP={best_f1_th[1]['tp']}, FP={best_f1_th[1]['fp']})", flush=True)

    out_file = ROOT / 'runs/detector_exp03/audit_fase1_fase2.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nAudit completed. Results saved to {out_file}", flush=True)

if __name__ == '__main__':
    audit_models()
