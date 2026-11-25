"""Calibración de umbrales por clase para Exp07 sobre las 670 imágenes canónicas de validación."""
import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from residuos import CLASSES
from scripts.experiment3_runner import DetectorS8, decode_grid32, device_for
from scripts.experiment7_runner import load_exp07_splits, sanear_cajas

def run_calibration(exp_dir_name='runs/detector_exp07'):
    device = device_for('auto')
    orig_val, _, _, _, _, _ = load_exp07_splits()
    val_npy = ROOT / 'runs/detector_exp01/val.npy'
    val_imgs = np.load(val_npy)  # (670, 256, 256, 3) uint8

    exp_dir = ROOT / exp_dir_name
    ckpt_path = exp_dir / 'best.pt'
    saved = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = DetectorS8().to(device)
    model.load_state_dict(saved['model'])
    model.eval()

    print(f"\n================ CALIBRACIÓN PER-CLASE PARA {exp_dir.name} EN {device} ================")
    print(f"Cargadas {len(orig_val)} imágenes de validación...")

    # Extraer todas las predicciones candidatas con umbral bajo (th=0.05)
    all_candidates = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(val_imgs), batch_size):
            batch_np = val_imgs[i:i+batch_size]
            batch_t = torch.from_numpy(batch_np).permute(0, 3, 1, 2).float().to(device) / 255.0
            preds = model(batch_t)
            for j in range(len(batch_t)):
                img_idx = i + j
                meta = orig_val[img_idx]
                w, h = meta.get('width', 256), meta.get('height', 256)
                # decode_grid32 at low threshold 0.05
                cands = decode_grid32(preds[j], w, h, threshold=0.05, nms=0.45)
                # Filter degenerate
                cands = [b for b in cands if b['xyxy'][2] > b['xyxy'][0] and b['xyxy'][3] > b['xyxy'][1]]
                all_candidates.append(cands)

    # Extraer ground truth saneado
    all_gt = []
    for meta in orig_val:
        w, h = meta.get('width', 256), meta.get('height', 256)
        clean_gt = sanear_cajas(meta.get('boxes', []), w, h)
        all_gt.append(clean_gt)

    def evaluate_thresholds(th_map):
        # th_map can be a float or dict {class: th}
        metrics = {c: {'tp': 0, 'fp': 0, 'fn': 0} for c in CLASSES}
        for cands, gt in zip(all_candidates, all_gt):
            for c in CLASSES:
                c_th = th_map[c] if isinstance(th_map, dict) else th_map
                pr = sorted([b for b in cands if b['class'] == c and b['score'] >= c_th], key=lambda x: x['score'], reverse=True)
                gt_c = [g for g in gt if g['class'] == c]
                matched = set()
                for p in pr:
                    best_iou = 0.0
                    best_gi = -1
                    px1, py1, px2, py2 = p['xyxy']
                    for gi, g in enumerate(gt_c):
                        if gi in matched:
                            continue
                        gx1, gy1, gx2, gy2 = g['xyxy']
                        ix1, iy1 = max(gx1, px1), max(gy1, py1)
                        ix2, iy2 = min(gx2, px2), min(gy2, py2)
                        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                        union = (gx2 - gx1)*(gy2 - gy1) + (px2 - px1)*(py2 - py1) - inter
                        iou = inter / max(union, 1e-6)
                        if iou > best_iou:
                            best_iou = iou
                            best_gi = gi
                    if best_iou >= 0.5:
                        metrics[c]['tp'] += 1
                        matched.add(best_gi)
                    else:
                        metrics[c]['fp'] += 1
                metrics[c]['fn'] += len(gt_c) - len(matched)

        tot_tp = sum(m['tp'] for m in metrics.values())
        tot_fp = sum(m['fp'] for m in metrics.values())
        tot_fn = sum(m['fn'] for m in metrics.values())
        prec = tot_tp / max(tot_tp + tot_fp, 1)
        rec = tot_tp / max(tot_tp + tot_fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-6)

        per_class = {}
        for c in CLASSES:
            c_tp = metrics[c]['tp']
            c_fp = metrics[c]['fp']
            c_fn = metrics[c]['fn']
            c_prec = c_tp / max(c_tp + c_fp, 1)
            c_rec = c_tp / max(c_tp + c_fn, 1)
            c_f1 = 2 * c_prec * c_rec / max(c_prec + c_rec, 1e-6)
            per_class[c] = {
                'tp': c_tp, 'fp': c_fp, 'fn': c_fn,
                'precision': c_prec, 'recall': c_rec, 'f1': c_f1
            }

        return {
            'tot_tp': tot_tp, 'tot_fp': tot_fp, 'tot_fn': tot_fn,
            'precision': prec, 'recall': rec, 'micro_f1': f1,
            'classes': per_class
        }

    # 1. Medir baseline con umbral global 0.35
    base_res = evaluate_thresholds(0.35)
    print("\n--- BASELINE EXP07 (Umbral Global 0.35) ---")
    print(f"Total TP: {base_res['tot_tp']}, FP: {base_res['tot_fp']}, FN: {base_res['tot_fn']}")
    print(f"Precisión Global: {base_res['precision']*100:.2f}%, Recall: {base_res['recall']*100:.2f}%, Micro-F1: {base_res['micro_f1']*100:.2f}%")
    for c in CLASSES:
        cm = base_res['classes'][c]
        print(f"  {c:10s}: TP={cm['tp']:3d}, FP={cm['fp']:4d}, FN={cm['fn']:3d} | Prec={cm['precision']*100:5.1f}%, Rec={cm['recall']*100:5.1f}%, F1={cm['f1']*100:5.1f}%")

    # 2. Calibración por clase: buscar el umbral óptimo independiente para cada clase
    candidate_thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    best_per_class_th = {}
    
    print("\n--- BÚSQUEDA DE UMBRAL ÓPTIMO POR CLASE ---")
    for c in CLASSES:
        best_c_f1 = -1.0
        best_th = 0.35
        best_stats = None
        for th in candidate_thresholds:
            # Evaluar solo esta clase
            single_th_map = {k: 0.35 for k in CLASSES}
            single_th_map[c] = th
            res = evaluate_thresholds(single_th_map)
            cm = res['classes'][c]
            # Priorizamos F1, y en caso de empate a 0, penalizamos FPs
            score_crit = cm['f1'] - (0.001 * cm['fp'])
            if score_crit > best_c_f1:
                best_c_f1 = score_crit
                best_th = th
                best_stats = cm
        best_per_class_th[c] = best_th
        print(f"  {c:10s} -> Umbral Óptimo: {best_th:.2f} (TP={best_stats['tp']}, FP={best_stats['fp']}, Prec={best_stats['precision']*100:.1f}%, Rec={best_stats['recall']*100:.1f}%, F1={best_stats['f1']*100:.1f}%)")

    # 3. Evaluar combinación óptima de umbrales calibrados
    opt_res = evaluate_thresholds(best_per_class_th)
    print("\n--- EXP07 CALIBRADO POR CLASE (Modo Equilibrado) ---")
    print(f"Umbrales: {best_per_class_th}")
    print(f"Total TP: {opt_res['tot_tp']}, FP: {opt_res['tot_fp']}, FN: {opt_res['tot_fn']}")
    print(f"Precisión Global: {opt_res['precision']*100:.2f}%, Recall: {opt_res['recall']*100:.2f}%, Micro-F1: {opt_res['micro_f1']*100:.2f}%")
    for c in CLASSES:
        cm = opt_res['classes'][c]
        print(f"  {c:10s}: TP={cm['tp']:3d}, FP={cm['fp']:4d}, FN={cm['fn']:3d} | Prec={cm['precision']*100:5.1f}%, Rec={cm['recall']*100:5.1f}%, F1={cm['f1']*100:5.1f}%")

    # 4. Modo Alta Precisión Calibrado (Reducir FPs a mínimo para video en vivo)
    precision_th = {
        'plastico': max(0.40, best_per_class_th['plastico']),
        'papel': max(0.35, best_per_class_th['papel']),
        'carton': max(0.35, best_per_class_th['carton']),
        'vidrio': max(0.35, best_per_class_th['vidrio']),
        'metal': max(0.35, best_per_class_th['metal']),
        'organico': 0.50
    }
    prec_res = evaluate_thresholds(precision_th)
    print("\n--- EXP07 MODO ALTA PRECISIÓN CALIBRADO ---")
    print(f"Umbrales: {precision_th}")
    print(f"Total TP: {prec_res['tot_tp']}, FP: {prec_res['tot_fp']}, FN: {prec_res['tot_fn']}")
    print(f"Precisión Global: {prec_res['precision']*100:.2f}%, Recall: {prec_res['recall']*100:.2f}%, Micro-F1: {prec_res['micro_f1']*100:.2f}%")
    for c in CLASSES:
        cm = prec_res['classes'][c]
        print(f"  {c:10s}: TP={cm['tp']:3d}, FP={cm['fp']:4d}, FN={cm['fn']:3d} | Prec={cm['precision']*100:5.1f}%, Rec={cm['recall']*100:5.1f}%, F1={cm['f1']*100:5.1f}%")

    # Guardar resultados en {exp_dir}/calibrated_thresholds.json
    out_file = exp_dir / 'calibrated_thresholds.json'
    out_data = {
        'checkpoint': str(ckpt_path),
        'baseline_th_035': base_res,
        'calibrated_balanced': {
            'thresholds': best_per_class_th,
            'metrics': opt_res
        },
        'calibrated_high_precision': {
            'thresholds': precision_th,
            'metrics': prec_res
        }
    }
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, indent=2)
    print(f"\nResultados guardados exitosamente en {out_file.relative_to(ROOT)}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Calibración por clase para detectores S8.")
    parser.add_argument('--exp-dir', type=str, default='runs/detector_exp07', help='Directorio del experimento')
    args = parser.parse_args()
    run_calibration(args.exp_dir)
