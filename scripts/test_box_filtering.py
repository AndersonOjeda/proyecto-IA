"""Evalúa el efecto de filtrar cajas minúsculas en inferencia de Exp 05."""
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from residuos import CLASSES
from residuos.detector import iou
from scripts.experiment3_runner import DetectorS8, decode_grid32, device_for
from residuos.data import Store

def main():
    device = device_for('auto')
    cp_path = ROOT / 'runs/detector_exp05/best.pt'
    saved = torch.load(cp_path, map_location=device, weights_only=False)
    model = DetectorS8().to(device)
    model.load_state_dict(saved['state'])
    model.eval()
    
    store = Store(ROOT / 'data/preparado/deteccion')
    val_rows = store.records('val')
    val_npy = ROOT / 'runs/detector_exp01/val.npy'
    pixels = np.load(val_npy, mmap_mode='r')
    
    print("Evaluando efecto del filtro de tamaño de caja mínimo en validación (670 fotos reales, th=0.35)...")
    
    with torch.no_grad():
        x_all = torch.from_numpy(np.array(pixels)).permute(0, 3, 1, 2).float() / 255.0
        # inferencia por lotes de 32
        all_outputs = []
        for i in range(0, len(x_all), 32):
            batch = x_all[i:i+32].to(device)
            out = model(batch).cpu()
            all_outputs.extend(out)
            
    for min_size in [0, 16, 24, 32]:
        tp, fp, fn = 0, 0, 0
        for i, r in enumerate(val_rows):
            w_orig, h_orig = r['width'], r['height']
            raw_boxes = decode_grid32(all_outputs[i], w_orig, h_orig, threshold=0.35, nms=0.45)
            
            # Filtro de tamaño mínimo de caja
            filt_boxes = [b for b in raw_boxes if (b['xyxy'][2] - b['xyxy'][0]) >= min_size and (b['xyxy'][3] - b['xyxy'][1]) >= min_size]
            
            matched = set()
            for pb in sorted(filt_boxes, key=lambda b: b['score'], reverse=True):
                best_iou = 0.0
                best_gt = -1
                for gt_idx, gt in enumerate(r['boxes']):
                    if gt_idx in matched or gt['class'] != pb['class']:
                        continue
                    ov = iou(pb['xyxy'], gt['xyxy'])
                    if ov > best_iou:
                        best_iou = ov
                        best_gt = gt_idx
                if best_iou >= 0.5:
                    matched.add(best_gt)
                    tp += 1
                else:
                    fp += 1
            for gt_idx in range(len(r['boxes'])):
                if gt_idx not in matched:
                    fn += 1
                    
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0
        print(f"Min Box Size: {min_size:2d} px | Prec: {p*100:6.2f}% | Rec: {rec*100:6.2f}% | F1: {f1*100:6.2f}% | TP: {tp:4d} | FP: {fp:5d} (-{1806-fp:4d}) | FN: {fn:4d}")

if __name__ == '__main__':
    main()
