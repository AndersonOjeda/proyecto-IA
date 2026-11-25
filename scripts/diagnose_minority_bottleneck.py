import json
from pathlib import Path
import cv2
import numpy as np
import torch
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from residuos import CLASSES
from residuos.data import Store
from residuos.detector import load_model, predict, iou, device_for
from scripts.experiment3_runner import DetectorS8, decode_grid32

device = device_for('auto')
store = Store('data/preparado/deteccion')
val_records = store.records('val')

# Cargar Exp05
model, metadata = load_model('runs/detector/best.pt', device)
model.eval()

target_classes = ['papel', 'carton', 'vidrio', 'metal']

print(f"Total imágenes de validación: {len(val_records)}")
print(f"Analizando cuello de botella en Exp05 para clases minoritarias: {target_classes}")

# Denominadores: cuántos GT hay en val
gt_counts = {c: 0 for c in target_classes}
for r in val_records:
    for b in r['boxes']:
        if b['class'] in gt_counts:
            gt_counts[b['class']] += 1

print(f"Objetos reales (GT) en validación: {gt_counts}")

# Evaluar por etapas
# Etapa 1: Caja candidata con IoU >= 0.5 (cualquier clase, umbral muy bajo 0.01)
# Etapa 2: Caja candidata con IoU >= 0.5 y clase correcta
# Etapa 3: Caja correcta con score >= 0.35 (umbral operativo)
# Etapa 4: Sobrevive NMS y matching 1 a 1

stage1_candidates = {c: 0 for c in target_classes} # IoU >= 0.5 (any class)
stage2_correct_cls = {c: 0 for c in target_classes} # IoU >= 0.5 & correct class
stage3_above_th = {c: 0 for c in target_classes}    # correct class & score >= 0.35
stage4_survived_nms = {c: 0 for c in target_classes} # survived NMS & matched

confusion_classes = {c: {} for c in target_classes} # what class was predicted instead
failure_previews = {c: [] for c in target_classes}

for r_idx, r in enumerate(val_records):
    frame = cv2.imdecode(np.fromfile(store.root / r['path'], dtype=np.uint8), cv2.IMREAD_COLOR)
    h_orig, w_orig = r['height'], r['width']
    
    # 1. Inferencia raw con umbral bajo para ver candidatos geométricos
    # Usar predict con umbral 0.01
    raw_preds_low = predict(model, frame, device, threshold=0.01)
    # Inferencia con umbral operativo 0.35
    preds_oper = predict(model, frame, device, threshold=0.35)
    
    # Para cada GT de clase minoritaria
    for gt_idx, gt in enumerate(r['boxes']):
        c = gt['class']
        if c not in target_classes:
            continue
            
        gt_box = gt['xyxy']
        
        # Buscar mejor solapamiento con cualquier candidato de raw_preds_low
        candidates = []
        for p in raw_preds_low:
            ov = iou(p['xyxy'], gt_box)
            if ov >= 0.5:
                candidates.append((ov, p))
                
        if candidates:
            stage1_candidates[c] += 1
            # Ver qué clases predijeron los candidatos con IoU >= 0.5
            best_cand = max(candidates, key=lambda x: x[0])[1]
            pred_cls = best_cand['class']
            confusion_classes[c][pred_cls] = confusion_classes[c].get(pred_cls, 0) + 1
            
            # Chequear si alguno tiene la clase correcta
            same_cls_cands = [p for ov, p in candidates if p['class'] == c]
            if same_cls_cands:
                stage2_correct_cls[c] += 1
                best_same = max(same_cls_cands, key=lambda x: x['score'])
                if best_same['score'] >= 0.35:
                    stage3_above_th[c] += 1
            else:
                if len(failure_previews[c]) < 3:
                    failure_previews[c].append({
                        'img_id': r['id'],
                        'gt_xyxy': gt_box,
                        'best_candidate_class': pred_cls,
                        'best_candidate_score': best_cand['score'],
                        'best_candidate_iou': max(candidates, key=lambda x: x[0])[0],
                        'reason': f"Confundido con {pred_cls}"
                    })
        else:
            # Ni siquiera hubo caja con IoU >= 0.5
            # Ver mejor IoU disponible
            all_ious = [(iou(p['xyxy'], gt_box), p) for p in raw_preds_low]
            best_sub = max(all_ious, key=lambda x: x[0]) if all_ious else (0, None)
            if len(failure_previews[c]) < 3:
                failure_previews[c].append({
                    'img_id': r['id'],
                    'gt_xyxy': gt_box,
                    'max_iou_found': best_sub[0],
                    'reason': "Sin candidato geométrico (IoU < 0.5)"
                })

# Ahora etapa 4 oficial desde el evaluador
from residuos.evaluate import metrics
pairs = [(predict(model, cv2.imdecode(np.fromfile(store.root / r['path'], dtype=np.uint8), cv2.IMREAD_COLOR), device, 0.35), r['boxes']) for r in val_records]
m = metrics(pairs)
for c in target_classes:
    stage4_survived_nms[c] = m['classes'][c]['tp']

print("\n" + "="*85)
print("DIAGNÓSTICO DE CUELLO DE BOTELLA POR ETAPA (Exp05 en Validación)")
print("="*85)
print(f"{'Clase':<10} | {'GT':>4} | {'1. IoU>=0.5 (Cualq)':>18} | {'2. Clase Correcta':>18} | {'3. Score>=0.35':>14} | {'4. TP Final':>11}")
print("-" * 85)
for c in target_classes:
    tot = gt_counts[c]
    s1 = f"{stage1_candidates[c]}/{tot} ({stage1_candidates[c]/tot*100:.1f}%)" if tot else "0"
    s2 = f"{stage2_correct_cls[c]}/{tot} ({stage2_correct_cls[c]/tot*100:.1f}%)" if tot else "0"
    s3 = f"{stage3_above_th[c]}/{tot} ({stage3_above_th[c]/tot*100:.1f}%)" if tot else "0"
    s4 = f"{stage4_survived_nms[c]}/{tot} ({stage4_survived_nms[c]/tot*100:.1f}%)" if tot else "0"
    print(f"{c:<10} | {tot:>4} | {s1:>18} | {s2:>18} | {s3:>14} | {s4:>11}")

print("\n" + "="*85)
print("CLASES ASIGNADAS A LOS CANDIDATOS CON IoU >= 0.5:")
print("="*85)
for c in target_classes:
    print(f"{c} (GT={gt_counts[c]}): {confusion_classes[c]}")

print("\n" + "="*85)
print("VISTA PREVIA COMPACTA DE FALLOS REPRESENTATIVOS:")
print("="*85)
for c in target_classes:
    print(f"\n--- Fallos en {c.upper()} ---")
    for f in failure_previews[c]:
        print(f)
