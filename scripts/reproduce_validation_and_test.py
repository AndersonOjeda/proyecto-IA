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
from residuos.evaluate import metrics, collect
from app.detector import WasteDetector

device = device_for('auto')
store = Store('data/preparado/deteccion')

print("=== EVALUACIÓN REPRODUCIBLE DE VALIDACIÓN (670 IMÁGENES REALES) ===")
val_records = store.records('val')
print(f"Total imágenes de validación: {len(val_records)}")

modes = [
    ('cobertura', 0.35, 0),
    ('precision', 0.40, 32)
]

val_results = {}

for mode_name, th, min_size in modes:
    print(f"\n--- Modo {mode_name.upper()} (th={th}, min_size={min_size}px) ---")
    detector = WasteDetector(checkpoint='runs/detector/best.pt', device=str(device), mode=mode_name, min_box_size=min_size)
    
    pairs_app = []
    for r in val_records:
        frame = cv2.imdecode(np.fromfile(store.root / r['path'], dtype=np.uint8), cv2.IMREAD_COLOR)
        preds = detector.detect(frame)
        pairs_app.append((preds, r['boxes']))
        
    m_app = metrics(pairs_app)
    
    # Comprobar contra residuos.evaluate.collect
    model, _ = load_model('runs/detector/best.pt', device)
    pairs_eval, _ = collect(store, model, device, 'val', threshold=th, min_box_size=min_size)
    m_eval = metrics(pairs_eval)
    
    assert m_app['tp'] == m_eval['tp'], f"Discrepancia TP app ({m_app['tp']}) vs eval ({m_eval['tp']})"
    assert m_app['fp'] == m_eval['fp'], f"Discrepancia FP app ({m_app['fp']}) vs eval ({m_eval['fp']})"
    assert m_app['fn'] == m_eval['fn'], f"Discrepancia FN app ({m_app['fn']}) vs eval ({m_eval['fn']})"
    print(f"[OK] Coincidencia 100% idéntica entre WasteDetector (app) y collect (evaluador).")
    
    val_results[mode_name] = m_app

print("\n" + "="*85)
print("TABLA OFICIAL DE VALIDACIÓN POR CLASE Y MODO (Mismo posprocesamiento de la App)")
print("="*85)

for mode_name, th, min_size in modes:
    res = val_results[mode_name]
    print(f"\nMODO: {mode_name.upper()} (th={th:.2f}, min_size={min_size}px)")
    print(f"{'Clase':<12} | {'Objetos':>8} | {'TP':>6} | {'FP':>6} | {'FN':>6} | {'Precisión (%)':>14} | {'Recall (%)':>12} | {'F1 (%)':>8}")
    print("-" * 85)
    for c in CLASSES:
        st = res['classes'][c]
        p_str = f"{st['precision']*100:.2f}%" if st['precision'] is not None else "N/D"
        r_str = f"{st['recall']*100:.2f}%" if st['recall'] is not None else "N/D"
        f1_c = (2 * st['precision'] * st['recall'] / (st['precision'] + st['recall']) * 100) if (st['precision'] and st['recall'] and (st['precision'] + st['recall']) > 0) else 0.0
        print(f"{c:<12} | {st['objects']:>8} | {st['tp']:>6} | {st['fp']:>6} | {st['fn']:>6} | {p_str:>14} | {r_str:>12} | {f1_c:>7.2f}%")
    print("-" * 85)
    tp, fp, fn = res['tp'], res['fp'], res['fn']
    p_tot = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
    r_tot = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    f1_tot = res['micro_f1'] * 100
    tot_obj = tp + fn
    print(f"{'GLOBAL':<12} | {tot_obj:>8} | {tp:>6} | {fp:>6} | {fn:>6} | {p_tot:>13.2f}% | {r_tot:>11.2f}% | {f1_tot:>7.2f}%")

# Guardar resultados reproducibles en JSON
out_path = ROOT / 'runs/verificacion_app_exp03_exp05/validacion_modos_oficial.json'
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(val_results, indent=2, ensure_ascii=False), encoding='utf-8')
print(f"\nResultados oficiales de validación guardados en: {out_path}")

print("\n" + "="*85)
print("EVALUACIÓN FINAL EN CONJUNTO TEST RESERVADO (CONGELADO Y SIN REAJUSTE)")
print("="*85)

test_records = store.records('test')
print(f"Total imágenes de test reservadas: {len(test_records)}")

test_results = {}
for mode_name, th, min_size in modes:
    print(f"\n--- TEST: Modo {mode_name.upper()} (th={th}, min_size={min_size}px) ---")
    detector = WasteDetector(checkpoint='runs/detector/best.pt', device=str(device), mode=mode_name, min_box_size=min_size)
    pairs_test = []
    for r in test_records:
        frame = cv2.imdecode(np.fromfile(store.root / r['path'], dtype=np.uint8), cv2.IMREAD_COLOR)
        preds = detector.detect(frame)
        pairs_test.append((preds, r['boxes']))
        
    m_test = metrics(pairs_test)
    test_results[mode_name] = m_test
    
    print(f"{'Clase':<12} | {'Objetos':>8} | {'TP':>6} | {'FP':>6} | {'FN':>6} | {'Precisión (%)':>14} | {'Recall (%)':>12} | {'F1 (%)':>8}")
    print("-" * 85)
    for c in CLASSES:
        st = m_test['classes'][c]
        p_str = f"{st['precision']*100:.2f}%" if st['precision'] is not None else "N/D"
        r_str = f"{st['recall']*100:.2f}%" if st['recall'] is not None else "N/D"
        f1_c = (2 * st['precision'] * st['recall'] / (st['precision'] + st['recall']) * 100) if (st['precision'] and st['recall'] and (st['precision'] + st['recall']) > 0) else 0.0
        print(f"{c:<12} | {st['objects']:>8} | {st['tp']:>6} | {st['fp']:>6} | {st['fn']:>6} | {p_str:>14} | {r_str:>12} | {f1_c:>7.2f}%")
    print("-" * 85)
    tp, fp, fn = m_test['tp'], m_test['fp'], m_test['fn']
    p_tot = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
    r_tot = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    f1_tot = m_test['micro_f1'] * 100
    tot_obj = tp + fn
    print(f"{'GLOBAL':<12} | {tot_obj:>8} | {tp:>6} | {fp:>6} | {fn:>6} | {p_tot:>13.2f}% | {r_tot:>11.2f}% | {f1_tot:>7.2f}%")

test_out_path = ROOT / 'runs/evaluation/test_modos_oficial.json'
test_out_path.parent.mkdir(parents=True, exist_ok=True)
test_out_path.write_text(json.dumps(test_results, indent=2, ensure_ascii=False), encoding='utf-8')
print(f"\nResultados oficiales de test guardados en: {test_out_path}")
