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
from residuos.evaluate import metrics

device = device_for('auto')
store = Store('data/preparado/deteccion')
val_records = store.records('val')

model, _ = load_model('runs/detector/best.pt', device)
model.eval()

# Obtener predicciones raw con umbral muy bajo (0.05) para todas las 670 imagenes
print("Extrayendo predicciones base (th=0.05) para simulación de umbrales por clase...")
val_pairs_raw = []
for i, r in enumerate(val_records):
    frame = cv2.imdecode(np.fromfile(store.root / r['path'], dtype=np.uint8), cv2.IMREAD_COLOR)
    preds = predict(model, frame, device, threshold=0.05)
    val_pairs_raw.append((preds, r['boxes']))

# Línea base Exp05 uniforme th=0.35
def eval_thresholds(th_dict, min_size=0):
    pairs_filt = []
    for preds, truths in val_pairs_raw:
        kept = []
        for p in preds:
            c = p['class']
            th = th_dict.get(c, 0.35)
            if p['score'] >= th:
                w = p['xyxy'][2] - p['xyxy'][0]
                h = p['xyxy'][3] - p['xyxy'][1]
                if min_size == 0 or (w >= min_size and h >= min_size):
                    kept.append(p)
        pairs_filt.append((kept, truths))
    return metrics(pairs_filt)

m_base = eval_thresholds({c: 0.35 for c in CLASSES})
print(f"Línea base uniforme 0.35: TP={m_base['tp']}, FP={m_base['fp']}, P={m_base['tp']/(m_base['tp']+m_base['fp'])*100:.2f}%, R={m_base['tp']/(m_base['tp']+m_base['fn'])*100:.2f}%, F1={m_base['micro_f1']*100:.2f}%")
for c in CLASSES:
    st = m_base['classes'][c]
    print(f"  {c:<10}: TP={st['tp']:3d}, FP={st['fp']:4d}, FN={st['fn']:4d}, Rec={st['recall']*100 if st['recall'] else 0:.2f}%")

# Probar umbrales adaptativos por clase:
# Para metal: probar 0.15, 0.20
# Para carton: probar 0.15, 0.20
# Para papel: probar 0.15, 0.20
# Para vidrio: probar 0.15, 0.20
for th_min in [0.10, 0.15, 0.20, 0.25]:
    th_map = {'plastico': 0.35, 'organico': 0.35, 'metal': th_min, 'carton': th_min, 'papel': th_min, 'vidrio': th_min}
    m_adap = eval_thresholds(th_map)
    print(f"\nUmbrales adaptativos (minoría th={th_min:.2f}): TP={m_adap['tp']} (+{m_adap['tp']-m_base['tp']}), FP={m_adap['fp']} (+{m_adap['fp']-m_base['fp']}), P={m_adap['tp']/(m_adap['tp']+m_adap['fp'])*100:.2f}%, R={m_adap['tp']/(m_adap['tp']+m_adap['fn'])*100:.2f}%, F1={m_adap['micro_f1']*100:.2f}%")
    for c in ['metal', 'carton', 'papel', 'vidrio']:
        st = m_adap['classes'][c]
        print(f"  {c:<10}: TP={st['tp']:3d}, FP={st['fp']:4d}, FN={st['fn']:4d}, Rec={st['recall']*100 if st['recall'] else 0:.2f}%")
