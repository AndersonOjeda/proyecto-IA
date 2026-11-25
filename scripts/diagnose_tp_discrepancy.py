import json
import torch
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from residuos import CLASSES
from residuos.detector import load_model, predict, iou, device_for
from residuos.data import Store
from residuos.evaluate import metrics
from scripts.experiment3_runner import DetectorS8, decode_grid32

device = device_for('auto')
cp = torch.load('runs/detector/best.pt', map_location=device, weights_only=False)
model = DetectorS8().to(device)
model.load_state_dict(cp['state'])
model.eval()

store = Store('data/preparado/deteccion')
val_records = store.records('val')
val_npy = np.load('runs/detector_exp01/val.npy', mmap_mode='r')
x_all = torch.from_numpy(np.array(val_npy)).permute(0, 3, 1, 2).float() / 255.0

print(f"Total val records: {len(val_records)}")

# 1. Batched inference (batch size 32)
with torch.no_grad():
    all_outputs_batched = []
    for i in range(0, len(x_all), 32):
        all_outputs_batched.extend(model(x_all[i:i+32].to(device)).cpu())

# 2. Single inference (batch size 1)
all_outputs_single = []
with torch.no_grad():
    for i in range(len(x_all)):
        all_outputs_single.append(model(x_all[i:i+1].to(device))[0].cpu())

diffs = [torch.max(torch.abs(b - s)).item() for b, s in zip(all_outputs_batched, all_outputs_single)]
print("Max absolute difference (batched vs single):", max(diffs))

# Method A: metrics() with single decode
pairs_single = []
for i, r in enumerate(val_records):
    boxes = decode_grid32(all_outputs_single[i], r['width'], r['height'], threshold=0.35, nms=0.45)
    pairs_single.append((boxes, r['boxes']))
m_single = metrics(pairs_single)
print(f"Single evaluate metrics: TP={m_single['tp']}, FP={m_single['fp']}, FN={m_single['fn']}")

# Method B: metrics() with batched decode
pairs_batched = []
for i, r in enumerate(val_records):
    boxes = decode_grid32(all_outputs_batched[i], r['width'], r['height'], threshold=0.35, nms=0.45)
    pairs_batched.append((boxes, r['boxes']))
m_batched = metrics(pairs_batched)
print(f"Batched evaluate metrics: TP={m_batched['tp']}, FP={m_batched['fp']}, FN={m_batched['fn']}")

# Method C: Look at threshold.json
tj = json.load(open('runs/detector/threshold.json', encoding='utf-8'))
sp = tj['selected_operating_point']
print(f"threshold.json selected: TP={sp['tp']}, FP={sp['fp']}, FN={sp['fn']}")

# Method D: Frame by frame from disk (App pipeline)
import cv2
pairs_app = []
for i, r in enumerate(val_records):
    frame = cv2.imdecode(np.fromfile(store.root / r['path'], dtype=np.uint8), cv2.IMREAD_COLOR)
    boxes = predict(model, frame, device, threshold=0.35)
    pairs_app.append((boxes, r['boxes']))
m_app = metrics(pairs_app)
print(f"App (imdecode from disk) evaluate metrics: TP={m_app['tp']}, FP={m_app['fp']}, FN={m_app['fn']}")

# Find per-image differences between pairs_batched, pairs_single, and threshold.json
# Let's inspect where the TP differences are!
for i, r in enumerate(val_records):
    m_s = metrics([(pairs_single[i][0], r['boxes'])])
    m_b = metrics([(pairs_batched[i][0], r['boxes'])])
    m_a = metrics([(pairs_app[i][0], r['boxes'])])
    if m_s['tp'] != m_b['tp'] or m_s['tp'] != m_a['tp']:
        print(f"Image diff at idx {i}, ID={r['id']}: single TP={m_s['tp']}, batched TP={m_b['tp']}, app TP={m_a['tp']}")
