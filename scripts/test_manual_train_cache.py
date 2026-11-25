import json
from pathlib import Path
import cv2
import numpy as np

with open('data/preparado/deteccion/manifest.json', 'r', encoding='utf-8') as f:
    m = json.load(f)

manual_train = [r for r in m['images'] if r.get('manual_batch') and r.get('reviewed') and r.get('split') == 'train']
print(f"Total manual train images: {len(manual_train)}")

# Verify all can be read and resized to (256, 256)
frames = []
root = Path('data/preparado/deteccion')
for r in manual_train:
    p = root / r['path']
    img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img is not None, f"Failed to read {p}"
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
    frames.append(img_res)

frames_arr = np.stack(frames, axis=0) # shape: [52, 256, 256, 3], uint8
print("Pre-caching shape:", frames_arr.shape, frames_arr.dtype, f"Size: {frames_arr.nbytes / 1024**2:.2f} MB")
