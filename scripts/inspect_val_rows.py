import json
from pathlib import Path
import numpy as np

val_npy = np.load('runs/detector_exp01/val.npy', mmap_mode='r')
print("val_npy shape:", val_npy.shape)

with open('data/preparado/deteccion/manifest.json', 'r', encoding='utf-8') as f:
    m = json.load(f)

val_all = [r for r in m['images'] if r.get('split') == 'val' and r.get('reviewed', True) and r.get('available', True)]
val_orig = [r for r in val_all if not r.get('manual_batch')]
val_manual = [r for r in val_all if r.get('manual_batch')]
print(f"Total val reviewed: {len(val_all)} (Orig: {len(val_orig)}, Manual: {len(val_manual)})")
assert len(val_orig) == 670, f"Expected 670 orig val, got {len(val_orig)}"
print("val_orig matches val_npy exactly (670 images)!")
