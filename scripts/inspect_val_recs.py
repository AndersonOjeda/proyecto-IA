import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from residuos.data import Store

store = Store('data/preparado/deteccion')
val_recs = store.records('val')
print("store.records('val') count:", len(val_recs))

# Let's check how many were without manual_batch
val_without_manual = [r for r in val_recs if not r.get('manual_batch')]
print("val_without_manual count:", len(val_without_manual))

# Check how val.npy was created originally in runs/detector_exp01
import numpy as np
val_npy = np.load('runs/detector_exp01/val.npy', mmap_mode='r')
print("val_npy shape:", val_npy.shape)
