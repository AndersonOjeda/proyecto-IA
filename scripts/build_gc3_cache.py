"""Construye caché npy pre-redimensionado (256x256 uint8) para las 7.293 imágenes limpias de GC3."""
import json
import time
from pathlib import Path
import cv2
import numpy as np

def main():
    root = Path(__file__).resolve().parents[1]
    gc3_json = root / 'runs/detector_gc3_integration/gc3_clean_train.json'
    with open(gc3_json, 'r', encoding='utf-8') as f:
        records = json.load(f)
        
    out_npy = root / 'runs/detector_gc3_integration/gc3_train.npy'
    N = len(records)
    print(f"Construyendo caché NPY de píxeles (256x256x3 uint8) para {N} imágenes de GC3...", flush=True)
    
    t0 = time.perf_counter()
    arr = np.zeros((N, 256, 256, 3), dtype=np.uint8)
    
    for i, r in enumerate(records):
        if (i + 1) % 1500 == 0 or (i + 1) == N:
            print(f"  Procesadas {i+1}/{N} imágenes ({time.perf_counter()-t0:.1f}s)...", flush=True)
        img_path = root / r['path']
        frame = cv2.imread(str(img_path))
        if frame is None:
            raise FileNotFoundError(f"No se pudo leer imagen: {img_path}")
        arr[i] = cv2.resize(frame, (256, 256))
        
    np.save(out_npy, arr)
    dt = time.perf_counter() - t0
    size_mb = out_npy.stat().st_size / (1024 * 1024)
    print(f"\n[OK] Caché guardada en {out_npy} ({size_mb:.1f} MB) en {dt:.1f}s.", flush=True)

if __name__ == '__main__':
    main()
