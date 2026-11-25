import json
import cv2
import numpy as np
from pathlib import Path

with open('data/datos_separados_limpios/clean_records.json', 'r', encoding='utf-8') as f:
    clean_records = json.load(f)

print(f"Pre-procesando {len(clean_records)} imágenes limpias de DATOS_SEPARADOS...")

cache_dir = Path('data/datos_separados_limpios')
npy_path = cache_dir / 'datos_separados_256.npy'

frames = []
valid_records = []

for r in clean_records:
    p = Path(r['path'])
    img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Advertencia: no se pudo leer {p}")
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
    frames.append(img_res)
    valid_records.append(r)

arr = np.stack(frames, axis=0)
print(f"Forma del tensor en memoria: {arr.shape}, dtype: {arr.dtype}, tamaño: {arr.nbytes / 1024**2:.1f} MB")
np.save(npy_path, arr)
print(f"Guardado exitoso en {npy_path}")

with open(cache_dir / 'valid_records.json', 'w', encoding='utf-8') as f:
    json.dump(valid_records, f, indent=2, ensure_ascii=False)
print("Guardado valid_records.json con", len(valid_records), "registros.")
