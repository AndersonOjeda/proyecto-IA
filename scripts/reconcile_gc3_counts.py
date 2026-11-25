import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

orig_manifest = json.loads((ROOT / 'data/preparado/deteccion/manifest.json').read_text(encoding='utf-8'))
base_train = [r for r in orig_manifest['images'] if r['split'] == 'train' and not r.get('id', '').startswith(('salsa', 'taco_ext', 'gc3'))][:3645]

ext_man = json.loads((ROOT / 'runs/detector_exp05/manifiesto_descargas.json').read_text(encoding='utf-8'))
ext_train = ext_man['accepted_images']

gc3_clean = json.loads((ROOT / 'runs/detector_gc3_integration/gc3_clean_train.json').read_text(encoding='utf-8'))

print("=== RECONCILIACIÓN EXACTA DE IMÁGENES GC3 Y DATASET UNIFICADO ===")
print(f"1. Imágenes base de entrenamiento: {len(base_train)}")
print(f"2. Imágenes exteriores de Exp05 (SALSA / TACO): {len(ext_train)}")
print(f"   -> Subtotal dataset canónico Exp05: {len(base_train) + len(ext_train)} imágenes")
print(f"3. Imágenes GC3 limpias (Roboflow auditadas): {len(gc3_clean)}")
total_train = len(base_train) + len(ext_train) + len(gc3_clean)
print(f"4. TOTAL entrenamiento unificado GC3: {total_train} imágenes ({len(base_train)} + {len(ext_train)} + {len(gc3_clean)})")

gc3_raw_dir = ROOT / 'data/externos/GARBAGE CLASSIFICATION 3.v2-gc1.coco'
if gc3_raw_dir.exists():
    for sp in ['train', 'valid', 'test']:
        sp_json = gc3_raw_dir / sp / '_annotations.coco.json'
        if sp_json.exists():
            d = json.loads(sp_json.read_text(encoding='utf-8'))
            print(f"   GC3 raw {sp}: {len(d.get('images', []))} imágenes, {len(d.get('annotations', []))} anotaciones")

gc3_boxes = sum(len(r['boxes']) for r in gc3_clean)
base_boxes = sum(len(r['boxes']) for r in base_train)
ext_boxes = sum(len(r['boxes']) for r in ext_train)
print(f"Cajas base: {base_boxes}, Cajas ext: {ext_boxes}, Cajas GC3: {gc3_boxes}")
print(f"TOTAL CAJAS entrenamiento unificado GC3: {base_boxes + ext_boxes + gc3_boxes}")
