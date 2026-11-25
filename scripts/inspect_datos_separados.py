import json
from pathlib import Path

p = Path('data/DATOS_SEPARADOS_6_CATEGORIAS_20261124_v1')
print("Directorio existe:", p.exists())

subdirs = [x for x in p.iterdir() if x.is_dir()]
print(f"Subdirectorios ({len(subdirs)}):")
for s in subdirs:
    json_files = list(s.glob('**/*.json'))
    img_files = list(s.glob('**/*.jpg')) + list(s.glob('**/*.png')) + list(s.glob('**/*.jpeg'))
    print(f" - {s.name}: {len(img_files)} imágenes, {len(json_files)} jsons: {[j.name for j in json_files]}")

# Also check if there are json files in the root of p
root_jsons = list(p.glob('*.json'))
print(f"JSONs en la raíz de DATOS_SEPARADOS ({len(root_jsons)}): {[j.name for j in root_jsons]}")
