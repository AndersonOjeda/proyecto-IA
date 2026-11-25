import json
from pathlib import Path

with open('data/preparado/deteccion/manifest.json', 'r', encoding='utf-8') as f:
    m = json.load(f)

manual = [r for r in m['images'] if r.get('manual_batch')]
reviewed = [r for r in manual if r.get('reviewed')]
print(f"Lote manual: {len(reviewed)}/{len(manual)} revisadas por el usuario.")
boxes_by_class = {}
for r in reviewed:
    for b in r.get('boxes', []):
        c = b.get('class')
        boxes_by_class[c] = boxes_by_class.get(c, 0) + 1
print("Cajas dibujadas por clase:", boxes_by_class)

# Search for any files or folders containing 'separad' or 'stratified'
for p in Path('.').glob('**/*separad*'):
    print("Found match in workspace:", p)
for p in Path('.').glob('**/*stratified*'):
    print("Found match in workspace:", p)
