import json

with open('data/preparado/deteccion/manifest.json', 'r', encoding='utf-8') as f:
    m = json.load(f)

manual = [r for r in m['images'] if r.get('manual_batch')]
reviewed = [r for r in manual if r.get('reviewed')]
print(f"Total manual: {len(manual)} | Revisadas: {len(reviewed)} | Pendientes: {len(manual) - len(reviewed)}")

splits = {}
for r in reviewed:
    sp = r.get('split')
    splits[sp] = splits.get(sp, 0) + 1
print("Partición de las revisadas:", splits)
