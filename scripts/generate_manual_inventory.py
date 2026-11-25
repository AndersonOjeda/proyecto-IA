import json
from pathlib import Path

with open('data/preparado/deteccion/manifest.json', 'r', encoding='utf-8') as f:
    m = json.load(f)

manual = [r for r in m['images'] if r.get('manual_batch')]

inventory = []
for i, r in enumerate(manual, 1):
    inventory.append({
        "num": i,
        "id": r['id'],
        "clase_sugerida": r.get('target_hint'),
        "particion": r.get('split'),
        "ancho": r['width'],
        "alto": r['height'],
        "ruta_relativa": r['path'],
        "ruta_completa": str((Path('data/preparado/deteccion') / r['path']).resolve())
    })

with open('data/inventario_90_fotos_manuales.json', 'w', encoding='utf-8') as f:
    json.dump(inventory, f, ensure_ascii=False, indent=2)

print(f"Inventario generado con {len(inventory)} fotos en data/inventario_90_fotos_manuales.json")
