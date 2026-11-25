import json
from collections import Counter

with open('data/preparado/deteccion/manifest.json', 'r', encoding='utf-8') as f:
    m = json.load(f)

manual = [r for r in m['images'] if r.get('manual_batch')]
reviewed = [r for r in manual if r.get('reviewed')]
pending = [r for r in manual if not r.get('reviewed')]

print(f"Total lote manual: {len(manual)}")
print(f"Revisadas por el usuario: {len(reviewed)} (78.9% del lote manual)")
print(f"Pendientes restantes: {len(pending)}")

box_classes = Counter([b['class'] for r in reviewed for b in r.get('boxes', [])])
print("Cajas dibujadas por clase:", dict(box_classes))

split_counts = Counter([r.get('split') for r in reviewed])
print("Distribución por partición de las revisadas:", dict(split_counts))

# Let's check pending by suggested class
pending_classes = Counter([r.get('target_hint') for r in pending])
print("Pendientes por clase sugerida:", dict(pending_classes))
