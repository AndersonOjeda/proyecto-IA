import json
from pathlib import Path

p1 = Path('data/preparado/deteccion/manifest.json')
p2 = Path('data/manifest.json')

found_any = False
for p in (p1, p2):
    if p.exists():
        with open(p, 'r', encoding='utf-8') as f:
            m = json.load(f)
        images = m.get('images', [])
        manual = [r for r in images if r.get('manual_batch')]
        reviewed = [r for r in manual if r.get('reviewed')]
        boxes = sum(len(r.get('boxes', [])) for r in reviewed)
        print(f"[{p}] Total manual batch: {len(manual)} | Revisadas: {len(reviewed)} | Cajas guardadas: {boxes}")
        
        # Check all reviewed images overall
        all_reviewed = [r for r in images if r.get('reviewed')]
        print(f"[{p}] Total imágenes revisadas en todo el dataset: {len(all_reviewed)}")
        
        if reviewed:
            found_any = True
            print("Detalle de las revisadas:")
            for r in reviewed:
                print(f" - ID: {r['id']} | split: {r.get('split')} | clase: {r.get('target_hint')} | cajas: {r.get('boxes')}")

# Also check if any temp or backup files were modified recently
for p in Path('.').glob('**/*manifest*'):
    if 'node_modules' not in str(p) and '.venv' not in str(p):
        print("Manifest file found:", p, "mtime:", p.stat().st_mtime)
