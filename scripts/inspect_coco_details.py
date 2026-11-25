import json
from pathlib import Path

p = Path('data/DATOS_SEPARADOS_6_CATEGORIAS_20261124_v1')

for c in sorted([x for x in p.iterdir() if x.is_dir()], key=lambda x: x.name):
    print(f"\n=== Clase: {c.name} ===")
    sub = [x for x in c.iterdir() if x.is_dir()]
    for s in sub:
        js = list(s.glob('*.json'))
        imgs = list(s.glob('*.jpg')) + list(s.glob('*.png'))
        print(f"  Sub: {s.name} | Imgs: {len(imgs)} | JSONs: {[j.name for j in js]}")
        for j in js:
            with open(j, 'r', encoding='utf-8') as f:
                data = json.load(f)
            cats = [cat['name'] for cat in data.get('categories', [])]
            print(f"    -> {j.name}: {len(data.get('images', []))} images, {len(data.get('annotations', []))} annotations, categories: {cats}")
