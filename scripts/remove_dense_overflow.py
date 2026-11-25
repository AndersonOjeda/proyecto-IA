import json
from pathlib import Path

for path in [Path('data/preparado/deteccion/manifest.json'), Path('data/real_waste/annotations.json')]:
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            db = json.load(f)
        orig_len = len(db['images'])
        db['images'] = [img for img in db['images'] if img['id'] != 'salsa_2791']
        new_len = len(db['images'])
        print(f"{path}: {orig_len} -> {new_len}")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(db, f, indent=2)
