import json
from pathlib import Path

with open('data/preparado/deteccion/manifest.json', 'r', encoding='utf-8') as f:
    m = json.load(f)

manual = [img for img in m['images'] if img.get('manual_batch')]
first = manual[0]
print("path:", first.get('path'))
print("exists as data/path?", (Path('data') / first.get('path', '')).exists())
print("original_path:", first.get('original_path'))
print("exists as original_path?", Path(first.get('original_path', '')).exists())

# Check how Store loads images
import residuos.data as rd
store = rd.Store('data')
print("Store root:", store.root)
print("Store db images count:", len(store.db.get('images', [])))
sample_record = next(r for r in store.db.get('images', []) if r.get('manual_batch'))
print("Sample record in store:", sample_record['id'], sample_record['path'])
sample_img_path = store.root / sample_record['path']
print("Does sample_img_path exist?", sample_img_path.exists())
if not sample_img_path.exists():
    print("Does original_path exist?", Path(sample_record.get('original_path', '')).exists())
