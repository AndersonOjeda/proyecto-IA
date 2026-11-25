import json
import hashlib
from pathlib import Path
from collections import Counter

root = Path('data/DATOS_SEPARADOS_6_CATEGORIAS_20261124_v1')

all_json_files = list(root.glob('**/*.json'))
print(f"Total JSON files found: {len(all_json_files)}")

# Check category mapping across all JSON files
cat_names = set()
for jf in all_json_files:
    with open(jf, 'r', encoding='utf-8') as f:
        d = json.load(f)
    for c in d.get('categories', []):
        cat_names.add(c['name'])
print("All category names in JSONs:", cat_names)

# Check all images on disk in the dataset
all_disk_images = list(root.glob('**/*.jpg')) + list(root.glob('**/*.png')) + list(root.glob('**/*.jpeg'))
print(f"Total image files on disk in DATOS_SEPARADOS: {len(all_disk_images)}")

# Check sha256 of all disk images and check against our current project val and test!
with open('runs/detector_exp01/snapshot/manifest.json', 'r', encoding='utf-8') as f:
    m01 = json.load(f)

val_hashes = {r['sha256']: r['id'] for r in m01['images'] if r.get('split') == 'val'}
test_hashes = {r['sha256']: r['id'] for r in m01['images'] if r.get('split') == 'test'}
print(f"Our project: {len(val_hashes)} val hashes, {len(test_hashes)} test hashes")

disk_hashes = {}
collisions_val = []
collisions_test = []

for img_p in all_disk_images:
    h = hashlib.sha256(img_p.read_bytes()).hexdigest()
    disk_hashes[img_p.name] = h
    if h in val_hashes:
        collisions_val.append((img_p.name, val_hashes[h]))
    if h in test_hashes:
        collisions_test.append((img_p.name, test_hashes[h]))

print(f"Unique images on disk by SHA-256: {len(set(disk_hashes.values()))}")
print(f"Collisions with project VAL: {len(collisions_val)}")
print(f"Collisions with project TEST: {len(collisions_test)}")
if collisions_val:
    print("Sample val collision:", collisions_val[:3])
if collisions_test:
    print("Sample test collision:", collisions_test[:3])
