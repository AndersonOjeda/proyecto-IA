"""Audit download candidates for SALSA and TACO unofficial for Experiment 5.
Evaluates category mappings, image integrity, overlap with official TACO / val / test splits,
and selects the high-priority images to download.
"""
from pathlib import Path
import json
import collections

ROOT = Path(__file__).resolve().parents[1]

# 1. Load original validation and test images to obtain their SHA-256 hashes and filenames
with open(ROOT / 'data/preparado/deteccion/manifest.json', encoding='utf-8') as f:
    orig_manifest = json.load(f)

val_hashes = {img['sha256'] for img in orig_manifest['images'] if img['split'] == 'val'}
test_hashes = {img['sha256'] for img in orig_manifest['images'] if img['split'] == 'test'}
train_hashes = {img['sha256'] for img in orig_manifest['images'] if img['split'] == 'train'}

val_test_stems = {Path(img['path']).stem for img in orig_manifest['images'] if img['split'] in ('val', 'test')}

print(f"Original splits in prepared manifest: train={len(train_hashes)}, val={len(val_hashes)}, test={len(test_hashes)}")

# 2. SALSA Analysis
salsa_path = ROOT / 'data/externos/salsa/src/salsa_utils/data/annotations.json'
with open(salsa_path, encoding='utf-8') as f:
    salsa = json.load(f)

salsa_cats = {c['id']: c['name'] for c in salsa['categories']}
salsa_licenses = {lic['id']: lic['name'] for lic in salsa.get('licenses', [])}
print("\n=== SALSA CATEGORIES ===")
for cid, cname in sorted(salsa_cats.items()):
    print(f"  {cid}: {cname}")

salsa_img_anns = collections.defaultdict(list)
for a in salsa['annotations']:
    salsa_img_anns[a['image_id']].append(a)

# Identify clean images: only contains {Glass, Metal, Paper, Plastic, Organic}
AMBIGUOUS_SALSA = {'Cigarette', 'Clothes', 'Electronics', 'Garbage bag', 'Miscellaneous'}
salsa_clean = []
salsa_clean_by_class = collections.Counter()
salsa_ann_by_class = collections.Counter()

for img in salsa['images']:
    anns = salsa_img_anns[img['id']]
    if not anns:
        continue
    cats = {salsa_cats[a['category_id']] for a in anns}
    if cats.intersection(AMBIGUOUS_SALSA):
        continue
    # Check bounding box validity
    valid_boxes = True
    for a in anns:
        x, y, w, h = a['bbox']
        if w < 2 or h < 2 or x + w > img['width'] + 5 or y + h > img['height'] + 5 or x < -5 or y < -5:
            valid_boxes = False
            break
    if not valid_boxes:
        continue
    
    salsa_clean.append(img)
    for c in cats:
        salsa_clean_by_class[c] += 1
    for a in anns:
        salsa_ann_by_class[salsa_cats[a['category_id']]] += 1

print(f"\nSALSA Total Images: {len(salsa['images'])}, Clean Unambiguous: {len(salsa_clean)}")
print(f"SALSA Clean Images by class presence: {dict(salsa_clean_by_class)}")
print(f"SALSA Clean Annotations count: {dict(salsa_ann_by_class)}")

# 3. TACO unofficial Analysis
taco_path = ROOT / 'data/externos/TACO-master/TACO-master/data/annotations_unofficial.json'
with open(taco_path, encoding='utf-8') as f:
    taco = json.load(f)

# 4. Check overlap between TACO unofficial, TACO official, and SALSA
with open(ROOT / 'data/externos/TACO-master/TACO-master/data/annotations.json', encoding='utf-8') as f:
    oficial = json.load(f)

taco_oficial_urls = set()
for img in oficial['images']:
    if img.get('flickr_url'): taco_oficial_urls.add(img['flickr_url'])
    if img.get('flickr_640_url'): taco_oficial_urls.add(img['flickr_640_url'])

unoff_urls = set()
for img in taco['images']:
    if img.get('flickr_url'): unoff_urls.add(img['flickr_url'])
    if img.get('flickr_640_url'): unoff_urls.add(img['flickr_640_url'])

taco_unoff_oficial_overlap = unoff_urls.intersection(taco_oficial_urls)
print(f"\nTACO unofficial vs TACO official URL overlap: {len(taco_unoff_oficial_overlap)}")

# SALSA OLM images
salsa_olm_clean = []
for img in salsa_clean:
    fn = img['file_name'].split('.')[0]
    if not fn.startswith('batch_'):
        salsa_olm_clean.append(img)

print(f"SALSA clean OLM images (non-batch): {len(salsa_olm_clean)}")
salsa_clean_boxes = collections.Counter()
for img in salsa_olm_clean:
    for a in salsa_img_anns[img['id']]:
        salsa_clean_boxes[salsa_cats[a['category_id']]] += 1
print(f"SALSA clean OLM annotations: {dict(salsa_clean_boxes)}")


from audit_external_datasets import source_mapping

taco_cats = {c['id']: c['name'] for c in taco['categories']}
taco_img_anns = collections.defaultdict(list)
for a in taco['annotations']:
    taco_img_anns[a['image_id']].append(a)

taco_clean = []
taco_clean_by_class = collections.Counter()
taco_ann_by_class = collections.Counter()

for img in taco['images']:
    anns = taco_img_anns[img['id']]
    if not anns:
        continue
    all_mapped = True
    img_classes = set()
    for a in anns:
        cname = taco_cats[a['category_id']]
        target, status, _ = source_mapping('TACO oficial', cname)
        if status not in ('propuesto', 'condicional') or not target:
            all_mapped = False
            break
        img_classes.add(target)
    
    if not all_mapped:
        continue
    
    # Check bounding boxes
    valid_boxes = True
    for a in anns:
        x, y, w, h = a['bbox']
        if w < 2 or h < 2 or x + w > img['width'] + 5 or y + h > img['height'] + 5 or x < -5 or y < -5:
            valid_boxes = False
            break
    if not valid_boxes:
        continue
    
    taco_clean.append(img)
    for c in img_classes:
        taco_clean_by_class[c] += 1
    for a in anns:
        target, _, _ = source_mapping('TACO oficial', taco_cats[a['category_id']])
        taco_ann_by_class[target] += 1

print(f"\nTACO Unofficial Total Images: {len(taco['images'])}, Clean Unambiguous: {len(taco_clean)}")
print(f"TACO Clean Images by class presence: {dict(taco_clean_by_class)}")
print(f"TACO Clean Annotations count: {dict(taco_ann_by_class)}")
