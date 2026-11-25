import json
from pathlib import Path
from collections import Counter

gc_root = Path('data/externos/GARBAGE CLASSIFICATION 3.v2-gc1.coco')
total_images = 0
total_annotations = 0

print("="*70)
print("AUDITORÍA DE DATASET: GARBAGE CLASSIFICATION 3 (Roboflow)")
print("="*70)

for split in ['train', 'valid', 'test']:
    ann_file = gc_root / split / '_annotations.coco.json'
    if not ann_file.exists():
        print(f"Split {split}: No encontrado")
        continue
    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    cats = {c['id']: c['name'] for c in data['categories']}
    cat_counts = Counter(cats.get(a['category_id'], 'UNKNOWN') for a in data['annotations'])
    n_imgs = len(data['images'])
    n_anns = len(data['annotations'])
    total_images += n_imgs
    total_annotations += n_anns
    print(f"\n--- Split: {split} ---")
    print(f"Total imágenes: {n_imgs}")
    print(f"Total anotaciones: {n_anns}")
    print("Conteo por categoría:")
    for cat_name, cnt in sorted(cat_counts.items(), key=lambda kv: -kv[1]):
        print(f"  - {cat_name:<25}: {cnt:5d} cajas")

print("\n" + "="*70)
print(f"TOTAL GLOBAL: {total_images} imágenes, {total_annotations} anotaciones")
print("="*70)
