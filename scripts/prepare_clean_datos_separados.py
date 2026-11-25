import os
import glob
import json
import hashlib
from pathlib import Path
from collections import Counter

base = Path("data/DATOS_SEPARADOS_6_CATEGORIAS_20261124_v1")

# 1. Load project val and test hashes (snapshot and manifest)
with open('runs/detector_exp01/snapshot/manifest.json', 'r', encoding='utf-8') as f:
    m01 = json.load(f)

val_hashes = {r['sha256'] for r in m01['images'] if r.get('split') == 'val'}
test_hashes = {r['sha256'] for r in m01['images'] if r.get('split') == 'test'}

# Also add current manifest val/test
with open('data/preparado/deteccion/manifest.json', 'r', encoding='utf-8') as f:
    m_cur = json.load(f)
for r in m_cur['images']:
    if r.get('split') == 'val': val_hashes.add(r['sha256'])
    if r.get('split') == 'test': test_hashes.add(r['sha256'])

print(f"Total forbidden hashes (Val + Test): {len(val_hashes)} val, {len(test_hashes)} test")

# 2. Map all images on disk
disk_images = {}
disk_hashes = {}
for root, dirs, files in os.walk(base):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            full_p = os.path.join(root, f)
            disk_images[f] = full_p
            h = hashlib.sha256(open(full_p, 'rb').read()).hexdigest()
            disk_hashes[f] = h

json_files = glob.glob(f"{base}/**/_annotations.coco.json", recursive=True)

cat_normalization = {
    'organicos': 'organico', 'org-nicos': 'organico', 'biodegradable': 'organico',
    'plastico': 'plastico', 'plastic': 'plastico',
    'papel': 'papel', 'paper': 'papel',
    'carton': 'carton', 'cardboard': 'carton',
    'vidrio': 'vidrio', 'glass': 'vidrio',
    'metal': 'metal'
}

clean_records = {}
rejected_val_leak = 0
rejected_test_leak = 0

for jf in json_files:
    with open(jf, "r", encoding="utf-8") as f:
        d = json.load(f)

    cat_map = {}
    for c in d.get("categories", []):
        raw = c["name"].lower().replace("ó", "o").replace("á", "a").strip()
        mapped = cat_normalization.get(raw, "desconocido")
        cat_map[c["id"]] = mapped

    img_map = {img["id"]: img for img in d.get("images", []) if img["file_name"] in disk_images}

    for ann in d.get("annotations", []):
        img_id = ann.get("image_id")
        if img_id not in img_map:
            continue
        img_info = img_map[img_id]
        fname = img_info["file_name"]
        fpath = disk_images[fname]
        fhash = disk_hashes[fname]

        # ANTI-LEAKAGE CHECK
        if fhash in val_hashes:
            rejected_val_leak += 1
            continue
        if fhash in test_hashes:
            rejected_test_leak += 1
            continue

        cid = ann.get("category_id")
        cname = cat_map.get(cid, "desconocido")
        if cname == "desconocido":
            continue

        bbox = ann.get("bbox", [])
        if len(bbox) != 4 or bbox[2] <= 2 or bbox[3] <= 2:
            continue

        w = float(img_info.get("width", 512))
        h = float(img_info.get("height", 512))
        x1 = max(0.0, float(bbox[0]))
        y1 = max(0.0, float(bbox[1]))
        x2 = min(w, float(bbox[0] + bbox[2]))
        y2 = min(h, float(bbox[1] + bbox[3]))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue

        if fname not in clean_records:
            clean_records[fname] = {
                "file_name": fname,
                "path": fpath,
                "sha256": fhash,
                "width": int(w),
                "height": int(h),
                "boxes": []
            }

        clean_records[fname]["boxes"].append({
            "xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            "class": cname
        })

print(f"\nResultados de filtrado anti-fuga:")
print(f" - Anotaciones rechazadas por colisión con Val: {rejected_val_leak}")
print(f" - Anotaciones rechazadas por colisión con Test: {rejected_test_leak}")
print(f" - Imágenes limpias aprobadas: {len(clean_records)}")
total_boxes = sum(len(r['boxes']) for r in clean_records.values())
print(f" - Total cajas limpias aprobadas: {total_boxes}")

box_class_counter = Counter(b['class'] for r in clean_records.values() for b in r['boxes'])
print(f" - Cajas por clase:", dict(box_class_counter))

out_dir = Path("data/datos_separados_limpios")
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "clean_records.json", "w", encoding="utf-8") as f:
    json.dump(list(clean_records.values()), f, indent=2, ensure_ascii=False)
print("Guardado en data/datos_separados_limpios/clean_records.json")
