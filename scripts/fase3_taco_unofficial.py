"""Phase 3: estimate useful objects in TACO *unofficial* annotations (same taxonomy/domain as val).
Uses the project's existing source_mapping for TACO (no new label rules). Images are NOT downloaded."""
from pathlib import Path
import sys, json, collections
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from audit_external_datasets import source_mapping
from residuos import CLASSES
from residuos.data import write_json

T = ROOT / 'data/externos/TACO-master/TACO-master/data'
un = json.load(open(T / 'annotations_unofficial.json', encoding='utf8'))
of = json.load(open(T / 'annotations.json', encoding='utf8'))
cat = {c['id']: c['name'] for c in un['categories']}
imgs = {i['id']: i for i in un['images']}
off_urls = {i.get('flickr_url') for i in of['images']} | {i.get('flickr_640_url') for i in of['images']}
off_urls.discard(None); off_urls.discard('')
dup_imgs = {i['id'] for i in un['images'] if i.get('flickr_url') in off_urls or i.get('flickr_640_url') in off_urls}

per_img = collections.defaultdict(list)
for a in un['annotations']:
    m, status, _ = source_mapping('TACO oficial', cat[a['category_id']])
    per_img[a['image_id']].append((m if status in ('propuesto', 'condicional') and m in CLASSES else None, a, cat[a['category_id']]))

stats = {c: collections.Counter() for c in CLASSES}
complete_imgs = set(); amb_names = collections.Counter(); lic = collections.Counter()
for iid, anns in per_img.items():
    im = imgs[iid]; lic[str(im.get('license'))] += 1
    complete = all(m is not None for m, _, _ in anns) and iid not in dup_imgs
    if complete: complete_imgs.add(iid)
    for m, a, n in anns:
        if m is None: amb_names[n] += 1; continue
        x, y, w, h = a['bbox']; s = max(w, h) * 256 / max(im['width'], im['height'])
        st = stats[m]; st['objects_mapped'] += 1
        st['small<32' if s < 32 else 'medium32-96' if s < 96 else 'large>=96'] += 1
        st['has_polygon'] += bool(a.get('segmentation'))
        if complete: st['objects_in_complete_images'] += 1
for c in CLASSES:
    stats[c]['images_complete_with_class'] = sum(1 for i in complete_imgs if any(m == c for m, _, _ in per_img[i]))

rep = {'source': 'TACO annotations_unofficial.json (local)', 'images': len(un['images']), 'annotations': len(un['annotations']),
       'images_overlapping_official_by_url': len(dup_imgs), 'images_all_boxes_unambiguous': len(complete_imgs),
       'license_field_counts': dict(lic), 'images_downloaded': False,
       'per_class': {c: dict(v) for c, v in stats.items()}, 'unmapped_or_ambiguous_names': dict(amb_names.most_common(20)),
       'caveats': ['Anotaciones no oficiales (comunidad); requieren revision visual por muestreo antes de usarlas.',
                   'Imagenes alojadas en Flickr: descargar y verificar SHA-256/dHash contra val, test y lote manual antes de usar.',
                   'Solo imagenes con todas las cajas mapeables (sin Unlabeled litter ni clases ambiguas) se proponen para entrenamiento.']}
write_json(ROOT / 'runs/consolidacion/fase3_taco_unofficial.json', rep)
print(json.dumps({k: v for k, v in rep.items() if k != 'caveats'}, indent=1, ensure_ascii=False))

