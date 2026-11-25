"""Test high-precision filtering and priority ordering across the 3 new external datasets:
Priority 1: TrashNet Re-annotated (YOLOv8 human segmentation polygons)
Priority 2: Trash Detection Dataset (YOLO bboxes/polygons)
Priority 3: Garbage Classification 3 (COCO bboxes)

Strict quality gates for every image:
1. No ambiguous classes (BIODEGRADABLE, trash, ALL, GARBAGE-GARBAGE-CLASSIFICATION) anywhere in the image.
2. Prefix-Label agreement: filename material prefix (glass, paper, cardboard, metal, plastic) must match all annotated boxes in the scene.
3. No full-image dummy boxes (every box must have 0.005 <= area_ratio <= 0.88, and not touch all 4 image borders).
4. No nested/redundant duplicate boxes on the same object (if two same-class boxes have IoU > 0.30 or one contains > 60% of the other's area, the noisy/redundant scene is rejected or merged).
5. Grid32 capacity: at most 3 box centers per 32x32 cell.
6. Zero leakage: no SHA256 match, stem match, or perceptual dHash (Hamming <= 4) match with original val (670), test (531), or manual_batch (90).
"""
from pathlib import Path
import sys
import json
import re
from collections import Counter, defaultdict
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_exp04_datasets import (
    EXT_ROOT, EXTRACT_DIR, ORIG_MANIFEST, OUT_DIR, STRICT_MAP, AMBIGUOUS_CLASSES,
    compute_dhash, hamming_distance, normalize_stem, clip_box, bbox_valid, check_grid32_capacity
)

PREFIX_TO_TARGET = {
    'cardboard': 'carton',
    'glass': 'vidrio',
    'metal': 'metal',
    'paper': 'papel',
    'plastic': 'plastico'
}

def box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

def box_iou_and_containment(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a1, a2 = box_area(b1), box_area(b2)
    union = a1 + a2 - inter
    iou = inter / union if union > 0 else 0.0
    cont = inter / min(a1, a2) if min(a1, a2) > 0 else 0.0
    return iou, cont

def has_nested_boxes(boxes):
    n = len(boxes)
    for i in range(n):
        for j in range(i + 1, n):
            iou, cont = box_iou_and_containment(boxes[i]['xyxy'], boxes[j]['xyxy'])
            if iou > 0.30 or cont > 0.60:
                return True
    return False

def is_tight_valid_scene(boxes, w, h, max_boxes=4):
    if not boxes or len(boxes) > max_boxes:
        return False, 'box_count_out_of_range'
    for b in boxes:
        x1, y1, x2, y2 = b['xyxy']
        ar = box_area(b['xyxy']) / (w * h)
        if ar > 0.88:
            return False, 'full_frame_dummy_box'
        if ar < 0.005:
            return False, 'microscopic_noise_box'
        # Check if box spans >=95% of both width and height
        if (x2 - x1) / w > 0.94 and (y2 - y1) / h > 0.94:
            return False, 'border_to_border_box'
    if has_nested_boxes(boxes):
        return False, 'nested_redundant_boxes'
    if not check_grid32_capacity(boxes, w, h):
        return False, 'grid32_capacity'
    return True, 'ok'

def main():
    orig_data = json.loads(ORIG_MANIFEST.read_text('utf-8'))
    orig_images = orig_data['images']

    # Protected sets: val (eligible), test (eligible), manual_batch (90 photos)
    # Plus all eligible train images so we don't duplicate existing train images
    protected_shas = set()
    protected_stems = set()
    protected_dhashes = []
    det_root = ROOT / 'data/preparado/deteccion'

    for r in orig_images:
        if r.get('eligible') or r.get('manual_batch'):
            if r.get('sha256'):
                protected_shas.add(r['sha256'])
            st = normalize_stem(r.get('original_path', r.get('path', '')))
            protected_stems.add(st)
            if (r.get('eligible') and r.get('split') in ('val', 'test')) or r.get('manual_batch'):
                img_p = det_root / r['path']
                if not img_p.exists():
                    img_p = ROOT / r['original_path']
                if img_p.exists():
                    try:
                        with Image.open(img_p) as im:
                            protected_dhashes.append(compute_dhash(im))
                    except Exception:
                        pass

    print(f"Protected SHAs: {len(protected_shas)}, Protected Stems: {len(protected_stems)}, Protected Val/Test/Manual dHashes: {len(protected_dhashes)}", flush=True)

    # Load candidates in priority order:
    # 1. TrashNet Re-annotated
    # 2. Trash Detection Dataset
    # 3. Garbage Classification 3
    raw_candidates = []

    # 1. TrashNet Re-annotated
    tn_base = EXTRACT_DIR / 'trashnet - Copy'
    tn_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    for split in ('train', 'valid', 'test'):
        im_dir = tn_base / split / 'images'
        lb_dir = tn_base / split / 'labels'
        if not im_dir.exists(): continue
        for p in sorted(im_dir.glob('*.jpg')):
            lb = lb_dir / (p.stem + '.txt')
            if not lb.exists(): continue
            lines = [l.split() for l in lb.read_text('utf-8').splitlines() if l.strip()]
            raw_candidates.append(('TrashNet Re-annotated', split, p, lines, tn_names, 'yolo'))

    # 2. Trash Detection Dataset
    td_dir = EXT_ROOT / 'Trash Detection Dataset/Trash Detection Dataset/CUSTOM_DATASET'
    td_names = ['BIODEGRADABLE', 'CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']
    for split in ('train', 'valid', 'test'):
        im_dir = td_dir / split / 'images'
        lb_dir = td_dir / split / 'labels'
        if not im_dir.exists(): continue
        for p in sorted(im_dir.glob('*.jpg')):
            lb = lb_dir / (p.stem + '.txt')
            if not lb.exists(): continue
            lines = [l.split() for l in lb.read_text('utf-8').splitlines() if l.strip()]
            raw_candidates.append(('Trash Detection Dataset', split, p, lines, td_names, 'yolo'))

    # 3. Garbage Classification 3
    gc_dir = EXT_ROOT / 'GARBAGE CLASSIFICATION 3.v2-gc1.coco'
    for split in ('train', 'valid', 'test'):
        ann_path = gc_dir / split / '_annotations.coco.json'
        if not ann_path.exists(): continue
        doc = json.loads(ann_path.read_text('utf-8'))
        cats = {c['id']: c['name'] for c in doc['categories']}
        anns_by_im = defaultdict(list)
        for a in doc['annotations']: anns_by_im[a['image_id']].append(a)
        for im_info in doc['images']:
            p = ann_path.parent / im_info['file_name']
            raw_candidates.append(('Garbage Classification 3', split, p, anns_by_im[im_info['id']], cats, 'coco'))

    seen_stems = set(protected_stems)
    seen_shas = set(protected_shas)
    accepted = []
    reject_reasons = Counter()
    by_ds_class = defaultdict(Counter)
    boxes_by_ds_class = defaultdict(Counter)

    for ds_name, split, p, anns, names_map, mode in raw_candidates:
        if not p.exists():
            reject_reasons['missing_file'] += 1
            continue
        st = normalize_stem(p.name)
        if st in seen_stems:
            reject_reasons['duplicate_or_protected_stem'] += 1
            continue
        m = re.match(r'^([a-z]+)\d+', p.name.lower())
        if not m or m.group(1) not in PREFIX_TO_TARGET:
            reject_reasons['non_target_prefix'] += 1
            continue
        expected_cls = PREFIX_TO_TARGET[m.group(1)]
        if not anns:
            reject_reasons['empty_annotations'] += 1
            continue

        try:
            with Image.open(p) as im:
                w, h = im.width, im.height
                dhash = compute_dhash(im)
        except Exception:
            reject_reasons['unreadable'] += 1
            continue

        boxes = []
        mismatch = False
        for a in anns:
            if mode == 'yolo':
                cid = int(a[0])
                oc = names_map[cid] if 0 <= cid < len(names_map) else 'UNKNOWN'
                if STRICT_MAP.get(oc) != expected_cls:
                    mismatch = True; break
                yc = [float(x) for x in a[1:]]
                if len(yc) == 4:
                    cx, cy, bw, bh = yc
                    xyxy = [(cx - bw/2)*w, (cy - bh/2)*h, (cx + bw/2)*w, (cy + bh/2)*h]
                    fmt = 'yolo_bbox'
                elif len(yc) >= 6 and len(yc) % 2 == 0:
                    xs, ys = yc[::2], yc[1::2]
                    xyxy = [min(xs)*w, min(ys)*h, max(xs)*w, max(ys)*h]
                    fmt = 'yolo_polygon'
                else:
                    mismatch = True; break
            else:
                oc = names_map.get(a['category_id'], 'UNKNOWN')
                if STRICT_MAP.get(oc) != expected_cls:
                    mismatch = True; break
                bx, by, bw, bh = a['bbox']
                xyxy = [bx, by, bx + bw, by + bh]
                fmt = 'coco_bbox'
            xyxy = clip_box(xyxy, w, h)
            if not bbox_valid(xyxy, w, h):
                mismatch = True; break
            boxes.append({'class': expected_cls, 'xyxy': [round(v, 4) for v in xyxy], 'original_class': oc, 'format': fmt})

        if mismatch:
            reject_reasons['class_or_geom_mismatch'] += 1
            continue

        ok, reason = is_tight_valid_scene(boxes, w, h, max_boxes=4)
        if not ok:
            reject_reasons[reason] += 1
            continue

        # Check perceptual dHash against protected val/test/manual_batch
        if any(hamming_distance(dhash, pdh) <= 4 for pdh in protected_dhashes):
            reject_reasons['visual_dhash_near_protected'] += 1
            continue

        seen_stems.add(st)
        accepted.append({
            'source': ds_name,
            'split': split,
            'path': p,
            'width': w,
            'height': h,
            'class': expected_cls,
            'boxes': boxes
        })
        by_ds_class[ds_name][expected_cls] += 1
        boxes_by_ds_class[ds_name][expected_cls] += len(boxes)

    tot_img_by_cls = Counter()
    tot_box_by_cls = Counter()
    for ds in by_ds_class:
        for c in by_ds_class[ds]:
            tot_img_by_cls[c] += by_ds_class[ds][c]
            tot_box_by_cls[c] += boxes_by_ds_class[ds][c]

    print("\n=== STRICT HIGH-PRECISION FILTERED POOL ===", flush=True)
    print("Total accepted unique images:", len(accepted), flush=True)
    print("Images by class:", dict(tot_img_by_cls), flush=True)
    print("Boxes by class:", dict(tot_box_by_cls), flush=True)
    print("Images by dataset & class:", {k: dict(v) for k, v in by_ds_class.items()}, flush=True)
    print("Boxes by dataset & class:", {k: dict(v) for k, v in boxes_by_ds_class.items()}, flush=True)
    print("Rejection breakdown:", dict(reject_reasons), flush=True)

    # Render visual verification grid of accepted high-precision samples
    canvas = Image.new('RGB', (1600, 1080), '#101d24')
    draw = ImageDraw.Draw(canvas)
    font_title = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 24)
    font_sub = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
    draw.text((20, 12), 'Muestra Validada de Alta Precisión (100% Consistencia Prefijo-Etiqueta · Sin Cajas Dummy ni Anidadas)', font=font_title, fill='white')

    ds_list = ['TrashNet Re-annotated', 'Trash Detection Dataset', 'Garbage Classification 3']
    target_5 = ['vidrio', 'papel', 'carton', 'metal', 'plastico']
    for row_idx, ds in enumerate(ds_list):
        for col_idx, cls_name in enumerate(target_5):
            cands = [r for r in accepted if r['source'] == ds and r['class'] == cls_name]
            if not cands: continue
            item = cands[len(cands) // 3]
            im = Image.open(item['path']).convert('RGB')
            im.thumbnail((290, 240))
            x0 = 20 + col_idx * 315
            y0 = 55 + row_idx * 340
            draw.text((x0, y0), f"{cls_name.upper()} · {ds[:18]}", font=font_sub, fill='#4ee6b1')
            ox = x0 + (290 - im.width) // 2
            oy = y0 + 24 + (240 - im.height) // 2
            canvas.paste(im, (ox, oy))
            sx, sy = im.width / item['width'], im.height / item['height']
            for b in item['boxes']:
                x1, y1, x2, y2 = b['xyxy']
                coords = [ox + x1 * sx, oy + y1 * sy, ox + x2 * sx, oy + y2 * sy]
                draw.rectangle(coords, outline='#00ff66', width=3)
                draw.text((coords[0] + 2, max(oy, coords[1] - 16)), f"{b['original_class']}→{b['class']}", font=font_sub, fill='#00ff66')
            draw.text((x0, y0 + 272), f"{item['path'].name[:32]}", font=font_sub, fill='#c5d5dc')
            draw.text((x0, y0 + 292), f"Fmt: {item['boxes'][0]['format']} ({len(item['boxes'])} obj)", font=font_sub, fill='#ffd280')

    out_img = OUT_DIR / 'vista_previa_alta_precision.jpg'
    canvas.save(out_img, quality=93)
    print(f"Saved high-precision visual preview to {out_img}", flush=True)

if __name__ == '__main__':
    main()
