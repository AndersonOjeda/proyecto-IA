"""Visual and statistical validation of bounding boxes from the 3 new external datasets.
Checks box area ratios (to detect lazy full-image boxes or microscopic noise) and renders
a 3x5 visual grid (3 datasets x 5 classes: vidrio, papel, carton, metal, plastico).
"""
from pathlib import Path
import sys
import json
from collections import Counter, defaultdict
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_exp04_datasets import (
    EXT_ROOT, EXTRACT_DIR, OUT_DIR, STRICT_MAP, AMBIGUOUS_CLASSES,
    normalize_stem, clip_box, bbox_valid, check_grid32_capacity
)

TARGET_5 = ['vidrio', 'papel', 'carton', 'metal', 'plastico']

def load_clean_samples():
    samples_by_ds_cls = defaultdict(lambda: defaultdict(list))
    area_stats = defaultdict(list)
    full_image_boxes = Counter()
    tiny_boxes = Counter()
    total_boxes_checked = Counter()

    # 1. Garbage Classification 3
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
            if not p.exists(): continue
            w, h = im_info['width'], im_info['height']
            anns = anns_by_im[im_info['id']]
            if not anns: continue
            if any(cats.get(a['category_id']) not in STRICT_MAP for a in anns): continue
            boxes = []
            for a in anns:
                bx, by, bw, bh = a['bbox']
                xyxy = clip_box([bx, by, bx + bw, by + bh], w, h)
                if not bbox_valid(xyxy, w, h):
                    boxes = []; break
                area_ratio = ((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])) / (w * h)
                area_stats['Garbage Classification 3'].append(area_ratio)
                total_boxes_checked['Garbage Classification 3'] += 1
                if area_ratio > 0.95: full_image_boxes['Garbage Classification 3'] += 1
                if area_ratio < 0.0005: tiny_boxes['Garbage Classification 3'] += 1
                boxes.append({'class': STRICT_MAP[cats[a['category_id']]], 'xyxy': xyxy, 'orig': cats[a['category_id']], 'fmt': 'COCO bbox'})
            if boxes and check_grid32_capacity(boxes, w, h):
                for c in {b['class'] for b in boxes}:
                    samples_by_ds_cls['Garbage Classification 3'][c].append({'path': p, 'w': w, 'h': h, 'boxes': boxes, 'split': split})

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
            if not lines: continue
            if any(td_names[int(pts[0])] not in STRICT_MAP for pts in lines): continue
            with Image.open(p) as im:
                w, h = im.width, im.height
            boxes = []
            for pts in lines:
                oc = td_names[int(pts[0])]
                yc = [float(x) for x in pts[1:]]
                if len(yc) == 4:
                    cx, cy, bw, bh = yc
                    xyxy = [(cx - bw/2)*w, (cy - bh/2)*h, (cx + bw/2)*w, (cy + bh/2)*h]
                    fmt = 'YOLO bbox'
                elif len(yc) >= 6 and len(yc) % 2 == 0:
                    xs, ys = yc[::2], yc[1::2]
                    xyxy = [min(xs)*w, min(ys)*h, max(xs)*w, max(ys)*h]
                    fmt = 'YOLO polygon'
                else:
                    boxes = []; break
                xyxy = clip_box(xyxy, w, h)
                if not bbox_valid(xyxy, w, h):
                    boxes = []; break
                area_ratio = ((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])) / (w * h)
                area_stats['Trash Detection Dataset'].append(area_ratio)
                total_boxes_checked['Trash Detection Dataset'] += 1
                if area_ratio > 0.95: full_image_boxes['Trash Detection Dataset'] += 1
                if area_ratio < 0.0005: tiny_boxes['Trash Detection Dataset'] += 1
                boxes.append({'class': STRICT_MAP[oc], 'xyxy': xyxy, 'orig': oc, 'fmt': fmt})
            if boxes and check_grid32_capacity(boxes, w, h):
                for c in {b['class'] for b in boxes}:
                    samples_by_ds_cls['Trash Detection Dataset'][c].append({'path': p, 'w': w, 'h': h, 'boxes': boxes, 'split': split})

    # 3. TrashNet Re-annotated
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
            if not lines: continue
            if any(tn_names[int(pts[0])] not in STRICT_MAP for pts in lines): continue
            with Image.open(p) as im:
                w, h = im.width, im.height
            boxes = []
            for pts in lines:
                oc = tn_names[int(pts[0])]
                yc = [float(x) for x in pts[1:]]
                if len(yc) == 4:
                    cx, cy, bw, bh = yc
                    xyxy = [(cx - bw/2)*w, (cy - bh/2)*h, (cx + bw/2)*w, (cy + bh/2)*h]
                    fmt = 'YOLOv8 bbox'
                elif len(yc) >= 6 and len(yc) % 2 == 0:
                    xs, ys = yc[::2], yc[1::2]
                    xyxy = [min(xs)*w, min(ys)*h, max(xs)*w, max(ys)*h]
                    fmt = 'YOLOv8 polygon'
                else:
                    boxes = []; break
                xyxy = clip_box(xyxy, w, h)
                if not bbox_valid(xyxy, w, h):
                    boxes = []; break
                area_ratio = ((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])) / (w * h)
                area_stats['TrashNet Re-annotated'].append(area_ratio)
                total_boxes_checked['TrashNet Re-annotated'] += 1
                if area_ratio > 0.95: full_image_boxes['TrashNet Re-annotated'] += 1
                if area_ratio < 0.0005: tiny_boxes['TrashNet Re-annotated'] += 1
                boxes.append({'class': STRICT_MAP[oc], 'xyxy': xyxy, 'orig': oc, 'fmt': fmt})
            if boxes and check_grid32_capacity(boxes, w, h):
                for c in {b['class'] for b in boxes}:
                    samples_by_ds_cls['TrashNet Re-annotated'][c].append({'path': p, 'w': w, 'h': h, 'boxes': boxes, 'split': split})

    geom_report = {}
    for ds in ('Garbage Classification 3', 'Trash Detection Dataset', 'TrashNet Re-annotated'):
        arr = np.array(area_stats[ds])
        geom_report[ds] = {
            'total_boxes': total_boxes_checked[ds],
            'mean_area_ratio': float(np.mean(arr)),
            'median_area_ratio': float(np.median(arr)),
            'p10_area_ratio': float(np.percentile(arr, 10)),
            'p90_area_ratio': float(np.percentile(arr, 90)),
            'full_image_boxes_gt_95pct': full_image_boxes[ds],
            'tiny_boxes_lt_005pct': tiny_boxes[ds]
        }
        print(f"[{ds}] Geometry audit: {geom_report[ds]}", flush=True)

    (OUT_DIR / 'geometria_cajas_nuevos_datasets.json').write_text(json.dumps(geom_report, indent=2), encoding='utf-8')

    # Render 3x5 visual grid
    canvas = Image.new('RGB', (1600, 1080), '#101d24')
    draw = ImageDraw.Draw(canvas)
    font_title = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 24)
    font_sub = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
    draw.text((20, 12), 'Validación Visual Fase 1 — Nuevos Datasets Externos (Cajas Originales y Polígonos Convertidos)', font=font_title, fill='white')

    ds_list = ['Garbage Classification 3', 'Trash Detection Dataset', 'TrashNet Re-annotated']
    selected_meta = []
    for row_idx, ds in enumerate(ds_list):
        for col_idx, cls_name in enumerate(TARGET_5):
            cands = samples_by_ds_cls[ds][cls_name]
            # Pick a representative sample with 1-4 boxes and reasonable box area
            cands_sorted = sorted(cands, key=lambda x: (abs(len(x['boxes']) - 2), x['path'].name))
            item = cands_sorted[min(5, len(cands_sorted) - 1)]
            im = Image.open(item['path']).convert('RGB')
            im.thumbnail((290, 240))
            x0 = 20 + col_idx * 315
            y0 = 55 + row_idx * 340
            draw.text((x0, y0), f"{cls_name.upper()} · {ds[:18]}", font=font_sub, fill='#4ee6b1')
            ox = x0 + (290 - im.width) // 2
            oy = y0 + 24 + (240 - im.height) // 2
            canvas.paste(im, (ox, oy))
            sx, sy = im.width / item['w'], im.height / item['h']
            for b in item['boxes']:
                x1, y1, x2, y2 = b['xyxy']
                coords = [ox + x1 * sx, oy + y1 * sy, ox + x2 * sx, oy + y2 * sy]
                draw.rectangle(coords, outline='#ffee33', width=2)
                draw.text((coords[0] + 2, max(oy, coords[1] - 15)), f"{b['orig']}→{b['class']}", font=font_sub, fill='#ffee33')
            draw.text((x0, y0 + 272), f"{item['path'].name[:32]}", font=font_sub, fill='#c5d5dc')
            draw.text((x0, y0 + 292), f"Fmt: {item['boxes'][0]['fmt']} ({len(item['boxes'])} obj)", font=font_sub, fill='#ffd280')
            selected_meta.append({
                'dataset': ds, 'class': cls_name, 'file': item['path'].name,
                'num_boxes': len(item['boxes']), 'format': item['boxes'][0]['fmt']
            })

    out_img = OUT_DIR / 'vista_previa_nuevos_datasets.jpg'
    canvas.save(out_img, quality=93)
    (OUT_DIR / 'muestra_visual_validada.json').write_text(json.dumps(selected_meta, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"Saved visual validation panel to {out_img}", flush=True)

if __name__ == '__main__':
    load_clean_samples()
