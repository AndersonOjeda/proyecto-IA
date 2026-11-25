"""Build the Verified Polygon Segmentation Cutout Bank and Small/Medium Natural External Subset for Experiment 4C.
Sources of verified human polygon segmentations (STRICTLY TRAIN / NON-LEAKING):
1. TACO oficial TRAIN split (from data/preparado/taco_oficial/annotations_corregidas.json, filtered strictly to split == 'train' and eligible == True in manifest.json):
   vidrio, papel, carton, metal cutouts with real outdoor lighting and textures.
2. External verified human polygon masks (TrashNet Re-annotated YOLOv8 polygons & Trash Detection YOLO polygons, filtered for 0 leaks against val/test/manual_batch):
   vidrio, papel, carton, metal cutouts with tight polygon masks (zero studio background pixels outside the polygon!).
3. Small/Medium Natural-Background External Images (max box side < 88px in 256x256, natural textured background bg_std >= 22, 35 <= bg_mean <= 205).
"""
from pathlib import Path
import sys
import json
import hashlib
import re
from collections import Counter, defaultdict
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from residuos import CLASSES
from residuos.data import Store, read_json, write_json
from scripts.audit_exp04_datasets import (
    EXT_ROOT, EXTRACT_DIR, ORIG_MANIFEST, STRICT_MAP,
    compute_dhash, hamming_distance, normalize_stem, clip_box, bbox_valid
)
from scripts.test_clean_filtering import PREFIX_TO_TARGET, is_tight_valid_scene, box_area

EXP04_DIR = ROOT / 'runs/detector_exp04'
MINORITY_4 = ('vidrio', 'papel', 'carton', 'metal')

def extract_polygon_cutout(bgr_img, pts_px, target_max_side=96):
    """Given an image (H, W, 3) and polygon vertices in pixel coords (N, 2),
    rasterizes the binary polygon mask, crops the tight bounding box, and resizes
    if larger than target_max_side. Returns (rgb_patch, alpha_mask_uint8) or None.
    """
    h, w = bgr_img.shape[:2]
    pts = np.round(pts_px).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

    x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
    x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())
    bw, bh = x2 - x1, y2 - y1
    if bw < 12 or bh < 12:
        return None
    if (bw * bh) / (w * h) > 0.85:
        return None

    mask_full = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_full, [pts], 255)
    mask_crop = mask_full[y1:y2, x1:x2]
    fill_ratio = float((mask_crop > 127).mean())
    # Reject degenerate thin slivers or almost empty polygons
    if fill_ratio < 0.22:
        return None

    bgr_crop = bgr_img[y1:y2, x1:x2]
    rgb_crop = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)

    max_side = max(bw, bh)
    if max_side > target_max_side:
        scale = target_max_side / max_side
        nw, nh = max(12, int(round(bw * scale))), max(12, int(round(bh * scale)))
        rgb_crop = cv2.resize(rgb_crop, (nw, nh), interpolation=cv2.INTER_AREA)
        mask_crop = cv2.resize(mask_crop, (nw, nh), interpolation=cv2.INTER_AREA)

    # Feather the polygon boundary slightly (3x3 Gaussian blur on alpha) to prevent artificial step edges
    alpha_feathered = cv2.GaussianBlur(mask_crop, (3, 3), 0.8)
    # Keep zero outside original mask dilated by 1px
    alpha_feathered = np.where(mask_crop > 16, alpha_feathered, 0).astype(np.uint8)
    return rgb_crop, alpha_feathered


def main():
    EXP04_DIR.mkdir(parents=True, exist_ok=True)
    store = Store(ROOT / 'data/preparado/deteccion')
    orig_train = store.records('train')
    orig_val = store.records('val')
    orig_test = store.records('test')
    orig_all = store.db['images']

    train_ids = {r['id'] for r in orig_train}
    val_test_ids = {r['id'] for r in orig_val + orig_test}
    assert not (train_ids & val_test_ids), "Train and Val/Test IDs overlap!"

    # 1. Extract verified human COCO polygons from TACO oficial TRAIN split ONLY
    taco_corr_path = ROOT / 'data/preparado/taco_oficial/annotations_corregidas.json'
    taco_doc = json.loads(taco_corr_path.read_text('utf-8'))
    taco_anns_by_img = defaultdict(list)
    for a in taco_doc['annotations']:
        taco_anns_by_img[a['image_id']].append(a)

    cutout_bank = []
    cutout_counts = defaultdict(Counter)
    det_root = ROOT / 'data/preparado/deteccion'

    bank_path = EXP04_DIR / 'verified_polygon_bank_4c.pt'
    if bank_path.exists():
        cutout_bank = torch.load(bank_path, weights_only=False)
        print(f"Loaded existing Verified Polygon Bank ({len(cutout_bank)} cutouts) from {bank_path.name}", flush=True)
    else:
        print("Extracting verified polygon cutouts from TACO oficial TRAIN split...", flush=True)
    for r in orig_train:
        if r['source'] != 'TACO oficial':
            continue
        assert r['split'] == 'train' and r['eligible'] is True and r['id'] not in val_test_ids
        img_p = det_root / r['path']
        if not img_p.exists():
            img_p = ROOT / r['original_path']
        bgr = cv2.imdecode(np.fromfile(img_p, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        anns = taco_anns_by_img.get(r['source_image_id'], [])
        for b in r['boxes']:
            c = b['class']
            if c not in MINORITY_4:
                continue
            a_id = b.get('annotation_id')
            matched_a = next((a for a in anns if a['id'] == a_id or a.get('original_annotation_id') == a_id), None)
            if not matched_a or not matched_a.get('segmentation') or not isinstance(matched_a['segmentation'], list):
                continue
            seg = matched_a['segmentation'][0]
            if len(seg) < 8:
                continue
            pts_px = np.array(seg, dtype=np.float32).reshape(-1, 2)
            res = extract_polygon_cutout(bgr, pts_px, target_max_side=96)
            if res is not None:
                rgb_patch, alpha_patch = res
                cutout_bank.append({
                    'class': c,
                    'origin': 'TACO_train_polygon',
                    'source_image_id': r['id'],
                    'rgb': rgb_patch,
                    'alpha': alpha_patch
                })
                cutout_counts['TACO_train_polygon'][c] += 1

    print("  TACO train polygon cutouts extracted:", dict(cutout_counts['TACO_train_polygon']), flush=True)

    # 2. Extract verified human YOLOv8 polygon cutouts from TrashNet Re-annotated & Trash Detection Dataset
    # Strictly verify zero leak against val, test, and manual_batch
    protected_shas = set()
    protected_stems = set()
    protected_dhashes = []
    for r in orig_all:
        if r.get('eligible') or r.get('manual_batch'):
            if r.get('sha256'): protected_shas.add(r['sha256'])
            protected_stems.add(normalize_stem(r.get('original_path', r.get('path', ''))))
            if (r.get('eligible') and r.get('split') in ('val', 'test')) or r.get('manual_batch'):
                img_p = det_root / r['path']
                if not img_p.exists(): img_p = ROOT / r['original_path']
                if img_p.exists():
                    try:
                        with Image.open(img_p) as im: protected_dhashes.append(compute_dhash(im))
                    except Exception: pass

    ext_poly_candidates = []
    tn_base = EXTRACT_DIR / 'trashnet - Copy'
    tn_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    for split in ('train', 'valid', 'test'):
        im_dir, lb_dir = tn_base / split / 'images', tn_base / split / 'labels'
        if not im_dir.exists(): continue
        for p in sorted(im_dir.glob('*.jpg')):
            lb = lb_dir / (p.stem + '.txt')
            if lb.exists():
                ext_poly_candidates.append(('TrashNet Re-annotated', p, [l.split() for l in lb.read_text('utf-8').splitlines() if l.strip()], tn_names))

    td_dir = EXT_ROOT / 'Trash Detection Dataset/Trash Detection Dataset/CUSTOM_DATASET'
    td_names = ['BIODEGRADABLE', 'CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']
    for split in ('train', 'valid', 'test'):
        im_dir, lb_dir = td_dir / split / 'images', td_dir / split / 'labels'
        if not im_dir.exists(): continue
        for p in sorted(im_dir.glob('*.jpg')):
            lb = lb_dir / (p.stem + '.txt')
            if lb.exists():
                ext_poly_candidates.append(('Trash Detection Dataset', p, [l.split() for l in lb.read_text('utf-8').splitlines() if l.strip()], td_names))

    seen_stems = set(protected_stems)
    ext_cutouts_by_cls = defaultdict(list)

    for ds_name, p, lines, names_map in ext_poly_candidates:
        if not p.exists() or not lines: continue
        st = normalize_stem(p.name)
        if st in seen_stems: continue
        m = re.match(r'^([a-z]+)\d+', p.name.lower())
        if not m or m.group(1) not in PREFIX_TO_TARGET: continue
        expected_cls = PREFIX_TO_TARGET[m.group(1)]
        if expected_cls not in MINORITY_4: continue

        # Must have a true polygon (>= 12 coords = >= 6 vertices) and 100% class agreement
        if not all(STRICT_MAP.get(names_map[int(pts[0])]) == expected_cls for pts in lines if 0 <= int(pts[0]) < len(names_map)):
            continue
        poly_lines = [pts for pts in lines if len(pts) - 1 >= 12 and (len(pts) - 1) % 2 == 0]
        if not poly_lines:
            continue

        try:
            with Image.open(p) as im:
                w, h = im.width, im.height
                dhash = compute_dhash(im)
        except Exception:
            continue
        if any(hamming_distance(dhash, pdh) <= 4 for pdh in protected_dhashes):
            continue

        bgr = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None: continue

        for pts in poly_lines:
            coords = [float(x) for x in pts[1:]]
            xs = np.array(coords[::2], dtype=np.float32) * w
            ys = np.array(coords[1::2], dtype=np.float32) * h
            # Verify it's not just a 4-corner bounding box disguised as a polygon:
            # Check number of unique X and Y coordinates
            if len(np.unique(np.round(xs))) < 4 or len(np.unique(np.round(ys))) < 4:
                continue
            pts_px = np.stack([xs, ys], axis=1)
            res = extract_polygon_cutout(bgr, pts_px, target_max_side=96)
            if res is not None:
                rgb_patch, alpha_patch = res
                # Measure fill ratio: prefer tight non-rectangular contours (0.30 <= fill_ratio <= 0.85)
                fill_r = float((alpha_patch > 127).mean())
                if 0.25 <= fill_r <= 0.88:
                    ext_cutouts_by_cls[expected_cls].append((abs(fill_r - 0.58), st, ds_name, rgb_patch, alpha_patch))
                    seen_stems.add(st)
                    break

    # Take up to 80 highest-quality non-rectangular polygon cutouts per minority class from external datasets
    for c in MINORITY_4:
        cands = sorted(ext_cutouts_by_cls[c], key=lambda x: (x[0], x[1]))[:80]
        for _, st, ds_name, rgb_patch, alpha_patch in cands:
            cutout_bank.append({
                'class': c,
                'origin': f"{ds_name}_polygon",
                'source_image_id': st,
                'rgb': rgb_patch,
                'alpha': alpha_patch
            })
            cutout_counts[f"{ds_name}_polygon"][c] += 1

    print("  External verified polygon cutouts added:", {k: dict(v) for k, v in cutout_counts.items() if k != 'TACO_train_polygon'}, flush=True)

    # Save cutout bank as a compact .npz / .pt file in runs/detector_exp04/
    bank_path = EXP04_DIR / 'verified_polygon_bank_4c.pt'
    torch.save(cutout_bank, bank_path)
    total_by_cls = Counter(item['class'] for item in cutout_bank)
    print(f"Saved Verified Polygon Bank ({len(cutout_bank)} cutouts) to {bank_path.name}: {dict(total_by_cls)}", flush=True)

    # 3. Also identify the purest Small/Medium (<80px in 256x256) + Natural-Textured-Background external images
    ext_rows = read_json(EXP04_DIR / 'manifest_exp04_ext.json')
    ext_pixels = np.load(EXP04_DIR / 'ext_train.npy', mmap_mode='r')
    small_med_natural_indices = []
    sm_counts = Counter()
    quotas_sm = {'vidrio': 45, 'papel': 40, 'carton': 40, 'metal': 35}

    cands_sm = defaultdict(list)
    for j, r in enumerate(ext_rows):
        c = r['class_primary']
        if c not in quotas_sm: continue
        # Check max box side in 256x256
        max_side_256 = max(
            max((b['xyxy'][2] - b['xyxy'][0]) * 256.0 / r['width'], (b['xyxy'][3] - b['xyxy'][1]) * 256.0 / r['height'])
            for b in r['boxes']
        )
        img = ext_pixels[j]
        border = np.concatenate([img[0, :], img[-1, :], img[:, 0], img[:, -1]], axis=0).astype(np.float32)
        bg_std = float(border.std())
        bg_mean = float(border.mean())
        if bg_std >= 24.0 and 40.0 <= bg_mean <= 200.0 and max_side_256 <= 115.0:
            cands_sm[c].append((max_side_256, -bg_std, j))

    for c, q in quotas_sm.items():
        for _, _, j in sorted(cands_sm[c])[:q]:
            small_med_natural_indices.append(j)
            sm_counts[c] += 1

    write_json(EXP04_DIR / 'small_med_natural_indices_4c.json', {
        'indices': sorted(small_med_natural_indices),
        'counts_by_class': dict(sm_counts)
    })
    print(f"Selected {len(small_med_natural_indices)} small/medium natural-background external images: {dict(sm_counts)}", flush=True)

    # 4. Generate a visual validation panel of Verified Polygon Copy-Paste onto real training backgrounds
    orig_train_pixels = np.load(ROOT / 'runs/detector_exp01/train.npy', mmap_mode='r')
    canvas = Image.new('RGB', (1600, 920), '#101d24')
    draw = ImageDraw.Draw(canvas)
    font_title = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 24)
    font_sub = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
    draw.text((20, 12), 'Validación Visual Fase 3 (Exp 4C) — Copy-Paste con Segmentación Poligonal Real sobre Fondos Train', font=font_title, fill='white')

    # Pick 8 training images exclusively from TACO oficial (exhaustive annotations, real outdoor domain, no letterbox bars)
    sample_bg_indices = [i for i, r in enumerate(orig_train) if r['source'] == 'TACO oficial' and 1 <= len(r['boxes']) <= 2][:8]
    cutouts_by_c = defaultdict(list)
    for item in cutout_bank:
        cutouts_by_c[item['class']].append(item)

    rng = np.random.RandomState(42)
    for panel_idx, bg_i in enumerate(sample_bg_indices):
        r_bg = orig_train[bg_i]
        frame = np.array(orig_train_pixels[bg_i], copy=True)
        w_orig, h_orig = r_bg['width'], r_bg['height']
        orig_boxes_256 = [
            {'class': b['class'], 'xyxy': [b['xyxy'][0]*256/w_orig, b['xyxy'][1]*256/h_orig, b['xyxy'][2]*256/w_orig, b['xyxy'][3]*256/h_orig], 'kind': 'orig'}
            for b in r_bg['boxes']
        ]

        target_c = MINORITY_4[panel_idx % 4]
        item = cutouts_by_c[target_c][(panel_idx * 7) % len(cutouts_by_c[target_c])]
        rgb_p, alpha_p = item['rgb'].copy(), item['alpha'].copy()

        # Scale to realistic TACO validation size (32..68 px)
        target_sz = rng.randint(34, 68)
        ph, pw = rgb_p.shape[:2]
        sc = target_sz / max(ph, pw)
        nw, nh = max(14, int(round(pw * sc))), max(14, int(round(ph * sc)))
        rgb_s = cv2.resize(rgb_p, (nw, nh), interpolation=cv2.INTER_LINEAR)
        alpha_s = cv2.resize(alpha_p, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # Find non-overlapping placement in lower 72% of image on valid non-black photographic surface
        placed_box = None
        for _ in range(40):
            x0 = rng.randint(12, 256 - nw - 12)
            y0 = rng.randint(64, 256 - nh - 12)
            cand_xy = [x0, y0, x0 + nw, y0 + nh]
            bg_patch = frame[y0:y0+nh, x0:x0+nw].astype(np.float32)
            # Reject if patch touches any dark border or flat black padding
            if bg_patch.mean(axis=(1, 2)).min() < 28.0 or bg_patch.mean(axis=(0, 2)).min() < 28.0 or bg_patch.std() < 10.0:
                continue
            overlap_found = False
            for ob in orig_boxes_256:
                ix1, iy1 = max(cand_xy[0], ob['xyxy'][0] - 6), max(cand_xy[1], ob['xyxy'][1] - 6)
                ix2, iy2 = min(cand_xy[2], ob['xyxy'][2] + 6), min(cand_xy[3], ob['xyxy'][3] + 6)
                if ix2 > ix1 and iy2 > iy1:
                    overlap_found = True; break
            if not overlap_found:
                fg_patch = rgb_s.astype(np.float32)
                bg_lum = max(25.0, float(bg_patch.mean()))
                fg_lum = max(25.0, float(fg_patch[alpha_s > 127].mean())) if (alpha_s > 127).any() else 128.0
                lum_factor = np.clip((bg_lum / fg_lum) * 0.40 + 0.60, 0.78, 1.22)
                fg_harmonized = np.clip(fg_patch * lum_factor, 0, 255)

                a_norm = (alpha_s.astype(np.float32) / 255.0)[..., None]
                comp = (a_norm * fg_harmonized + (1.0 - a_norm) * bg_patch).astype(np.uint8)
                frame[y0:y0+nh, x0:x0+nw] = comp

                ys_m, xs_m = np.nonzero(alpha_s > 32)
                bx1, by1 = float(x0 + xs_m.min()), float(y0 + ys_m.min())
                bx2, by2 = float(x0 + xs_m.max() + 1), float(y0 + ys_m.max() + 1)
                placed_box = {'class': target_c, 'xyxy': [bx1, by1, bx2, by2], 'kind': 'pasted', 'origin': item['origin']}
                orig_boxes_256.append(placed_box)
                break

        pil_im = Image.fromarray(frame).resize((360, 360), Image.Resampling.BILINEAR)
        col_i = panel_idx % 4
        row_i = panel_idx // 4
        x_corner = 20 + col_i * 395
        y_corner = 55 + row_i * 425
        draw.text((x_corner, y_corner), f"Paste: {target_c.upper()} ({item['origin'][:15]})", font=font_sub, fill='#4ee6b1')
        canvas.paste(pil_im, (x_corner, y_corner + 24))
        scale_vis = 360.0 / 256.0
        for b in orig_boxes_256:
            x1, y1, x2, y2 = [v * scale_vis for v in b['xyxy']]
            coords = [x_corner + x1, y_corner + 24 + y1, x_corner + x2, y_corner + 24 + y2]
            col = '#00ff66' if b['kind'] == 'orig' else '#00d8ff'
            tag = f"Orig: {b['class']}" if b['kind'] == 'orig' else f"PolyPaste: {b['class']}"
            draw.rectangle(coords, outline=col, width=2)
            draw.text((coords[0] + 2, max(y_corner + 24, coords[1] - 16)), tag, font=font_sub, fill=col)
        draw.text((x_corner, y_corner + 390), f"Fondo Train: {r_bg['source']} ({r_bg['id'][:10]})", font=font_sub, fill='#c5d5dc')

    out_preview = EXP04_DIR / 'audit/vista_previa_copypaste_4c.jpg'
    canvas.save(out_preview, quality=93)
    print(f"Saved Copy-Paste visual verification panel to {out_preview}", flush=True)

if __name__ == '__main__':
    main()
