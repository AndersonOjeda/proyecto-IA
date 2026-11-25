"""Phase 2: Dataset Balancing, Deduplication, Verification, and Cache Generation for Experiment 4.
Preserves all original manifests and files intact.
Builds:
- runs/detector_exp04/manifest_exp04_ext.json (the 1,600 curated high-precision external records)
- runs/detector_exp04/ext_train.npy (256x256 RGB uint8 cache for the 1,600 external records)
- runs/detector_exp04/fase2_balanceo_datos.json (complete distribution before/after import & balancing)
"""
from pathlib import Path
import sys
import json
import hashlib
import re
from collections import Counter, defaultdict
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from residuos import CLASSES
from residuos.data import Store, read_json, write_json
from scripts.audit_exp04_datasets import (
    EXT_ROOT, EXTRACT_DIR, ORIG_MANIFEST, STRICT_MAP,
    compute_dhash, hamming_distance, normalize_stem, clip_box, bbox_valid, check_grid32_capacity
)
from scripts.test_clean_filtering import PREFIX_TO_TARGET, is_tight_valid_scene, box_area

EXP04_DIR = ROOT / 'runs/detector_exp04'
SEED = 42

QUOTAS_EXT = {
    'vidrio': 400,
    'papel': 400,
    'carton': 400,
    'metal': 320,
    'plastico': 80
}

def main():
    EXP04_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Verify original manifest integrity
    store = Store(ROOT / 'data/preparado/deteccion')
    orig_train = store.records('train')
    orig_val = store.records('val')
    orig_test = store.records('test')
    assert len(orig_train) == 3645, f"Expected 3645 train images, got {len(orig_train)}"
    assert len(orig_val) == 670, f"Expected 670 val images, got {len(orig_val)}"
    assert len(orig_test) == 531, f"Expected 531 test images, got {len(orig_test)}"

    orig_train_boxes = Counter()
    orig_train_imgs_per_cls = Counter()
    for r in orig_train:
        for b in r['boxes']:
            orig_train_boxes[b['class']] += 1
        for c in {b['class'] for b in r['boxes']}:
            orig_train_imgs_per_cls[c] += 1

    orig_val_boxes = Counter()
    for r in orig_val:
        for b in r['boxes']:
            orig_val_boxes[b['class']] += 1

    # 2. Collect protected hashes, stems, and dHashes
    orig_all = store.db['images']
    protected_shas = set()
    protected_stems = set()
    protected_val_test_manual_shas = set()
    protected_dhashes = []
    det_root = ROOT / 'data/preparado/deteccion'

    for r in orig_all:
        if r.get('eligible') or r.get('manual_batch'):
            if r.get('sha256'):
                protected_shas.add(r['sha256'])
            st = normalize_stem(r.get('original_path', r.get('path', '')))
            protected_stems.add(st)
            if (r.get('eligible') and r.get('split') in ('val', 'test')) or r.get('manual_batch'):
                if r.get('sha256'):
                    protected_val_test_manual_shas.add(r['sha256'])
                img_p = det_root / r['path']
                if not img_p.exists():
                    img_p = ROOT / r['original_path']
                if img_p.exists():
                    try:
                        with Image.open(img_p) as im:
                            protected_dhashes.append(compute_dhash(im))
                    except Exception:
                        pass

    # 3. Gather all high-precision candidates across the 3 external sources
    raw_candidates = []

    # Priority 1: TrashNet Re-annotated (YOLOv8 polygons)
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
            raw_candidates.append(('TrashNet Re-annotated', split, p, lines, tn_names, 'yolo', 'CC BY 4.0 (Mendeley / Roboflow trashnetsegrohit-2)'))

    # Priority 2: Trash Detection Dataset (YOLO)
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
            raw_candidates.append(('Trash Detection Dataset', split, p, lines, td_names, 'yolo', 'CC BY 4.0 (Mendeley z732f9pwxt/1)'))

    # Priority 3: Garbage Classification 3 (COCO)
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
            raw_candidates.append(('Garbage Classification 3', split, p, anns_by_im[im_info['id']], cats, 'coco', 'CC BY 4.0 (Roboflow Garbage Classification 3 v2)'))

    seen_stems = set(protected_stems)
    seen_shas = set(protected_shas)
    pool_by_class = defaultdict(list)

    print("Filtering and deduplicating external candidates...", flush=True)
    for ds_name, split, p, anns, names_map, mode, license_str in raw_candidates:
        if not p.exists(): continue
        st = normalize_stem(p.name)
        if st in seen_stems: continue
        m = re.match(r'^([a-z]+)\d+', p.name.lower())
        if not m or m.group(1) not in PREFIX_TO_TARGET: continue
        expected_cls = PREFIX_TO_TARGET[m.group(1)]
        if not anns: continue

        try:
            sha = hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
            if sha in seen_shas: continue
            with Image.open(p) as im:
                w, h = im.width, im.height
                dhash = compute_dhash(im)
        except Exception:
            continue

        boxes = []
        mismatch = False
        for idx_a, a in enumerate(anns):
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
            boxes.append({
                'class': expected_cls,
                'xyxy': [round(v, 4) for v in xyxy],
                'original_class': oc,
                'annotation_id': idx_a,
                'format': fmt
            })

        if mismatch: continue
        ok, _ = is_tight_valid_scene(boxes, w, h, max_boxes=4)
        if not ok: continue
        if any(hamming_distance(dhash, pdh) <= 4 for pdh in protected_dhashes):
            continue

        seen_stems.add(st)
        seen_shas.add(sha)

        max_ar = max(box_area(b['xyxy']) / (w * h) for b in boxes)
        mean_ar = float(np.mean([box_area(b['xyxy']) / (w * h) for b in boxes]))
        rec_id = hashlib.sha256(f"exp04|{ds_name}|{p.name}|{sha}".encode()).hexdigest()[:20]

        pool_by_class[expected_cls].append({
            'id': rec_id,
            'path': p.relative_to(ROOT).as_posix(),
            'original_path': p.relative_to(ROOT).as_posix(),
            'source': ds_name,
            'original_split': split,
            'split': 'train',
            'width': w,
            'height': h,
            'sha256': sha,
            'dhash': dhash,
            'stem': st,
            'class_primary': expected_cls,
            'max_area_ratio': max_ar,
            'mean_area_ratio': mean_ar,
            'boxes': boxes,
            'reviewed': True,
            'eligible': True,
            'manual_batch': False,
            'provenance': {
                'license': license_str,
                'dataset': ds_name
            }
        })

    # 4. Select balanced subset prioritizing natural object scales (closer to TACO validation distribution)
    # and balanced representation across the 3 external sources
    selected_ext = []
    selected_by_src_cls = defaultdict(Counter)
    selected_boxes_by_cls = Counter()
    selected_imgs_by_cls = Counter()

    for c, quota in QUOTAS_EXT.items():
        cands = pool_by_class[c]
        # Sort candidates to prioritize natural scales (0.015 <= mean_area_ratio <= 0.45 first),
        # then deterministic hash tie-breaker for full diversity across sources
        def score_cand(r):
            ar = r['mean_area_ratio']
            # Distance from ideal TACO-like scale band [0.02, 0.35]
            if 0.02 <= ar <= 0.35:
                scale_tier = 0
            elif 0.01 <= ar <= 0.55:
                scale_tier = 1
            else:
                scale_tier = 2
            tie = hashlib.sha256(f"{SEED}:{r['id']}".encode()).hexdigest()
            return (scale_tier, tie)

        cands_sorted = sorted(cands, key=score_cand)
        chosen = cands_sorted[:min(quota, len(cands_sorted))]
        for r in chosen:
            selected_ext.append(r)
            selected_by_src_cls[r['source']][c] += 1
            selected_imgs_by_cls[c] += 1
            selected_boxes_by_cls[c] += len(r['boxes'])

    # Sort selected_ext deterministically by ID
    selected_ext.sort(key=lambda r: r['id'])

    # 5. Double-check ZERO leakage with original val (670), test (531), and manual_batch (90)
    for r in selected_ext:
        assert r['sha256'] not in protected_val_test_manual_shas, f"SHA leak detected: {r['id']}"
        assert r['stem'] not in protected_stems, f"Stem leak detected: {r['id']}"
        assert all(hamming_distance(r['dhash'], pdh) > 4 for pdh in protected_dhashes), f"dHash leak detected: {r['id']}"

    print(f"Selected {len(selected_ext)} external training images with 0 leaks.", flush=True)

    # 6. Build 256x256 RGB uint8 cache for the selected external images
    ext_npy_path = EXP04_DIR / 'ext_train.npy'
    print(f"Building 256x256 pixel cache at {ext_npy_path}...", flush=True)
    cache_arr = np.empty((len(selected_ext), 256, 256, 3), dtype=np.uint8)
    for idx, r in enumerate(selected_ext):
        img_p = ROOT / r['path']
        bgr = cv2.imdecode(np.fromfile(img_p, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to decode image {img_p}")
        resized = cv2.resize(bgr, (256, 256), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        cache_arr[idx] = rgb
    np.save(ext_npy_path, cache_arr)
    print(f"Saved {ext_npy_path.name} ({cache_arr.shape}, {cache_arr.nbytes / (1024*1024):.1f} MB).", flush=True)

    # Save external manifest
    # Strip integer dhash or keep it as hex string for JSON
    for r in selected_ext:
        r['dhash_hex'] = f"{r['dhash']:016x}"
        del r['dhash']
    write_json(EXP04_DIR / 'manifest_exp04_ext.json', selected_ext)

    # 7. Calculate complete pre-import vs post-import class distributions
    combined_train_boxes = Counter(orig_train_boxes)
    combined_train_imgs = Counter(orig_train_imgs_per_cls)
    for c in CLASSES:
        combined_train_boxes[c] += selected_boxes_by_cls[c]
        combined_train_imgs[c] += selected_imgs_by_cls[c]

    # Also compute effective epoch distribution when TACO minority images (132 images) are oversampled 2x
    taco_minority_indices = [
        i for i, r in enumerate(orig_train)
        if r['source'] == 'TACO oficial' and any(b['class'] in ('vidrio', 'papel', 'carton', 'metal') for b in r['boxes'])
    ]
    epoch_boxes = Counter(combined_train_boxes)
    for i in taco_minority_indices:
        for b in orig_train[i]['boxes']:
            epoch_boxes[b['class']] += 1

    # Compute smooth bounded class weights (power 0.25) on the new balanced distribution
    counts_arr = np.array([epoch_boxes[c] for c in CLASSES], dtype=np.float64)
    raw_w = np.power(np.median(counts_arr) / counts_arr, 0.25)
    raw_w = np.clip(raw_w, 0.60, 1.50)
    norm_w = raw_w / np.mean(raw_w)
    class_weights_exp04 = {c: round(float(w), 4) for c, w in zip(CLASSES, norm_w)}

    balance_report = {
        'seed': SEED,
        'original_train_2010': {
            'total_images': len(orig_train),
            'images_per_class': dict(orig_train_imgs_per_cls),
            'boxes_per_class': dict(orig_train_boxes),
            'total_boxes': sum(orig_train_boxes.values()),
            'max_to_min_box_ratio': round(max(orig_train_boxes.values()) / min(orig_train_boxes.values()), 2)
        },
        'imported_high_precision_pool_5205': {
            'total_unique_images': sum(len(v) for v in pool_by_class.values()),
            'images_per_class': {c: len(pool_by_class[c]) for c in CLASSES},
            'boxes_per_class': {c: sum(len(r['boxes']) for r in pool_by_class[c]) for c in CLASSES}
        },
        'selected_balanced_external_1600': {
            'total_images': len(selected_ext),
            'quotas_per_class': QUOTAS_EXT,
            'images_per_class': dict(selected_imgs_by_cls),
            'boxes_per_class': dict(selected_boxes_by_cls),
            'images_by_source_and_class': {k: dict(v) for k, v in selected_by_src_cls.items()}
        },
        'combined_exp04_train_3610': {
            'total_unique_images': len(orig_train) + len(selected_ext),
            'images_per_class': dict(combined_train_imgs),
            'boxes_per_class': dict(combined_train_boxes),
            'total_boxes': sum(combined_train_boxes.values()),
            'max_to_min_box_ratio': round(max(combined_train_boxes.values()) / min(combined_train_boxes.values()), 2),
            'improvement_factors_boxes': {
                c: f"{orig_train_boxes[c]} -> {combined_train_boxes[c]} ({combined_train_boxes[c]/orig_train_boxes[c]:.1f}x)"
                for c in CLASSES
            }
        },
        'effective_epoch_with_taco_anchoring': {
            'taco_minority_oversampled_images': len(taco_minority_indices),
            'total_samples_per_epoch': len(orig_train) + len(selected_ext) + len(taco_minority_indices),
            'effective_boxes_per_epoch': dict(epoch_boxes),
            'computed_class_weights_exp04': class_weights_exp04
        },
        'validation_set_unchanged_670': {
            'total_images': len(orig_val),
            'boxes_per_class': dict(orig_val_boxes),
            'total_boxes': sum(orig_val_boxes.values())
        },
        'leakage_verification': {
            'exact_sha256_overlaps_with_val_test_manual': 0,
            'stem_overlaps_with_orig': 0,
            'perceptual_dhash_leaks_hamming_le_4': 0,
            'manual_90_photos_included': False,
            'ambiguous_biodegradable_or_trash_included': False
        }
    }

    write_json(EXP04_DIR / 'fase2_balanceo_datos.json', balance_report)
    print("\n=== PHASE 2 BALANCING REPORT ===", flush=True)
    print(json.dumps(balance_report['combined_exp04_train_3610'], indent=2, ensure_ascii=False), flush=True)
    print("Effective epoch & class weights:", json.dumps(balance_report['effective_epoch_with_taco_anchoring'], indent=2), flush=True)

if __name__ == '__main__':
    main()
