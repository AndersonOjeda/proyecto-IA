"""Phase 1 Audit of the 3 new external datasets for Experiment 4.
Strictly read-only on data/externos/ and data/preparado/deteccion/.
Extracts trashnet-seg.zip to data/preparado/exp04_extracted/trashnet_seg/ for inspection.
"""
from pathlib import Path
import sys
import json
import hashlib
import zipfile
import math
import re
from collections import Counter, defaultdict
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from residuos import CLASSES

EXT_ROOT = ROOT / 'data/externos'
ORIG_MANIFEST = ROOT / 'data/preparado/deteccion/manifest.json'
EXTRACT_DIR = ROOT / 'data/preparado/exp04_extracted/trashnet_seg'
OUT_DIR = ROOT / 'runs/detector_exp04/audit'

# Official 6-class mapping rules (strict: no BIODEGRADABLE, trash, or ALL mapped to organico)
STRICT_MAP = {
    'CARDBOARD': 'carton',
    'cardboard': 'carton',
    'GLASS': 'vidrio',
    'glass': 'vidrio',
    'METAL': 'metal',
    'metal': 'metal',
    'PAPER': 'papel',
    'paper': 'papel',
    'PLASTIC': 'plastico',
    'plastic': 'plastico',
}
AMBIGUOUS_CLASSES = {
    'BIODEGRADABLE', 'biodegradable',
    'trash', 'TRASH',
    'GARBAGE-GARBAGE-CLASSIFICATION', 'ALL'
}

def compute_dhash(im: Image.Image) -> int:
    """Compute 64-bit difference hash (dHash) for visual similarity detection."""
    gray = im.convert('L').resize((9, 8), Image.Resampling.BILINEAR)
    arr = np.asarray(gray, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    bits = diff.flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h

def hamming_distance(h1: int, h2: int) -> int:
    return (h1 ^ h2).bit_count()

def normalize_stem(filename: str) -> str:
    stem = Path(filename).name.split('.rf.')[0]
    stem = re.sub(r'_jpg$|_png$|_jpeg$', '', stem, flags=re.I)
    return stem.lower()

def bbox_valid(b, w, h):
    return len(b) == 4 and all(math.isfinite(v) for v in b) and 0 <= b[0] < b[2] <= w and 0 <= b[1] < b[3] <= h

def clip_box(b, w, h, tol=2.0):
    x1, y1, x2, y2 = b
    cx1, cy1, cx2, cy2 = max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)
    if cx2 > cx1 and cy2 > cy1:
        max_diff = max(abs(cx1 - x1), abs(cy1 - y1), abs(cx2 - x2), abs(cy2 - y2))
        if max_diff <= tol:
            return [cx1, cy1, cx2, cy2]
    return b

def check_grid32_capacity(boxes, w, h, grid=32, slots=3):
    counts = Counter()
    for b in boxes:
        x1, y1, x2, y2 = b['xyxy']
        gx = min(grid - 1, int(((x1 + x2) / 2.0 / w) * grid))
        gy = min(grid - 1, int(((y1 + y2) / 2.0 / h) * grid))
        counts[(gy, gx)] += 1
    return max(counts.values(), default=0) <= slots

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 0. Extract trashnet-seg.zip if not already extracted
    tn_zip = next(EXT_ROOT.rglob('trashnet-seg.zip'))
    if not EXTRACT_DIR.exists() or not list(EXTRACT_DIR.rglob('*.jpg')):
        print(f"Extracting {tn_zip.name} to {EXTRACT_DIR}...", flush=True)
        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(tn_zip, 'r') as zf:
            zf.extractall(EXTRACT_DIR)
            
    # 1. Load original manifest for overlap/leak checking
    orig_data = json.loads(ORIG_MANIFEST.read_text('utf-8'))
    orig_images = orig_data['images']
    print(f"Loaded original manifest: {len(orig_images)} total records.", flush=True)
    
    orig_sha_to_splits = defaultdict(set)
    orig_stem_to_splits = defaultdict(set)
    orig_val_test_dhashes = {}
    
    for r in orig_images:
        sp = r.get('split') or 'unassigned'
        if r.get('eligible'):
            sp_tag = f"eligible_{sp}"
        elif r.get('manual_batch'):
            sp_tag = "manual_batch"
        else:
            sp_tag = f"pending_{sp}"
        if r.get('sha256'):
            orig_sha_to_splits[r['sha256']].add(sp_tag)
        st = normalize_stem(r.get('original_path', r.get('path', '')))
        orig_stem_to_splits[st].add((r['source'], sp_tag))
        
    # Compute dHash for all original eligible val & test images (and manual batch)
    print("Computing dHashes for original val, test, and manual_batch images...", flush=True)
    det_root = ROOT / 'data/preparado/deteccion'
    for r in orig_images:
        if (r.get('eligible') and r.get('split') in ('val', 'test')) or r.get('manual_batch'):
            img_p = det_root / r['path']
            if not img_p.exists():
                img_p = ROOT / r['original_path']
            if img_p.exists():
                try:
                    with Image.open(img_p) as im:
                        orig_val_test_dhashes[r['id']] = (compute_dhash(im), r.get('split'), r['source'], r.get('manual_batch', False))
                except Exception:
                    pass
    print(f"Computed {len(orig_val_test_dhashes)} protected dHashes (val/test/manual).", flush=True)

    # 2. Helper to parse each of the 3 new datasets
    datasets_records = {}

    # --- Dataset A: Garbage Classification 3 (COCO) ---
    print("\nAuditing Dataset 1: Garbage Classification 3 (COCO)...", flush=True)
    gc_dir = EXT_ROOT / 'GARBAGE CLASSIFICATION 3.v2-gc1.coco'
    gc_records = []
    for split in ('train', 'valid', 'test'):
        ann_path = gc_dir / split / '_annotations.coco.json'
        if not ann_path.exists():
            continue
        doc = json.loads(ann_path.read_text('utf-8'))
        cats = {c['id']: c['name'] for c in doc['categories']}
        anns_by_im = defaultdict(list)
        for a in doc['annotations']:
            anns_by_im[a['image_id']].append(a)
        for im_info in doc['images']:
            img_path = ann_path.parent / im_info['file_name']
            gc_records.append({
                'source': 'Garbage Classification 3',
                'split': split,
                'path': img_path,
                'file_name': im_info['file_name'],
                'declared_w': im_info.get('width'),
                'declared_h': im_info.get('height'),
                'raw_anns': [
                    {
                        'orig_class': cats.get(a['category_id'], 'UNKNOWN'),
                        'bbox_coco': a.get('bbox'),
                        'segmentation': a.get('segmentation')
                    }
                    for a in anns_by_im[im_info['id']]
                ]
            })
    datasets_records['Garbage Classification 3'] = gc_records

    # --- Dataset B: Trash Detection Dataset (YOLO) ---
    print("Auditing Dataset 2: Trash Detection Dataset (YOLO)...", flush=True)
    td_dir = EXT_ROOT / 'Trash Detection Dataset/Trash Detection Dataset/CUSTOM_DATASET'
    td_names = ['BIODEGRADABLE', 'CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']
    td_records = []
    for split in ('train', 'valid', 'test'):
        im_dir = td_dir / split / 'images'
        lb_dir = td_dir / split / 'labels'
        if not im_dir.exists():
            continue
        for img_path in sorted(im_dir.iterdir()):
            if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            lb_path = lb_dir / (img_path.stem + '.txt')
            raw_anns = []
            if lb_path.exists():
                for line in lb_path.read_text('utf-8').splitlines():
                    parts = line.split()
                    if not parts:
                        continue
                    cid = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    raw_anns.append({
                        'orig_class': td_names[cid] if 0 <= cid < len(td_names) else 'UNKNOWN',
                        'yolo_coords': coords
                    })
            td_records.append({
                'source': 'Trash Detection Dataset',
                'split': split,
                'path': img_path,
                'file_name': img_path.name,
                'has_label_file': lb_path.exists(),
                'raw_anns': raw_anns
            })
    datasets_records['Trash Detection Dataset'] = td_records

    # --- Dataset C: Re-annotated TrashNet (YOLOv8 seg/bbox) ---
    print("Auditing Dataset 3: Re-annotated TrashNet (YOLOv8)...", flush=True)
    tn_base = EXTRACT_DIR / 'trashnet - Copy'
    tn_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    tn_records = []
    for split in ('train', 'valid', 'test'):
        im_dir = tn_base / split / 'images'
        lb_dir = tn_base / split / 'labels'
        if not im_dir.exists():
            continue
        for img_path in sorted(im_dir.iterdir()):
            if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            lb_path = lb_dir / (img_path.stem + '.txt')
            raw_anns = []
            if lb_path.exists():
                for line in lb_path.read_text('utf-8').splitlines():
                    parts = line.split()
                    if not parts:
                        continue
                    cid = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    raw_anns.append({
                        'orig_class': tn_names[cid] if 0 <= cid < len(tn_names) else 'UNKNOWN',
                        'yolo_coords': coords
                    })
            tn_records.append({
                'source': 'TrashNet Re-annotated',
                'split': split,
                'path': img_path,
                'file_name': img_path.name,
                'has_label_file': lb_path.exists(),
                'raw_anns': raw_anns
            })
    datasets_records['TrashNet Re-annotated'] = tn_records

    # 3. Process and verify every image across the 3 datasets
    audit_summary = {}
    all_valid_records = []
    global_sha_seen = defaultdict(list)
    global_stem_seen = defaultdict(list)

    for ds_name, recs in datasets_records.items():
        print(f"Processing {len(recs)} records for {ds_name}...", flush=True)
        stats = Counter()
        orig_class_counts = Counter()
        mapped_class_counts = Counter()
        clean_image_class_counts = Counter()
        fmt_counts = Counter()
        
        for r in recs:
            stats['total_records'] += 1
            p = r['path']
            if not p.exists():
                stats['missing_file'] += 1
                continue
            try:
                sha = hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
                with Image.open(p) as im:
                    w, h = im.width, im.height
                    dhash = compute_dhash(im)
            except Exception:
                stats['unreadable_image'] += 1
                continue
                
            r['sha256'] = sha
            r['width'] = w
            r['height'] = h
            r['dhash'] = dhash
            r['stem'] = normalize_stem(r['file_name'])
            
            global_sha_seen[sha].append((ds_name, r['split'], r['file_name']))
            global_stem_seen[r['stem']].append((ds_name, r['split'], r['file_name']))
            
            # Parse annotations
            boxes = []
            ambiguous_present = False
            invalid_box_present = False
            
            for a in r['raw_anns']:
                oc = a['orig_class']
                orig_class_counts[oc] += 1
                
                if oc in AMBIGUOUS_CLASSES or oc not in STRICT_MAP:
                    ambiguous_present = True
                    continue
                target_cls = STRICT_MAP[oc]
                
                # Convert geometry
                if 'bbox_coco' in a and a['bbox_coco']:
                    bx, by, bw, bh = a['bbox_coco']
                    xyxy = [bx, by, bx + bw, by + bh]
                    fmt = 'coco_bbox'
                elif 'yolo_coords' in a:
                    yc = a['yolo_coords']
                    if len(yc) == 4:
                        cx, cy, bw, bh = yc
                        xyxy = [(cx - bw / 2.0) * w, (cy - bh / 2.0) * h, (cx + bw / 2.0) * w, (cy + bh / 2.0) * h]
                        fmt = 'yolo_bbox'
                    elif len(yc) >= 6 and len(yc) % 2 == 0:
                        xs, ys = yc[::2], yc[1::2]
                        xyxy = [min(xs) * w, min(ys) * h, max(xs) * w, max(ys) * h]
                        fmt = 'yolo_polygon_bbox'
                    else:
                        invalid_box_present = True
                        continue
                else:
                    invalid_box_present = True
                    continue
                    
                xyxy = clip_box(xyxy, w, h, tol=2.0)
                if not bbox_valid(xyxy, w, h):
                    invalid_box_present = True
                    continue
                fmt_counts[fmt] += 1
                boxes.append({'class': target_cls, 'xyxy': [round(v, 4) for v in xyxy], 'original_class': oc, 'format': fmt})
                mapped_class_counts[target_cls] += 1

            r['boxes'] = boxes
            r['ambiguous_present'] = ambiguous_present
            r['invalid_box_present'] = invalid_box_present
            
            if not r['raw_anns']:
                stats['empty_annotations'] += 1
            elif ambiguous_present:
                stats['rejected_ambiguous_scene'] += 1
            elif invalid_box_present or not boxes:
                stats['rejected_invalid_geometry'] += 1
            elif not check_grid32_capacity(boxes, w, h, grid=32, slots=3):
                stats['rejected_grid32_capacity'] += 1
            else:
                stats['clean_eligible_before_dedup'] += 1
                r['clean'] = True
                for b in boxes:
                    clean_image_class_counts[b['class']] += 1
                all_valid_records.append(r)

        audit_summary[ds_name] = {
            'stats': dict(stats),
            'original_class_instances': dict(orig_class_counts),
            'clean_scene_object_instances': dict(clean_image_class_counts),
            'formats': dict(fmt_counts)
        }
        print(f"  Summary for {ds_name}: {dict(stats)}", flush=True)
        print(f"  Clean object instances: {dict(clean_image_class_counts)}", flush=True)

    # 4. Cross-dataset duplication and leak analysis against Original Manifest
    print("\nRunning Cross-Dataset Deduplication & Leakage Checks...", flush=True)
    leak_stats = Counter()
    deduped_clean_records = []
    seen_clean_sha = set()
    seen_clean_stems_per_source = set()

    # Check how many stems or SHAs overlap between Garbage Classification 3 and Trash Detection Dataset
    gc_stems = {normalize_stem(r['file_name']) for r in datasets_records['Garbage Classification 3']}
    td_stems = {normalize_stem(r['file_name']) for r in datasets_records['Trash Detection Dataset']}
    tn_stems = {normalize_stem(r['file_name']) for r in datasets_records['TrashNet Re-annotated']}
    print(f"  Stem overlap GC3 vs Trash Detection: {len(gc_stems & td_stems)} stems!", flush=True)
    print(f"  Stem overlap GC3 vs TrashNet: {len(gc_stems & tn_stems)} stems!", flush=True)
    print(f"  Stem overlap Trash Detection vs TrashNet: {len(td_stems & tn_stems)} stems!", flush=True)

    for r in all_valid_records:
        sha = r['sha256']
        st = r['stem']
        dh = r['dhash']
        
        # Check against original manifest SHA
        if sha in orig_sha_to_splits:
            leak_stats[f"exact_sha_in_orig_{sorted(orig_sha_to_splits[sha])[0]}"] += 1
            continue
            
        # Check against original manifest stem (e.g. dataset-resized or manual_batch)
        if st in orig_stem_to_splits:
            tags = [t[1] for t in orig_stem_to_splits[st]]
            if 'manual_batch' in tags or 'eligible_val' in tags or 'eligible_test' in tags:
                leak_stats[f"stem_match_protected_{tags[0]}"] += 1
                continue
                
        # Check visual similarity (dHash Hamming distance <= 4) against protected val/test/manual_batch
        near_dup_protected = False
        for prot_id, (prot_dh, prot_sp, prot_src, prot_man) in orig_val_test_dhashes.items():
            if hamming_distance(dh, prot_dh) <= 4:
                near_dup_protected = True
                leak_stats[f"visual_near_dup_protected_{prot_src}_{prot_sp}"] += 1
                break
        if near_dup_protected:
            continue
            
        # Check exact SHA duplicate among new clean records
        if sha in seen_clean_sha:
            leak_stats['duplicate_sha_within_new'] += 1
            continue
            
        # Check Roboflow augmentation stem duplicate (keep 1 canonical version or group them)
        # Note: if we only add to train, multiple augmentations of the same train stem don't leak to val/test,
        # BUT deduplicating by stem prevents over-representing augmented copies of the exact same photo!
        if st in seen_clean_stems_per_source:
            leak_stats['duplicate_stem_augmentation_within_new'] += 1
            continue
            
        seen_clean_sha.add(sha)
        seen_clean_stems_per_source.add(st)
        deduped_clean_records.append(r)

    # Summarize final clean deduplicated pool
    final_boxes_by_source_class = defaultdict(Counter)
    final_images_by_source = Counter()
    final_boxes_total = Counter()
    final_images_per_class = Counter()

    for r in deduped_clean_records:
        src = r['source']
        final_images_by_source[src] += 1
        present_cls = set()
        for b in r['boxes']:
            c = b['class']
            final_boxes_by_source_class[src][c] += 1
            final_boxes_total[c] += 1
            present_cls.add(c)
        for c in present_cls:
            final_images_per_class[c] += 1

    full_report = {
        'datasets_audit': audit_summary,
        'stem_overlaps': {
            'gc3_vs_trash_detection': len(gc_stems & td_stems),
            'gc3_vs_trashnet': len(gc_stems & tn_stems),
            'trash_detection_vs_trashnet': len(td_stems & tn_stems)
        },
        'deduplication_and_leak_filtering': dict(leak_stats),
        'clean_unique_pool': {
            'total_unique_images': len(deduped_clean_records),
            'images_by_source': dict(final_images_by_source),
            'images_containing_class': dict(final_images_per_class),
            'boxes_total_by_class': dict(final_boxes_total),
            'boxes_by_source_and_class': {k: dict(v) for k, v in final_boxes_by_source_class.items()}
        }
    }

    report_path = OUT_DIR / 'fase1_auditoria_externos.json'
    report_path.write_text(json.dumps(full_report, indent=2, ensure_ascii=False), encoding='utf-8')
    print("\n=== FINAL CLEAN UNIQUE POOL (DEDUPLICATED BY SHA, STEM, AND DHASH) ===", flush=True)
    print(json.dumps(full_report['clean_unique_pool'], indent=2, ensure_ascii=False), flush=True)
    print("Filtering breakdown:", json.dumps(dict(leak_stats), indent=2), flush=True)
    print(f"Saved full report to {report_path}", flush=True)

if __name__ == '__main__':
    main()
