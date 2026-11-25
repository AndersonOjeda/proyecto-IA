"""Phase 1 and Phase 2 Audit for Experiment 4C:
1. Homogeneous comparison of Exp 3C, Exp 4A, and Exp 4B across identical thresholds,
   and Precision / False Positives at equivalent Recall levels (10%, 15%, 18.7%, 20%, 25%, 27.4%, 30%, 35%).
2. Detailed audit of the 5,205 clean external images (6,227 boxes) + TACO train polygons:
   - Natural vs studio backgrounds
   - Object scale distribution (<32px, 32-96px, >=96px in 256x256)
   - Valid human polygon segmentation masks available per class and dataset
   - Identification of domain-shift sources vs clean polygon cutouts.
"""
from pathlib import Path
import sys
import json
import re
from collections import Counter, defaultdict
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from residuos import CLASSES
from residuos.data import Store, read_json, write_json
from scripts.experiment3_runner import DetectorS8, CachedDataset32, GRID
from scripts.audit_evaluation import ap50_voc101, overlaps, nms_indices, extract_candidates_generic
from scripts.audit_exp04_datasets import (
    EXT_ROOT, EXTRACT_DIR, ORIG_MANIFEST, STRICT_MAP,
    compute_dhash, hamming_distance, normalize_stem, clip_box, bbox_valid
)
from scripts.test_clean_filtering import PREFIX_TO_TARGET, is_tight_valid_scene, box_area

EXP04_DIR = ROOT / 'runs/detector_exp04'

def evaluate_equal_recall_and_breakdown():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    store = Store(ROOT / 'data/preparado/deteccion')
    val_rows = store.records('val')
    val_npy = ROOT / 'runs/detector_exp01/val.npy'
    valset = CachedDataset32(val_rows, val_npy, augment=False)
    loader = torch.utils.data.DataLoader(valset, batch_size=16, shuffle=False)

    gt_counts = Counter(b['class'] for r in val_rows for b in r['boxes'])
    total_gt = sum(gt_counts.values())

    checkpoints = {
        'Exp3C': ROOT / 'runs/detector_exp03/corrected/best.pt',
        'Exp4A_Ep17': EXP04_DIR / 'variant_4a_broad/best_val_loss.pt',
        'Exp4A_Ep40': EXP04_DIR / 'variant_4a_broad/best.pt',
        'Exp4B_Ep34': EXP04_DIR / 'variant_4b_domain_aligned/best.pt'
    }

    phase1_report = {}

    for name, ckpt_path in checkpoints.items():
        saved = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = DetectorS8().to(device)
        model.load_state_dict(saved['state'])
        model.eval()

        outputs = []
        with torch.no_grad():
            for x, _ in loader:
                outputs.extend(model(x.to(device)).cpu())

        all_global_scores = []
        all_global_hits = []
        class_scores = {c: [] for c in CLASSES}
        class_hits = {c: [] for c in CLASSES}
        source_scores = defaultdict(lambda: defaultdict(list))
        source_hits = defaultdict(lambda: defaultdict(list))
        source_gt = defaultdict(Counter)

        pos_p_obj = {c: [] for c in CLASSES}
        pos_p_cls = {c: [] for c in CLASSES}
        neg_p_obj_by_src = defaultdict(list)

        for i, r in enumerate(val_rows):
            out = outputs[i]
            w, h = r['width'], r['height']
            src = r['source']
            xy, cls, scores, p_obj, p_cls = extract_candidates_generic(out, w, h, GRID)
            neg_p_obj_by_src[src].append(float(np.mean(p_obj)))

            gt_boxes = [b['xyxy'] for b in r['boxes']]
            gt_classes = [CLASSES.index(b['class']) for b in r['boxes']]
            for b in r['boxes']:
                source_gt[src][b['class']] += 1
                c_idx = CLASSES.index(b['class'])
                cx = (b['xyxy'][0] + b['xyxy'][2]) / 2.0 / w
                cy = (b['xyxy'][1] + b['xyxy'][3]) / 2.0 / h
                gx, gy = min(int(cx * GRID), GRID - 1), min(int(cy * GRID), GRID - 1)
                cell_z = out[gy, gx]
                co = torch.sigmoid(cell_z[:, 0]).numpy()
                cp = torch.softmax(cell_z[:, 5:], dim=-1)[:, c_idx].numpy()
                bs = np.argmax(co * cp)
                pos_p_obj[b['class']].append(float(co[bs]))
                pos_p_cls[b['class']].append(float(cp[bs]))

            kept = nms_indices(xy, cls, scores, threshold=0.45, max_detections=100)
            n_xy, n_cls, n_sc = xy[kept], cls[kept], scores[kept]
            used = set()
            if len(gt_boxes) > 0 and len(n_xy) > 0:
                ov = overlaps(n_xy, gt_boxes)
                for p_i in np.argsort(-n_sc, kind='stable'):
                    cp = n_cls[p_i]
                    cn = CLASSES[cp]
                    elig = [g_i for g_i in range(len(gt_boxes)) if gt_classes[g_i] == cp and g_i not in used]
                    vo, bg = max([(ov[p_i, g_i], g_i) for g_i in elig], default=(0.0, -1))
                    hit = (vo >= 0.5)
                    if hit:
                        used.add(bg)
                    sc_val = float(n_sc[p_i])
                    class_scores[cn].append(sc_val)
                    class_hits[cn].append(hit)
                    source_scores[src][cn].append(sc_val)
                    source_hits[src][cn].append(hit)
                    all_global_scores.append(sc_val)
                    all_global_hits.append(hit)
            elif len(n_xy) > 0:
                for p_i in range(len(n_xy)):
                    cn = CLASSES[n_cls[p_i]]
                    sc_val = float(n_sc[p_i])
                    class_scores[cn].append(sc_val)
                    class_hits[cn].append(False)
                    source_scores[src][cn].append(sc_val)
                    source_hits[src][cn].append(False)
                    all_global_scores.append(sc_val)
                    all_global_hits.append(False)

        # Compute equal-recall table
        g_sc = np.array(all_global_scores)
        g_hi = np.array(all_global_hits, dtype=int)
        order = np.argsort(-g_sc, kind='stable')
        g_sc_sorted = g_sc[order]
        cum_tp = np.cumsum(g_hi[order])
        cum_fp = np.arange(1, len(order) + 1) - cum_tp
        recalls = cum_tp / total_gt
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1)
        f1s = 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-12)

        equal_recall_points = {}
        for target_r in [0.10, 0.15, 0.1873, 0.20, 0.25, 0.2737, 0.30, 0.35, 0.40]:
            idx = int(np.searchsorted(recalls, target_r))
            if idx < len(recalls):
                equal_recall_points[f"recall_{target_r*100:.1f}pct"] = {
                    'threshold': round(float(g_sc_sorted[idx]), 4),
                    'tp': int(cum_tp[idx]),
                    'fp': int(cum_fp[idx]),
                    'precision': round(float(precisions[idx]), 4),
                    'actual_recall': round(float(recalls[idx]), 4),
                    'micro_f1': round(float(f1s[idx]), 4)
                }

        ap50_per_cls = {c: round(ap50_voc101(class_scores[c], class_hits[c], gt_counts[c]), 5) for c in CLASSES}
        src_ap50 = {
            s: {c: round(ap50_voc101(source_scores[s][c], source_hits[s][c], source_gt[s][c]), 5)
                for c in CLASSES if source_gt[s][c] > 0}
            for s in sorted(source_gt.keys())
        }

        phase1_report[name] = {
            'mAP50': round(float(np.mean(list(ap50_per_cls.values()))), 5),
            'ap50_per_class': ap50_per_cls,
            'ap50_by_source': src_ap50,
            'equal_recall_comparison': equal_recall_points,
            'mean_pos_p_obj': {c: round(float(np.mean(pos_p_obj[c])), 4) for c in CLASSES},
            'mean_pos_p_cls': {c: round(float(np.mean(pos_p_cls[c])), 4) for c in CLASSES},
            'mean_cell_p_obj_by_source': {s: round(float(np.mean(neg_p_obj_by_src[s])), 5) for s in sorted(neg_p_obj_by_src.keys())}
        }

    write_json(EXP04_DIR / 'fase1_verificacion_4c.json', phase1_report)
    print("=== PHASE 1 EQUAL-RECALL COMPARISON (Exp3C vs Exp4B_Ep34) ===", flush=True)
    for r_key in phase1_report['Exp3C']['equal_recall_comparison']:
        p3c = phase1_report['Exp3C']['equal_recall_comparison'][r_key]
        p4b = phase1_report['Exp4B_Ep34']['equal_recall_comparison'].get(r_key, {})
        print(f"  {r_key:16s} | 3C: TP={p3c['tp']:3d}, FP={p3c['fp']:5d}, Prec={p3c['precision']*100:5.2f}%, F1={p3c['micro_f1']*100:5.2f}% (th={p3c['threshold']:.3f}) | 4B: TP={p4b.get('tp',0):3d}, FP={p4b.get('fp',0):5d}, Prec={p4b.get('precision',0)*100:5.2f}%, F1={p4b.get('micro_f1',0)*100:5.2f}% (th={p4b.get('threshold',0):.3f})", flush=True)
    print("  Mean pos p_obj 3C:", phase1_report['Exp3C']['mean_pos_p_obj'], flush=True)
    print("  Mean pos p_obj 4B:", phase1_report['Exp4B_Ep34']['mean_pos_p_obj'], flush=True)
    print("  Mean pos p_cls 3C:", phase1_report['Exp3C']['mean_pos_p_cls'], flush=True)
    print("  Mean pos p_cls 4B:", phase1_report['Exp4B_Ep34']['mean_pos_p_cls'], flush=True)


def analyze_external_pool_and_segmentations():
    print("\n=== PHASE 2: AUDITING 5,205 EXTERNAL IMAGES & SEGMENTATION POLYGONS ===", flush=True)
    orig_data = json.loads(ORIG_MANIFEST.read_text('utf-8'))
    orig_images = orig_data['images']
    det_root = ROOT / 'data/preparado/deteccion'

    protected_shas = set()
    protected_stems = set()
    protected_dhashes = []
    for r in orig_images:
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

    # Also check how many TACO oficial TRAINING images have verified human COCO polygon segmentations!
    taco_corr_path = ROOT / 'data/preparado/taco_oficial/annotations_corregidas.json'
    taco_doc = json.loads(taco_corr_path.read_text('utf-8'))
    taco_anns_by_img = defaultdict(list)
    for a in taco_doc['annotations']:
        taco_anns_by_img[a['image_id']].append(a)

    taco_train_polygons_by_cls = Counter()
    for r in orig_images:
        if r.get('eligible') and r.get('split') == 'train' and r['source'] == 'TACO oficial':
            src_id = r['source_image_id']
            anns = taco_anns_by_img.get(src_id, [])
            for b in r['boxes']:
                # Match box to original TACO annotation with segmentation
                a_id = b.get('annotation_id')
                matched_a = next((a for a in anns if a['id'] == a_id or a.get('original_annotation_id') == a_id), None)
                if matched_a and matched_a.get('segmentation') and isinstance(matched_a['segmentation'], list) and len(matched_a['segmentation'][0]) >= 6:
                    taco_train_polygons_by_cls[b['class']] += 1

    print("  TACO oficial TRAIN verified COCO polygons by class:", dict(taco_train_polygons_by_cls), flush=True)

    # Scan the 5,205 clean external images for:
    # 1. Valid polygon segmentations (TrashNet Re-annotated YOLOv8 polygons & Trash Detection YOLO polygons)
    # 2. Natural vs Studio backgrounds
    # 3. Object scales (<32px, 32-96px, >=96px in 256x256)
    raw_candidates = []
    tn_base = EXTRACT_DIR / 'trashnet - Copy'
    tn_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    for split in ('train', 'valid', 'test'):
        im_dir, lb_dir = tn_base / split / 'images', tn_base / split / 'labels'
        if not im_dir.exists(): continue
        for p in sorted(im_dir.glob('*.jpg')):
            lb = lb_dir / (p.stem + '.txt')
            if lb.exists():
                raw_candidates.append(('TrashNet Re-annotated', split, p, [l.split() for l in lb.read_text('utf-8').splitlines() if l.strip()], tn_names, 'yolo'))

    td_dir = EXT_ROOT / 'Trash Detection Dataset/Trash Detection Dataset/CUSTOM_DATASET'
    td_names = ['BIODEGRADABLE', 'CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']
    for split in ('train', 'valid', 'test'):
        im_dir, lb_dir = td_dir / split / 'images', td_dir / split / 'labels'
        if not im_dir.exists(): continue
        for p in sorted(im_dir.glob('*.jpg')):
            lb = lb_dir / (p.stem + '.txt')
            if lb.exists():
                raw_candidates.append(('Trash Detection Dataset', split, p, [l.split() for l in lb.read_text('utf-8').splitlines() if l.strip()], td_names, 'yolo'))

    gc_dir = EXT_ROOT / 'GARBAGE CLASSIFICATION 3.v2-gc1.coco'
    for split in ('train', 'valid', 'test'):
        ann_path = gc_dir / split / '_annotations.coco.json'
        if not ann_path.exists(): continue
        doc = json.loads(ann_path.read_text('utf-8'))
        cats = {c['id']: c['name'] for c in doc['categories']}
        anns_by_im = defaultdict(list)
        for a in doc['annotations']: anns_by_im[a['image_id']].append(a)
        for im_info in doc['images']:
            raw_candidates.append(('Garbage Classification 3', split, ann_path.parent / im_info['file_name'], anns_by_im[im_info['id']], cats, 'coco'))

    seen_stems = set(protected_stems)
    seen_shas = set(protected_shas)

    stats_bg = defaultdict(Counter)
    stats_size = defaultdict(Counter)
    stats_seg_by_ds_cls = defaultdict(Counter)
    valid_polygon_cutouts = []

    for ds_name, split, p, anns, names_map, mode in raw_candidates:
        if not p.exists(): continue
        st = normalize_stem(p.name)
        if st in seen_stems: continue
        m = re.match(r'^([a-z]+)\d+', p.name.lower())
        if not m or m.group(1) not in PREFIX_TO_TARGET: continue
        expected_cls = PREFIX_TO_TARGET[m.group(1)]
        if not anns: continue

        try:
            with Image.open(p) as im:
                w, h = im.width, im.height
                dhash = compute_dhash(im)
                thumb = np.asarray(im.convert('RGB').resize((64, 64)), dtype=np.float32)
        except Exception:
            continue

        boxes = []
        polygons = []
        mismatch = False
        for a in anns:
            if mode == 'yolo':
                cid = int(a[0])
                oc = names_map[cid] if 0 <= cid < len(names_map) else 'UNKNOWN'
                if STRICT_MAP.get(oc) != expected_cls: mismatch = True; break
                yc = [float(x) for x in a[1:]]
                if len(yc) == 4:
                    cx, cy, bw, bh = yc
                    xyxy = [(cx - bw/2)*w, (cy - bh/2)*h, (cx + bw/2)*w, (cy + bh/2)*h]
                    poly = None
                elif len(yc) >= 6 and len(yc) % 2 == 0:
                    xs, ys = yc[::2], yc[1::2]
                    xyxy = [min(xs)*w, min(ys)*h, max(xs)*w, max(ys)*h]
                    # Check if polygon is a true multi-vertex contour (>4 vertices or non-axis-aligned)
                    pts_norm = list(zip(xs, ys))
                    poly = yc if len(pts_norm) >= 5 else None
                else:
                    mismatch = True; break
            else:
                oc = names_map.get(a['category_id'], 'UNKNOWN')
                if STRICT_MAP.get(oc) != expected_cls: mismatch = True; break
                bx, by, bw, bh = a['bbox']
                xyxy = [bx, by, bx + bw, by + bh]
                poly = a.get('segmentation')[0] if (a.get('segmentation') and isinstance(a['segmentation'], list) and len(a['segmentation'][0]) >= 10) else None
            xyxy = clip_box(xyxy, w, h)
            if not bbox_valid(xyxy, w, h): mismatch = True; break
            boxes.append({'class': expected_cls, 'xyxy': [round(v, 4) for v in xyxy]})
            polygons.append(poly)

        if mismatch: continue
        ok, _ = is_tight_valid_scene(boxes, w, h, max_boxes=4)
        if not ok: continue
        if any(hamming_distance(dhash, pdh) <= 4 for pdh in protected_dhashes): continue

        seen_stems.add(st)

        # Background classification
        border = np.concatenate([thumb[0, :], thumb[-1, :], thumb[:, 0], thumb[:, -1]], axis=0)
        bg_std = float(border.std())
        bg_mean = float(border.mean())
        if bg_std >= 22.0 and 35.0 <= bg_mean <= 210.0:
            bg_type = 'natural_textured'
        elif bg_mean > 210.0 and bg_std < 22.0:
            bg_type = 'studio_white_flat'
        else:
            bg_type = 'indoor_moderate'
        stats_bg[expected_cls][bg_type] += 1

        # Size classification in 256x256
        for b, poly in zip(boxes, polygons):
            bw256 = (b['xyxy'][2] - b['xyxy'][0]) * 256.0 / w
            bh256 = (b['xyxy'][3] - b['xyxy'][1]) * 256.0 / h
            area256 = bw256 * bh256
            sbin = 'small_<32px' if area256 < 32**2 else 'medium_32-96px' if area256 < 96**2 else 'large_>=96px'
            stats_size[expected_cls][sbin] += 1
            if poly is not None:
                stats_seg_by_ds_cls[ds_name][expected_cls] += 1
                valid_polygon_cutouts.append({
                    'source': ds_name,
                    'path': p.relative_to(ROOT).as_posix(),
                    'width': w,
                    'height': h,
                    'class': expected_cls,
                    'xyxy': b['xyxy'],
                    'polygon_norm': poly if mode == 'yolo' else [v / (w if idx_v % 2 == 0 else h) for idx_v, v in enumerate(poly)]
                })

    phase2_report = {
        'taco_train_verified_polygons_by_class': dict(taco_train_polygons_by_cls),
        'external_background_distribution_by_class': {k: dict(v) for k, v in stats_bg.items()},
        'external_box_size_distribution_256x256': {k: dict(v) for k, v in stats_size.items()},
        'external_verified_polygon_masks_by_dataset_and_class': {k: dict(v) for k, v in stats_seg_by_ds_cls.items()},
        'total_external_verified_polygon_instances': len(valid_polygon_cutouts)
    }
    write_json(EXP04_DIR / 'fase2_analisis_dataset_4c.json', phase2_report)
    print(json.dumps(phase2_report, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    evaluate_equal_recall_and_breakdown()
    analyze_external_pool_and_segmentations()
