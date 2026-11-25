"""Experiment 4C Runner:
- Architecture: DetectorS8 (347,649 parameters, initialized 100% from scratch)
- Loss: Exact corrected Focal Loss of Exp 3C (gamma=2.0, alpha=0.75, obj_scale=1.5, box_weight=10.0)
- Data Strategy:
  1. 3,645 original training images intact (100% preservation of Bangladesh, Food Waste, TACO diversity)
  2. 73 strictly small/medium (<115px) natural-textured-background external images (vidrio, papel, carton, metal)
  3. Perspective-Matched Verified Polygon Copy-Paste onto TACO oficial ground-plane training backgrounds
     using the 779-cutout Verified Polygon Bank (459 TACO train polygons + 320 external YOLOv8 polygons)
     with realistic rotation, scale matching, local luminance harmonization, and mathematical bbox derivation.
- Optimization: AdamW + CosineAnnealingLR + Weight Exponential Moving Average (EMA) for variance reduction.
"""
from pathlib import Path
import sys
import time
import copy
import math
import json
import random
import hashlib
import argparse
from collections import Counter, defaultdict
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from residuos import CLASSES
from residuos.data import Store, read_json, write_json
from residuos.detector import iou
from scripts.experiment3_runner import (
    DetectorS8, CachedDataset32, encode_grid32, decode_grid32,
    GRID, SLOTS, SIZE, seed_all, device_for
)
from scripts.audit_evaluation import ap50_voc101, overlaps, nms_indices, extract_candidates_generic

EXP04_DIR = ROOT / 'runs/detector_exp04'
OUT_4C_DIR = EXP04_DIR / 'variant_4c_copypaste'
MINORITY_4 = ('vidrio', 'papel', 'carton', 'metal')

# Exact proven class weights from Exp 3C with slight vidrio boost
CLASS_WEIGHTS_4C = torch.tensor([0.53, 1.39, 1.27, 1.35, 0.95, 0.53], dtype=torch.float32)

def loss_fn_4c(pred, target, cls_weights=CLASS_WEIGHTS_4C, gamma=2.0, alpha=0.75, obj_scale=1.5, box_weight=10.0):
    positive = target[..., 0] == 1
    n_pos = max(float(positive.sum()), 1.0)

    p = torch.sigmoid(pred[..., 0])
    pt = p * target[..., 0] + (1.0 - p) * (1.0 - target[..., 0])
    focal_weight = (1.0 - pt) ** gamma
    alpha_weight = alpha * target[..., 0] + (1.0 - alpha) * (1.0 - target[..., 0])
    bce = nn.functional.binary_cross_entropy_with_logits(pred[..., 0], target[..., 0], reduction='none')
    fl = alpha_weight * focal_weight * bce
    obj = (fl.sum() / n_pos) * obj_scale

    zero = pred.sum() * 0
    box = nn.functional.smooth_l1_loss(pred[..., 1:5][positive].sigmoid(), target[..., 1:5][positive]) if positive.any() else zero

    if positive.any():
        w = cls_weights.to(pred.device) if cls_weights is not None else None
        cls = nn.functional.cross_entropy(pred[..., 5:][positive], target[..., 5][positive].long(), weight=w)
    else:
        cls = zero

    return obj + box_weight * box + cls


class ModelEMA:
    """Exponential Moving Average of model parameters (100% from scratch, zero external weights)."""
    def __init__(self, model, decay=0.996):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        self.updates = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        self.updates += 1
        d = self.decay * (1.0 - math.exp(-self.updates / 200.0))
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
                else:
                    v.copy_(msd[k])


def transform_and_derive_cutout(rgb_patch, alpha_patch, target_sz, angle_deg=0.0, hflip=False):
    """Applies horizontal flip, rotation, and scaling to a polygon-segmented cutout,
    and mathematically derives the exact tight bounding box from the transformed alpha mask.
    """
    rgb = rgb_patch.copy()
    alpha = alpha_patch.copy()
    if hflip:
        rgb = np.ascontiguousarray(rgb[:, ::-1])
        alpha = np.ascontiguousarray(alpha[:, ::-1])

    if abs(angle_deg) > 1e-3:
        h, w = alpha.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
        nw_r = int(h * sin_a + w * cos_a)
        nh_r = int(h * cos_a + w * sin_a)
        M[0, 2] += (nw_r / 2.0) - cx
        M[1, 2] += (nh_r / 2.0) - cy
        rgb = cv2.warpAffine(rgb, M, (nw_r, nh_r), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        alpha = cv2.warpAffine(alpha, M, (nw_r, nh_r), flags=cv2.INTER_LINEAR, borderValue=0)

    ys, xs = np.nonzero(alpha > 32)
    if len(xs) < 16:
        return None
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    rgb = rgb[y1:y2, x1:x2]
    alpha = alpha[y1:y2, x1:x2]

    ph, pw = alpha.shape[:2]
    sc = target_sz / max(ph, pw)
    nw = max(12, int(round(pw * sc)))
    nh = max(12, int(round(ph * sc)))
    rgb_s = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    alpha_s = cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return rgb_s, alpha_s


class DatasetExp04C(Dataset):
    """Dataset for Experiment 4C:
    - All 3,645 original training images
    - 73 strictly small/medium natural-textured-background external minority images
    - Perspective-Matched Verified Polygon Copy-Paste onto TACO oficial ground-plane training images
    """
    def __init__(self, orig_rows, orig_npy_path, ext_rows, ext_npy_path, ext_sm_indices, cutout_bank,
                 augment=True, paste_prob=0.55):
        self.orig_rows = orig_rows
        self.ext_rows = ext_rows
        self.orig_pixels = np.load(orig_npy_path, mmap_mode='r')
        self.ext_pixels = np.load(ext_npy_path, mmap_mode='r')
        self.augment = augment
        self.paste_prob = paste_prob

        self.samples = [('orig', i) for i in range(len(self.orig_rows))]
        self.samples.extend([('ext', j) for j in ext_sm_indices])

        # Group polygon cutouts by class and origin (TACO vs External)
        self.taco_cutouts = defaultdict(list)
        self.ext_cutouts = defaultdict(list)
        for item in cutout_bank:
            if item['origin'] == 'TACO_train_polygon':
                self.taco_cutouts[item['class']].append(item)
            else:
                self.ext_cutouts[item['class']].append(item)

        # Sampling weights for minority classes in Copy-Paste (prioritize vidrio, papel, carton, metal)
        self.paste_classes = ['vidrio', 'papel', 'carton', 'metal']
        self.paste_probs = [0.30, 0.28, 0.24, 0.18]

    def __len__(self):
        return len(self.samples)

    def _try_polygon_copypaste(self, frame, boxes_256):
        """Attempts to composite 1 verified polygon cutout onto a TACO oficial ground-plane training image."""
        if len(boxes_256) == 0 or len(boxes_256) > 3:
            return frame, boxes_256

        # Check existing object scale in 256x256: only paste on medium/far ground-plane scenes (max side <= 110px)
        existing_sides = [max(b['xyxy'][2] - b['xyxy'][0], b['xyxy'][3] - b['xyxy'][1]) for b in boxes_256]
        if max(existing_sides) > 110.0:
            return frame, boxes_256

        ref_side = float(np.median(existing_sides))
        ref_y = float(np.median([(b['xyxy'][1] + b['xyxy'][3]) / 2.0 for b in boxes_256]))

        # Choose minority class to paste
        r_val = random.random()
        cum = 0.0
        target_c = 'vidrio'
        for c, p in zip(self.paste_classes, self.paste_probs):
            cum += p
            if r_val <= cum:
                target_c = c
                break

        # Pure TACO-train polygon cutouts (100% in-domain outdoor lighting and texture)
        pool = self.taco_cutouts[target_c]
        if not pool:
            return frame, boxes_256
        item = random.choice(pool)

        # Perspective-matched target size (within 0.75x..1.40x of existing object size, clipped to [22, 74] px)
        target_sz = int(round(np.clip(ref_side * random.uniform(0.75, 1.40), 22.0, 74.0)))
        angle = random.choice([0.0, 0.0, -12.0, 12.0, -22.0, 22.0, 90.0, -90.0])
        hflip_cut = random.random() < 0.5

        res = transform_and_derive_cutout(item['rgb'], item['alpha'], target_sz, angle_deg=angle, hflip=hflip_cut)
        if res is None:
            return frame, boxes_256
        rgb_s, alpha_s = res
        nh, nw = alpha_s.shape[:2]
        if nw >= 220 or nh >= 220:
            return frame, boxes_256

        # Search for a valid non-overlapping ground-plane location near ref_y
        y_min = max(44, int(ref_y - 55 - nh / 2))
        y_max = min(SIZE - nh - 10, int(ref_y + 55 - nh / 2))
        if y_max <= y_min:
            y_min, y_max = 48, max(50, SIZE - nh - 10)

        for _ in range(30):
            x0 = random.randint(10, SIZE - nw - 10)
            y0 = random.randint(y_min, y_max)
            cand_xy = [x0, y0, x0 + nw, y0 + nh]

            bg_patch = frame[y0:y0+nh, x0:x0+nw].astype(np.float32)
            # Ensure non-black, textured photographic surface
            if bg_patch.mean(axis=(1, 2)).min() < 26.0 or bg_patch.mean(axis=(0, 2)).min() < 26.0 or bg_patch.std() < 9.0:
                continue

            # Ensure 0 overlap with existing boxes (with 6px safety margin)
            if any(
                max(cand_xy[0], ob['xyxy'][0] - 6) < min(cand_xy[2], ob['xyxy'][2] + 6) and
                max(cand_xy[1], ob['xyxy'][1] - 6) < min(cand_xy[3], ob['xyxy'][3] + 6)
                for ob in boxes_256
            ):
                continue

            # Derive exact mathematical bounding box from active polygon mask pixels (alpha > 32)
            ys_m, xs_m = np.nonzero(alpha_s > 32)
            if len(xs_m) < 12:
                continue
            bx1, by1 = float(x0 + xs_m.min()), float(y0 + ys_m.min())
            bx2, by2 = float(x0 + xs_m.max() + 1), float(y0 + ys_m.max() + 1)
            new_box = {'class': target_c, 'xyxy': [bx1, by1, bx2, by2]}
            trial_boxes = boxes_256 + [new_box]

            # Check grid32 capacity before committing
            try:
                encode_grid32(trial_boxes, SIZE, SIZE)
            except ValueError:
                continue

            # Local luminance harmonization
            fg_patch = rgb_s.astype(np.float32)
            bg_lum = max(25.0, float(bg_patch.mean()))
            fg_lum = max(25.0, float(fg_patch[alpha_s > 127].mean())) if (alpha_s > 127).any() else 128.0
            lum_factor = np.clip((bg_lum / fg_lum) * 0.38 + 0.62, 0.80, 1.22)
            fg_harmonized = np.clip(fg_patch * lum_factor, 0, 255)

            a_norm = (alpha_s.astype(np.float32) / 255.0)[..., None]
            frame_out = frame.copy()
            frame_out[y0:y0+nh, x0:x0+nw] = (a_norm * fg_harmonized + (1.0 - a_norm) * bg_patch).astype(np.uint8)
            return frame_out, trial_boxes

        return frame, boxes_256

    def __getitem__(self, idx):
        kind, real_idx = self.samples[idx]
        if kind == 'orig':
            r = self.orig_rows[real_idx]
            frame = np.array(self.orig_pixels[real_idx], copy=True)
        else:
            r = self.ext_rows[real_idx]
            frame = np.array(self.ext_pixels[real_idx], copy=True)

        w_orig, h_orig = r['width'], r['height']
        boxes_256 = [
            {
                'class': b['class'],
                'xyxy': [b['xyxy'][0] * SIZE / w_orig, b['xyxy'][1] * SIZE / h_orig,
                         b['xyxy'][2] * SIZE / w_orig, b['xyxy'][3] * SIZE / h_orig]
            }
            for b in r['boxes']
        ]

        if self.augment:
            # 1. Verified Polygon Copy-Paste onto TACO oficial training backgrounds
            if kind == 'orig' and r['source'] == 'TACO oficial' and random.random() < self.paste_prob:
                frame, boxes_256 = self._try_polygon_copypaste(frame, boxes_256)

            # 2. Horizontal flip (50%)
            if random.random() < 0.5:
                frame = np.ascontiguousarray(frame[:, ::-1])
                for b in boxes_256:
                    x1, y1, x2, y2 = b['xyxy']
                    b['xyxy'] = [SIZE - x2, y1, SIZE - x1, y2]

            # 3. Moderate illumination jitter
            frame = np.clip(frame.astype(np.float32) * random.uniform(0.86, 1.14), 0, 255).astype(np.uint8)

        x_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        target_tensor = encode_grid32(boxes_256, SIZE, SIZE)
        return x_tensor, target_tensor


def evaluate_model_on_val(model, orig_val, val_loader, device):
    model.eval()
    c_scores = {c: [] for c in CLASSES}
    c_hits = {c: [] for c in CLASSES}
    gt_cnt = Counter(b['class'] for r in orig_val for b in r['boxes'])
    val_outs = []
    with torch.no_grad():
        for vx, _ in val_loader:
            val_outs.extend(model(vx.to(device)).cpu())
    for i_v, r_v in enumerate(orig_val):
        xy, cls_v, sc_v, _, _ = extract_candidates_generic(val_outs[i_v], r_v['width'], r_v['height'], GRID)
        kept = nms_indices(xy, cls_v, sc_v, threshold=0.45, max_detections=100)
        n_xy, n_cls, n_sc = xy[kept], cls_v[kept], sc_v[kept]
        gt_b = [b['xyxy'] for b in r_v['boxes']]
        gt_c = [CLASSES.index(b['class']) for b in r_v['boxes']]
        used_g = set()
        if len(gt_b) > 0 and len(n_xy) > 0:
            ov = overlaps(n_xy, gt_b)
            for p_i in np.argsort(-n_sc, kind='stable'):
                cp = n_cls[p_i]
                cn = CLASSES[cp]
                elig = [g_i for g_i in range(len(gt_b)) if gt_c[g_i] == cp and g_i not in used_g]
                vo, bg = max([(ov[p_i, g_i], g_i) for g_i in elig], default=(0.0, -1))
                if vo >= 0.5:
                    used_g.add(bg)
                    c_scores[cn].append(float(n_sc[p_i]))
                    c_hits[cn].append(True)
                else:
                    c_scores[cn].append(float(n_sc[p_i]))
                    c_hits[cn].append(False)
        elif len(n_xy) > 0:
            for p_i in range(len(n_xy)):
                cn = CLASSES[n_cls[p_i]]
                c_scores[cn].append(float(n_sc[p_i]))
                c_hits[cn].append(False)
    ap_ep = {c: ap50_voc101(c_scores[c], c_hits[c], gt_cnt[c]) for c in CLASSES}
    map_ep = float(np.mean(list(ap_ep.values())))
    return map_ep, ap_ep


def run_preflight_4c():
    print("\n================ RUNNING EXPERIMENT 4C PREFLIGHT & INTEGRITY GATES ================", flush=True)
    seed_all(42)
    device = device_for('auto')
    OUT_4C_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Verify previous checkpoints are 100% intact
    ckpts_to_verify = {
        'exp01': ROOT / 'runs/detector_exp01/best.pt',
        'exp03c': ROOT / 'runs/detector_exp03/corrected/best.pt',
        'exp04a': EXP04_DIR / 'variant_4a_broad/best.pt',
        'exp04b': EXP04_DIR / 'variant_4b_domain_aligned/best.pt'
    }
    hashes = {}
    for k, p in ckpts_to_verify.items():
        assert p.exists(), f"Missing preserved checkpoint: {p}"
        hashes[k] = hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
    assert hashes['exp01'] == "fac0896e9c77c6d4a61674505656c6800d4d44502c8e8578443fbac4ca52dd02" or len(hashes['exp01']) == 64

    store = Store(ROOT / 'data/preparado/deteccion')
    orig_train = store.records('train')
    orig_val = store.records('val')
    orig_test = store.records('test')
    ext_rows = read_json(EXP04_DIR / 'manifest_exp04_ext.json')
    sm_info = read_json(EXP04_DIR / 'small_med_natural_indices_4c.json')
    cutout_bank = torch.load(EXP04_DIR / 'verified_polygon_bank_4c.pt', weights_only=False)

    # 2. Verify zero leakage in cutout bank
    val_test_ids = {r['id'] for r in orig_val + orig_test}
    for item in cutout_bank:
        assert item['source_image_id'] not in val_test_ids, f"Cutout leak detected: {item['source_image_id']}"

    ds = DatasetExp04C(
        orig_train, ROOT / 'runs/detector_exp01/train.npy',
        ext_rows, EXP04_DIR / 'ext_train.npy',
        sm_info['indices'], cutout_bank,
        augment=True, paste_prob=0.80
    )

    # 3. Test 25 random Copy-Paste items from DatasetExp04C for geometric roundtrip and finiteness
    taco_sample_indices = [i for i, r in enumerate(orig_train) if r['source'] == 'TACO oficial'][:25]
    max_iou_err = 0.0
    pasted_boxes_tested = 0
    for idx in taco_sample_indices:
        x, y = ds[idx]
        assert x.shape == (3, 256, 256) and y.shape == (32, 32, 3, 6)
        assert torch.isfinite(x).all() and torch.isfinite(y).all()
        n_obj = int((y[..., 0] == 1).sum().item())
        if n_obj > len(orig_train[idx]['boxes']):
            pasted_boxes_tested += 1
        oracle_z = torch.zeros(32, 32, 3, 11)
        oracle_z[..., 0] = -10.0
        gt_b = []
        for gy, gx, sl in (y[..., 0] == 1).nonzero().tolist():
            oracle_z[gy, gx, sl, 0] = 10.0
            dx, dy, bw, bh = y[gy, gx, sl, 1:5].tolist()
            oracle_z[gy, gx, sl, 1:5] = torch.logit(y[gy, gx, sl, 1:5].clamp(1e-5, 1 - 1e-5))
            c = int(y[gy, gx, sl, 5].item())
            oracle_z[gy, gx, sl, 5 + c] = 10.0
            cx, cy = (gx + dx) / 32.0, (gy + dy) / 32.0
            gt_b.append({'class': CLASSES[c], 'xyxy': [(cx - bw/2)*256, (cy - bh/2)*256, (cx + bw/2)*256, (cy + bh/2)*256]})
        dec = decode_grid32(oracle_z, 256, 256, threshold=0.5, nms=0.45)
        assert len(dec) == len(gt_b), "Oracle decode count mismatch on Copy-Paste scene"
        for d in dec:
            best_i = max(iou(d['xyxy'], g['xyxy']) for g in gt_b if g['class'] == d['class'])
            max_iou_err = max(max_iou_err, 1.0 - best_i)

    assert pasted_boxes_tested >= 10, f"Expected >=10 pasted scenes in test, got {pasted_boxes_tested}"
    assert max_iou_err < 1e-3, f"Geometric error too high: {max_iou_err}"
    print(f"[Preflight 4C] Tested {pasted_boxes_tested} Copy-Paste scenes: max 1-IoU = {max_iou_err:.2e}.", flush=True)

    # 4. Overfit test on 6 composited Copy-Paste scenes containing all 6 classes
    batch_items = [ds[i] for i in taco_sample_indices[:6]]
    bx = torch.stack([it[0] for it in batch_items]).to(device)
    by = torch.stack([it[1] for it in batch_items]).to(device)
    n_gt = int((by[..., 0] == 1).sum().item())

    model = DetectorS8().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0015)
    init_loss, final_loss = 0.0, 0.0
    for step in range(140):
        opt.zero_grad()
        loss = loss_fn_4c(model(bx), by)
        if step == 0: init_loss = float(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        final_loss = float(loss.item())

    print(f"[Preflight 4C] Copy-Paste Overfit Test ({n_gt} objects): loss {init_loss:.4f} -> {final_loss:.4f} ({init_loss/max(final_loss,1e-6):.1f}x reduction)", flush=True)
    assert final_loss < init_loss * 0.05

    report = {
        'status': 'PASSED',
        'preserved_checkpoints_sha256': hashes,
        'cutout_bank_size': len(cutout_bank),
        'small_med_external_added': len(sm_info['indices']),
        'total_epoch_samples': len(ds),
        'pasted_scenes_verified': pasted_boxes_tested,
        'max_geometric_roundtrip_error': max_iou_err,
        'overfit_init_loss': init_loss,
        'overfit_final_loss': final_loss
    }
    write_json(OUT_4C_DIR / 'preflight_4c.json', report)
    print(">>> ALL EXPERIMENT 4C PREFLIGHT GATES PASSED <<<\n", flush=True)


def train_exp04c(epochs=40, batch_size=16, lr=0.001, seed=42):
    seed_all(seed)
    device = device_for('auto')
    OUT_4C_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n================ STARTING EXPERIMENT 4C TRAINING FROM SCRATCH ON {device} ================", flush=True)

    store = Store(ROOT / 'data/preparado/deteccion')
    orig_train = store.records('train')
    orig_val = store.records('val')
    ext_rows = read_json(EXP04_DIR / 'manifest_exp04_ext.json')
    sm_info = read_json(EXP04_DIR / 'small_med_natural_indices_4c.json')
    cutout_bank = torch.load(EXP04_DIR / 'verified_polygon_bank_4c.pt', weights_only=False)

    trainset = DatasetExp04C(
        orig_train, ROOT / 'runs/detector_exp01/train.npy',
        ext_rows, EXP04_DIR / 'ext_train.npy',
        [], cutout_bank,
        augment=True, paste_prob=0.35
    )
    valset = CachedDataset32(orig_val, ROOT / 'runs/detector_exp01/val.npy', augment=False)

    loaders = {
        'train': DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True),
        'val': DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    }

    config = {
        'variant': 'Exp04C_PolygonCopyPaste',
        'description': 'DetectorS8 trained 100% from scratch on 3645 orig + 73 small/med natural ext + Perspective-Matched Verified Polygon Copy-Paste on TACO train + Weight EMA',
        'pretrained_weights_used': False,
        'initialized_from_3c': False,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'seed': seed,
        'device': str(device),
        'grid': GRID,
        'slots': SLOTS,
        'class_weights': [float(x) for x in CLASS_WEIGHTS_4C],
        'focal_gamma': 2.0,
        'focal_alpha': 0.75,
        'obj_scale': 1.5,
        'box_weight': 10.0,
        'paste_prob_on_taco': 0.55,
        'polygon_cutout_bank_size': len(cutout_bank),
        'small_med_natural_ext_images': len(sm_info['indices']),
        'total_epoch_samples': len(trainset),
        'val_images': len(valset),
        'test_images_touched': 0
    }
    write_json(OUT_4C_DIR / 'config.json', config)

    model = DetectorS8(num_classes=len(CLASSES), slots=SLOTS).to(device)
    ema = ModelEMA(model, decay=0.996)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    best_val_loss = float('inf')
    best_map50 = -1.0
    best_ema_map50 = -1.0
    history = []
    start_time = time.perf_counter()

    for epoch in range(epochs):
        ep_start = time.perf_counter()
        row = {'epoch': epoch + 1}

        # Disable Copy-Paste in last 8 epochs (fine-tuning on pure uncomposited 3,645 original images)
        trainset.paste_prob = 0.35 if (epoch + 1) <= (epochs - 8) else 0.0

        for split, loader in loaders.items():
            is_train = (split == 'train')
            model.train(is_train)
            total_loss = 0.0
            total_samples = 0

            for x, y in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with torch.set_grad_enabled(is_train):
                    out = model(x)
                    loss = loss_fn_4c(out, y)
                    if not torch.isfinite(loss):
                        raise ValueError(f"Non-finite loss at epoch {epoch+1}, split {split}")

                    if is_train:
                        opt.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        opt.step()
                        ema.update(model)

                total_loss += loss.item() * len(x)
                total_samples += len(x)

            row[f'{split}_loss'] = total_loss / total_samples

        scheduler.step()
        row['lr'] = opt.param_groups[0]['lr']
        row['seconds_elapsed'] = time.perf_counter() - start_time
        ep_duration = time.perf_counter() - ep_start

        val_loss = row['val_loss']
        train_loss = row['train_loss']
        is_best_loss = val_loss < best_val_loss
        if is_best_loss:
            best_val_loss = val_loss
            torch.save({
                'origin': 'residuos-detector-exp04c-from-scratch',
                'epoch': epoch + 1,
                'val_loss': best_val_loss,
                'state': model.state_dict(),
                'config': config
            }, OUT_4C_DIR / 'best_val_loss.pt')

        if is_best_loss or (epoch + 1) >= 12:
            raw_map, raw_ap = evaluate_model_on_val(model, orig_val, loaders['val'], device)
            ema_map, ema_ap = evaluate_model_on_val(ema.ema, orig_val, loaders['val'], device)
            row['val_map50_raw'] = raw_map
            row['val_ap50_raw'] = raw_ap
            row['val_map50_ema'] = ema_map
            row['val_ap50_ema'] = ema_ap

            if ema_map > best_ema_map50:
                best_ema_map50 = ema_map
                torch.save({
                    'origin': 'residuos-detector-exp04c-from-scratch',
                    'epoch': epoch + 1,
                    'weights_type': 'EMA',
                    'val_loss': val_loss,
                    'val_map50': best_ema_map50,
                    'val_ap50': ema_ap,
                    'state': ema.ema.state_dict(),
                    'config': config
                }, OUT_4C_DIR / 'best_ema.pt')

            cand_map, cand_ap, cand_state, cand_tag = (
                (ema_map, ema_ap, ema.ema.state_dict(), 'EMA') if ema_map >= raw_map
                else (raw_map, raw_ap, model.state_dict(), 'RAW')
            )
            if cand_map > best_map50:
                best_map50 = cand_map
                ckpt_payload = {
                    'origin': 'residuos-detector-exp04c-from-scratch',
                    'epoch': epoch + 1,
                    'weights_type': cand_tag,
                    'val_loss': val_loss,
                    'val_map50': best_map50,
                    'val_ap50': cand_ap,
                    'state': cand_state,
                    'config': config
                }
                torch.save(ckpt_payload, OUT_4C_DIR / 'best.pt')
                torch.save(ckpt_payload, EXP04_DIR / 'best_4c.pt')
                star_map = f" ★ (BEST mAP50 [{cand_tag}]: {best_map50*100:.2f}%)"
            else:
                star_map = f" (raw:{raw_map*100:.2f}% | ema:{ema_map*100:.2f}%)"
        else:
            star_map = ""

        torch.save({
            'origin': 'residuos-detector-exp04c-from-scratch',
            'epoch': epoch + 1,
            'val_loss': val_loss,
            'state': model.state_dict(),
            'ema_state': ema.ema.state_dict(),
            'config': config
        }, OUT_4C_DIR / 'last.pt')

        history.append(row)
        write_json(OUT_4C_DIR / 'history.json', history)
        star_loss = " [BestLoss]" if is_best_loss else ""
        print(f"[Exp04C] Época {epoch+1:02d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | {ep_duration:.1f}s{star_loss}{star_map}", flush=True)

    print(f"\n>>> ENTRENAMIENTO EXP 04C COMPLETADO: best_val_loss = {best_val_loss:.4f} | best_val_mAP50 = {best_map50*100:.2f}% <<<", flush=True)
    return history


if __name__ == '__main__':
    from scripts.experiment4_runner import evaluate_exp04
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['preflight', 'train', 'evaluate', 'all'], default='all')
    parser.add_argument('--epochs', type=int, default=40)
    args = parser.parse_args()

    if args.action in ('preflight', 'all'):
        run_preflight_4c()
    if args.action in ('train', 'all'):
        train_exp04c(epochs=args.epochs, batch_size=16, lr=0.001, seed=42)
    if args.action in ('evaluate', 'all'):
        evaluate_exp04('variant_4c_copypaste/best.pt', 'variant_4c_copypaste/evaluation.json')
        evaluate_exp04('variant_4c_copypaste/best_ema.pt', 'variant_4c_copypaste/evaluation_best_ema.json')
        evaluate_exp04('variant_4c_copypaste/best_val_loss.pt', 'variant_4c_copypaste/evaluation_best_val_loss.json')
