"""Experiment 3: Resolution enhancement (stride 8, 32x32 grid) and class-balance remedies.
Trained genuinely from scratch, no pretrained backbones, test partition strictly reserved.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader

from residuos import CLASSES
from residuos.data import Store, read_json, write_json, digest
from residuos.detector import seed_all, device_for
from residuos.evaluate import metrics, panel, comparison

SIZE = 256
GRID = 32
SLOTS = 3

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(4, channels)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(4, channels)

    def forward(self, x):
        return x + self.gn2(self.conv2(self.act(self.gn1(self.conv1(x)))))


class DetectorS8(nn.Module):
    """Compact CNN detector designed from scratch with stride 8 (32x32 grid).
    Input: (B, 3, 256, 256)
    Output: (B, 32, 32, SLOTS, 11) where 11 = 1 presence + 4 box (dx, dy, w, h) + 6 class logits.
    """
    def __init__(self, slots=SLOTS, num_classes=len(CLASSES)):
        super().__init__()
        self.slots = slots
        self.num_classes = num_classes
        self.entry = nn.Sequential(
            # Stage 1: stride 2 -> 128x128
            nn.Conv2d(3, 24, 3, stride=2, padding=1),
            nn.GroupNorm(4, 24),
            nn.SiLU(),
            # Stage 2: stride 2 -> 64x64
            nn.Conv2d(24, 48, 3, stride=2, padding=1),
            nn.GroupNorm(4, 48),
            nn.SiLU(),
            ResBlock(48),
            # Stage 3: stride 2 -> 32x32
            nn.Conv2d(48, 96, 3, stride=2, padding=1),
            nn.GroupNorm(4, 96),
            nn.SiLU(),
            ResBlock(96),
            # Spatial refinement at 32x32 resolution
            nn.Conv2d(96, 96, 3, padding=1),
            nn.GroupNorm(4, 96),
            nn.SiLU(),
        )
        out_ch = slots * (5 + num_classes)
        self.head = nn.Conv2d(96, out_ch, 1)
        # Strict local initialization from scratch
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias.view(slots, -1)[:, 0] = -3.5

    def forward(self, x):
        z = self.head(self.entry(x))
        # z: (B, slots*(5+C), 32, 32)
        return z.view(len(x), self.slots, 5 + self.num_classes, GRID, GRID).permute(0, 3, 4, 1, 2)


def encode_grid32(boxes, width, height, grid=GRID, slots=SLOTS):
    target = torch.zeros(grid, grid, slots, 6)
    cells = {}
    for box in boxes:
        x1, y1, x2, y2 = box['xyxy']
        cx, cy, w, h = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height
        gx, gy = min(int(cx * grid), grid - 1), min(int(cy * grid), grid - 1)
        cells.setdefault((gy, gx), []).append((cx, cy, w, h, CLASSES.index(box['class'])))
    for (gy, gx), entries in cells.items():
        if len(entries) > slots:
            raise ValueError(f'Más de {slots} centros en celda ({gx},{gy}).')
        for slot, (cx, cy, w, h, c) in enumerate(sorted(entries)):
            target[gy, gx, slot] = torch.tensor([1, cx * grid - gx, cy * grid - gy, w, h, c])
    return target


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.


def decode_grid32(output, width, height, threshold=0.25, nms=0.45, grid=GRID):
    out = output.detach().cpu()
    probs, classes = out[..., 5:].softmax(-1).max(-1)
    scores = out[..., 0].sigmoid() * probs
    boxes = []
    for gy, gx, slot in (scores >= threshold).nonzero().tolist():
        dx, dy, w, h = out[gy, gx, slot, 1:5].sigmoid().tolist()
        cx, cy = (gx + dx) / grid, (gy + dy) / grid
        coords = [
            max(0, (cx - w / 2) * width),
            max(0, (cy - h / 2) * height),
            min(width, (cx + w / 2) * width),
            min(height, (cy + h / 2) * height)
        ]
        if coords[2] > coords[0] and coords[3] > coords[1]:
            boxes.append({
                'class': CLASSES[int(classes[gy, gx, slot])],
                'xyxy': coords,
                'score': float(scores[gy, gx, slot])
            })
    kept = []
    for b in sorted(boxes, key=lambda b: b['score'], reverse=True):
        if not any(k['class'] == b['class'] and iou(k['xyxy'], b['xyxy']) > nms for k in kept):
            kept.append(b)
        if len(kept) == 100:
            break
    return kept


def loss_fn_3a(pred, target):
    """Loss for Experiment 3A: exact same formulation as baseline, adapted to 32x32."""
    positive = target[..., 0] == 1
    bce = nn.functional.binary_cross_entropy_with_logits(pred[..., 0], target[..., 0], reduction='none')
    obj = (bce * torch.where(positive, 5., 0.2)).mean()
    zero = pred.sum() * 0
    box = nn.functional.smooth_l1_loss(pred[..., 1:5][positive].sigmoid(), target[..., 1:5][positive]) if positive.any() else zero
    cls = nn.functional.cross_entropy(pred[..., 5:][positive], target[..., 5][positive].long()) if positive.any() else zero
    return obj + 8 * box + cls


def loss_fn_3b(pred, target, class_weights=None):
    """Loss for Experiment 3B: incorporates class weighting on classification loss and adjusted presence balance for 32x32."""
    positive = target[..., 0] == 1
    bce = nn.functional.binary_cross_entropy_with_logits(pred[..., 0], target[..., 0], reduction='none')
    obj = (bce * torch.where(positive, 15., 0.5)).mean()
    zero = pred.sum() * 0
    box = nn.functional.smooth_l1_loss(pred[..., 1:5][positive].sigmoid(), target[..., 1:5][positive]) if positive.any() else zero
    if positive.any():
        w = class_weights.to(pred.device) if class_weights is not None else None
        cls = nn.functional.cross_entropy(pred[..., 5:][positive], target[..., 5][positive].long(), weight=w)
    else:
        cls = zero
    return obj + 8 * box + cls


class CachedDataset32(Dataset):
    def __init__(self, rows, npy_path, augment=False):
        self.rows = rows
        self.pixels = np.load(npy_path, mmap_mode='r')
        self.augment = augment
        self.targets = [encode_grid32(r['boxes'], r['width'], r['height']) for r in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        pixels = self.pixels[i].copy()
        r = self.rows[i]
        target = self.targets[i]
        if self.augment and random.random() < 0.5:
            pixels = pixels[:, ::-1].copy()
            boxes = [
                dict(b, xyxy=[r['width'] - b['xyxy'][2], b['xyxy'][1], r['width'] - b['xyxy'][0], b['xyxy'][3]])
                for b in r['boxes']
            ]
            target = encode_grid32(boxes, r['width'], r['height'])
        x = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255.0
        if self.augment:
            x = (x * random.uniform(0.85, 1.15)).clamp(0, 1)
        return x, target


def run_preflight_checks(out_dir, rows, npy_path, device, seed=20261123):
    print("--- INICIANDO COMPUERTAS PREFLIGHT PARA EXPERIMENTO 3 ---", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_all(seed)

    # 1. Compuerta de dimensiones
    model = DetectorS8().to(device)
    dummy_in = torch.randn(2, 3, SIZE, SIZE, device=device)
    dummy_out = model(dummy_in)
    assert dummy_out.shape == (2, GRID, GRID, SLOTS, 11), f"Forma inesperada: {dummy_out.shape}"
    print(f"[OK] 1. Compuerta de dimensiones: salida correcta {tuple(dummy_out.shape)}")

    # 2. Compuerta de inversión geométrica y reconstrucción
    dataset = CachedDataset32(rows['train'], npy_path)
    chosen = []
    for c in CLASSES:
        candidates = [i for i, r in enumerate(rows['train']) if any(b['class'] == c for b in r['boxes']) and i not in chosen]
        chosen.append(min(candidates, key=lambda i: (len(rows['train'][i]['boxes']), rows['train'][i]['id'])))

    max_err = 0.0
    for k, i in enumerate(chosen):
        r = rows['train'][i]
        target = dataset.targets[i]
        for gy, gx, slot in (target[..., 0] > 0).nonzero().tolist():
            _, dx, dy, w, h, c = target[gy, gx, slot].tolist()
            cx, cy = (gx + dx) / GRID, (gy + dy) / GRID
            xy = [(cx - w / 2) * r['width'], (cy - h / 2) * r['height'], (cx + w / 2) * r['width'], (cy + h / 2) * r['height']]
            err = min(np.max(np.abs(np.array(xy) - np.array(b['xyxy']))) for b in r['boxes'] if b['class'] == CLASSES[int(c)])
            max_err = max(max_err, float(err))
    assert max_err < 0.001, f"Error geométrico excesivo: {max_err}"
    print(f"[OK] 2. Compuerta de inversión geométrica: error máx = {max_err:.7f} px")

    # 3. Compuerta Oracle
    truth_oracle = [{'class': c, 'xyxy': [10 + 35 * i, 15, 30 + 35 * i, 55]} for i, c in enumerate(CLASSES)]
    t_oracle = encode_grid32(truth_oracle, SIZE, SIZE)
    z_oracle = torch.full((1, GRID, GRID, SLOTS, 11), -20.0, device=device)
    pos = t_oracle[..., 0] > 0
    z_oracle[0, ..., 1:5][pos] = torch.logit(t_oracle[..., 1:5][pos].clamp(1e-6, 1 - 1e-6)).to(device)
    z_oracle[0, ..., 0][pos] = 20.0
    for gy, gx, sl in pos.nonzero().tolist():
        z_oracle[0, gy, gx, sl, 5 + int(t_oracle[gy, gx, sl, 5])] = 20.0
    preds_oracle = decode_grid32(z_oracle[0], SIZE, SIZE, threshold=0.5, grid=GRID)
    single_met = metrics([(preds_oracle, truth_oracle)])
    assert single_met['tp'] == 6 and single_met['fp'] == 0, f"Fallo en Oracle: {single_met}"
    print(f"[OK] 3. Compuerta Oracle: 6/6 TP, 0 FP, F1=1.0")

    # 4. Compuerta de sobreajuste corto (Overfit Gate)
    x_sub = torch.stack([dataset[i][0] for i in chosen]).to(device)
    y_sub = torch.stack([dataset[i][1] for i in chosen]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.003)
    trace = []
    initial_loss = float(loss_fn_3a(model(x_sub), y_sub))
    for step in range(800):
        model.train()
        loss = loss_fn_3a(model(x_sub), y_sub)
        assert torch.isfinite(loss), "Pérdida no finita en preflight"
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % 50 == 0 or step == 799:
            model.eval()
            with torch.no_grad():
                z = model(x_sub)
                val_loss = float(loss_fn_3a(z, y_sub))
                pairs = [
                    (decode_grid32(p, rows['train'][i]['width'], rows['train'][i]['height'], threshold=0.25, grid=GRID), rows['train'][i]['boxes'])
                    for p, i in zip(z, chosen)
                ]
                met = metrics(pairs)
            trace.append(dict(step=step + 1, loss=val_loss, tp=met['tp'], fp=met['fp'], fn=met['fn'], f1=met['micro_f1']))
            if step >= 150 and val_loss < initial_loss * 0.2 and met['micro_f1'] >= 0.9:
                break

    last_f1 = trace[-1]['f1']
    last_loss = trace[-1]['loss']
    last_tp = trace[-1]['tp']
    print(f"[OK] 4. Compuerta de sobreajuste: pérdida reducida de {initial_loss:.4f} a {last_loss:.6f}, TP = {last_tp}/6, F1 = {last_f1:.4f} (pasos: {trace[-1]['step']})")
    assert last_tp >= 6 and last_loss < initial_loss * 0.2, "Fallo compuerta de sobreajuste"

    res = {
        "dimensions_verified": True,
        "max_reconstruction_error_px": max_err,
        "oracle_tp": single_met['tp'],
        "oracle_fp": single_met['fp'],
        "overfit_gate_passed": True,
        "initial_loss": initial_loss,
        "final_loss": last_loss,
        "final_f1": last_f1,
        "trace": trace,
        "grid": GRID,
        "slots": SLOTS,
        "stride": 8,
        "parameters": sum(p.numel() for p in model.parameters())
    }
    write_json(out_dir / "preflight.json", res)
    del model, opt, x_sub, y_sub
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("--- TODAS LAS COMPUERTAS PREFLIGHT SUPERADAS EXITOSAMENTE ---", flush=True)
    return res


def train_experiment(variant, out_dir, store, rows, base_npy_dir, epochs=40, batch=16, lr=0.001, seed=20261123, class_weights=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    snap = out_dir / 'snapshot'
    snap.mkdir(parents=True, exist_ok=True)

    # Freeze manifest snapshot
    write_json(snap / 'manifest.json', store.db)
    write_json(snap / 'eligible_ids.json', {s: [r['id'] for r in rr] for s, rr in rows.items()})

    device = device_for('auto')
    seed_all(seed)
    model = DetectorS8().to(device)

    trainset = CachedDataset32(rows['train'], base_npy_dir / 'train.npy', augment=True)
    valset = CachedDataset32(rows['val'], base_npy_dir / 'val.npy', augment=False)

    train_loader = DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=0, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(valset, batch_size=batch, shuffle=False, num_workers=0, pin_memory=(device.type == 'cuda'))

    config = {
        "experiment": variant,
        "architecture": "DetectorS8 (stride 8, grid 32x32, 3 slots, 347,649 params)",
        "epochs": epochs,
        "batch": batch,
        "lr": lr,
        "seed": seed,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "grid": GRID,
        "slots": SLOTS,
        "candidates_per_image": GRID * GRID * SLOTS,
        "classes": list(CLASSES),
        "class_weights": [float(w) for w in class_weights] if class_weights is not None else None,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "initialization": "random all layers from scratch, no pretrained backbone",
        "checkpoint_selection": "minimum validation loss"
    }
    write_json(out_dir / 'config.json', config)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    best_loss = float('inf')
    history = []
    start_time = time.perf_counter()

    loss_fn = (lambda p, t: loss_fn_3b(p, t, class_weights.to(device))) if class_weights is not None else loss_fn_3a

    print(f"\n>>> INICIANDO ENTRENAMIENTO {variant} (40 épocas, batch={batch}, device={device}) <<<", flush=True)
    for epoch in range(epochs):
        ep_start = time.perf_counter()
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch+1}")
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_loss += loss.item() * len(x)
        train_loss /= len(trainset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                loss = loss_fn(model(x), y)
                val_loss += loss.item() * len(x)
        val_loss /= len(valset)

        ep_duration = time.perf_counter() - ep_start
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "seconds": ep_duration,
            "seconds_elapsed": time.perf_counter() - start_time
        }
        history.append(row)
        write_json(out_dir / 'history.json', history)
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            torch.save({
                'origin': 'residuos-exp03-scratch',
                'variant': variant,
                'epoch': epoch + 1,
                'val_loss': val_loss,
                'train_loss': train_loss,
                'state': model.state_dict(),
                'classes': list(CLASSES),
                'config': config
            }, out_dir / 'best.pt')

        torch.save({
            'epoch': epoch + 1,
            'state': model.state_dict(),
            'optimizer': opt.state_dict(),
            'config': config
        }, out_dir / 'last.pt')

        star = " ★ (MEJOR)" if is_best else ""
        print(f"[{variant}] Época {epoch+1:02d}/40 | Train: {train_loss:.4f} | Val: {val_loss:.4f} | {ep_duration:.2f}s{star}", flush=True)

    print(f"\n>>> ENTRENAMIENTO FINALIZADO {variant}: mejor val_loss = {best_loss:.4f} <<<", flush=True)
    return history


def evaluate_and_diagnose(variant, out_dir, store, rows, npy_path, checkpoint_path):
    print(f"\n--- EVALUANDO Y DIAGNOSTICANDO {variant} ---", flush=True)
    device = device_for('auto')
    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = DetectorS8().to(device)
    model.load_state_dict(saved['state'])
    model.eval()

    valset = CachedDataset32(rows['val'], npy_path, augment=False)
    val_loader = DataLoader(valset, batch_size=16, shuffle=False)

    outputs = []
    with torch.no_grad():
        for x, _ in val_loader:
            outputs.extend(model(x.to(device)).cpu().numpy())

    raw_preds = outputs
    # 1. Sweep thresholds to find best operating point on validation
    threshold_grid = [0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.65, 0.8]
    sweep_results = []
    all_threshold_pairs = {}

    for th in threshold_grid:
        pairs = []
        for i, r in enumerate(rows['val']):
            preds = decode_grid32(torch.tensor(raw_preds[i]), r['width'], r['height'], threshold=th, nms=0.45, grid=GRID)
            pairs.append((preds, r['boxes']))
        met = metrics(pairs)
        sweep_results.append(dict(threshold=th, **met))
        all_threshold_pairs[th] = pairs

    best_sweep = max(sweep_results, key=lambda x: (x['micro_f1'], x['threshold']))
    chosen_th = best_sweep['threshold']

    # Also compute at baseline threshold 0.65
    base_th_met = next(x for x in sweep_results if abs(x['threshold'] - 0.65) < 1e-5)

    # 2. Detailed candidate audit: 3 moments (raw any, raw correct, final filtered)
    stages = {c: dict(objects=0, raw_any_class=0, raw_correct_class=0, final_correct=0) for c in CLASSES}
    size_breakdown = defaultdict(lambda: Counter())
    source_breakdown = defaultdict(lambda: Counter())

    all_scores = {c: [] for c in CLASSES}
    all_hits = {c: [] for c in CLASSES}

    # Extract all candidate boxes for each class to compute exact 101-point AP50
    final_pairs = all_threshold_pairs[chosen_th]
    for i, r in enumerate(rows['val']):
        gt = r['boxes']
        gt_classes = [b['class'] for b in gt]
        gt_boxes = [b['xyxy'] for b in gt]

        z = torch.tensor(raw_preds[i])
        probs, cls_indices = z[..., 5:].softmax(-1).max(-1)
        scores = z[..., 0].sigmoid() * probs

        # Flatten grid candidates
        wh = z[..., 1:5].sigmoid().numpy().reshape(-1, 4)
        gy, gx, slot = np.indices((GRID, GRID, SLOTS))
        gx, gy = gx.ravel(), gy.ravel()
        cx = (gx + wh[:, 0]) / GRID
        cy = (gy + wh[:, 1]) / GRID
        cands_xy = np.stack([
            np.clip(cx - wh[:, 2] / 2, 0, 1) * r['width'],
            np.clip(cy - wh[:, 3] / 2, 0, 1) * r['height'],
            np.clip(cx + wh[:, 2] / 2, 0, 1) * r['width'],
            np.clip(cy + wh[:, 3] / 2, 0, 1) * r['height']
        ], axis=1)
        cands_cls = cls_indices.numpy().ravel()
        cands_scores = scores.numpy().ravel()

        # Update stage stats for each GT
        for j, b in enumerate(gt):
            c = b['class']
            stages[c]['objects'] += 1

            # Check IoU with all candidates
            cand_ious = np.array([iou(cands_xy[k], b['xyxy']) for k in range(len(cands_xy))])
            has_geom = bool((cand_ious >= 0.5).any())
            correct_cls = cands_cls == CLASSES.index(c)
            has_correct = bool(((cand_ious >= 0.5) & correct_cls).any())

            if has_geom:
                stages[c]['raw_any_class'] += 1
            if has_correct:
                stages[c]['raw_correct_class'] += 1

            # Size classification
            w_norm = (b['xyxy'][2] - b['xyxy'][0]) * 256.0 / r['width']
            h_norm = (b['xyxy'][3] - b['xyxy'][1]) * 256.0 / r['height']
            area = w_norm * h_norm
            s_bin = 'small_<32px' if area < 32**2 else 'medium_32-96px' if area < 96**2 else 'large_>=96px'
            size_breakdown[(c, s_bin)]['objects'] += 1
            if has_geom:
                size_breakdown[(c, s_bin)]['geom'] += 1
            if has_correct:
                size_breakdown[(c, s_bin)]['correct'] += 1
            source_breakdown[(c, r['source'])]['objects'] += 1

        # Final predictions for this image
        final_preds_img = final_pairs[i][0]
        for c in CLASSES:
            gt_c = [b for b in gt if b['class'] == c]
            pr_c = sorted([b for b in final_preds_img if b['class'] == c], key=lambda x: x['score'], reverse=True)
            matched = set()
            for p in pr_c:
                eligible = [(iou(p['xyxy'], g['xyxy']), idx) for idx, g in enumerate(gt_c) if idx not in matched]
                val_iou, idx = max(eligible, default=(0, -1))
                if val_iou >= 0.5:
                    matched.add(idx)
                    all_scores[c].append(p['score'])
                    all_hits[c].append(True)
                else:
                    all_scores[c].append(p['score'])
                    all_hits[c].append(False)
            stages[c]['final_correct'] += len(matched)

    # 3. Compute AP50 per class using 101-point interpolated precision-recall curve
    ap50_dict = {}
    for c in CLASSES:
        n_gt = stages[c]['objects']
        if n_gt == 0 or len(all_scores[c]) == 0:
            ap50_dict[c] = 0.0
            continue
        order = np.argsort(-np.array(all_scores[c]))
        tp_arr = np.array(all_hits[c], dtype=int)[order]
        cumsum_tp = np.cumsum(tp_arr)
        cumsum_fp = np.arange(1, len(order) + 1) - cumsum_tp
        recall = cumsum_tp / n_gt
        precision = cumsum_tp / (cumsum_tp + cumsum_fp)
        # 101-point envelope
        envelope = np.maximum.accumulate(precision[::-1])[::-1]
        interp_rec = np.linspace(0, 1, 101)
        ap = float(np.mean([
            envelope[np.searchsorted(recall, r)] if np.searchsorted(recall, r) < len(envelope) else 0.0
            for r in interp_rec
        ]))
        ap50_dict[c] = ap

    map50 = float(np.mean(list(ap50_dict.values())))

    diag_result = {
        "variant": variant,
        "checkpoint": str(checkpoint_path),
        "best_epoch": saved['epoch'],
        "val_loss": saved['val_loss'],
        "chosen_threshold": chosen_th,
        "metrics_at_chosen_threshold": best_sweep,
        "metrics_at_base_threshold_065": base_th_met,
        "threshold_sweep": sweep_results,
        "stages": stages,
        "ap50": ap50_dict,
        "map50": map50,
        "size_breakdown": {f"{c}_{s}": dict(v) for (c, s), v in sorted(size_breakdown.items())},
        "source_breakdown": {f"{c}_{src}": dict(v) for (c, src), v in sorted(source_breakdown.items())}
    }
    write_json(out_dir / 'evaluation.json', diag_result)
    print(f"[{variant}] mAP50 = {map50*100:.2f}% | Micro-F1 (th={chosen_th}) = {best_sweep['micro_f1']*100:.2f}% | Micro-F1 (th=0.65) = {base_th_met['micro_f1']*100:.2f}%")
    return diag_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['preflight', 'train_3a', 'evaluate_3a', 'train_3b', 'evaluate_3b', 'full_pipeline'], default='full_pipeline')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=20261123)
    args = parser.parse_args()

    base_dir = ROOT / 'runs/detector_exp03'
    base_dir.mkdir(parents=True, exist_ok=True)
    exp01_dir = ROOT / 'runs/detector_exp01'

    # Load frozen manifest & splits
    store = Store(ROOT / 'data/preparado/deteccion')
    store.db = read_json(exp01_dir / 'snapshot/manifest.json')
    rows = {s: store.records(s) for s in ('train', 'val')}

    # Preflight
    if args.action in ('preflight', 'full_pipeline'):
        run_preflight_checks(base_dir / 'preflight', rows, exp01_dir / 'train.npy', device_for('auto'), args.seed)

    # Train 3A
    dir_3a = base_dir / 'architecture'
    if args.action in ('train_3a', 'full_pipeline'):
        train_experiment('Exp3A_Architecture_32x32', dir_3a, store, rows, exp01_dir, epochs=args.epochs, batch=args.batch, lr=args.lr, seed=args.seed)

    # Evaluate 3A
    if args.action in ('evaluate_3a', 'full_pipeline'):
        diag_3a = evaluate_and_diagnose('Exp3A', dir_3a, store, rows, exp01_dir / 'val.npy', dir_3a / 'best.pt')

    # Train 3B (with class weighting)
    dir_3b = base_dir / 'balance'
    if args.action in ('train_3b', 'full_pipeline'):
        # Class distribution in train: {'plastico': 3398, 'organico': 5084, 'metal': 260, 'vidrio': 71, 'carton': 82, 'papel': 57}
        # Smoothed inverse class weights: w_c = (N_total / (N_c * 6)) ** 0.5
        train_counts = [3398.0, 57.0, 82.0, 71.0, 260.0, 5084.0]
        total_objs = sum(train_counts)
        inv_w = [np.sqrt(total_objs / (6.0 * max(1.0, cnt))) for cnt in train_counts]
        # Normalize so mean weight is 1.0
        norm_w = [w / np.mean(inv_w) for w in inv_w]
        class_weights = torch.tensor(norm_w, dtype=torch.float32)
        print("Pesos de clase calculados para Exp 3B:", {c: round(float(w), 3) for c, w in zip(CLASSES, class_weights)}, flush=True)

        train_experiment('Exp3B_ClassWeighted', dir_3b, store, rows, exp01_dir, epochs=args.epochs, batch=args.batch, lr=args.lr, seed=args.seed, class_weights=class_weights)

    # Evaluate 3B
    if args.action in ('evaluate_3b', 'full_pipeline'):
        diag_3b = evaluate_and_diagnose('Exp3B', dir_3b, store, rows, exp01_dir / 'val.npy', dir_3b / 'best.pt')

    print("\n================ PIPELINE EXPERIMENTO 3 COMPLETADO ================", flush=True)


if __name__ == '__main__':
    main()
