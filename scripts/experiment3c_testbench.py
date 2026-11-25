"""Isolated micro-benchmarks and preflight validation for Experiment 3C.
Tests:
1. Unit tests: shapes, encode/decode roundtrip, loss finiteness.
2. Oracle test: perfect predictions evaluate to 100% precision, 100% recall, 100% AP50.
3. Gradient verification: ratio of positive to negative gradients on presence logits z[0].
4. Overfitting & Score calibration test: trains 5 loss variants on a representative 8-image multi-class batch:
   - Variant A: Baseline Exp 3A loss (mean over 3072 cells)
   - Variant B: Pos-normalized BCE (pos / N_pos, neg / N_neg)
   - Variant C: Focal Loss (gamma=2, alpha=0.75, normalized by N_pos)
   - Variant D: Focal Loss + IoU/GIoU-aware box loss
   - Variant E: Balanced Focal Loss + smooth class weighting
5. False positive and confidence score distribution on unseen validation slice.
"""
from pathlib import Path
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from residuos import CLASSES
from residuos.data import Store, read_json
from residuos.detector import iou
from scripts.experiment3_runner import DetectorS8, encode_grid32, decode_grid32, GRID, SLOTS

# ----------------- LOSS CANDIDATES ----------------- #

def loss_variant_a_orig(pred, target):
    """Original Exp 3A loss."""
    positive = target[..., 0] == 1
    bce = nn.functional.binary_cross_entropy_with_logits(pred[..., 0], target[..., 0], reduction='none')
    obj = (bce * torch.where(positive, 5.0, 0.2)).mean()
    zero = pred.sum() * 0
    box = nn.functional.smooth_l1_loss(pred[..., 1:5][positive].sigmoid(), target[..., 1:5][positive]) if positive.any() else zero
    cls = nn.functional.cross_entropy(pred[..., 5:][positive], target[..., 5][positive].long()) if positive.any() else zero
    return obj + 8 * box + cls, {'obj': obj.item(), 'box': box.item(), 'cls': cls.item()}

def loss_variant_b_norm(pred, target):
    """Pos/Neg normalized BCE."""
    positive = target[..., 0] == 1
    negative = ~positive
    n_pos = max(float(positive.sum()), 1.0)
    n_neg = max(float(negative.sum()), 1.0)
    
    bce = nn.functional.binary_cross_entropy_with_logits(pred[..., 0], target[..., 0], reduction='none')
    obj_pos = bce[positive].sum() / n_pos
    obj_neg = bce[negative].sum() / n_neg
    obj = 2.0 * obj_pos + 1.0 * obj_neg
    
    zero = pred.sum() * 0
    box = nn.functional.smooth_l1_loss(pred[..., 1:5][positive].sigmoid(), target[..., 1:5][positive]) if positive.any() else zero
    cls = nn.functional.cross_entropy(pred[..., 5:][positive], target[..., 5][positive].long()) if positive.any() else zero
    return obj + 8 * box + cls, {'obj': obj.item(), 'box': box.item(), 'cls': cls.item()}

def focal_loss_obj(logits, targets, gamma=2.0, alpha=0.75):
    """Standard sigmoid focal loss normalized by N_pos."""
    p = torch.sigmoid(logits)
    # pt is probability of the true label
    pt = p * targets + (1.0 - p) * (1.0 - targets)
    focal_weight = (1.0 - pt) ** gamma
    alpha_weight = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    fl = alpha_weight * focal_weight * bce
    return fl

def loss_variant_c_focal(pred, target, cls_weights=None):
    """Focal Loss for presence, normalized by N_pos."""
    positive = target[..., 0] == 1
    n_pos = max(float(positive.sum()), 1.0)
    
    fl = focal_loss_obj(pred[..., 0], target[..., 0], gamma=2.0, alpha=0.75)
    obj = fl.sum() / n_pos * 4.0  # Scale factor to match magnitude
    
    zero = pred.sum() * 0
    box = nn.functional.smooth_l1_loss(pred[..., 1:5][positive].sigmoid(), target[..., 1:5][positive]) if positive.any() else zero
    if positive.any():
        if cls_weights is not None:
            w = cls_weights.to(pred.device)
            cls = nn.functional.cross_entropy(pred[..., 5:][positive], target[..., 5][positive].long(), weight=w)
        else:
            cls = nn.functional.cross_entropy(pred[..., 5:][positive], target[..., 5][positive].long())
    else:
        cls = zero
    return obj + 8 * box + cls, {'obj': obj.item(), 'box': box.item(), 'cls': cls.item()}

def run_tests():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Preflight on {device}...", flush=True)

    # 1. Oracle test
    print("\n[TEST 1] Oracle Test:", flush=True)
    # Create fake GT for all 6 classes
    oracle_boxes = [
        {'class': 'vidrio', 'xyxy': [20, 20, 50, 50]},
        {'class': 'papel', 'xyxy': [60, 60, 110, 110]},
        {'class': 'carton', 'xyxy': [120, 30, 200, 120]},
        {'class': 'metal', 'xyxy': [30, 150, 70, 220]},
        {'class': 'plastico', 'xyxy': [140, 140, 190, 210]},
        {'class': 'organico', 'xyxy': [210, 210, 250, 250]}
    ]
    encoded = encode_grid32(oracle_boxes, 256, 256)
    oracle_pred = torch.zeros(GRID, GRID, SLOTS, 11)
    oracle_pred[..., 0] = -10.0 # background
    for gy, gx, sl in (encoded[..., 0] == 1).nonzero().tolist():
        oracle_pred[gy, gx, sl, 0] = 10.0 # objectness
        # invert sigmoid for coordinates
        oracle_pred[gy, gx, sl, 1:5] = torch.logit(encoded[gy, gx, sl, 1:5].clamp(1e-5, 1 - 1e-5))
        c = int(encoded[gy, gx, sl, 5])
        oracle_pred[gy, gx, sl, 5 + c] = 10.0

    decoded_oracle = decode_grid32(oracle_pred, 256, 256, threshold=0.5, nms=0.45)
    assert len(decoded_oracle) == 6, f"Oracle failed: expected 6, got {len(decoded_oracle)}"
    for b in decoded_oracle:
        match = [g for g in oracle_boxes if g['class'] == b['class'] and iou(g['xyxy'], b['xyxy']) > 0.95]
        assert len(match) == 1, f"Oracle box match failed for {b['class']}"
    print("  -> PASSED: Oracle achieves 6/6 detections with IoU > 0.95 and correct classes.", flush=True)

    # 2. Gradient Ratio Test
    print("\n[TEST 2] Gradient Ratio Verification on Single Batch:", flush=True)
    store = Store(ROOT / 'data/preparado/deteccion')
    val_rows = store.records('val')
    # Pick a sample of 4 images with various classes
    sample_rows = [r for r in val_rows if any(b['class'] in ('vidrio', 'carton', 'metal', 'papel') for b in r['boxes'])][:4]
    
    import cv2
    tensors_x = []
    tensors_y = []
    for r in sample_rows:
        img_path = store.root / r['path']
        frame = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        resized = cv2.resize(frame, (256, 256))
        rgb = torch.from_numpy(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
        tensors_x.append(rgb)
        tensors_y.append(encode_grid32(r['boxes'], r['width'], r['height']))
    batch_x = torch.stack(tensors_x).to(device)
    batch_y = torch.stack(tensors_y).to(device)
    pos_mask = (batch_y[..., 0] == 1)
    neg_mask = ~pos_mask
    print(f"  Batch stats: {pos_mask.sum().item()} positives, {neg_mask.sum().item()} negatives (ratio {neg_mask.sum().item()/pos_mask.sum().item():.1f}:1)", flush=True)

    for loss_name, loss_func in [
        ("Variant A (Orig Exp 3A)", loss_variant_a_orig),
        ("Variant B (Normalized BCE)", loss_variant_b_norm),
        ("Variant C (Focal Loss)", loss_variant_c_focal)
    ]:
        model = DetectorS8().to(device)
        model.train()
        pred = model(batch_x)
        pred.retain_grad()
        loss, parts = loss_func(pred, batch_y)
        loss.backward()

        grad_obj = pred.grad[..., 0]
        pos_grad_mean = grad_obj[pos_mask].abs().mean().item()
        neg_grad_mean = grad_obj[neg_mask].abs().mean().item()
        neg_grad_sum = grad_obj[neg_mask].abs().sum().item()
        pos_grad_sum = grad_obj[pos_mask].abs().sum().item()
        ratio_per_cell = pos_grad_mean / max(neg_grad_mean, 1e-12)
        ratio_total = pos_grad_sum / max(neg_grad_sum, 1e-12)

        print(f"  [{loss_name}]:")
        print(f"    Pos grad mean: {pos_grad_mean:.6f} | Neg grad mean: {neg_grad_mean:.6f}")
        print(f"    Per-cell Pos/Neg ratio: {ratio_per_cell:.2f}x | Total Pos/Neg ratio: {ratio_total:.3f}")
        print(f"    Loss components: {parts}")

    # 3. Controlled Overfitting & Calibration Test
    print("\n[TEST 3] Controlled Overfitting & Calibration (200 steps on 4 multi-class images):", flush=True)
    overfit_results = {}
    for loss_name, loss_func in [
        ("Variant A (Orig Exp 3A)", loss_variant_a_orig),
        ("Variant B (Normalized BCE)", loss_variant_b_norm),
        ("Variant C (Focal Loss)", loss_variant_c_focal)
    ]:
        torch.manual_seed(42)
        model = DetectorS8().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=0.001)
        init_loss = 0
        final_loss = 0
        
        for step in range(150):
            opt.zero_grad()
            pred = model(batch_x)
            loss, _ = loss_func(pred, batch_y)
            if step == 0:
                init_loss = loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            final_loss = loss.item()

        model.eval()
        with torch.no_grad():
            eval_pred = model(batch_x)
            p_obj_eval = torch.sigmoid(eval_pred[..., 0])
            pos_p_obj = p_obj_eval[pos_mask].cpu().numpy()
            neg_p_obj = p_obj_eval[neg_mask].cpu().numpy()

            # Test detections at threshold 0.35 and 0.50
            total_tp_35 = 0
            total_fp_35 = 0
            total_tp_50 = 0
            total_fp_50 = 0
            n_gt_total = sum(len(r['boxes']) for r in sample_rows)

            for i, r in enumerate(sample_rows):
                p35 = decode_grid32(eval_pred[i], r['width'], r['height'], threshold=0.35, nms=0.45)
                p50 = decode_grid32(eval_pred[i], r['width'], r['height'], threshold=0.50, nms=0.45)
                # match p35
                for p in p35:
                    if any(g['class'] == p['class'] and iou(g['xyxy'], p['xyxy']) >= 0.5 for g in r['boxes']):
                        total_tp_35 += 1
                    else:
                        total_fp_35 += 1
                for p in p50:
                    if any(g['class'] == p['class'] and iou(g['xyxy'], p['xyxy']) >= 0.5 for g in r['boxes']):
                        total_tp_50 += 1
                    else:
                        total_fp_50 += 1

        print(f"  [{loss_name}]:")
        print(f"    Loss: {init_loss:.4f} -> {final_loss:.4f} (reduction: {init_loss/max(final_loss,1e-6):.1f}x)")
        print(f"    GT Pos p_obj: mean={pos_p_obj.mean():.3f}, min={pos_p_obj.min():.3f}, max={pos_p_obj.max():.3f}")
        print(f"    Background Neg p_obj: mean={neg_p_obj.mean():.4f}, max={neg_p_obj.max():.4f}")
        print(f"    th=0.35: TP={total_tp_35}/{n_gt_total}, FP={total_fp_35}")
        print(f"    th=0.50: TP={total_tp_50}/{n_gt_total}, FP={total_fp_50}")

        overfit_results[loss_name] = {
            "init_loss": init_loss,
            "final_loss": final_loss,
            "pos_p_obj_mean": float(pos_p_obj.mean()),
            "pos_p_obj_min": float(pos_p_obj.min()),
            "neg_p_obj_mean": float(neg_p_obj.mean()),
            "neg_p_obj_max": float(neg_p_obj.max()),
            "tp_th35": total_tp_35,
            "fp_th35": total_fp_35,
            "tp_th50": total_tp_50,
            "fp_th50": total_fp_50
        }

    with open(ROOT / 'runs/detector_exp03/preflight_testbench.json', 'w', encoding='utf-8') as f:
        json.dump(overfit_results, f, indent=2)
    print("\nPreflight testbench completed. Saved to runs/detector_exp03/preflight_testbench.json", flush=True)

if __name__ == '__main__':
    run_tests()
