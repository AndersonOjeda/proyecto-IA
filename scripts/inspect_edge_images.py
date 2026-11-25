import json, torch, numpy as np, cv2
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from residuos.data import Store
from residuos.detector import load_model, predict, iou, device_for
from scripts.experiment3_runner import DetectorS8, decode_grid32

device = device_for('auto')
cp = torch.load('runs/detector/best.pt', map_location=device, weights_only=False)
model = DetectorS8().to(device)
model.load_state_dict(cp['state'])
model.eval()

store = Store('data/preparado/deteccion')
val_records = store.records('val')
val_npy = np.load('runs/detector_exp01/val.npy', mmap_mode='r')

for idx in [203, 437]:
    r = val_records[idx]
    print(f"=== Image idx {idx}: ID={r['id']} ===")
    print("Ground truth boxes:", r['boxes'])
    
    # Single NPY
    x_s = torch.from_numpy(np.array(val_npy[idx])).permute(2, 0, 1).float() / 255.0
    with torch.no_grad():
        out_s = model(x_s[None].to(device))[0].cpu()
    b_s = decode_grid32(out_s, r['width'], r['height'], threshold=0.34, nms=0.45)
    print("Single NPY predictions (th=0.34):")
    for b in b_s:
        print("  ", b['class'], f"score={b['score']:.5f}", "xyxy=", [round(v, 1) for v in b['xyxy']])
        for g_idx, g in enumerate(r['boxes']):
            print(f"    vs GT {g_idx} ({g['class']}): IoU={iou(b['xyxy'], g['xyxy']):.4f}")
            
    # Batched NPY
    x_b = torch.from_numpy(np.array(val_npy[idx:idx+32])).permute(0, 3, 1, 2).float() / 255.0
    with torch.no_grad():
        out_b = model(x_b.to(device))[0].cpu()
    b_b = decode_grid32(out_b, r['width'], r['height'], threshold=0.34, nms=0.45)
    print("Batched NPY predictions (th=0.34):")
    for b in b_b:
        print("  ", b['class'], f"score={b['score']:.5f}", "xyxy=", [round(v, 1) for v in b['xyxy']])
        for g_idx, g in enumerate(r['boxes']):
            print(f"    vs GT {g_idx} ({g['class']}): IoU={iou(b['xyxy'], g['xyxy']):.4f}")

    # Direct App Frame
    frame = cv2.imdecode(np.fromfile(store.root / r['path'], dtype=np.uint8), cv2.IMREAD_COLOR)
    b_a = predict(model, frame, device, threshold=0.34)
    print("App disk frame predictions (th=0.34):")
    for b in b_a:
        print("  ", b['class'], f"score={b['score']:.5f}", "xyxy=", [round(v, 1) for v in b['xyxy']])
        for g_idx, g in enumerate(r['boxes']):
            print(f"    vs GT {g_idx} ({g['class']}): IoU={iou(b['xyxy'], g['xyxy']):.4f}")
