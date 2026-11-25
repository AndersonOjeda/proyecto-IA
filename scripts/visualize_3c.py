"""Generate visual comparison panels and consolidate metrics for Experiment 3C.
"""
from pathlib import Path
import sys
import json
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from residuos import CLASSES
from residuos.data import Store, read_json, write_json
from residuos.detector import Detector, decode
from scripts.experiment3_runner import DetectorS8, decode_grid32

def run_visuals_3c():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = ROOT / 'runs/detector_exp03/visuals_3c'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    store = Store(ROOT / 'data/preparado/deteccion')
    val_rows = store.records('val')
    
    # Load Baseline Exp 1
    base_model = Detector().to(device)
    base_ckpt = torch.load(ROOT / 'runs/detector_exp01/best.pt', map_location=device, weights_only=False)
    base_model.load_state_dict(base_ckpt.get('state', base_ckpt.get('state_dict')))
    base_model.eval()
    
    # Load Exp 3C
    m3c = DetectorS8().to(device)
    ckpt3c = torch.load(ROOT / 'runs/detector_exp03/corrected/best.pt', map_location=device, weights_only=False)
    m3c.load_state_dict(ckpt3c.get('state', ckpt3c.get('state_dict')))
    m3c.eval()
    
    targets = [
        {"name": "01_metal_lata", "id": "96297912b97623ccad89", "focus": "Metal (TACO)"},
        {"name": "02_organico_comida", "id": "41a7ff550dd06d4a71a6", "focus": "Organico <32px (Food Waste)"},
        {"name": "03_papel_residuo", "id": "9935ac60f69cfc916ced", "focus": "Papel (TACO)"},
        {"name": "04_carton_caja", "id": "eba0cb1763b43d3fb0cc", "focus": "Carton (TACO)"},
        {"name": "05_plastico_botellas", "id": "9bc7093c53f63dff3483", "focus": "Plastico (Bangladesh)"},
        {"name": "06_vidrio_botella", "id": "e9144430b0e8dafe56fe", "focus": "Vidrio (TACO)"}
    ]
    
    COLORS = {
        'GT': (0, 255, 0),        # Green
        'Exp1': (255, 140, 0),     # Orange
        'Exp3C': (0, 215, 255)     # Gold / Bright Yellow
    }
    
    for tgt in targets:
        r = next((x for x in val_rows if x['id'] == tgt['id']), None)
        if not r: continue
        
        img_path = store.root / r['path']
        raw_bgr = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if raw_bgr is None: continue
        h, w = raw_bgr.shape[:2]
        
        resized = cv2.resize(raw_bgr, (256, 256))
        rgb = torch.from_numpy(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        rgb = rgb.to(device)
        
        with torch.no_grad():
            out1 = base_model(rgb)
            preds1 = decode(out1[0], w, h, threshold=0.25, nms=0.45)
            
            out3c = m3c(rgb)
            preds3c = decode_grid32(out3c[0], w, h, threshold=0.25, nms=0.45)
            
        def render_panel(title, bboxes, tag):
            canvas = raw_bgr.copy()
            for b in bboxes:
                xyxy = [int(v) for v in b['xyxy']]
                col = COLORS[tag]
                if tag == 'GT':
                    label = f"GT: {b['class']}"
                else:
                    sc = b.get('score', 0.0)
                    label = f"{b['class']} {sc:.2f}"
                cv2.rectangle(canvas, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), col, 2)
                txt_sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
                cv2.rectangle(canvas, (xyxy[0], max(0, xyxy[1] - 18)), (xyxy[0] + txt_sz[0] + 6, max(0, xyxy[1])), col, -1)
                cv2.putText(canvas, label, (xyxy[0] + 3, max(12, xyxy[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.rectangle(canvas, (0, 0), (w, 30), (30, 30, 30), -1)
            cv2.putText(canvas, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            return cv2.resize(canvas, (450, 450))
            
        p_gt = render_panel("Ground Truth (Anotacion Real)", r['boxes'], 'GT')
        p_base = render_panel(f"Baseline Exp 1 (s16, {len(preds1)} det)", preds1, 'Exp1')
        p_3c = render_panel(f"Exp 3C Focal Loss (s8, {len(preds3c)} det)", preds3c, 'Exp3C')
        
        row_img = np.hstack([p_gt, p_base, p_3c])
        banner = np.zeros((45, row_img.shape[1], 3), dtype=np.uint8)
        banner[:] = (20, 20, 20)
        cv2.putText(banner, f"Comparativa Experimento 3C - {tgt['focus']} | ID: {tgt['id']}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2, cv2.LINE_AA)
        
        combined = np.vstack([banner, row_img])
        out_file = out_dir / f"{tgt['name']}_{tgt['id']}.jpg"
        cv2.imwrite(str(out_file), combined)
        print(f"Saved: {out_file.name}", flush=True)

if __name__ == '__main__':
    run_visuals_3c()
