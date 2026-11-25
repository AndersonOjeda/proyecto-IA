"""Generate comparative visual panels and charts for Experiment 5 vs Experiment 3C."""
from pathlib import Path
import sys
import json
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from residuos import CLASSES
from residuos.data import Store, read_json, write_json
from scripts.experiment3_runner import DetectorS8, decode_grid32

EXP05_DIR = ROOT / 'runs/detector_exp05'
VIS_DIR = EXP05_DIR / 'visuals'
VIS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_COLORS = {
    'plastico': (255, 120, 0),
    'papel': (0, 200, 255),
    'carton': (0, 140, 255),
    'vidrio': (255, 0, 255),
    'metal': (0, 255, 255),
    'organico': (0, 255, 0)
}

def render_comparison_panels():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    store = Store(ROOT / 'data/preparado/deteccion')
    val_rows = {r['id']: r for r in store.records('val')}

    # Load 3C
    m3c = DetectorS8().to(device)
    ckpt3c = torch.load(ROOT / 'runs/detector_exp03/corrected/best.pt', map_location=device, weights_only=False)
    m3c.load_state_dict(ckpt3c['state'])
    m3c.eval()

    # Load Exp 5
    m5 = DetectorS8().to(device)
    ckpt5 = torch.load(EXP05_DIR / 'best.pt', map_location=device, weights_only=False)
    m5.load_state_dict(ckpt5['state'])
    m5.eval()

    targets = [
        {"id": "96297912b97623ccad89", "focus": "Metal y Reduccion FP en TACO"},
        {"id": "41a7ff550dd06d4a71a6", "focus": "Organico Food Waste (TP vs FP)"},
        {"id": "9bc7093c53f63dff3483", "focus": "Plastico Bangladesh en Exteriores"},
        {"id": "e9144430b0e8dafe56fe", "focus": "Vidrio en Escenario Exterior (TACO)"},
        {"id": "9935ac60f69cfc916ced", "focus": "Papel y Tamano de Objeto (TACO)"},
        {"id": "eba0cb1763b43d3fb0cc", "focus": "Carton en Fondo Complejo (TACO)"}
    ]

    for tgt in targets:
        tid = tgt['id']
        r = val_rows.get(tid)
        if not r:
            continue
        img_p = ROOT / 'data/preparado/deteccion' / r['path']
        im = cv2.imdecode(np.fromfile(str(img_p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            continue
        h, w = im.shape[:2]

        im_resized = cv2.resize(im, (256, 256))
        x = torch.from_numpy(cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

        with torch.no_grad():
            preds_3c = decode_grid32(m3c(x)[0].cpu(), w, h, threshold=0.40, nms=0.45)
            preds_5 = decode_grid32(m5(x)[0].cpu(), w, h, threshold=0.40, nms=0.45)

        def draw_box_panel(title, boxes, is_gt=False):
            canvas = im.copy()
            for b in boxes:
                c = b['class']
                col = CLASS_COLORS.get(c, (0, 255, 0))
                x1, y1, x2, y2 = [int(v) for v in b['xyxy']]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), col, max(2, w // 250))
                label = f"GT: {c}" if is_gt else f"{c} {b['score']:.2f}"
                cv2.putText(canvas, label, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, max(0.42, w / 1100), col, 2)
            panel = cv2.resize(canvas, (400, 400))
            header = np.zeros((26, 400, 3), dtype=np.uint8)
            cv2.putText(header, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            return np.vstack([header, panel])

        p_gt = draw_box_panel(f"Ground Truth ({len(r['boxes'])} obj)", r['boxes'], is_gt=True)
        p_3c = draw_box_panel(f"Exp 3C th=0.40 ({len(preds_3c)} det)", preds_3c)
        p_5 = draw_box_panel(f"Exp 5 (SALSA+TACO) th=0.40 ({len(preds_5)} det)", preds_5)

        banner = np.zeros((32, 1200, 3), dtype=np.uint8)
        banner[:] = (20, 20, 20)
        cv2.putText(banner, f"Comparativa Exp 5 vs Exp 3C - {tgt['focus']} | ID: {tid}", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 1)

        trio = np.hstack([p_gt, p_3c, p_5])
        final_panel = np.vstack([banner, trio])
        out_file = VIS_DIR / f"comparativa_{tid}.jpg"
        cv2.imwrite(str(out_file), final_panel)
        print(f"Panel visual guardado: {out_file.name}")

    # Generate AP50 bar chart
    W, H, M = 850, 400, 60
    chart = np.full((H, W, 3), 255, np.uint8)
    e3 = read_json(ROOT / 'runs/detector_exp04/eval_3c_homogeneous.json')
    e5 = read_json(EXP05_DIR / 'evaluation.json')
    ap3 = e3['true_unthresholded_ap50']
    ap5 = e5['true_unthresholded_ap50']

    gw = (W - 2 * M) / len(CLASSES)
    bw = gw / 3.0
    for ci, c in enumerate(CLASSES):
        v3 = ap3[c] * 100
        v5 = ap5[c] * 100
        x3 = int(M + ci * gw)
        y3 = int(H - M - v3 / 20.0 * (H - 2 * M))
        x5 = int(x3 + bw + 4)
        y5 = int(H - M - v5 / 20.0 * (H - 2 * M))

        cv2.rectangle(chart, (x3, max(y3, M)), (int(x3 + bw), H - M), (255, 165, 0), -1)
        cv2.rectangle(chart, (x5, max(y5, M)), (int(x5 + bw), H - M), (0, 200, 255), -1)

        cv2.putText(chart, f"{v3:.1f}", (x3, max(y3 - 4, M + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.putText(chart, f"{v5:.1f}", (x5, max(y5 - 4, M + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.putText(chart, c, (int(M + ci * gw + 4), H - M + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    cv2.line(chart, (M, H - M), (W - M, H - M), (0, 0, 0), 1)
    # Legend
    cv2.rectangle(chart, (M, 15), (M + 14, 29), (255, 165, 0), -1)
    cv2.putText(chart, f"Exp 3C (mAP50 = {e3['true_unthresholded_map50']*100:.2f}%)", (M + 20, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.rectangle(chart, (M + 300, 15), (M + 314, 29), (0, 200, 255), -1)
    cv2.putText(chart, f"Exp 5 Outdoor (mAP50 = {e5['true_unthresholded_map50']*100:.2f}%)", (M + 320, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    cv2.imwrite(str(VIS_DIR / 'ap50_exp05_vs_exp3c.png'), chart)
    print("Gráfica comparativa AP50 guardada.")

if __name__ == '__main__':
    render_comparison_panels()
