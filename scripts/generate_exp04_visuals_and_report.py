"""Generate comparative visual panels (True Positives and Errors across classes)
and consolidated comparison JSON for Experiment 4 vs Experiment 3C.
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
from scripts.experiment3_runner import DetectorS8, decode_grid32

EXP04_DIR = ROOT / 'runs/detector_exp04'

def generate_visuals():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vis_dir = EXP04_DIR / 'visuals'
    vis_dir.mkdir(parents=True, exist_ok=True)

    store = Store(ROOT / 'data/preparado/deteccion')
    val_rows = store.records('val')

    # Load Exp 3C
    m3c = DetectorS8().to(device)
    ckpt3c = torch.load(ROOT / 'runs/detector_exp03/corrected/best.pt', map_location=device, weights_only=False)
    m3c.load_state_dict(ckpt3c['state'])
    m3c.eval()

    # Load Exp 4B (best.pt)
    m4b = DetectorS8().to(device)
    ckpt4b = torch.load(EXP04_DIR / 'best.pt', map_location=device, weights_only=False)
    m4b.load_state_dict(ckpt4b['state'])
    m4b.eval()

    # Load Exp 4A (best_val_loss.pt - 6/6 classes >0% AP50, vidrio 1.08%)
    m4a = DetectorS8().to(device)
    ckpt4a = torch.load(EXP04_DIR / 'variant_4a_broad/best_val_loss.pt', map_location=device, weights_only=False)
    m4a.load_state_dict(ckpt4a['state'])
    m4a.eval()

    targets = [
        {"name": "01_metal_tp_taco", "id": "96297912b97623ccad89", "focus": "Metal TP y Reduccion FP (TACO)"},
        {"name": "02_organico_fp_reduction", "id": "41a7ff550dd06d4a71a6", "focus": "Organico TP y Control de FP (Food Waste)"},
        {"name": "03_plastico_tp_bangladesh", "id": "9bc7093c53f63dff3483", "focus": "Plastico Alta Precision (Bangladesh)"},
        {"name": "04_vidrio_taco", "id": "e9144430b0e8dafe56fe", "focus": "Vidrio Candidato Desbloqueado (TACO)"},
        {"name": "05_papel_taco", "id": "9935ac60f69cfc916ced", "focus": "Papel Error de Clasificacion / Escala (TACO)"},
        {"name": "06_carton_taco", "id": "eba0cb1763b43d3fb0cc", "focus": "Carton Localizacion vs Confianza (TACO)"}
    ]

    COLORS = {
        'GT': (0, 255, 0),
        'Exp3C': (255, 165, 0),
        'Exp4A': (255, 50, 200),
        'Exp4B': (0, 230, 255)
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
            p3c = decode_grid32(m3c(rgb)[0], w, h, threshold=0.25, nms=0.45)
            p4a = decode_grid32(m4a(rgb)[0], w, h, threshold=0.20, nms=0.45)
            p4b = decode_grid32(m4b(rgb)[0], w, h, threshold=0.25, nms=0.45)

        def render_panel(title, bboxes, tag):
            canvas = raw_bgr.copy()
            for b in bboxes:
                xyxy = [int(v) for v in b['xyxy']]
                col = COLORS[tag]
                if tag == 'GT':
                    label = f"GT: {b['class']}"
                else:
                    label = f"{b['class']} {b.get('score', 0.0):.2f}"
                cv2.rectangle(canvas, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), col, 2)
                tsz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
                cv2.rectangle(canvas, (xyxy[0], max(0, xyxy[1] - 18)), (xyxy[0] + tsz[0] + 6, max(0, xyxy[1])), col, -1)
                cv2.putText(canvas, label, (xyxy[0] + 3, max(12, xyxy[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.rectangle(canvas, (0, 0), (w, 30), (25, 25, 25), -1)
            cv2.putText(canvas, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            return cv2.resize(canvas, (420, 420))

        pan_gt = render_panel("Ground Truth (Anotacion Real)", r['boxes'], 'GT')
        pan_3c = render_panel(f"Exp 3C (th=0.25, {len(p3c)} det)", p3c, 'Exp3C')
        pan_4a = render_panel(f"Exp 4A 1600-Ext (th=0.20, {len(p4a)} det)", p4a, 'Exp4A')
        pan_4b = render_panel(f"Exp 4B 400-DomAlign (th=0.25, {len(p4b)} det)", p4b, 'Exp4B')

        grid_img = np.vstack([np.hstack([pan_gt, pan_3c]), np.hstack([pan_4a, pan_4b])])
        banner = np.zeros((45, grid_img.shape[1], 3), dtype=np.uint8)
        banner[:] = (18, 18, 18)
        cv2.putText(banner, f"Experimento 4 vs 3C - {tgt['focus']} | ID: {tgt['id']}", (15, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 230, 255), 2, cv2.LINE_AA)
        final_img = np.vstack([banner, grid_img])
        out_p = vis_dir / f"{tgt['name']}_{tgt['id']}.jpg"
        cv2.imwrite(str(out_p), final_img)
        print(f"Saved visual panel: {out_p.name}", flush=True)

def consolidate_comparison():
    eval_3c = read_json(ROOT / 'runs/detector_exp03/corrected/evaluation.json')
    eval_4a_ep17 = read_json(EXP04_DIR / 'variant_4a_broad/evaluation_best_val_loss.json')
    eval_4a_ep40 = read_json(EXP04_DIR / 'variant_4a_broad/evaluation.json')
    eval_4b = read_json(EXP04_DIR / 'evaluation.json')

    comp = {
        'models': {
            'Exp3C_Baseline_Reference': {
                'checkpoint': 'runs/detector_exp03/corrected/best.pt',
                'train_images_unique': 3645,
                'mAP50_unthresholded': eval_3c['true_unthresholded_map50'],
                'ap50_per_class': eval_3c['true_unthresholded_ap50'],
                'classes_with_positive_ap50': sum(1 for v in eval_3c['true_unthresholded_ap50'].values() if v > 0),
                'operating_point': eval_3c['best_operating_point']
            },
            'Exp04A_Broad1600_Ep17_All6Classes': {
                'checkpoint': 'runs/detector_exp04/variant_4a_broad/best_val_loss.pt',
                'train_images_unique': 5245,
                'mAP50_unthresholded': eval_4a_ep17['true_unthresholded_map50'],
                'ap50_per_class': eval_4a_ep17['true_unthresholded_ap50'],
                'classes_with_positive_ap50': sum(1 for v in eval_4a_ep17['true_unthresholded_ap50'].values() if v > 0),
                'operating_point': eval_4a_ep17['best_operating_point']
            },
            'Exp04A_Broad1600_Ep40': {
                'checkpoint': 'runs/detector_exp04/variant_4a_broad/best.pt',
                'train_images_unique': 5245,
                'mAP50_unthresholded': eval_4a_ep40['true_unthresholded_map50'],
                'ap50_per_class': eval_4a_ep40['true_unthresholded_ap50'],
                'classes_with_positive_ap50': sum(1 for v in eval_4a_ep40['true_unthresholded_ap50'].values() if v > 0),
                'operating_point': eval_4a_ep40['best_operating_point']
            },
            'Exp04B_DomainAligned400_Ep34': {
                'checkpoint': 'runs/detector_exp04/best.pt',
                'train_images_unique': 4045,
                'mAP50_unthresholded': eval_4b['true_unthresholded_map50'],
                'ap50_per_class': eval_4b['true_unthresholded_ap50'],
                'classes_with_positive_ap50': sum(1 for v in eval_4b['true_unthresholded_ap50'].values() if v > 0),
                'operating_point': eval_4b['best_operating_point'],
                'confusion_matrix': eval_4b['confusion_matrix_at_op_th'],
                'recall_by_size': eval_4b['recall_by_size'],
                'results_by_provenance': eval_4b['results_by_provenance']
            }
        }
    }
    write_json(EXP04_DIR / 'comparacion_exp04_vs_exp3c.json', comp)
    print("Saved consolidated comparison to runs/detector_exp04/comparacion_exp04_vs_exp3c.json", flush=True)

if __name__ == '__main__':
    generate_visuals()
    consolidate_comparison()
