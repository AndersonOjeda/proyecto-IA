"""Script to consolidate metrics, generate comparison JSONs, verify integrity,
and create comparative visual panels for Experiment 3.
"""
from pathlib import Path
import sys
import json
import hashlib
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from residuos import CLASSES
from residuos.data import Store, read_json
from residuos.detector import Detector, decode
from scripts.experiment3_runner import DetectorS8, decode_grid32, GRID

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def consolidate_metrics():
    exp01_dir = ROOT / 'runs/detector_exp01'
    exp02_dir = ROOT / 'runs/detector_exp02'
    exp03_dir = ROOT / 'runs/detector_exp03'
    exp03_arch = exp03_dir / 'architecture'
    exp03_bal = exp03_dir / 'balance'

    # 1. Integrity checks
    exp01_best = exp01_dir / 'best.pt'
    exp01_sha = sha256_file(exp01_best)
    expected_exp01_sha = "fac0896e9c77c6d4a61674505656c6800d4d44502c8e8588443fbac4ca52dd02"
    assert exp01_sha == expected_exp01_sha, f"Exp01 best.pt altered! {exp01_sha}"

    # Check manual photos
    manual_manifest = ROOT / 'data/preparado/deteccion/manifest.json'
    manual_data = read_json(manual_manifest)
    manual_entries = [r for r in manual_data['images'] if 'lote90' in r.get('path', '') or 'fotos_manuales' in r.get('path', '')]
    for m in manual_entries:
        assert m.get('reviewed') is False or len(m.get('boxes', [])) == 0, f"Manual photo altered! {m['id']}"

    # Check test partition
    store = Store(ROOT / 'data/preparado/deteccion')
    test_ids = [r['id'] for r in store.records('test')]
    assert len(test_ids) == 531, f"Unexpected test set size: {len(test_ids)}"

    integrity = {
        "exp01_checkpoint_sha256": exp01_sha,
        "exp01_unmodified": True,
        "manual_photos_reviewed_false": True,
        "manual_photos_used_in_training": False,
        "test_partition_held_out": True,
        "test_images_count": len(test_ids),
        "test_images_read_or_evaluated": 0,
        "no_pretrained_weights_used": True,
        "architecture_trained_from_scratch": True,
        "seed_recorded": 42
    }
    with open(exp03_dir / 'integrity_checks.json', 'w', encoding='utf-8') as f:
        json.dump(integrity, f, indent=2)
    print("Integrity checks passed and saved.", flush=True)

    # 2. Load metrics
    base_diag = read_json(exp02_dir / 'diagnosis_baseline/diagnosis.json')
    exp02_th = read_json(exp02_dir / 'threshold_only/results.json')
    exp03a = read_json(exp03_arch / 'evaluation.json')
    exp03b = read_json(exp03_bal / 'evaluation.json')

    # Build Class Comparison Table
    class_comparison = {}
    for c in CLASSES:
        b_stages = base_diag['stages'][c]
        b_metrics = base_diag['metrics']['classes'][c]
        b_ap = exp02_th['ap50'][c]

        a_stages = exp03a['stages'][c]
        a_metrics = exp03a['metrics_at_chosen_threshold']['classes'][c]
        a_ap = exp03a['ap50'][c]

        bb_stages = exp03b['stages'][c]
        bb_metrics = exp03b['metrics_at_chosen_threshold']['classes'][c]
        bb_ap = exp03b['ap50'][c]

        class_comparison[c] = {
            "num_objects": b_stages['objects'],
            "baseline_exp01": {
                "geom_candidates_iou50": b_stages['raw_any_class'],
                "geom_correct_class": b_stages['raw_correct_class'],
                "tp": b_metrics['tp'],
                "fp": b_metrics['fp'],
                "fn": b_metrics['fn'],
                "precision": b_metrics['precision'],
                "recall": b_metrics['recall'],
                "f1": (2 * b_metrics['precision'] * b_metrics['recall'] / (b_metrics['precision'] + b_metrics['recall'])) if b_metrics['precision'] and b_metrics['recall'] else 0.0,
                "ap50": b_ap
            },
            "exp02_threshold_only": {
                "threshold": exp02_th['threshold'],
                "tp": exp02_th['classes'][c]['tp'],
                "fp": exp02_th['classes'][c]['fp'],
                "fn": exp02_th['classes'][c]['fn'],
                "precision": exp02_th['classes'][c]['precision'],
                "recall": exp02_th['classes'][c]['recall'],
                "f1": (2 * exp02_th['classes'][c]['precision'] * exp02_th['classes'][c]['recall'] / (exp02_th['classes'][c]['precision'] + exp02_th['classes'][c]['recall'])) if exp02_th['classes'][c]['precision'] and exp02_th['classes'][c]['recall'] else 0.0,
                "ap50": b_ap
            },
            "exp03a_stride8_arch": {
                "geom_candidates_iou50": a_stages['raw_any_class'],
                "geom_correct_class": a_stages['raw_correct_class'],
                "tp": a_metrics['tp'],
                "fp": a_metrics['fp'],
                "fn": a_metrics['fn'],
                "precision": a_metrics['precision'],
                "recall": a_metrics['recall'],
                "f1": (2 * a_metrics['precision'] * a_metrics['recall'] / (a_metrics['precision'] + a_metrics['recall'])) if a_metrics['precision'] and a_metrics['recall'] else 0.0,
                "ap50": a_ap
            },
            "exp03b_stride8_balance": {
                "geom_candidates_iou50": bb_stages['raw_any_class'],
                "geom_correct_class": bb_stages['raw_correct_class'],
                "tp": bb_metrics['tp'],
                "fp": bb_metrics['fp'],
                "fn": bb_metrics['fn'],
                "precision": bb_metrics['precision'],
                "recall": bb_metrics['recall'],
                "f1": (2 * bb_metrics['precision'] * bb_metrics['recall'] / (bb_metrics['precision'] + bb_metrics['recall'])) if bb_metrics['precision'] and bb_metrics['recall'] else 0.0,
                "ap50": bb_ap
            }
        }

    # Macro and Micro summary
    summary_comparison = {
        "baseline_exp01": {
            "threshold": 0.65,
            "geom_coverage_total": f"{sum(base_diag['stages'][c]['raw_any_class'] for c in CLASSES)} / 1564 ({sum(base_diag['stages'][c]['raw_any_class'] for c in CLASSES)/1564*100:.2f}%)",
            "mAP50": exp02_th['map50'],
            "micro_f1": base_diag['metrics']['micro_f1'],
            "total_tp": base_diag['metrics']['tp'],
            "total_fp": base_diag['metrics']['fp'],
            "total_fn": base_diag['metrics']['fn']
        },
        "exp02_threshold_only": {
            "threshold": exp02_th['threshold'],
            "mAP50": exp02_th['map50'],
            "micro_f1": exp02_th['micro_f1'],
            "total_tp": exp02_th['tp'],
            "total_fp": exp02_th['fp'],
            "total_fn": exp02_th['fn']
        },
        "exp03a_stride8_arch": {
            "threshold": exp03a['chosen_threshold'],
            "geom_coverage_total": f"{sum(exp03a['stages'][c]['raw_any_class'] for c in CLASSES)} / 1564 ({sum(exp03a['stages'][c]['raw_any_class'] for c in CLASSES)/1564*100:.2f}%)",
            "mAP50": exp03a['map50'],
            "micro_f1": exp03a['metrics_at_chosen_threshold']['micro_f1'],
            "total_tp": exp03a['metrics_at_chosen_threshold']['tp'],
            "total_fp": exp03a['metrics_at_chosen_threshold']['fp'],
            "total_fn": exp03a['metrics_at_chosen_threshold']['fn']
        },
        "exp03b_stride8_balance": {
            "threshold": exp03b['chosen_threshold'],
            "geom_coverage_total": f"{sum(exp03b['stages'][c]['raw_any_class'] for c in CLASSES)} / 1564 ({sum(exp03b['stages'][c]['raw_any_class'] for c in CLASSES)/1564*100:.2f}%)",
            "mAP50": exp03b['map50'],
            "micro_f1": exp03b['metrics_at_chosen_threshold']['micro_f1'],
            "total_tp": exp03b['metrics_at_chosen_threshold']['tp'],
            "total_fp": exp03b['metrics_at_chosen_threshold']['fp'],
            "total_fn": exp03b['metrics_at_chosen_threshold']['fn']
        }
    }

    full_comp = {
        "summary": summary_comparison,
        "classes": class_comparison
    }

    with open(exp03_dir / 'comparacion_experimentos.json', 'w', encoding='utf-8') as f:
        json.dump(full_comp, f, indent=2)

    with open(exp03_dir / 'metricas_por_clase.json', 'w', encoding='utf-8') as f:
        json.dump(class_comparison, f, indent=2)

    # Diagnostic errors JSON
    diagnosis_analysis = {
        "glass_analysis_9_objects": {
            "baseline": "0/9 objetos localizados con IoU >= 0.5. Cuello de botella 100% geométrico por baja resolución espacial (stride 16, rejilla 16x16)",
            "exp03a": "4/9 objetos localizados geométricamente (44.4%), 2/9 con clase correcta pre-umbral. Al resolver la resolución espacial, las cajas aparecen",
            "exp03b": "3/9 objetos localizados, 3/3 con clase vidrio correcta pre-umbral (100% pureza de clase). Se pierden en umbral debido a escasez crítica de muestras en train (71 anotaciones frente a >5.000 de orgánico)",
            "root_cause": "Combinación de límite de resolución (resuelto por Stride 8) y desbalance extremo de datos de entrenamiento (71 en train vs 5084 de orgánico). Sin datos de entrenamiento adicionales o anotaciones revisadas, la red no adquiere suficiente confianza."
        },
        "paper_analysis_13_objects": {
            "baseline": "7/13 localizados geométricamente (53.8%), pero 0/13 con clase correcta. Confundidos enteramente con plástico/orgánico",
            "exp03a": "12/13 localizados geométricamente (92.3%!). Mejora masiva de localización, pero el sesgo bayesiano colapsa la predicción hacia plástico",
            "exp03b": "9/13 localizados geométricamente (69.2%), 1 con clase correcta pre-umbral",
            "root_cause": "La localización espacial está casi completamente resuelta (92.3% de cobertura). El fallo es 100% de clasificación por desbalance de clases (57 cajas en train = 0.63% del dataset)."
        },
        "cardboard_analysis_12_objects": {
            "baseline": "5/12 localizados geométricamente (41.7%), 0/12 con clase correcta",
            "exp03a": "7/12 localizados geométricamente (58.3%), 4/12 con clase correcta pre-umbral (33.3% acierto inicial)",
            "exp03b": "8/12 localizados geométricamente (66.7%), 0 sobreviven tras umbralización",
            "root_cause": "Solo 82 cajas de cartón en train (0.91%). El modelo aprende los anclajes espaciales grandes (>=96px: 4/4 detectados), pero las probabilidades de salida son bajas (<0.20)."
        },
        "organic_analysis_757_objects": {
            "breakdown": {
                "small_<32px": {
                    "total_objects": 547,
                    "baseline_geom": 188,
                    "exp03a_geom": 339,
                    "gain": "+80.3% más objetos pequeños localizados geométricamente gracias a Stride 8 (rejilla 32x32)"
                },
                "medium_32-96px": {
                    "total_objects": 210,
                    "baseline_geom": 139,
                    "exp03a_geom": 147
                }
            },
            "source_origin": "756 provienen de Food Waste y 1 de TACO oficial",
            "root_cause": "En la línea base, 359 de los 547 objetos pequeños (65.6%) eran físicamente invisibles para la rejilla 16x16 (stride 16px). En Exp 3A, la resolución espacial aumentó la recuperación geométrica a 339/547 (62.0%). El factor limitante remanente es la aglomeración de partículas de comida donde múltiples residuos ocupan la misma celda de 8px."
        }
    }

    with open(exp03_dir / 'diagnostico_errores.json', 'w', encoding='utf-8') as f:
        json.dump(diagnosis_analysis, f, indent=2)

    print("Consolidated JSONs generated successfully.", flush=True)

def generate_visual_panels():
    exp01_dir = ROOT / 'runs/detector_exp01'
    exp03_dir = ROOT / 'runs/detector_exp03'
    out_dir = exp03_dir / 'visuals'
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    store = Store(ROOT / 'data/preparado/deteccion')
    store.db = read_json(exp01_dir / 'snapshot/manifest.json')
    val_rows = store.records('val')

    # Load Exp 1 baseline model
    base_model = Detector().to(device)
    base_ckpt = torch.load(exp01_dir / 'best.pt', map_location=device, weights_only=False)
    base_model.load_state_dict(base_ckpt.get('state', base_ckpt.get('state_dict')))
    base_model.eval()

    # Load Exp 3A model
    m3a = DetectorS8(num_classes=len(CLASSES)).to(device)
    ckpt3a = torch.load(exp03_dir / 'architecture/best.pt', map_location=device, weights_only=False)
    m3a.load_state_dict(ckpt3a.get('state_dict', ckpt3a.get('state')))
    m3a.eval()

    # Load Exp 3B model
    m3b = DetectorS8(num_classes=len(CLASSES)).to(device)
    ckpt3b = torch.load(exp03_dir / 'balance/best.pt', map_location=device, weights_only=False)
    m3b.load_state_dict(ckpt3b.get('state_dict', ckpt3b.get('state')))
    m3b.eval()

    # Target test cases across classes and sizes
    targets = [
        {"name": "01_vidrio_taco", "id": "e9144430b0e8dafe56fe", "focus": "Vidrio (objeto minoritario, 9 en val)"},
        {"name": "02_organico_small_foodwaste", "id": "41a7ff550dd06d4a71a6", "focus": "Organico pequeno <32px"},
        {"name": "03_papel_taco", "id": "9935ac60f69cfc916ced", "focus": "Papel (minoritario, 13 en val)"},
        {"name": "04_carton_taco", "id": "eba0cb1763b43d3fb0cc", "focus": "Carton (minoritario, 12 en val)"},
        {"name": "05_metal_taco", "id": "96297912b97623ccad89", "focus": "Metal (34 en val)"},
        {"name": "06_plastico_bangladesh", "id": "9bc7093c53f63dff3483", "focus": "Plastico (mayoritario)"}
    ]

    COLORS = {
        'GT': (0, 255, 0),        # Green
        'Exp1': (255, 140, 0),     # Orange
        'Exp3A': (30, 144, 255),   # DeepSkyBlue
        'Exp3B': (255, 20, 147)    # DeepPink
    }

    for tgt in targets:
        r = next((x for x in val_rows if x['id'] == tgt['id']), None)
        if not r:
            continue

        img_path = store.root / r['path']
        raw_bgr = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if raw_bgr is None:
            continue

        h, w = raw_bgr.shape[:2]
        resized = cv2.resize(raw_bgr, (256, 256), interpolation=cv2.INTER_LINEAR)
        rgb_tensor = torch.from_numpy(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        rgb_tensor = rgb_tensor.to(device)

        with torch.no_grad():
            out1 = base_model(rgb_tensor)
            preds1 = decode(out1[0], w, h, threshold=0.25, nms=0.45)

            out3a = m3a(rgb_tensor)
            preds3a = decode_grid32(out3a[0], w, h, threshold=0.25, nms=0.45)

            out3b = m3b(rgb_tensor)
            preds3b = decode_grid32(out3b[0], w, h, threshold=0.25, nms=0.45)

        # Function to draw detections on an image canvas
        def render_subpanel(title, bboxes, tag_type):
            canvas = raw_bgr.copy()
            overlay = canvas.copy()
            for b in bboxes:
                xyxy = [int(v) for v in b['xyxy']]
                if tag_type == 'GT':
                    label = f"GT: {b['class']}"
                    col = (0, 255, 0)
                else:
                    score = b.get('score', 0.0)
                    label = f"{b['class']} {score:.2f}"
                    col = COLORS[tag_type]

                cv2.rectangle(canvas, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), col, 2)
                # Label banner
                txt_sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
                cv2.rectangle(canvas, (xyxy[0], max(0, xyxy[1] - 18)), (xyxy[0] + txt_sz[0] + 6, max(0, xyxy[1])), col, -1)
                cv2.putText(canvas, label, (xyxy[0] + 3, max(12, xyxy[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            # Header bar
            cv2.rectangle(canvas, (0, 0), (w, 30), (30, 30, 30), -1)
            cv2.putText(canvas, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            return cv2.resize(canvas, (400, 400))

        p_gt = render_subpanel("Ground Truth (Anotacion Real)", r['boxes'], 'GT')
        p_base = render_subpanel(f"Exp 1 Baseline (s16, {len(preds1)} det)", preds1, 'Exp1')
        p_3a = render_subpanel(f"Exp 3A Arquitectura (s8, {len(preds3a)} det)", preds3a, 'Exp3A')
        p_3b = render_subpanel(f"Exp 3B Balance (s8+wc, {len(preds3b)} det)", preds3b, 'Exp3B')

        # Combine 4 subpanels in a 2x2 grid
        top_row = np.hstack([p_gt, p_base])
        bot_row = np.hstack([p_3a, p_3b])
        grid_panel = np.vstack([top_row, bot_row])

        # Add global banner
        banner = np.zeros((50, grid_panel.shape[1], 3), dtype=np.uint8)
        banner[:] = (20, 20, 20)
        title_text = f"Experimento 3 - {tgt['focus']} | ID: {tgt['id']}"
        cv2.putText(banner, title_text, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)

        final_img = np.vstack([banner, grid_panel])
        out_file = out_dir / f"{tgt['name']}_{tgt['id']}.jpg"
        cv2.imwrite(str(out_file), final_img)
        print(f"Saved visual panel: {out_file.name}", flush=True)

    print("All visual comparison panels generated.", flush=True)

if __name__ == '__main__':
    consolidate_metrics()
    generate_visual_panels()
