"""Exp 4C: comparison JSON, homogeneous-threshold charts and visual panels (val only)."""
from pathlib import Path
import sys, json
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from residuos import CLASSES
from residuos.data import Store, read_json, write_json
from scripts.experiment3_runner import DetectorS8, decode_grid32

E4 = ROOT / 'runs/detector_exp04'
OUT = E4 / 'visuals_4c'
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    'Exp3C': ('runs/detector_exp03/corrected/best.pt', 'runs/detector_exp04/eval_3c_homogeneous.json'),
    'Exp4B': ('runs/detector_exp04/best.pt', 'runs/detector_exp04/evaluation.json'),
    'Exp4C_pure': ('runs/detector_exp04/variant_4c_copypaste/best.pt', 'runs/detector_exp04/variant_4c_copypaste/evaluation.json'),
}
COL = {'Exp3C': (255, 165, 0), 'Exp4B': (0, 230, 255), 'Exp4C1_hybrid': (255, 50, 200), 'Exp4C_pure': (80, 80, 255)}


def summarize(ev):
    op = ev['best_operating_point']
    s035 = ev['metrics_at_th_035']
    return {
        'checkpoint': ev.get('checkpoint'), 'best_epoch': ev.get('best_epoch'),
        'mAP50': ev['true_unthresholded_map50'], 'ap50_per_class': ev['true_unthresholded_ap50'],
        'best_operating_point': {k: op[k] for k in ('threshold', 'micro_f1', 'precision', 'recall', 'tp', 'fp', 'fn')},
        'at_th_035': {k: s035.get(k) for k in ('micro_f1', 'precision', 'recall', 'tp', 'fp', 'fn')},
        'per_class_op': op['classes'], 'recall_by_size': ev['recall_by_size'],
        'confusion_op': ev['confusion_matrix_at_op_th'], 'provenance': ev['results_by_provenance'],
        'threshold_sweep': ev['threshold_sweep'],
    }


def main():
    evs = {k: summarize(read_json(ROOT / v[1])) for k, v in MODELS.items()}
    ref = evs['Exp3C']
    decision = {
        'beats_3c_map50': {k: v['mAP50'] > ref['mAP50'] for k, v in evs.items() if k != 'Exp3C'},
        'beats_3c_f1': {k: v['best_operating_point']['micro_f1'] > ref['best_operating_point']['micro_f1'] for k, v in evs.items() if k != 'Exp3C'},
        'global_reference': 'Exp3C' if all(v['mAP50'] <= ref['mAP50'] for v in evs.values()) else max(evs, key=lambda k: evs[k]['mAP50']),
        'test_set_evaluations': 0,
    }
    write_json(E4 / 'comparacion_exp04c.json', {'models': evs, 'decision': decision})

    # charts
    names = list(evs)
    W, H, M = 900, 420, 60
    img = np.full((H, W, 3), 255, np.uint8)
    mx = max(evs[n]['ap50_per_class'][c] for n in names for c in CLASSES) * 110
    gw = (W - 2 * M) / len(CLASSES); bw = gw / (len(names) + 1)
    for ci, c in enumerate(CLASSES):
        for i, n in enumerate(names):
            v = evs[n]['ap50_per_class'][c] * 100; x0 = int(M + ci * gw + i * bw); y0 = int(H - M - v / mx * (H - 2 * M))
            cv2.rectangle(img, (x0, y0), (int(x0 + bw - 2), H - M), COL[n], -1)
            cv2.putText(img, f'{v:.1f}', (x0, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, .32, (0, 0, 0), 1)
        cv2.putText(img, c, (int(M + ci * gw + 5), H - M + 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
    for i, n in enumerate(names):
        cv2.rectangle(img, (M + i * 270, 12), (M + i * 270 + 14, 26), COL[n], -1)
        cv2.putText(img, f"{n} mAP50={100*evs[n]['mAP50']:.2f}%", (M + i * 270 + 18, 24), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1)
    cv2.putText(img, 'AP50 (%) por clase - val 670', (M, 48), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
    cv2.imwrite(str(OUT / 'ap50_por_clase.png'), img)

    img = np.full((H, W, 3), 255, np.uint8)
    cv2.rectangle(img, (M, M), (W - M, H - M), (0, 0, 0), 1)
    for n in names:
        sw = evs[n]['threshold_sweep']; ths = sorted(sw, key=float)
        pts = [(int(M + sw[t]['recall'] / .6 * (W - 2 * M)), int(H - M - sw[t]['precision'] / .6 * (H - 2 * M))) for t in ths]
        for q in range(len(pts) - 1): cv2.line(img, pts[q], pts[q + 1], COL[n], 2)
        for q in pts: cv2.circle(img, q, 3, COL[n], -1)
    for i, n in enumerate(names):
        cv2.putText(img, n, (W - M - 150, M + 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, .5, COL[n], 2)
    cv2.putText(img, 'Precision (y 0-60%) vs Recall (x 0-60%), umbrales 0.01..0.80', (M, 40), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
    cv2.imwrite(str(OUT / 'precision_recall.png'), img)

    # visual panels at homogeneous threshold 0.35
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nets = {}
    for n, (ck, _) in MODELS.items():
        m = DetectorS8().to(dev); m.load_state_dict(torch.load(ROOT / ck, map_location=dev, weights_only=False)['state']); m.eval(); nets[n] = m
    store = Store(ROOT / 'data/preparado/deteccion')
    rows = {r['id']: r for r in store.records('val')}
    targets = ['96297912b97623ccad89', '41a7ff550dd06d4a71a6', '9bc7093c53f63dff3483', 'e9144430b0e8dafe56fe', '9935ac60f69cfc916ced', 'eba0cb1763b43d3fb0cc']
    for tid in targets:
        r = rows.get(tid)
        if r is None: continue
        img = cv2.imdecode(np.fromfile(store.root / r['path'], dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w_ = img.shape[:2]
        t = torch.from_numpy(cv2.cvtColor(cv2.resize(img, (256, 256)), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0).to(dev) / 255
        def panel(title, boxes, col, gt=False):
            c = img.copy()
            for b in boxes:
                p = [int(v) for v in b['xyxy']]
                cv2.rectangle(c, (p[0], p[1]), (p[2], p[3]), col, max(2, w_ // 300))
                lab = b['class'] if gt else f"{b['class']} {b['score']:.2f}"
                cv2.putText(c, lab, (p[0], max(14, p[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, max(.45, w_ / 1200), col, 2)
            c = cv2.resize(c, (380, 380)); cv2.rectangle(c, (0, 0), (380, 24), (20, 20, 20), -1)
            cv2.putText(c, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1); return c
        pans = [panel('GT', r['boxes'], (0, 255, 0), True)]
        with torch.no_grad():
            for n, m in nets.items():
                d = decode_grid32(m(t)[0], w_, h, threshold=0.35, nms=0.45)
                pans.append(panel(f'{n} th=0.35 ({len(d)})', d, COL[n]))
        cv2.imwrite(str(OUT / f'panel_{tid}.jpg'), np.vstack([np.hstack(pans[:2]), np.hstack(pans[2:])]))
    print(json.dumps(decision, indent=1))
    for n, v in evs.items():
        op = v['best_operating_point']
        print(n, f"mAP={100*v['mAP50']:.2f}", op, {c: round(100 * v['ap50_per_class'][c], 2) for c in CLASSES}, {k: round(s.get('recall_at_op_th', 0), 3) for k, s in v['recall_by_size'].items()})


if __name__ == '__main__':
    main()


