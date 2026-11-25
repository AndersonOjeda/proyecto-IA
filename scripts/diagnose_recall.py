import json
from pathlib import Path

def inspect_eval(path_str):
    p = Path(path_str)
    if not p.exists():
        print(f"File not found: {path_str}")
        return
    data = json.loads(p.read_text(encoding='utf-8'))
    print(f"\n==========================================")
    print(f"Archivo: {path_str}")
    print(f"mAP50 no umbralizado: {data.get('true_unthresholded_map50', 0.0)*100:.2f}%")
    print(f"AP50 por clase:")
    for c, ap in data.get('true_unthresholded_ap50', {}).items():
        print(f"  {c:10s}: {ap*100:6.2f}%")
    
    print("\nBarrido de umbrales:")
    print(f"{'Th':>6} | {'Precision':>10} | {'Recall':>10} | {'Micro-F1':>10} | {'TP':>6} | {'FP':>6} | {'FN':>6}")
    print("-" * 65)
    sweep = data.get('threshold_sweep', {})
    if isinstance(sweep, dict):
        items = sorted(sweep.items(), key=lambda kv: float(kv[0]))
    else:
        items = [(r.get('threshold', 0.0), r) for r in sweep]
    for th_key, row in items:
        th = float(th_key) if isinstance(th_key, (int, float, str)) else 0.0
        prec = row.get('precision', 0.0) * 100
        rec = row.get('recall', 0.0) * 100
        f1 = row.get('micro_f1', 0.0) * 100
        tp = row.get('tp', 0)
        fp = row.get('fp', 0)
        fn = row.get('fn', 0)
        print(f"{th:6.2f} | {prec:9.2f}% | {rec:9.2f}% | {f1:9.2f}% | {tp:6d} | {fp:6d} | {fn:6d}")

if __name__ == '__main__':
    inspect_eval('runs/detector_exp03/corrected/evaluation.json')
    inspect_eval('runs/detector_exp05/evaluation.json')
