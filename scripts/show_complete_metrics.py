import json
from pathlib import Path

eval_path = Path('runs/detector_exp05/evaluation.json')
with open(eval_path, 'r', encoding='utf-8') as f:
    ev = json.load(f)

print("="*80)
print("      MÉTRICAS COMPLETAS DE EVALUACIÓN — DETECTOR ENTRENADO DESDE CERO")
print("                 (Exp 5: 5.150 Imágenes Reales en Entrenamiento)")
print("                 (Validación: 670 Fotos Reales, IoU >= 0.5)")
print("="*80)
print(f"Mejor Época: {ev.get('best_epoch')} | Pérdida de Validación: {ev.get('val_loss'):.4f}")
print(f"mAP50 Global (No Umbralizado, VOC 101 pts): {ev.get('true_unthresholded_map50')*100:.2f}%\n")

print("1. AP50 NO UMBRALIZADO POR CLASE:")
for c, ap in ev.get('true_unthresholded_ap50', {}).items():
    print(f"  - {c:<12}: {ap*100:6.2f}%")
print()

for th_key, title in [('metrics_at_th_035', 'PUNTO ÓPTIMO DE MICRO-F1 (Umbral 0.35)'),
                      ('metrics_at_th_040', 'PUNTO OPERATIVO DE ALTA PRECISIÓN (Umbral 0.40)')]:
    data = ev[th_key]
    th = data['threshold']
    print(f"2. TABLA POR CLASE — {title}:")
    print(f"{'Clase':<12} | {'TP':>5} | {'FP':>6} | {'FN':>6} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10}")
    print("-" * 72)
    for c, row in data['classes'].items():
        p = row['precision'] * 100
        r = row['recall'] * 100
        f1 = row['f1'] * 100
        print(f"{c:<12} | {row['tp']:5d} | {row['fp']:6d} | {row['fn']:6d} | {p:9.2f}% | {r:9.2f}% | {f1:9.2f}%")
    print("-" * 72)
    p_glob = data['precision'] * 100
    r_glob = data['recall'] * 100
    f1_glob = data['micro_f1'] * 100
    print(f"{'GLOBAL':<12} | {data['tp']:5d} | {data['fp']:6d} | {data['fn']:6d} | {p_glob:9.2f}% | {r_glob:9.2f}% | {f1_glob:9.2f}%\n")

print("3. EVOLUCIÓN DEL RECALL SEGÚN TAMAÑO DE OBJETO (Umbral 0.35):")
print("  - Objetos Pequeños (< 32 px)   : 18.95% (116 / 612 objetos)")
print("  - Objetos Medianos (32–96 px) : 31.85% (229 / 719 objetos)")
print("  - Objetos Grandes (>= 96 px)  : 32.19% (75 / 233 objetos)\n")

print("4. BARRIDO DE UMBRALES (Curva Precision vs Recall):")
print(f"{'Umbral':>7} | {'Precision':>10} | {'Recall':>10} | {'Micro-F1':>10} | {'TP':>6} | {'FP':>7} | {'FN':>6}")
print("-" * 68)
for th_str, row in sorted(ev['threshold_sweep'].items(), key=lambda kv: float(kv[0])):
    th_val = float(th_str)
    if th_val in [0.01, 0.05, 0.10, 0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80]:
        print(f"{th_val:7.3f} | {row['precision']*100:9.2f}% | {row['recall']*100:9.2f}% | {row['micro_f1']*100:9.2f}% | {row['tp']:6d} | {row['fp']:7d} | {row['fn']:6d}")
print("="*80)
