"""Diagnóstico y optimización de umbrales independientes por clase en Experimento 5."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def main():
    e5_eval_path = ROOT / 'runs/detector_exp05/evaluation.json'
    with open(e5_eval_path, 'r', encoding='utf-8') as f:
        e5 = json.load(f)
        
    sweep = e5['threshold_sweep']
    classes = ['plastico', 'papel', 'carton', 'vidrio', 'metal', 'organico']
    
    print("="*85)
    print("  ANÁLISIS DE UMBRALES POR CLASE EN EXPERIMENTO 5 (670 FOTOS REALES VAL)")
    print("="*85)
    
    # Para cada clase, encontrar el umbral que maximiza su propio F1-score
    best_per_class = {}
    for c in classes:
        best_f1 = -1.0
        best_row = None
        best_th = None
        for th_k, row in sorted(sweep.items(), key=lambda kv: float(kv[0])):
            c_data = row['classes'][c]
            f1 = c_data.get('f1', 0.0)
            rec = c_data.get('recall', 0.0)
            prec = c_data.get('precision', 0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_row = c_data
                best_th = float(th_k)
        best_per_class[c] = {
            'optimal_threshold': best_th,
            'f1': best_f1,
            'precision': best_row['precision'] if best_row else 0.0,
            'recall': best_row['recall'] if best_row else 0.0,
            'tp': best_row['tp'] if best_row else 0,
            'fp': best_row['fp'] if best_row else 0,
            'fn': best_row['fn'] if best_row else 0
        }
        
    print(f"{'Clase':<12} | {'Umbral Óptimo':>14} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10} | {'TP':>5} | {'FP':>6} | {'FN':>6}")
    print("-" * 85)
    for c, r in best_per_class.items():
        th = r['optimal_threshold']
        p = r['precision'] * 100 if r['precision'] else 0.0
        rec = r['recall'] * 100 if r['recall'] else 0.0
        f1 = r['f1'] * 100 if r['f1'] else 0.0
        print(f"{c:<12} | {th:14.2f} | {p:9.2f}% | {rec:9.2f}% | {f1:9.2f}% | {r['tp']:5d} | {r['fp']:6d} | {r['fn']:6d}")
    print("-" * 85)
    
    total_tp = sum(r['tp'] for r in best_per_class.values())
    total_fp = sum(r['fp'] for r in best_per_class.values())
    total_fn = sum(r['fn'] for r in best_per_class.values())
    macro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    macro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    macro_f1 = 2 * macro_p * macro_r / (macro_p + macro_r) if (macro_p + macro_r) > 0 else 0.0
    
    print(f"\nIMPACTO ESTIMADO CON UMBRALES CALIBRADOS INDEPENDIENTES POR CLASE:")
    print(f"  Micro-F1 Global : {macro_f1*100:.2f}% (frente al 22.12% con umbral único 0.35)")
    print(f"  Precisión Global: {macro_p*100:.2f}% (frente al 18.83% con umbral único 0.35)")
    print(f"  Recall Global   : {macro_r*100:.2f}%")
    print(f"  Aciertos (TP)   : {total_tp} (frente a 419)")
    print(f"  Falsos Positivos: {total_fp} (frente a 1.806 -> ¡Reducción de falsas alarmas!)")
    print("="*85)

if __name__ == '__main__':
    main()
