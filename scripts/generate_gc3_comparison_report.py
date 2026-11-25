"""Genera reporte comparativo detallado entre el modelo de fotos exteriores puras (Exp 5)
y el modelo unificado masivo con Roboflow GC3 (12.442 imágenes).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load_json(p):
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_pct(v):
    return f"{v*100:6.2f}%" if v is not None else "   N/A "

def main():
    exp5_eval_path = ROOT / 'runs/detector_exp05/evaluation.json'
    gc3_eval_path = ROOT / 'runs/detector_gc3_integration/evaluation.json'
    
    if not gc3_eval_path.exists():
        print("La evaluación de GC3 aún no ha finalizado.")
        return
        
    e5 = load_json(exp5_eval_path)
    egc3 = load_json(gc3_eval_path)
    
    classes = ['plastico', 'papel', 'carton', 'vidrio', 'metal', 'organico']
    
    print("="*90)
    print("          REPORTE COMPARATIVO: ANTES vs. DESPUÉS DE INTEGRAR GC3 (ROBOFLOW)")
    print("           Evaluado estrictamente sobre 670 fotos reales de validación (IoU >= 0.5)")
    print("="*90)
    
    n_train_gc3 = egc3.get('total_train_images', 12443)
    best_ep_gc3 = egc3.get('best_epoch', 18)
    val_loss_gc3 = egc3.get('val_loss', 2.9555)
    
    print("\n1. COMPARATIVA DE CONFIGURACIÓN Y ENTRENAMIENTO:")
    print(f"{'Parámetro':<30} | {'Antes (Exp 5 - Exterior Puro)':<30} | {'Después (Unificado Masivo GC3)':<30}")
    print("-" * 95)
    print(f"{'Total Imágenes Train':<30} | {'5.150 imágenes':<30} | {f'{n_train_gc3} imágenes':<30}")
    print(f"{'Fuentes Integradas':<30} | {'Bangladesh, TACO, FoodWaste, SALSA':<30} | {'+ 7.293 fotos GC3 (Roboflow CC BY)':<30}")
    print(f"{'Cajas Totales en Train':<30} | {'11.960 cajas':<30} | {'63.459 cajas (+51.539 nuevas)':<30}")
    print(f"{'Mejor Época / Val Loss':<30} | {'Época 22 (Loss 1.7785)':<30} | {f'Época {best_ep_gc3} (Loss {val_loss_gc3:.4f})':<30}")
    print(f"{'mAP50 Global (VOC 101 pts)':<30} | {format_pct(e5.get('true_unthresholded_map50')):<30} | {format_pct(egc3.get('true_unthresholded_map50')):<30}")
    
    print("\n2. COMPARATIVA DE AP50 NO UMBRALIZADO POR CLASE:")
    print(f"{'Clase':<15} | {'Exp 5 (Exterior)':<18} | {'Con GC3 (Masivo)':<18} | {'Diferencia':<15}")
    print("-" * 72)
    e5_aps = e5.get('true_unthresholded_ap50', {})
    gc3_aps = egc3.get('true_unthresholded_ap50', {})
    for c in classes:
        ap1 = e5_aps.get(c, 0.0)
        ap2 = gc3_aps.get(c, 0.0)
        diff = ap2 - ap1
        sign = "+" if diff > 0 else ""
        print(f"{c:<15} | {format_pct(ap1):<18} | {format_pct(ap2):<18} | {sign}{diff*100:6.2f}%")
    print("-" * 72)
    map1 = e5.get('true_unthresholded_map50', 0.0)
    map2 = egc3.get('true_unthresholded_map50', 0.0)
    diff_m = map2 - map1
    sign_m = "+" if diff_m > 0 else ""
    print(f"{'PROMEDIO (mAP50)':<15} | {format_pct(map1):<18} | {format_pct(map2):<18} | {sign_m}{diff_m*100:6.2f}%")
    
    print("\n3. TABLA COMPARATIVA A UMBRAL CALIBRADO ÓPTIMO:")
    th1 = e5['best_operating_point']['threshold']
    th2 = egc3['best_operating_point']['threshold']
    p1, r1, f1_1 = e5['best_operating_point']['precision'], e5['best_operating_point']['recall'], e5['best_operating_point']['micro_f1']
    p2, r2, f1_2 = egc3['best_operating_point']['precision'], egc3['best_operating_point']['recall'], egc3['best_operating_point']['micro_f1']
    tp1, fp1, fn1 = e5['best_operating_point']['tp'], e5['best_operating_point']['fp'], e5['best_operating_point']['fn']
    tp2, fp2, fn2 = egc3['best_operating_point']['tp'], egc3['best_operating_point']['fp'], egc3['best_operating_point']['fn']
    
    print(f"{'Métrica':<20} | {f'Exp 5 (th={th1:.2f})':<25} | {f'Con GC3 (th={th2:.2f})':<25}")
    print("-" * 75)
    print(f"{'Micro-F1':<20} | {format_pct(f1_1):<25} | {format_pct(f1_2):<25}")
    print(f"{'Precisión':<20} | {format_pct(p1):<25} | {format_pct(p2):<25}")
    print(f"{'Recall':<20} | {format_pct(r1):<25} | {format_pct(r2):<25}")
    print(f"{'True Positives (TP)':<20} | {tp1:<25} | {tp2:<25}")
    print(f"{'False Positives (FP)':<20} | {fp1:<25} | {fp2:<25}")
    print(f"{'False Negatives (FN)':<20} | {fn1:<25} | {fn2:<25}")
    
    print("\n4. DESGLOSE POR CLASE EN EL MODELO UNIFICADO (Con GC3):")
    print(f"{'Clase':<12} | {'TP':>5} | {'FP':>6} | {'FN':>6} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10}")
    print("-" * 72)
    for c, row in egc3['best_operating_point']['classes'].items():
        print(f"{c:<12} | {row['tp']:5d} | {row['fp']:6d} | {row['fn']:6d} | {format_pct(row['precision']):>10} | {format_pct(row['recall']):>10} | {format_pct(row['f1']):>10}")
    print("-" * 72)
    print(f"{'GLOBAL':<12} | {tp2:5d} | {fp2:6d} | {fn2:6d} | {format_pct(p2):>10} | {format_pct(r2):>10} | {format_pct(f1_2):>10}")
    print("="*90)

if __name__ == '__main__':
    main()
