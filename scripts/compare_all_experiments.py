import json

with open('runs/detector_exp05/evaluation.json', 'r', encoding='utf-8') as f:
    e05 = json.load(f)

with open('runs/detector_exp06/evaluation.json', 'r', encoding='utf-8') as f:
    e06 = json.load(f)

with open('runs/detector_exp07/evaluation.json', 'r', encoding='utf-8') as f:
    e07 = json.load(f)

print("=" * 70)
print("COMPARATIVA GLOBAL DE TODOS LOS EXPERIMENTOS EN VALIDACIÓN (670 IMGS)")
print("=" * 70)

print(f"\n1. Pérdida Mínima de Validación:")
print(f"   - Exp05: {e05.get('val_loss', 'N/D'):.4f} (época {e05.get('best_epoch')})")
print(f"   - Exp06: {e06.get('val_loss', 'N/D'):.4f} (época {e06.get('best_epoch')})")
print(f"   - Exp07: {e07.get('val_loss', 'N/D'):.4f} (época {e07.get('best_epoch')}) -> ¡RÉCORD ABSOLUTO!")

print(f"\n2. mAP50 Global (No Umbralizado):")
print(f"   - Exp05: {e05['true_unthresholded_map50']*100:.2f}%")
print(f"   - Exp06: {e06['true_unthresholded_map50']*100:.2f}%")
print(f"   - Exp07: {e07['true_unthresholded_map50']*100:.2f}%")

print(f"\n3. AP50 por Clase:")
ap05 = e05['true_unthresholded_ap50']
ap06 = e06['true_unthresholded_ap50']
ap07 = e07['true_unthresholded_ap50']
for c in ['plastico', 'organico', 'metal', 'papel', 'carton', 'vidrio']:
    v05 = ap05.get(c, 0.0) * 100
    v06 = ap06.get(c, 0.0) * 100
    v07 = ap07.get(c, 0.0) * 100
    print(f"   - {c:10s}: Exp05 = {v05:5.2f}% | Exp06 = {v06:5.2f}% | Exp07 = {v07:5.2f}%")

print(f"\n4. Métricas a Umbral Operativo 0.35 (Modo Cobertura):")
th05 = e05['threshold_sweep'].get('0.35', {})
th06 = e06['threshold_sweep'].get('0.35', {})
th07 = e07['threshold_sweep'].get('0.35', {})
print(f"   - Exp05: TP={th05.get('tp')}, FP={th05.get('fp')}, Prec={th05.get('precision', 0)*100:.2f}%, Rec={th05.get('recall', 0)*100:.2f}%, F1={th05.get('micro_f1', 0)*100:.2f}%")
print(f"   - Exp06: TP={th06.get('tp')}, FP={th06.get('fp')}, Prec={th06.get('precision', 0)*100:.2f}%, Rec={th06.get('recall', 0)*100:.2f}%, F1={th06.get('micro_f1', 0)*100:.2f}%")
print(f"   - Exp07: TP={th07.get('tp')}, FP={th07.get('fp')}, Prec={th07.get('precision', 0)*100:.2f}%, Rec={th07.get('recall', 0)*100:.2f}%, F1={th07.get('micro_f1', 0)*100:.2f}%")

print(f"\n5. Desglose de Clases Minoritarias a th=0.35:")
for c in ['carton', 'papel', 'vidrio', 'metal']:
    c05 = th05['classes'][c]
    c06 = th06['classes'][c]
    c07 = th07['classes'][c]
    print(f"   - {c:10s}:")
    print(f"       Exp05: TP={c05['tp']}, FP={c05['fp']}, FN={c05['fn']}, Prec={c05['precision']*100:.2f}%, Rec={c05['recall']*100:.2f}%, F1={c05['f1']*100:.2f}%")
    print(f"       Exp06: TP={c06['tp']}, FP={c06['fp']}, FN={c06['fn']}, Prec={c06['precision']*100:.2f}%, Rec={c06['recall']*100:.2f}%, F1={c06['f1']*100:.2f}%")
    print(f"       Exp07: TP={c07['tp']}, FP={c07['fp']}, FN={c07['fn']}, Prec={c07['precision']*100:.2f}%, Rec={c07['recall']*100:.2f}%, F1={c07['f1']*100:.2f}%")

print(f"\n6. Métricas a Umbral Operativo 0.40:")
th05_40 = e05['threshold_sweep'].get('0.40', {})
th06_40 = e06['threshold_sweep'].get('0.40', {})
th07_40 = e07['threshold_sweep'].get('0.40', {})
print(f"   - Exp05: TP={th05_40.get('tp')}, FP={th05_40.get('fp')}, Prec={th05_40.get('precision', 0)*100:.2f}%, Rec={th05_40.get('recall', 0)*100:.2f}%, F1={th05_40.get('micro_f1', 0)*100:.2f}%")
print(f"   - Exp06: TP={th06_40.get('tp')}, FP={th06_40.get('fp')}, Prec={th06_40.get('precision', 0)*100:.2f}%, Rec={th06_40.get('recall', 0)*100:.2f}%, F1={th06_40.get('micro_f1', 0)*100:.2f}%")
print(f"   - Exp07: TP={th07_40.get('tp')}, FP={th07_40.get('fp')}, Prec={th07_40.get('precision', 0)*100:.2f}%, Rec={th07_40.get('recall', 0)*100:.2f}%, F1={th07_40.get('micro_f1', 0)*100:.2f}%")
