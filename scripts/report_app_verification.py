"""Summarize the batch reproduction separately from the app-equivalent evaluation."""
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from residuos.data import Store,read_json,write_json,digest
from residuos.evaluate import metrics

out=ROOT/'runs/verificacion_app_exp03_exp05';result=read_json(out/'results.json')
store=Store(ROOT/'data/preparado/deteccion');rows=store.records('val');batch={}
for name in ('Exp03','Exp05'):
    saved=read_json(out/(name+'_batch16_predictions.json'))
    assert saved['compatibility']['checkpoint_sha256']==result['models'][name]['checkpoint_sha256']
    assert [r['id'] for r in rows]==[r['id'] for r in saved['images']]
    m=metrics([(p['predictions'],r['boxes']) for p,r in zip(saved['images'],rows)])
    m.update(precision=m['tp']/(m['tp']+m['fp']),recall=m['tp']/(m['tp']+m['fn']))
    batch[name]=m
assert (batch['Exp03']['tp'],batch['Exp03']['fp'],batch['Exp03']['fn'])==(331,1133,1233)
reference=read_json(ROOT/'runs/detector_exp05/evaluation.json')['threshold_sweep']['0.35']
for c,n in batch['Exp05']['classes'].items():
    assert all(n[k]==reference['classes'][c][k] for k in ('tp','fp','fn'))
assert all(batch['Exp05'][k]==reference[k] for k in ('tp','fp','fn'))
write_json(out/'batch16_report_reproduction.json',dict(models=batch,Exp05_report_all_class_counts_match=True,Exp03_report_040_aggregate_matches=True,
    note='Original reports use batches of 16. App uses one frame. Original report artifacts and checkpoints were not modified.'))
for path in ('app/main.py','app/detector.py','residuos/app.py','residuos/detector.py','residuos/detector_s8.py','residuos/evaluate.py'):
    target=out/'source_snapshot'/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/path).read_bytes())
fmt=lambda v:'N/D' if v is None else f'{v*100:.2f}%'
lines=['# Verificación de la aplicación: Exp03C y Exp05','',
'La aplicación cargaba por defecto `runs/detector/best.pt`, una copia de Exp01 (época 14, SHA256 fac0896e9c77c6d4a61674505656c6800d4d44502c8e8588443fbac4ca52dd02), con umbral 0,65. Su threshold.json sí correspondía a ese checkpoint; el problema era la elección del modelo y la ausencia de soporte DetectorS8 en el cargador de producción.',
'',
'Ahora `app/main.py` elige Exp03C, época 19, a 0,40 por defecto: tiene menos falsos positivos que Exp05 a 0,35, aunque su recall y micro-F1 son menores. No se declara superior en todas las métricas. La app muestra ruta absoluta, SHA256, época, umbral y dispositivo en una etiqueta permanente y en stdout. El selector de comparación usa validación, no test.',
'',
'Se añadieron asociaciones threshold.json específicas junto a los checkpoints Exp03C y Exp05. Ambos validan el SHA256 antes de inferir. Se eliminó el fallback silencioso a 0,35 de app/detector.py. El antiguo par Exp01/threshold.json permanece intacto.',
'',
'## Protocolo común y equivalencia con la app',
'670 imágenes y 1.564 objetos de la validación original, idénticos IDs, orden, grupos, dimensiones, cajas y huellas. IoU >= 0,5, misma clase, coincidencia uno a uno por confianza descendente, NMS por clase 0,45, máximo 100 detecciones por imagen. Se comprobó la igualdad exacta de píxeles entre originales redimensionados y caché. No se encontraron predicciones previas compatibles: se ejecutó inferencia y se guardaron resultados con huellas de compatibilidad para reutilizarlos.',
'',
'La tabla siguiente usa batch=1, igual que la app. Su trabajador real se ejecutó sobre cada imagen original: 670/670 coincidencias por modelo, con error máximo 0 en coordenadas y puntuaciones frente al evaluador. Las ventanas Tk se iniciaron ocultas para comprobar la identidad mostrada; se verificaron ambos modelos.',
'',
'| Modelo | Clase | Objetos | TP | FP | FN | Precisión | Recall |',
'|---|---|---:|---:|---:|---:|---:|---:|']
for name,res in result['models'].items():
    for c,n in res['classes'].items():lines.append(f"| {name} ({res['threshold']:.2f}) | {c} | {n['objects']} | {n['tp']} | {n['fp']} | {n['fn']} | {fmt(n['precision'])} | {fmt(n['recall'])} |")
    lines.append(f"| **{name}** | **Total** | **1564** | **{res['tp']}** | **{res['fp']}** | **{res['fn']}** | **{fmt(res['precision'])}** | **{fmt(res['recall'])}** |")
lines+=['','N/D significa precisión sin denominador porque no hubo predicciones de esa clase. Vidrio, cartón y papel tienen solo 9, 12 y 13 objetos: no hay aciertos de esas clases a estos puntos operativos.',
'','## Coincidencia con los informes históricos',
'Los informes se reproducen con batch=16. No se forzaron números ni se modificaron los informes originales. Al usar batch=1 para igualar la aplicación aparecen pequeñas diferencias de redondeo de GPU que afectan decisiones cercanas al umbral/NMS.',
'','| Modelo | Batch16: TP / FP / FN | App batch1: TP / FP / FN | Micro-F1 batch16 / app |',
'|---|---:|---:|---:|']
for name,r in result['models'].items():
    b=batch[name];lines.append(f"| {name} | {b['tp']} / {b['fp']} / {b['fn']} | {r['tp']} / {r['fp']} / {r['fn']} | {fmt(b['micro_f1'])} / {fmt(r['micro_f1'])} |")
lines+=['',
'Exp03 a 0,40 reproduce 1.133 FP, 22,61% precisión y 21,86% micro-F1 citados en Exp05/RESULTADOS.md. Su propio evaluation.json histórico no incluía 0,40 en la cuadrícula (incluía 0,35 y 0,50). Exp05 a 0,35 coincide en TP/FP/FN por cada clase con su evaluation.json. La diferencia batch16/app afecta el número de detecciones en 2 imágenes de Exp03 y 4 de Exp05. Los detalles están en batch16_vs_app_numerical_differences.json.',
'',
'Micro-F1 en el modo de la app: Exp03 21,85%; Exp05 22,09%. Estos resultados son de validación; no se ejecutó test ni entrenamiento.',
'','## Ejecución',
'Desde la raíz del proyecto:',
'```powershell',
'.\\.venv\\Scripts\\python.exe app\\main.py --detector runs/detector_exp03/corrected/best.pt --device auto',
'```','Pulse **Abrir video** para seleccionar su archivo. También se admite `--video "ruta_al_video.mp4"`.',
'','Archivos: `metrics.csv`, `results.json`, `batch16_report_reproduction.json`, `app_parity.json`, `gui_startup_checks.json`, `validation_identity.json`.']
(out/'RESULTADOS.md').write_text('\n'.join(lines),encoding='utf-8')
print(out/'RESULTADOS.md')
