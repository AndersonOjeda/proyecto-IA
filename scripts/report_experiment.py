"""Write a report exclusively from the completed experiment's measured artifacts."""
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
out=root/'runs/detector_exp01'
read=lambda name:json.loads((out/name).read_text(encoding='utf-8'))
config,pre,done=read('config.json'),read('preflight.json'),read('completed.json')
met=done['metrics'];history=read('history.json');conf=read('confusions.json')
best=history[met['checkpoint_epoch']-1]
lines=[
'# Experimento 1: detector compacto desde cero',
'',
f"Dispositivo: {config['device']} / {config['gpu']}. PyTorch {config['torch_version']}, CUDA {config['cuda_version']}. Semilla {config['seed']}.",
f"Modelo: {config['parameters']:,} parámetros, entrada {config['size']} x {config['size']}, malla {config['grid']} x {config['grid']}, {config['slots']} candidatos por celda. Todos los pesos se inicializaron al azar, incluido el backbone. Los pesos de la prueba corta se descartaron.",
f"Entrenamiento: {config['epochs']} épocas, batch {config['batch']}, AdamW lr={config['lr']}, weight_decay={config['weight_decay']}. Se usaron {config['counts']['train']['images']} imágenes / {config['counts']['train']['objects']} cajas de train y {met['images']} imágenes / {sum(x['objects'] for x in met['classes'].values())} cajas de validación.",
f"Mejor checkpoint por menor pérdida de validación: época {met['checkpoint_epoch']}, pérdida train={best['train_loss']:.6f}, val={best['val_loss']:.6f}. Archivo: `best.pt`.",
'',
'## Integridad y prueba corta',
f"Manifiesto congelado: `snapshot/manifest.json`, SHA256 `{config['manifest_sha256']}`. La raíz de resolución de rutas originales es `{config['data_root']}`. Las asignaciones están en `snapshot/partitions.csv` y los grupos originales en `snapshot/grupos_particiones.json`.",
'Se conservaron los originales y se excluyeron escenas pendientes completas. Las 90 fotos del lote manual no se usaron. No se leyeron imágenes de test ni se calcularon métricas sobre ellas. Se verificó la ausencia de cruces de grupo y huella entre particiones usando el manifiesto existente.',
f"Prueba de redimensionado: error máximo de reconstrucción {pre['resize_max_pixel_error']:.9f} píxeles; panel `preflight_resize.jpg`. La pérdida fue finita y pasó de {pre['trace'][0]['loss']:.8f} a {pre['trace'][-1]['loss']:.8f}; en el paso {pre['trace'][-1]['step']} obtuvo TP={pre['trace'][-1]['tp']}, FP={pre['trace'][-1]['fp']}, FN={pre['trace'][-1]['fn']} sobre seis imágenes de train (umbral 0,1, IoU >= 0,5). Esto prueba capacidad de memorizar, no generalización.",
'',
'## Validación',
f"Confianza = sigmoid(presencia) x máxima softmax de clase. Umbral elegido: **{met['threshold']}**, por mayor micro-F1 en la cuadrícula registrada en `threshold.json`; empates favorecen el mayor umbral. NMS por clase IoU > 0,45. Emparejamiento uno a uno, misma clase, IoU >= 0,5: predicciones ordenadas por confianza y mejor objeto disponible. Los duplicados cuentan como FP. Sin detecciones, precisión no definida (N/D).",
'Estas cifras usan la validación que seleccionó checkpoint y umbral; no son una estimación imparcial de evaluación final. Test sigue reservado.',
'',
'| Clase | Objetos | Aciertos | FP | Omitidos | Precisión | Exhaustividad |',
'|---|---:|---:|---:|---:|---:|---:|'
]
fmt=lambda x:'N/D' if x is None else f'{100*x:.2f}%'
for c,n in met['classes'].items():lines.append(f"| {c} | {n['objects']} | {n['tp']} | {n['fp']} | {n['fn']} | {fmt(n['precision'])} | {fmt(n['recall'])} |")
lines.extend(['','## Confusiones observadas','La matriz `confusions.json` es un diagnóstico adicional: empareja por IoU >= 0,5 sin exigir la misma clase, uno a uno. No sustituye el emparejamiento de la tabla. Un objeto omitido sin predicción localizada no demuestra confusión de material.'])
for a,row in conf['matrix'].items():
    for b,n in row.items():
        if a!=b and n:
            examples=[e for e in conf['examples'] if e['true_class']==a and e['predicted_class']==b]
            lines.append(f"- {a} → {b}: {n}; ejemplo `{examples[0]['id']}` (IoU {examples[0]['iou']:.3f}, confianza {examples[0]['score']:.3f}).")
if not conf['examples']:lines.append('No hubo confusiones de material localizadas con IoU >= 0,5 al umbral seleccionado.')
lines.extend(['','![Anotaciones y predicciones de las mismas imágenes de validación](validation_preview.jpg)',
'La vista previa prioriza papel, cartón y vidrio. Selecciona un caso con confusión localizada cuando existe y, en otro caso, el primer ID de validación de esa clase; no selecciona por apariencia de éxito. A la izquierda están las predicciones y a la derecha las anotaciones. Los IDs y cajas están en `preview_examples.json`.',
'','## Tiempos medidos',
f"Caché: {done['cache_seconds']:.2f} s; prueba corta: {done['preflight_seconds']:.2f} s; entrenamiento: {done['training_seconds']:.2f} s; inferencia y métricas de validación: {done['validation_seconds']:.2f} s. No incluyen instalación/descarga de CUDA ni la revisión visual. Historial completo: `history.json`.",
'','## Probar un video en PowerShell','```powershell',
'.\\.venv\\Scripts\\python.exe scripts\\predict_video.py --video "C:\\ruta\\video.mp4" --checkpoint runs/detector_exp01/best.pt --output runs/detector_exp01/video_predicciones.mp4 --device auto',
'```','Ejecutar desde la raíz del proyecto. El archivo de salida debe ser nuevo. La inferencia utiliza únicamente el detector; no entrena ni necesita el transformer.'])
(out/'RESULTADOS.md').write_text('\n'.join(lines),encoding='utf-8')
print(out/'RESULTADOS.md')
