# Registro de verificación técnica

Fecha: 22 de noviembre de 2026. Windows, Python 3.12.14, PyTorch 2.15.0, OpenCV 5.1.0.95, NumPy 2.5.5 y Pillow 13.0.0. `torch.cuda.is_available()` devolvió `False`; los entrenamientos de esta sesión usaron CPU. Versiones completas en `requirements-lock.txt`.

Este archivo registra pruebas del software. **No es una evaluación final con residuos reales ni sustituye la tabla pendiente del informe.**

## Datos recibidos

Guía PDF de seis páginas e imagen de ejemplo, sin código previo. Se extrajo el contenido y se revisaron las páginas relevantes de Proyecto 4 y criterios. Se inspeccionó la imagen: comparación de cajas predichas y humanas en el mismo fotograma, con varios objetos. No se recibió `referencias/` ni materiales adicionales.

`python -m residuos stats` confirmó cero videos, sesiones, imágenes y objetos reales en todas las particiones. `validate` confirmó estructura sin errores e indicó ausencia de imágenes. `train-detector --epochs 1` sobre `data/` terminó con mensaje explícito de falta de imágenes revisadas, sin generar pesos de detector.

## Suite automatizada

Comando: `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`.

**14 pruebas aprobadas**. Cobertura relevante:

1. Guardar y reabrir tres cajas con plástico, metal y orgánico, conservando coordenadas y clases.
2. Distinguir un fotograma pendiente de otro vacío revisado.
3. Rechazar distribución de una sesión o un video idéntico entre conjuntos.
4. Rechazar cajas invertidas.
5. Persistir cambio de clase y eliminación.
6. Mostrar comandos de entrenamiento al faltar pesos.
7. Dibujar las tres cajas a través de los eventos del editor Tkinter, guardar, reabrir mediante el selector de imagen simulado y verificar las tres filas. Corregir clase, redibujar y eliminar posteriormente.
8. Abrir un video de prueba en la app, ejecutar inferencia en el hilo de trabajo, procesar resultados y mostrar comparación doble. Se usaron pesos locales aleatorios explícitamente de prueba; no se validó acierto visual.
9. Comprobar respuestas contextuales, ambigüedad, detecciones ausentes y recomendaciones condicionales.
10. Ejecutar forward y backward con imágenes vacías y con varios objetos; gradientes presentes.
11. Verificar emparejamiento uno a uno: una duplicación cuenta como falso positivo.
12. Conservar dos predicciones de plástico separadas en la misma imagen.
13. Aceptar dos centros por celda y rechazar más de tres sin perder etiquetas.
14. Construir vocabulario sólo con train, verificar corpus sin duplicados y gradientes del transformer.

Las dos pruebas iniciales de Tk fallaron dentro del aislamiento de herramientas por búsqueda de `init.tcl`. La suite completa fuera de ese aislamiento pasó. No se cambió el algoritmo para eludir esos errores; la prueba necesita acceso al escritorio y a las bibliotecas Tcl/Tk del intérprete.

## Pipeline sintético

Comando: `.\.venv\Scripts\python.exe scripts\smoke_pipeline.py`.

Generó tres videos artificiales, sesiones distintas y tres recuadros por imagen, exclusivamente en `tmp/synthetic-smoke/`. Ejecutó entrenamiento de una época, guardado/carga del mejor checkpoint, selección de umbral de validación, evaluación en la partición sintética test, JSON de resultados y PNG de comparación. Terminó sin errores. El archivo `NOTICE.json` marca la finalidad técnica. No se publican esas métricas como desempeño del detector y no se copiaron esos pesos a la ruta de la aplicación.

## Transformer: experimento preliminar real sobre corpus semilla

Comandos ejecutados:

```powershell
.\.venv\Scripts\python.exe -m residuos train-assistant --epochs 100 --device cpu
.\.venv\Scripts\python.exe -m residuos evaluate-assistant
```

Se completaron 100 épocas en aproximadamente 7,62 segundos de bucle de entrenamiento (no incluye carga del proceso). El modelo tiene 46.517 parámetros. Se guardaron pesos propios, configuración e historial en `runs/assistant/`. La validación eligió un umbral de rechazo de 0,65.

La evaluación sobre las 20 frases reservadas del **corpus semilla redactado durante implementación** produjo 11 aciertos (55%). Cada intención tenía cuatro preguntas. Exhaustividad observada: inventario 3/4, separar 1/4, orgánicos 0/4, suciedad 3/4 y fuera de alcance 4/4. Se conservaron todas las predicciones, puntuaciones y matriz de confusión en `runs/assistant/evaluation.json`.

La escasa generalización observada exige ampliar el corpus con preguntas humanas diversas. No se ajustó el modelo utilizando las preguntas test después de medirlas. No se traslada el 55% al informe como resultado de una evaluación humana independiente. Las frases exactas solicitadas por el usuario se reconocieron, pero pertenecen al entrenamiento y ese hecho **no demuestra generalización**.

## Interfaz y exposición

Se verificaron controles y flujo de video/comparación con pruebas Tkinter. El navegador abrió `docs/presentacion.html`; las diez diapositivas pasaron la comprobación de límites a 1036×647 píxeles: sin desbordamiento horizontal ni contenido bajo la barra de navegación. Se exportó una copia local `docs/presentacion.pdf` de diez páginas, se renderizaron todas y se revisaron visualmente. Se ajustó el contraste de advertencias y enlaces en impresión. El contenido identifica explícitamente los resultados reales pendientes.

No se ha verificado una detección correcta de basura real ni una demostración completa en cámara física. No se ha grabado un video de demostración real. Estos pendientes requieren datos y entrenamiento del detector; no se sustituyeron por capturas inventadas.
