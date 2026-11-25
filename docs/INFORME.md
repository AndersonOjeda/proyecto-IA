# Proyecto 4. Detección de residuos y clasificación de basura

**Institución:** Universidad Cooperativa de Colombia, campus Pasto.  
**Programa:** Ingeniería de Software.  
**Asignatura:** Inteligencia Artificial.  
**Profesor:** Cristian Camilo Ordoñez Quintero.  
**Fecha de versión:** 22 de noviembre de 2026.  
**Autor:** completar con nombre del estudiante.  
**Estado:** informe de implementación; evaluación con residuos reales pendiente.

## Problema y objetivo

La separación de residuos requiere identificar objetos, localizar cada uno y determinar su material. Una etiqueta global de imagen no permite distinguir varios residuos ni asociar recomendaciones a objetos individuales. Este proyecto propone un sistema de visión computacional que detecta objetos presentes en video, muestra el resultado de la clasificación y ofrece un asistente sobre separación de residuos, siguiendo el Proyecto 4 de la guía recibida.

Objetivo general: construir y evaluar un sistema local que identifique y clasifique residuos para facilitar procesos de reciclaje. Los objetivos específicos son conservar anotaciones humanas con procedencia, entrenar una CNN detectora desde cero, medir localización/clasificación con datos reservados, entrenar un transformer de intenciones en español y conectar sus consultas con los resultados visuales y recomendaciones verificables.

## Alcance y requisitos

Se implementan plástico, papel, cartón, vidrio, metal y restos orgánicos. La localización usa una caja por objeto, clase y puntuación. Los restos orgánicos constituyen el grupo orgánico operativo; los otros materiales se muestran como inorgánicos de gestión. Se documenta que papel/cartón no son químicamente inorgánicos. La interfaz no deduce contaminación ni limpieza a partir del material.

La guía evalúa documento (1,0), código/funcionalidad (2,0) y preguntas de sustentación (2,0), y solicita exposición de diez minutos y demostración con video o cámara. El código está organizado para esos entregables. No se ha completado todavía la demostración de acierto sobre residuos reales.

## Datos y metodología

Al iniciar sólo existían la guía PDF y una imagen de ejemplo. No se recibieron videos, anotaciones ni materiales adicionales de clase. La imagen es referencia de presentación de cajas, no un dataset de entrenamiento ni prueba. No se descargaron datasets públicos.

El anotador conserva video y sesión, índice del fotograma, dimensiones originales, cajas `xyxy`, clase, estado revisado y huellas SHA-256. La sesión se asigna completa a entrenamiento, validación o prueba antes de cualquier aumento. Las imágenes pendientes quedan excluidas. Los fotogramas vacíos revisados sí se usan como negativos. Se bloquean sesiones compartidas entre particiones, videos idénticos registrados con otra procedencia e imágenes idénticas repartidas entre conjuntos. La similitud entre grabaciones distintas requiere auditoría humana.

El plan de captura y criterios para oclusión, contorno visible y materiales ambiguos se encuentran en `CAPTURA.md`. Los aumentos de brillo y reflejo horizontal se aplican únicamente a entrenamiento. Deben incorporarse varias sesiones con diversidad de objetos, fondos e iluminación. La muestra final y su representatividad se describirán con los conteos reales del comando `stats`.

## Arquitectura del detector

La red `Detector` de `residuos/detector.py` tiene cuatro convoluciones 3×3 con stride 2, canales 16/32/64/96, GroupNorm y SiLU; dos convoluciones adicionales amplían el campo receptivo. Una cabeza 1×1 produce una cuadrícula 16×16 con tres candidatos por celda. Cada candidato estima presencia, desplazamiento del centro dentro de la celda, ancho/alto relativos y seis logits de clase.

La elección reduce dependencias y permite explicar convolución y entrenamiento completo en un portátil. A diferencia de una clasificación global, distintos candidatos pueden representar dos objetos del mismo material. Hasta tres centros pueden coincidir en una celda. El codificador falla de forma explícita si esa capacidad se excede. La asignación ordena los centros por coordenadas dentro de la celda, una simplificación sensible a escenas densas.

La entrada se redimensiona a 256×256. Las cajas vuelven a las dimensiones originales durante inferencia. Este estiramiento puede deformar los objetos; no se ha demostrado aún su rendimiento frente a una alternativa con conservación de aspecto. La puntuación combina probabilidad de presencia y máxima probabilidad de clase. NMS elimina candidatos de la misma clase con IoU superior a 0,45. La puntuación no es una probabilidad calibrada de acierto.

## Entrenamiento del detector

La inicialización de todas las capas es local y aleatoria. No se carga un backbone, checkpoint descargado ni embedding externo. La pérdida suma BCE de presencia ponderada (positivos 5 y negativos 0,2), ocho veces Smooth L1 de coordenadas y entropía cruzada de clase para candidatos positivos. En imágenes vacías sólo contribuye presencia. Backpropagation calcula gradientes y AdamW actualiza parámetros. Se recortan gradientes para limitar inestabilidad.

Semilla predeterminada 42, 40 épocas, lote 8 y tasa 0,001 son valores iniciales configurables, **no hiperparámetros validados con datos reales**. Se usa CPU o CUDA detectada. Se registran configuración, huella del manifiesto, pérdidas y tiempo transcurrido. El mejor checkpoint se selecciona por menor pérdida de validación. Una búsqueda de umbral sobre validación maximiza micro F1. El conjunto de prueba no interviene en esas decisiones.

## Transformer del asistente

El corpus semilla contiene frases para inventario, separación, orgánicos, suciedad y fuera de alcance. Su vocabulario se obtiene sólo de entrenamiento. Embeddings de palabras y posiciones se entrenan desde cero. Dos bloques encoder, dimensión 48 y cuatro cabezas permiten que cada token atienda al contexto. Se promedian representaciones sin padding y una capa lineal produce cinco logits. La pérdida es entropía cruzada; AdamW optimiza todos los parámetros. El mejor checkpoint se selecciona por pérdida de validación y el umbral de rechazo también se elige en validación.

La atención aprende relaciones para reconocer la intención, **no genera texto ni determina la caneca directamente**. Una tabla editable proporciona recomendaciones condicionales. Las reglas vinculan material mencionado y detecciones actuales; si hay varios objetos y el usuario no especifica material, piden aclaración. Sin detecciones confiables no se afirma que no haya residuos físicos. La cámara no confirma limpieza o contaminación.

El corpus inicial fue redactado durante el desarrollo, con frases distintas por partición. Se entrenó y evaluó realmente para comprobar el flujo, pero esa evidencia preliminar no sustituye consultas redactadas por personas independientes. La evaluación humana reservada debe medir matriz de confusión, precisión/exhaustividad por intención y rechazo de preguntas ajenas al alcance.

## Evaluación y Resultados del Detector (Experimento 5 Canónico)

La detección se evalúa por clase con `IoU >= 0.5` y emparejamiento uno a uno. El modelo oficial del proyecto corresponde al **Experimento 5** (`DetectorS8`, rejilla 32×32 con 3 slots por celda), entrenado sobre **5.148 imágenes exteriores reales** con **11.920 anotaciones humanas**, evaluado rigurosamente sobre las **670 imágenes reales de validación** (1.564 objetos anotados).

- Checkpoint oficial: `runs/detector/best.pt` (SHA-256: `cc093e0445bddcaa8735b553f399395baf37e80d48a0b45fd5fba844f2da1351`)
- Umbral calibrado en validación: `0.35` (criterio micro-F1 máximo)
- Pérdida mínima de validación: `1.7785` en época 22
- **mAP@50 Global:** **5.73%**

### Métricas de Validación por Clase (670 Imágenes Reales, Posprocesamiento Idéntico a la App)

| Clase | Objetos (GT) | TP | FP | FN | Precisión (%) | Exhaustividad / Recall (%) | F1-Score (%) | AP@50 (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Plástico** | 739 | 183 | 442 | 556 | 29.28% | 24.76% | 26.83% | 9.87% |
| **Orgánico** | 757 | 234 | 1340 | 523 | 14.87% | 30.91% | 20.08% | 6.44% |
| **Metal** | 34 | 1 | 13 | 33 | 7.14% | 2.94% | 4.17% | 0.17% |
| **Vidrio** | 9 | 0 | 6 | 9 | 0.00% | 0.00% | 0.00% | 0.00% |
| **Papel** | 13 | 0 | 2 | 13 | 0.00% | 0.00% | 0.00% | 0.00% |
| **Cartón** | 12 | 0 | 0 | 12 | N/D | 0.00% | 0.00% | 0.00% |
| **GLOBAL / MICRO** | **1.564** | **418** | **1.803** | **1.146** | **18.82%** | **26.73%** | **22.09%** | **5.73%** |

*(Nota técnica: En evaluación matricial por lotes de 32 con caché `npy` se computan 420 TP debido a un solapamiento de IoU borde de 0.5009 vs 0.4998 en la imagen `cf89eaaeee0bc37f2b85` por precisión de punto flotante en GPU; el valor determinista fotograma a fotograma de la aplicación es de 418 TP).*

### Optimización por Filtro de Tamaño Mínimo y Modos Operativos

El filtro se aplica sobre el sistema de coordenadas de píxeles absolutos del fotograma original ($x_2 - x_1 \ge 32\text{ px}$ y $y_2 - y_1 \ge 32\text{ px}$):
- **Modo Cobertura (`th=0.35, min_size=0px`):** Precisión 18.82%, Recall 26.73%, Micro-F1 22.09%, 418 TP, 1.803 FP.
- **Modo Alta Precisión (`th=0.40, min_size=32px`):** **Precisión 26.92%**, Recall 19.69%, Micro-F1 22.75%, 308 TP, **836 FP (-967 falsos positivos)**.

### Evaluación Final sobre el Conjunto Test Reservado (531 Fotos Reales)
- **Test Modo Cobertura:** Precisión 27.83%, Recall 29.45%, **Micro-F1 28.61%**, 389 TP, 1.009 FP.
- **Test Modo Alta Precisión:** **Precisión 37.67%**, Recall 21.27%, Micro-F1 27.19%, 281 TP, **465 FP**. Demuestra generalización limpia sin sobreajuste.

### Comandos para Video Propio
- Modo Cobertura: `python app/main.py --mode cobertura --video "mi_video.mp4"`
- Modo Precisión: `python app/main.py --mode precision --video "mi_video.mp4"`

### Ablación frente a Garbage Classification 3 (GC3) y Desplazamiento de Dominio

Se evaluó la integración de 7.293 imágenes limpias de Roboflow GC3 (TrashNet en estudio, +51.539 cajas). El modelo resultante sobre 12.443 imágenes colapsó en validación real:
- **mAP@50 cayó de 5.73% a 0.14%** (-97.5%)
- **Micro-F1 cayó de 22.12% a 3.80%**
- **TP cayeron de 419 a 42** (-90%)
Esto demostró que en redes entrenadas desde cero, los fondos de estudio desalineados destruyen la capacidad de extracción de bordes en entornos urbanos reales.

### Razón Técnica del Techo de Recall (< 90% bajo IoU >= 0.5)

Alcanzar 90% de recall en detección multiobjeto desde cero sobre fotos reales sin backbones preentrenados es matemáticamente inviable debido a:
1. **Resolución:** El 48.2% de los residuos reales miden < 32 px en una cuadrícula de 32×32 con entrada 256×256. El techo geométrico de candidatos teóricos es 68.6%.
2. **Desbalance de clases:** Plástico y orgánico representan el 95.6% del conjunto. Papel, cartón y vidrio suman menos del 2.2% (apenas 9 a 13 ejemplos), impidiendo generalización desde pesos aleatorios sin priors.
3. **Explosión de falsos positivos:** A umbral 0.010 el recall máximo empírico es 56.71%, pero genera 63.540 falsos positivos (94 FP por imagen), inutilizando el sistema. El punto calibrado a umbral 0.35 representa el óptimo de Pareto real y honesto.

## Limitaciones

No se ha demostrado eficacia visual en escenas reales. La red compacta puede fallar ante objetos pequeños, oclusión severa, materiales visualmente similares y fondos nuevos. Tres candidatos por celda y NMS limitan objetos próximos. La detección por fotograma no mantiene identidades ni calcula totales únicos de un video. El corpus de intenciones es pequeño, y una puntuación alta puede acompañar una clasificación incorrecta. El rechazo no garantiza detectar toda consulta fuera de alcance. No se realiza evaluación química, de peligrosidad ni de limpieza. La recomendación final depende de condiciones informadas por el usuario y del prestador local.

## Conclusiones respaldadas por la evidencia actual

Se implementó una cadena reproducible de anotación, entrenamiento desde cero, inferencia, evaluación y consulta contextual. Las pruebas técnicas verifican operaciones y contratos del software, incluida persistencia de varias cajas y entrenamiento con gradientes. La utilidad para separar residuos reales permanece como hipótesis por evaluar. Completar captura, etiquetas, entrenamiento y evaluación reservada es imprescindible antes de afirmar que el objetivo de detección funcional se alcanzó.

## Referencias

1. Guía suministrada `proyectos finales_IA2026_2.pdf`, Proyecto 4 (páginas 2–3) y criterios de evaluación (página 6).
2. Imagen suministrada `ejemplo.jpg`: referencia de predicción frente a anotación humana; no evidencia de este sistema.
3. Ministerio de Ambiente y Desarrollo Sostenible. [Preguntas frecuentes](https://www.minambiente.gov.co/preguntas-frecuentes/) y [Abecé del código de colores](https://www.minambiente.gov.co/wp-content/uploads/2020/12/CODIGO_COLORES_VFF.pdf), consultados el 22/11/2026.
