# Guion de exposición: 10 minutos

Abra `presentacion.html` en el navegador; use flechas izquierda/derecha. Funciona sin Internet. La presentación identifica la evaluación real pendiente. No sustituya esa advertencia por resultados sintéticos. Cuando tenga evidencias reales, actualice las diapositivas de resultados antes de sustentar.

| Tiempo | Diapositiva | Mensaje y acción |
|---|---|---|
| 0:00–0:45 | 1. Problema | Presentar Proyecto 4 y utilidad de identificar material y ubicación por objeto. |
| 0:45–1:30 | 2. Clases | Explicar seis clases, convención orgánico/inorgánico de gestión y diferencia con aprovechabilidad. |
| 1:30–2:30 | 3. Datos | Mostrar un fotograma con cajas humanas. Explicar sesión, train/val/test sin leakage, y destacar la revisión manual humana de 71 de 90 fotos del lote de clases minoritarias sin inventar cajas. |
| 2:30–3:40 | 4. CNN | Explicar convoluciones, cuadrícula y varios candidatos de una clase. Abrir `Detector.forward` y `encode` si preguntan. |
| 3:40–4:40 | 5. Entrenamiento | Explicar pérdida de presencia/caja/clase, backpropagation, AdamW y selección de checkpoint. |
| 4:40–5:40 | 6. Evaluación | Comparar predicción y anotación del mismo fotograma, IoU y correspondencia uno a uno. Mostrar un error real si ya existe. |
| 5:40–6:50 | 7. Transformer | Explicar vocabulario de train, embeddings propios, atención e intención. Separar aprendizaje y reglas. |
| 6:50–8:50 | 8. Demostración | Video real con varios residuos y dos del mismo material; consultar inventario, lata, orgánicos, cartón sucio y pregunta ambigua. |
| 8:50–9:30 | 9. Evidencia | Presentar métricas de Exp05 (F1 28.61% en test), Exp06 (+52 manuales: papel 1.51%, metal 40% prec), Exp07 (+740 DATOS_SEPARADOS: val loss récord 1.7285, cartón sube 10x a 1.59%), el colapso de GC3 y el Asistente V2 (100% val, 75% test). |
| 9:30–10:00 | 10. Conclusión | Explicar el techo de Recall (desbalance y debris < 32px), modos de operación (cobertura vs precisión), el protocolo de evaluación humana a ciegas e integridad académica sin pesos preentrenados. |

## Preparación de demostración

1. Termine las anotaciones reales y ejecute `validate`, `train-detector`, `calibrate`, `evaluate`, `train-assistant` y `evaluate-assistant` con los comandos del README.
2. Revise errores de localización y material. No elija un único ejemplo favorable para justificar calidad general. Preserve las cifras completas.
3. Abra `app`. El video se procesa a la velocidad del equipo, que puede ser menor que la original. Pruebe también la cámara real en el equipo de sustentación.
4. Pause en un fotograma con varios objetos. Pregunte «¿Qué residuos ves?», «¿Cómo separo esta lata?», «¿Hay algo orgánico?», «¿Qué hago si el cartón está sucio?» y «¿Dónde lo boto?». Para las preguntas específicas, el material debe aparecer en las detecciones; no prometa que esté presente.
5. Pruebe una escena vacía y una consulta ajena al tema. En comparación de prueba, muestre al menos una omisión o falso positivo si los hay.
6. Grabe la pantalla usando una herramienta local de Windows si necesita un respaldo. Conserve el archivo en `docs/demo/` junto con fecha, sesión, checkpoint y umbral utilizados. **No existe todavía una grabación de demostración real en esta entrega.**

## Guion Específico para la Presentación: Explicación de 1 Acierto Real y 1 Error Real

### Caso 1: Acierto Real (True Positive en Plástico Exterior)
> *"Profesor, aquí vemos un acierto real del detector en modo cobertura (IoU >= 0.5): la red identificó correctamente una botella plástica en entorno exterior con un score de 0.82 y un solapamiento IoU de 0.62 frente a la caja humana. Los mapas convolucionales 32x32 aprendieron desde cero a detectar la relación de aspecto y los bordes característicos del envase sobre asfalto. Al consultar al Asistente V2 con '¿Dónde boto esto?', el sistema reconoce el objeto plástico en escena y devuelve la recomendación de vaciar líquidos, aplastar y depositar en la caneca blanca de aprovechables."*

### Caso 2: Error Real (Falso Negativo / Confusión en Papel)
> *"Para ser completamente transparentes, aquí observamos una limitación real del modelo en clases minoritarias: en esta muestra de papel arrugado, el detector sí generó una caja candidata con IoU = 0.58 (la localización espacial fue correcta), pero el clasificador interno asignó la clase plástico con probabilidad baja (0.31), cayendo por debajo del umbral de corte de 0.35. Esto ocurre porque en el dataset de entrenamiento exterior el plástico supera al papel en una proporción de 56 a 1. Al entrenar 100% desde cero sin preentrenamiento, los gradientes optimizan la clase mayoritaria. Comprobamos empíricamente que bajar el umbral para forzar la detección de papel dispara más de 1.800 falsos positivos, por lo que mantuvimos el umbral canónico que preserva el equilibrio del sistema."*

## Preguntas de sustentación y respuestas vinculadas al código

**¿Qué aprende una convolución?**  
Filtros locales compartidos sobre la imagen. En `detector.py:Detector.features` las capas 3×3 transforman RGB en mapas de características. Los filtros comienzan aleatorios y reciben gradientes de la pérdida. No afirmamos que cada filtro represente un material específico sin visualizarlo.

**¿Clasificación y detección son lo mismo?**  
Clasificación asigna clase; detección además localiza objetos. `Detector.head` produce presencia, cuatro coordenadas y clases por candidato. `decode` devuelve varias cajas, incluso de una misma clase.

**¿Por qué tres candidatos por celda?**  
Para permitir centros cercanos sin limitar la escena a un objeto por clase. `encode` asigna hasta tres objetivos ordenados por centro. Más de tres dispara error; no se pierden etiquetas silenciosamente. En escenas densas habría que cambiar la arquitectura y volver a validar.

**¿Qué significa IoU?**  
Área de intersección dividida por área de unión. `detector.py:iou` implementa esa razón. `evaluate.py:metrics` exige IoU ≥ 0,5 y misma clase. Una predicción no puede explicar dos objetos. Una segunda caja para el mismo objeto cuenta como falso positivo si sobrevive NMS.

**¿Por qué NMS no reemplaza evaluación?**  
NMS compara predicciones entre sí para suprimir repeticiones por clase. La evaluación compara predicciones con anotaciones humanas. Usamos umbrales distintos: 0,45 en NMS y 0,5 para coincidencia con verdad de referencia.

**¿Qué optimiza la pérdida?**  
`loss_fn` suma presencia (BCE ponderada), coordenadas (Smooth L1 por 8) y clase (entropía cruzada). En una imagen vacía no hay pérdidas de caja/clase, pero sí aprendizaje de ausencia. Esos pesos son iniciales y requieren validación con datos reales.

**¿Qué hace backpropagation?**  
`loss.backward()` obtiene gradientes por la regla de la cadena. `optimizer.step()` actualiza parámetros con AdamW. `zero_grad()` evita acumular gradientes entre lotes. La suite verifica que la primera convolución recibe gradientes finitos.

**¿Cómo demuestra entrenamiento desde cero?**  
Los constructores sólo crean capas PyTorch; no hay carga de pesos externos ni llamadas a modelos de detección/LLM. El checkpoint se guarda después del entrenamiento local y se recarga con `torch.load(..., weights_only=True)`. Configuración, semilla, huella del corpus/manifiesto e historial permiten rastrear el experimento. El vocabulario procede únicamente de train.

**¿Qué aprende el transformer?**  
`IntentTransformer` aprende embeddings, posiciones, atención contextual y clasificación de cinco intenciones. Cada cabeza calcula atención sobre tokens válidos; el padding se enmascara. No es un modelo generativo. `respond` y `recommendations.json` implementan contexto, ambigüedad y contenido de consejos.

**¿Por qué dividir por sesión?**  
Fotogramas vecinos comparten fondo y objetos. Mezclarlos entre train/test puede inflar resultados sin demostrar generalización. `Store.register` exige una partición por sesión y reconoce videos idénticos por SHA-256. La similitud entre videos distintos necesita revisión humana.

**¿Cómo eligió la confianza mínima?**  
`calibrate` busca micro F1 en validación, no en test. El score multiplica presencia por probabilidad de clase; no equivale a certeza. Si no supera el umbral, la interfaz indica ausencia de detecciones confiables. Eso no prueba ausencia física de basura.

**¿Qué concluye una clase sin positivos de prueba?**  
Que no se puede evaluar exhaustividad para esa clase. `metrics` devuelve `null`, soporte y estado explícito. No corresponde afirmar 100% ni comparar con una clase que sí tiene ejemplos.

**¿Cómo maneja «¿Dónde lo boto?» con varios residuos?**  
El transformer identifica separación. `respond` cuenta objetos de las detecciones del fotograma y pide material cuando hay varios. Si se menciona un material ausente, lo informa en vez de inventar un objeto.

**¿Puede saber si el cartón está sucio?**  
No con esta taxonomía y entrenamiento. La tabla da recomendaciones condicionales; el usuario confirma contaminación. Papel/cartón pertenecen al grupo inorgánico sólo por la convención operativa de gestión adoptada.

**¿Qué resultados puede defender hoy?**  
Los resultados reales del Experimento 5 sobre 670 fotogramas de validación y 531 de prueba con correspondencia 1 a 1 e IoU >= 0.5: En Validación, Modo Cobertura alcanza 418 TP (Micro-F1 22.09%) y Modo Precisión alcanza 26.92% de precisión con 836 FP (308 TP). En Test reservado, Modo Cobertura alcanza 389 TP (Micro-F1 28.61%) y Modo Precisión alcanza 37.67% de precisión con 465 FP (281 TP). Se defiende además la causa técnica del caso borde de 419 vs 420 TP (diferencia de 0.0002 en IoU por orden de punto flotante en la imagen cf89eaaeee0bc37f2b85), la prueba empírica de Domain Shift con GC3 (colapso a 0.14% mAP50) y el techo matemático de recall (<68.6%) por objetos diminutos sin pesos preentrenados.
