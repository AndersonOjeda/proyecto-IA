# Dataset derivado para detección

Esta preparación usa los CSV de `docs/auditoria_externos/` y los originales de `data/externos/`. No entrena redes ni usa modelos preentrenados. El manifiesto resultante está en `data/preparado/deteccion/manifest.json`. Los archivos de origen se mantienen intactos.

Las utilidades de preparación usan `requirements-preparacion.txt` (incluye `requests`, requerido por el descargador de TACO). Ya está instalado en este entorno. Las descargas grandes se transportan en rangos HTTP cuando el servidor lo permite, verificando `Content-Range`, longitud y ETag antes de entregar los bytes al descargador original. No se recodifican las fotos descargadas.

## Abrir y revisar

Desde la raíz del proyecto:

```powershell
.\.venv\Scripts\python.exe -m residuos annotate --data data/preparado/deteccion
```

También puede ejecutar `data/preparado/deteccion/abrir_anotador.cmd`. Pulse **Pendientes**: la tanda manual seleccionada aparece primero. **Reabrir imagen** admite ahora JPG, JPEG y PNG registrados. Las copias de la primera tanda están en `data/preparado/deteccion/images/lote_manual/`, organizadas por clase propuesta. Las demás imágenes se abren mediante el manifiesto, sin copiarlas ni modificar sus píxeles.

Las cajas verdes ya tienen una clase aceptada por la política de importación; las naranjas muestran la clase original pendiente. Seleccione cada caja naranja y asigne una de las seis clases sólo cuando pueda verificar el material. Añada cualquier objeto real omitido en las anotaciones de origen. Si no puede resolverlo, deje la escena pendiente. Eliminar una caja incierta no es una solución: enseñaría fondo donde existe un residuo. **Guardar como revisada** requiere todas las clases resueltas y confirmación de revisión de la escena completa. Guardar sin cajas confirma explícitamente un negativo real.

## Política aplicada

- Bangladesh: sólo canonical. Tres clases plásticas se agrupan como `plastico`. Las cajas se obtienen del mínimo/máximo de vértices humanos normalizados; no se infieren objetos nuevos. `Paper` queda pendiente, al igual que la escena completa que la contenga.
- Food Waste: se aceptan clases explícitas de restos, cáscaras, corazones de fruta, huesos y conchas de alimentos. El resto de clases (incluido `Vegetable`, que no distingue raíz descartada de alimento entero) requiere revisión de contexto. Las 888 imágenes sin cajas siguen pendientes; ninguna se declaró negativa automáticamente.
- TACO: se ejecuta su `download.py` original, con manifiestos de una sola imagen y límites de red desde un envoltorio local. Se descargan únicamente fotos oficiales, conservando bytes originales, URL, licencia declarada y demás metadatos. La geometría corresponde a la orientación EXIF de visualización cuando ésta explica las dimensiones originales. Los fallos quedan en `data/preparado/taco_oficial/fallos.json`.
- Las 48 cajas fuera de límites corresponden a 7 del JSON oficial y 41 del no oficial. Sólo se corrigen en copia las 7 oficiales mediante intersección con la imagen (desborde marginal de hasta 2 píxeles). La segmentación original se conserva y no se utiliza para el detector de cajas. Los IDs repetidos 309 y 4040 pertenecen al oficial: se conservan ambos objetos y se reasigna un ID nuevo a la segunda aparición. Antes/después e índices originales están en `correcciones_taco.json`.
- Envases multicapa y clases de material incierto de TACO mantienen sus cajas para revisión; toda la escena permanece fuera del entrenamiento. Las clases fuera de taxonomía también requieren revisión/exclusión explícita.
- `dataset-resized` es reserva, sin cajas. `trash` continúa ambiguo. La primera tanda prioriza las cuatro clases indicadas según imágenes listas efectivas después de TACO. La variedad se selecciona mediante color, textura y aspecto, con semilla fija; no se genera ninguna caja automática.

## Particiones y conteos

Semilla fija: **20261123**. Se conservan `original_split`, `original_path`, `source_image_id`, anotaciones originales, huella y procedencia. La asignación derivada usa grupos de origen, nunca filas aleatorias de fotogramas. Se unen huellas idénticas, nombres previos a `.rf` (aumentos), sufijos `copy` de Bangladesh e identificadores de grabación de Food Waste. Las muestras visuales de `evidencia_cruces.jpg` confirman cruces reales; distintos bytes no implican fotos independientes.

Dentro de cada fuente se ordenan los grupos por SHA-256 de semilla e ID y se asignan aproximadamente 70/15/15 por **número de grupos**, no por cantidad de imágenes. Por ello, un video grande puede producir proporciones de imágenes diferentes. No se optimizan las particiones mirando métricas de modelos. Los grupos son unidades de procedencia conocidas: las fotografías sin metadatos de sesión no prueban independencia de cámara, lugar o jornada. No deben interpretarse miles de versiones aumentadas como miles de escenas independientes.

`resumen_seis_clases.csv` cuenta imágenes únicas por SHA y cajas de escenas **listas**. Excluye escenas pendientes completas y duplicados binarios. Los pendientes asociados por clase pueden solaparse y no incluyen negativos por confirmar sin clase. `resumen.json` aporta el total y motivos. `grupos_particiones.json` conserva todos los miembros, su partición anterior y la nueva, y los grupos disponibles para evaluación. Si un duplicado exacto aporta etiquetas discrepantes o pendientes, se bloquea el contenido para revisión en vez de aceptar la copia más incompleta.

El detector actual admite tres centros por celda 16×16. Las pocas escenas que exceden esa capacidad permanecen pendientes por una limitación del modelo, no por ser etiquetas incorrectas; no elimine objetos para hacerlas caber. La siguiente etapa debe decidir cómo tratarlas o adaptar la arquitectura.

## Artefactos y conservación

- `manifiesto_imagenes.csv`: fuente, origen, grupo, ambas particiones, estado, cajas conocidas y ambiguas.
- `resumen_seis_clases.csv`, `resumen.json`: conteos efectivos y pendientes.
- `grupos_particiones.json`, `evidencia_grupos.json`: agrupación y evidencia de cruces.
- `correcciones_taco.json`: comprobación de ambos JSON y decisiones de corrección.
- `lote_manual.json`: prioridad, conteos de escasez y selección de fotos.
- `vista_previa_seis_clases.jpg`: ejemplos reales con anotaciones humanas; no predicciones.
- `validacion_preparacion.json`: comprobaciones de geometría, particiones y estados.

La preparación rechaza sobrescribir un manifiesto existente para proteger revisiones humanas. Para una nueva versión, conserve la anterior y reconcilie las revisiones explícitamente. No vuelva a ejecutar el generador sobre trabajo manual sin esa reconciliación.

Las licencias se conservan tal como vienen: `CC` sin versión o `null` no se convierten en una licencia inventada. La extracción de `dataset-resized` no contiene licencia documental; su reserva para anotación no resuelve esa procedencia.

**El entrenamiento del detector y del transformer queda para la siguiente etapa.**
