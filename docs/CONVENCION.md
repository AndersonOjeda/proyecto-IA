# Convención de gestión y fuentes

El detector aprende seis materiales: `plastico`, `papel`, `carton`, `vidrio`, `metal`, `organico`. La etiqueta `organico` representa restos alimentarios/cáscaras. En la interfaz, los cinco materiales restantes se agrupan como **Inorgánico (gestión)**. Esta es una convención de separación del proyecto, no una clasificación química: papel y cartón contienen materia de origen biológico.

Material, grupo operativo y destino son conceptos distintos. Papel y cartón contaminados no pasan a la clase visual `organico`. La recomendación cambia según aprovechabilidad, limpieza y recepción local. La cámara no demuestra esas condiciones.

## Convención de anotación ampliada: 23 de noviembre de 2026

Por decisión del proyecto, `organico` incluye restos alimentarios reconocibles: cáscaras, corazones de fruta, raíces descartadas, huesos y conchas de alimentos. En Food Waste se aceptan automáticamente sólo las clases que identifican esos restos explícitamente. `Vegetable` no permite distinguir una raíz descartada de un alimento entero y permanece pendiente de revisión del contexto. Lo mismo ocurre con alimentos enteros y `Other-waste` cuando no se puede confirmar su condición de residuo. Esta convención de material no supone que todos los restos sean admitidos por cualquier sistema de compostaje o ruta local.

Si una imagen contiene una anotación de material incierto o multicapa, la escena completa queda pendiente: el detector actual no admite regiones ignoradas. Se conservan tanto las cajas conocidas como las inciertas. No se convierte una región ambigua en fondo eliminando su anotación.

| Condición confirmada por el usuario | Orientación general |
|---|---|
| Material aprovechable limpio y seco admitido por el recolector | Blanco |
| Restos orgánicos aprovechables y ruta local disponible | Verde |
| No aprovechable, por ejemplo papel/cartón contaminado con comida | Negro |
| Residuo peligroso, envase químico o situación no cubierta | Consultar ruta específica; fuera del alcance de la clasificación ordinaria |

Fuentes oficiales consultadas el 22 de noviembre de 2026:

- Ministerio de Ambiente y Desarrollo Sostenible, [Preguntas frecuentes: código de colores](https://www.minambiente.gov.co/preguntas-frecuentes/).
- Ministerio de Ambiente, [Abecé código de colores](https://www.minambiente.gov.co/wp-content/uploads/2020/12/CODIGO_COLORES_VFF.pdf).

La tabla `residuos/recommendations.json` es revisable. No se ha confirmado una ruta específica del prestador de Pasto; verifique recepción, horarios y condiciones localmente antes de una recomendación operativa. El transformer identifica intenciones. Las reglas aportan la información de gestión; no es conocimiento aprendido de la legislación.
