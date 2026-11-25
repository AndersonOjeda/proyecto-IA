# Captura y etiquetado

## Organización antes de extraer imágenes

Grabe escenas propias con permiso. Elija objetos cotidianos seguros de las seis clases: plástico, papel, cartón, vidrio, metal y restos orgánicos (alimentos/cáscaras). No se necesitan rostros ni datos personales. Conserve los archivos originales en `data/videos/`.

Una **sesión** agrupa grabaciones cercanas de la misma preparación: cámara, fondo, objetos y jornada. Cambiar nombre o cortar un video no crea una sesión independiente. Asigne la sesión completa a una sola partición antes de extraer fotogramas. Como objetivo inicial de planificación, capture al menos 12 sesiones diversas y distribuya aproximadamente 8/2/2 entre train/val/test, procurando presencia de las seis clases en todas las particiones. Esto es un punto de partida, no una garantía estadística de suficiencia.

Use fondos, distancias, iluminación, orientaciones y ejemplares distintos. Reserve también objetos físicos nuevos para prueba cuando sea posible. Incluya escenas con 1–5 objetos, dos del mismo material, oclusiones moderadas y escenas vacías. El detector compacto no está diseñado para montones densos como los de la imagen de referencia. El estilo de comparación sí se reproduce.

En cada video extraiga fotogramas separados y con cambio visible. No copie una escena casi idéntica a otra partición. El programa verifica igualdad de videos, imágenes y consistencia de sesión, pero no detecta automáticamente similitud visual entre videos distintos. La revisión humana de las sesiones sigue siendo necesaria.

## Reglas de anotación

- Una caja por objeto reconocible, incluyendo dos objetos de la misma clase. Coordenadas ajustadas al contorno visible, sin incluir mucho fondo.
- Anote la porción visible de objetos parcialmente ocluidos. Si no puede determinar la clase con el objeto físico o evidencia suficiente, deje ese fotograma pendiente y revíselo; no lo declare vacío.
- Plástico: objetos conocidos de plástico. Vidrio: vidrio confirmado. Una botella transparente por sí sola no demuestra material.
- Metal: latas u objetos metálicos. Papel y cartón separados según el objeto físico. Restos orgánicos: alimentos, cáscaras y residuos alimentarios.
- Excluya escenas dominadas por residuos fuera de estas clases o materiales compuestos ambiguos hasta ampliar la taxonomía. No los etiquete como fondo vacío: podrían enseñar al detector a ignorarlos.
- Limpieza, grasa o peligrosidad no son clases visuales de este proyecto. Anote material, no destino de la caneca.
- Guarde como revisada sólo después de revisar **todos** los objetos. Una imagen vacía revisada tiene `reviewed=true` y `boxes=[]`. Una pendiente tiene `reviewed=false` y no participa en entrenamiento/evaluación.
- Una segunda persona debería revisar una muestra de las cajas y todos los casos ambiguos, usando el mismo criterio de contorno visible.

## Uso del editor

Ejecute `.\.venv\Scripts\python.exe -m residuos annotate`. Abra video, escriba sesión y partición. El campo Fotograma permite saltar a un índice; `+30` avanza 30 fotogramas, no 30 segundos. Cada captura se conserva como PNG. Seleccione clase antes de arrastrar el mouse. La lista permite cambiar clase, redibujar coordenadas o eliminar. Guardar confirma la revisión. Reabrir imagen recupera las cajas guardadas.

El editor muestra una vista reducida, pero transforma el mouse a los píxeles originales. No dibuje una caja para todo el conjunto de basura. Para el ejemplo mínimo solicitado: dibuje tres residuos con sus clases, guarde, cierre y use Reabrir imagen. Deben aparecer las tres filas y los tres recuadros.

## Auditoría y conjuntos

```powershell
.\.venv\Scripts\python.exe -m residuos stats
.\.venv\Scripts\python.exe -m residuos validate
```

Revise los conteos de objetos e imágenes por clase. Videos sin imágenes también se cuentan. Si una clase no tiene positivos en prueba, su exhaustividad no se puede estimar. Si sólo tiene un ejemplo, la cifra será muy inestable. No equilibre prueba duplicando imágenes ni aplique aumentos antes de particionar.

Sólo train recibe reflejo horizontal y variación de brillo durante la carga. Val/test usan imágenes originales. Mantenga el manifiesto junto con sus videos y PNG; editar o sustituir archivos provoca un error de integridad. No cambie el split a mano después de ver resultados de test.

## Datasets externos

Esta versión no descarga ni usa datasets públicos. Si se incorporan después, cree un registro de procedencia con URL, autor, versión, licencia, permisos, clases originales, mapeo a las seis clases y tipo de anotación. Se requieren cajas humanas por objeto para localización. Una etiqueta única por imagen no es una caja real y no sirve para evaluar detección. Respete grupos de captura también en los datos externos. No use pseudoetiquetas de modelos preentrenados.
