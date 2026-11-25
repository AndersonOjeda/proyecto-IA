# EcoDetect: Detección de Residuos y Asistente de Separación en Español

**Proyecto Final de Inteligencia Artificial**  
*Universidad Cooperativa de Colombia (UCC Pasto) — Ingeniería de Software*  
- **Docente:** Cristian Camilo Ordoñez Quintero  
- **Estudiante:** Anderson Ojeda  
- **Repositorio Oficial:** [https://github.com/AndersonOjeda/proyecto-IA](https://github.com/AndersonOjeda/proyecto-IA)

---

## 🌟 Características Principales

1. **Modelos 100% Entrenados Desde Cero:**
   - Cumplimiento estricto de la regla académica: **CERO modelos preentrenados**, cero pesos transferidos (*transfer learning*), cero APIs de visión externa.
   - **Detector de Objetos (`DetectorS8`):** Red convolucional profunda multiescala con rejilla de $32 \times 32$ y 3 candidatos por celda.
   - **Asistente Conversacional (`IntentTransformer`):** Red transformadora con embeddings posicionales y cabezal de atención entrenado desde cero para reconocer intenciones en español de Colombia.
2. **Aplicación Web en Streamlit (`streamlit_app.py`):**
   - Detección en vivo mediante cámara web o subida de imágenes.
   - Asistente conversacional con recomendaciones según el código de colores oficial de reciclaje en Colombia (Resolución 2184 de 2019: Blanco, Verde, Negro).
   - Panel de métricas académicas y comparativa de experimentos.
3. **Aplicación de Escritorio en Tkinter (`app/main.py`):**
   - Interfaz nativa local con procesamiento en vivo de cámara y video.

---

## 🏆 Hitos y Métricas del Experimento 7-B (Evaluación Canónica)

Evaluado sobre las **670 imágenes canónicas de validación** (estrictamente aisladas mediante escudo anti-fuga):

| Métrica | Exp05 (Línea Base) | Exp07 (Datos Separados) | Exp07-B (Balanceado) | Ganancia / Impacto |
| :--- | :---: | :---: | :---: | :---: |
| **mAP@50 Global** | 5.73% | 4.58% | **6.15%** 🏆 | **Récord histórico del proyecto** |
| **AP@50 Cartón** | 0.17% | 1.59% | **10.13%** 🏆 | **Mejora relativa de casi 60 veces (+5.850%)** |
| **AP@50 Papel** | 0.63% | 0.94% | **1.35%** 🏆 | **+114% de mejora** |
| **AP@50 Plástico** | 16.80% | 15.68% | **16.47%** | **Alta estabilidad y solidez** |
| **Precisión Plástico** | 29.9% | 29.9% | **50.0% a 61.3%** | **Más de la mitad de detecciones son aciertos reales** |
| **Falsos Positivos (th=0.35)** | 1.500+ FP | 1.985 FP | **1.007 FP** | **Reducción de casi 1.000 detecciones espurias** |

---

## 🚀 Cómo Ejecutar la Aplicación

### Opción 1: Ejecución Local de Streamlit

1. Clona el repositorio e instala las dependencias:
   ```bash
   git clone https://github.com/AndersonOjeda/proyecto-IA.git
   cd proyecto-IA
   pip install -r requirements.txt
   ```

2. Ejecuta la aplicación web:
   ```bash
   streamlit run streamlit_app.py
   ```
   *Se abrirá automáticamente en tu navegador en `http://localhost:8501`.*

### Opción 2: Despliegue en Streamlit Community Cloud (Gratis en la Nube)

1. Ingresa a [share.streamlit.io](https://share.streamlit.io/) e inicia sesión con tu cuenta de GitHub.
2. Pulsa en **"New app"** y completa los campos:
   - **Repository:** `AndersonOjeda/proyecto-IA`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
3. Haz clic en **"Deploy!"**.
   *Streamlit Cloud detectará `requirements.txt`, instalará las dependencias y cargará los checkpoints `runs/detector_exp07_b/best.pt` y `runs/assistant/best.pt` automáticamente.*

### Opción 3: Aplicación de Escritorio Tkinter

```bash
python app/main.py --mode cobertura
```

---

## 📁 Estructura del Repositorio

```text
├── streamlit_app.py            # Aplicación Web moderna en Streamlit
├── requirements.txt            # Dependencias optimizadas para Streamlit Cloud
├── ejemplo.jpg                 # Imagen de prueba rápida
├── app/
│   ├── main.py                 # Punto de entrada de la aplicación Tkinter
│   └── detector.py             # Envoltorio de inferencia
├── residuos/                   # Paquete central de la solución
│   ├── detector.py             # Arquitectura base y decodificador
│   ├── detector_s8.py          # Arquitectura DetectorS8 (convolucional densa)
│   ├── assistant.py            # IntentTransformer y motor de recomendaciones
│   ├── evaluate.py             # Módulo de métricas IoU 1 a 1 y curvas PR
│   └── recommendations.json    # Reglas del código de colores de reciclaje
├── runs/
│   ├── detector_exp07_b/       # Checkpoint récord Exp07-B y umbrales calibrados
│   │   ├── best.pt             # Pesos del modelo entrenado (1.4 MB)
│   │   └── calibrated_thresholds.json
│   ├── detector/               # Checkpoint Exp05 (Baseline canónico)
│   └── assistant/              # Transformer del Asistente en español
├── scripts/                    # Scripts de entrenamiento y auditoría
└── tests/                      # Suite de pruebas unitarias (33 tests automáticos)
```

---

## 🧪 Pruebas Unitarias

Para validar la integridad de la arquitectura, el escudo anti-fuga y la correspondencia 1 a 1:

```bash
python -m unittest discover -s tests -v
```
*(33 de 33 pruebas aprobadas).*
