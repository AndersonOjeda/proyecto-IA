"""Aplicación Web Streamlit: Detección de Residuos y Asistente de Separación en Español.
Proyecto Final de Inteligencia Artificial - Universidad Cooperativa de Colombia (UCC Pasto)
Profesor: Cristian Camilo Ordoñez Quintero | Estudiante: Anderson Ojeda
Modelos 100% entrenados desde cero (DetectorS8 Exp07-B + IntentTransformer).
"""
import sys
import time
import json
from pathlib import Path
from collections import Counter

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from residuos import CLASSES
from residuos.assistant import Assistant
from scripts.experiment3_runner import DetectorS8, decode_grid32, device_for

# Configuración de página de Streamlit
st.set_page_config(
    page_title="EcoDetect UCC · Detección y Separación de Residuos",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paleta de colores para cajas delimitadoras (RGB y Hex)
CLASS_COLORS = {
    'plastico': {'hex': '#00D2FF', 'rgb': (0, 210, 255), 'caneca': 'Blanca', 'tipo': 'Aprovechable'},
    'papel': {'hex': '#FFD200', 'rgb': (255, 210, 0), 'caneca': 'Blanca', 'tipo': 'Aprovechable'},
    'carton': {'hex': '#FF7A00', 'rgb': (255, 122, 0), 'caneca': 'Blanca', 'tipo': 'Aprovechable'},
    'vidrio': {'hex': '#00FF88', 'rgb': (0, 255, 136), 'caneca': 'Blanca', 'tipo': 'Aprovechable'},
    'metal': {'hex': '#FF0055', 'rgb': (255, 0, 85), 'caneca': 'Blanca', 'tipo': 'Aprovechable'},
    'organico': {'hex': '#2ECC71', 'rgb': (46, 204, 113), 'caneca': 'Verde', 'tipo': 'Orgánico Biodegradable'}
}

# Código de colores de reciclaje en Colombia (Resolución 2184 de 2019)
CANECA_INFO = {
    'Blanca': {
        'nombre': 'Caneca Blanca (Residuos Aprovechables)',
        'icono': '⚪',
        'desc': 'Plástico, cartón, papel, vidrio y metales. Deben estar LIMPIOS y SECOS para permitir su reciclaje.'
    },
    'Verde': {
        'nombre': 'Caneca Verde (Residuos Orgánicos)',
        'icono': '🟢',
        'desc': 'Restos de comida, cáscaras de frutas/verduras, residuos de café y restos de poda.'
    },
    'Negra': {
        'nombre': 'Caneca Negra (Residuos No Aprovechables)',
        'icono': '⚫',
        'desc': 'Papel higiénico, servilletas usadas, plásticos o cartones grasosos/contaminados y residuos de barrido.'
    }
}

@st.cache_resource
def load_detector_model(model_path, device_str='auto'):
    """Carga en caché el modelo DetectorS8."""
    device = device_for(device_str)
    path = Path(model_path)
    if not path.exists():
        st.error(f"No se encontró el checkpoint: {path}")
        return None, None
    saved = torch.load(path, map_location=device, weights_only=True)
    model = DetectorS8().to(device)
    state = saved.get('model', saved.get('state'))
    model.load_state_dict(state)
    model.eval()
    return model, device

@st.cache_resource
def load_assistant_model(model_path, device_str='cpu'):
    """Carga en caché el asistente transformer."""
    path = Path(model_path)
    if not path.exists():
        st.error(f"No se encontró el transformer del asistente: {path}")
        return None
    return Assistant(str(path), device_str)

def run_inference(model, device, image_bgr, thresholds_dict, min_box_size=0):
    """Ejecuta inferencia con DetectorS8 aplicando umbrales por clase."""
    h_orig, w_orig = image_bgr.shape[:2]
    # Preprocesamiento a 256x256 RGB normalizado
    img_resized = cv2.resize(image_bgr, (256, 256))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

    t_start = time.perf_counter()
    with torch.no_grad():
        pred = model(tensor)[0]
    t_elapsed = (time.perf_counter() - t_start) * 1000

    # Decodificar con umbral mínimo base (0.05) para filtrar después por clase
    raw_boxes = decode_grid32(pred, w_orig, h_orig, threshold=0.05, nms=0.45)
    
    # Filtrar según el umbral calibrado de cada clase y tamaño mínimo
    filtered_boxes = []
    for b in raw_boxes:
        cls_name = b['class']
        th = thresholds_dict.get(cls_name, 0.35)
        if b['score'] >= th:
            bx_w = b['xyxy'][2] - b['xyxy'][0]
            bx_h = b['xyxy'][3] - b['xyxy'][1]
            if bx_w >= min_box_size and bx_h >= min_box_size:
                filtered_boxes.append(b)

    # Ordenar por confianza decreciente
    filtered_boxes = sorted(filtered_boxes, key=lambda x: x['score'], reverse=True)
    return filtered_boxes, t_elapsed

def draw_detections(image_bgr, detections):
    """Dibuja cajas delimitadoras profesionales sobre la imagen."""
    im_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(im_pil)
    
    for b in detections:
        cls_name = b['class']
        score = b['score']
        x1, y1, x2, y2 = b['xyxy']
        color_info = CLASS_COLORS.get(cls_name, {'hex': '#00FF00', 'caneca': 'Blanca'})
        box_color = color_info['hex']
        
        # Grosor de línea adaptativo
        line_w = max(2, int(min(im_pil.size) / 250))
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=line_w)
        
        # Etiqueta con clase, score y caneca
        label = f"{cls_name.upper()} {score*100:.1f}% ({color_info['caneca']})"
        
        # Fondo para el texto
        text_bbox = draw.textbbox((x1, max(0, y1 - 22)), label)
        draw.rectangle(text_bbox, fill=box_color)
        draw.text((x1 + 3, max(0, y1 - 22) + 2), label, fill='#000000')
        
    return im_pil

# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Logo_UCC.png/400px-Logo_UCC.png", width=160)
    st.title("EcoDetect UCC")
    st.markdown("**Inteligencia Artificial · Proyecto Final**")
    st.caption("Universidad Cooperativa de Colombia · Pasto")
    st.markdown("---")
    
    st.subheader("🧠 Configuración del Detector")
    model_choice = st.selectbox(
        "Checkpoint del Detector:",
        options=[
            "Exp07-B (Recomendado · mAP 6.15%, Cartón 10.13%)",
            "Exp05 (Baseline · mAP 5.73%, Micro-F1 22.12%)"
        ],
        index=0
    )
    
    if "Exp07-B" in model_choice:
        model_path = ROOT / "runs/detector_exp07_b/best.pt"
    else:
        model_path = ROOT / "runs/detector/best.pt"

    st.subheader("🎯 Modo Operativo")
    op_mode = st.radio(
        "Estrategia de umbrales:",
        options=[
            "Modo Equilibrado (Calibrado por Clase)",
            "Modo Alta Precisión (Cero Falsas Alarmas)",
            "Modo Manual (Slider único)"
        ],
        index=0
    )

    if op_mode == "Modo Equilibrado (Calibrado por Clase)":
        thresholds = {
            'plastico': 0.35,
            'papel': 0.35,
            'carton': 0.20,
            'vidrio': 0.25,
            'metal': 0.20,
            'organico': 0.55
        }
        st.info("⚖️ **Plástico:** 0.35 | **Cartón:** 0.20 (100% prec) | **Metal:** 0.20 | **Orgánico:** 0.55")
    elif op_mode == "Modo Alta Precisión (Cero Falsas Alarmas)":
        thresholds = {
            'plastico': 0.40,
            'papel': 0.35,
            'carton': 0.35,
            'vidrio': 0.35,
            'metal': 0.35,
            'organico': 0.50
        }
        st.success("🎯 **Plástico al 61.3% de precisión** | Falsos positivos suprimidos")
    else:
        single_th = st.slider("Umbral de confianza general:", min_value=0.10, max_value=0.90, value=0.35, step=0.05)
        thresholds = {c: single_th for c in CLASSES}

    min_size = st.slider("Filtro tamaño mín. caja (px):", min_value=0, max_value=64, value=0, step=8)

    st.markdown("---")
    st.markdown("### 📋 Ficha Técnica Académica")
    st.markdown("""
    - **Profesor:** Cristian Camilo Ordoñez
    - **Estudiante:** Anderson Ojeda
    - **Backbone:** `DetectorS8` (100% scratch)
    - **Transformer:** `IntentTransformer`
    - **Regla:** 0 pesos externos / 0 preentrenados
    """)

# Carga de modelos
detector_model, device = load_detector_model(model_path, 'auto')
assistant_model = load_assistant_model(ROOT / "runs/assistant/best.pt", 'cpu')

# --- ENCABEZADO PRINCIPAL ---
st.title("♻️ EcoDetect: Detección y Separación Inteligente de Residuos")
st.markdown("""
Sistema inteligente en tiempo real para detección multiclase de residuos sólidos y asistencia conversacional en español.  
Implementado con redes neuronales convolucionales profundas (`DetectorS8`) y transformadores de atención (`IntentTransformer`), **entrenados 100% desde cero con pesos aleatorios**.
""")

# --- PESTAÑAS DE LA APLICACIÓN ---
tab_detect, tab_assistant, tab_metrics = st.tabs([
    "📸 Detección Visual",
    "💬 Asistente de Separación (NLP)",
    "📊 Resultados y Métricas del Proyecto"
])

# ==================== PESTAÑA 1: DETECCIÓN VISUAL ====================
with tab_detect:
    col_input, col_output = st.columns([1, 1.2])

    with col_input:
        st.subheader("1. Entrada de Imagen")
        source_type = st.radio("Selecciona el origen de la imagen:", ["Subir Archivo", "Cámara Web", "Imagen de Ejemplo"], horizontal=True)

        image_bgr = None

        if source_type == "Subir Archivo":
            uploaded_file = st.file_uploader("Elige una foto de residuos (JPG, PNG)...", type=['jpg', 'jpeg', 'png'])
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        elif source_type == "Cámara Web":
            camera_photo = st.camera_input("Capturar foto desde la cámara")
            if camera_photo is not None:
                file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
                image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        else:
            sample_path = ROOT / "ejemplo.jpg"
            if sample_path.exists():
                image_bgr = cv2.imread(str(sample_path))
                st.caption(f"Cargada imagen de muestra: `{sample_path.name}`")
            else:
                st.warning("No se encontró `ejemplo.jpg` en la raíz.")

    with col_output:
        st.subheader("2. Detecciones en Tiempo Real")
        if image_bgr is not None and detector_model is not None:
            # Inferencia
            detections, latency_ms = run_inference(detector_model, device, image_bgr, thresholds, min_size)
            result_img = draw_detections(image_bgr, detections)
            st.image(result_img, caption=f"Inferencia completada en {latency_ms:.1f} ms ({device})", use_container_width=True)

            # Métricas rápidas
            m1, m2, m3 = st.columns(3)
            m1.metric("Objetos Detectados", len(detections))
            clases_detectadas = set(d['class'] for d in detections)
            m2.metric("Clases Identificadas", len(clases_detectadas))
            m3.metric("Latencia", f"{latency_ms:.1f} ms")

            # Tabla de desglose de residuos
            if detections:
                st.markdown("#### 📦 Detalle de Residuos Identificados:")
                table_data = []
                for idx, d in enumerate(detections, 1):
                    c_info = CLASS_COLORS.get(d['class'], {'caneca': 'Blanca', 'tipo': 'Aprovechable'})
                    table_data.append({
                        "#": idx,
                        "Residuo": d['class'].capitalize(),
                        "Confianza": f"{d['score']*100:.1f}%",
                        "Tipo de Residuo": c_info['tipo'],
                        "Caneca Asignada": f"{CANECA_INFO[c_info['caneca']]['icono']} {c_info['caneca']}"
                    })
                st.dataframe(table_data, use_container_width=True)
            else:
                st.info("No se detectaron residuos por encima del umbral en este fotograma.")
        elif image_bgr is None:
            st.info("👈 Por favor, carga o captura una imagen en el panel izquierdo para comenzar.")

# ==================== PESTAÑA 2: ASISTENTE DE SEPARACIÓN (NLP) ====================
with tab_assistant:
    st.subheader("💬 Asistente de Reciclaje y Separación en la Fuente")
    st.markdown("Hazle preguntas en lenguaje natural sobre los residuos detectados o sobre cómo disponer de un material en Colombia:")

    # Sugerencias rápidas
    st.markdown("**Preguntas frecuentes rápidas:**")
    c_sug1, c_sug2, c_sug3, c_sug4 = st.columns(4)
    pregunta_sugerida = ""
    if c_sug1.button("🧴 ¿Dónde boto esta botella?"):
        pregunta_sugerida = "¿dónde boto esta botella?"
    if c_sug2.button("🍌 ¿Qué hago con esta cáscara?"):
        pregunta_sugerida = "¿dónde boto esta cáscara?"
    if c_sug3.button("📦 ¿Cómo reciclo una caja de cartón?"):
        pregunta_sugerida = "¿dónde boto una caja de cartón?"
    if c_sug4.button("🥫 ¿Qué residuos tengo en la escena?"):
        pregunta_sugerida = "¿qué residuos tengo?"

    user_query = st.text_input("Escribe tu consulta:", value=pregunta_sugerida, placeholder="Ej: ¿Dónde boto esta lata de cerveza?")

    if user_query and assistant_model is not None:
        # Usar las detecciones actuales si existen
        current_dets = detections if ('detections' in locals() and detections) else []
        ans = assistant_model.answer(user_query, current_dets)

        col_a1, col_a2 = st.columns([1, 2])
        with col_a1:
            st.markdown("#### 🧠 Análisis Transformer")
            st.metric("Intención Reconocida", ans['intent'])
            st.metric("Certeza del Modelo", f"{ans['confidence']*100:.1f}%")
        with col_a2:
            st.markdown("#### 📢 Recomendación del Asistente")
            st.info(ans['text'])

    st.markdown("---")
    st.subheader("🇨🇴 Código de Colores Oficial (Resolución 2184 de 2019)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"### {CANECA_INFO['Blanca']['icono']} {CANECA_INFO['Blanca']['nombre']}")
        st.write(CANECA_INFO['Blanca']['desc'])
    with c2:
        st.markdown(f"### {CANECA_INFO['Verde']['icono']} {CANECA_INFO['Verde']['nombre']}")
        st.write(CANECA_INFO['Verde']['desc'])
    with c3:
        st.markdown(f"### {CANECA_INFO['Negra']['icono']} {CANECA_INFO['Negra']['nombre']}")
        st.write(CANECA_INFO['Negra']['desc'])

# ==================== PESTAÑA 3: RESULTADOS Y MÉTRICAS ====================
with tab_metrics:
    st.subheader("📊 Comparativa Cuantitativa de Experimentos (Evaluación Canónica en 670 imágenes)")
    st.markdown(r"""
    Evaluación independiente y reproducible sobre las 670 imágenes reales canónicas de validación, utilizando correspondencia 1 a 1 ($IoU \ge 0.5$) y curvas PR unthresholded de 101 puntos VOC.
    """)

    metrics_table = [
        {"Métrica": "mAP@50 Global", "Exp05 (Baseline)": "5.73%", "Exp07 (Datos Separados)": "4.58%", "Exp07-B (Balanceado)": "6.15% 🏆 (Récord)"},
        {"Métrica": "AP@50 Cartón", "Exp05 (Baseline)": "0.17%", "Exp07 (Datos Separados)": "1.59%", "Exp07-B (Balanceado)": "10.13% 🏆 (x60 mejora)"},
        {"Métrica": "AP@50 Papel", "Exp05 (Baseline)": "0.63%", "Exp07 (Datos Separados)": "0.94%", "Exp07-B (Balanceado)": "1.35% 🏆 (+114%)"},
        {"Métrica": "AP@50 Plástico", "Exp05 (Baseline)": "16.80%", "Exp07 (Datos Separados)": "15.68%", "Exp07-B (Balanceado)": "16.47% (Sólido)"},
        {"Métrica": "AP@50 Metal", "Exp05 (Baseline)": "4.72%", "Exp07 (Datos Separados)": "1.63%", "Exp07-B (Balanceado)": "1.88% (3 aciertos directos)"},
        {"Métrica": "AP@50 Orgánico", "Exp05 (Baseline)": "12.07%", "Exp07 (Datos Separados)": "7.60%", "Exp07-B (Balanceado)": "7.03%"},
        {"Métrica": "Falsos Positivos (th=0.35)", "Exp05 (Baseline)": "1.500+ FP", "Exp07 (Datos Separados)": "1.985 FP", "Exp07-B (Balanceado)": "1.007 FP (-49% ruido)"},
        {"Métrica": "Precisión en Plástico", "Exp05 (Baseline)": "29.9%", "Exp07 (Datos Separados)": "29.9%", "Exp07-B (Balanceado)": "50.0% a 61.3% 🏆"},
        {"Métrica": "Imágenes de Entrenamiento", "Exp05 (Baseline)": "5.150", "Exp07 (Datos Separados)": "5.950", "Exp07-B (Balanceado)": "5.950 (+800 nuevas)"}
    ]
    st.dataframe(metrics_table, use_container_width=True)

    st.markdown("""
    > [!IMPORTANT]
    > **Conclusión del Proyecto:**
    > La integración del Experimento 7-B resolvió el desbalance crítico en las clases minoritarias:
    > - El **Cartón** multiplicó casi por 60 su rendimiento gracias a las imágenes balanceadas de `DATOS_SEPARADOS` y las anotaciones manuales del estudiante.
    > - La **supresión de fondo ($\alpha=0.60$)** redujo los falsos positivos a la mitad y duplicó la precisión en plásticos hasta alcanzar el **61.3%**.
    """)
