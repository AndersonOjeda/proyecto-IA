import json
from pathlib import Path
import cv2
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.detector import WasteDetector
from residuos.assistant import Assistant

print("=== SIMULACIÓN DEL RECORRIDO DE DEMOSTRACIÓN (10 MINUTOS) ===")

# Checkpoints
detector_cob = WasteDetector(checkpoint='runs/detector/best.pt', mode='cobertura')
detector_prec = WasteDetector(checkpoint='runs/detector/best.pt', mode='precision')
assistant = Assistant('runs/assistant_v2/best.pt')

video_path = ROOT / 'data/externos/original_videos/original_videos/0008.mp4'
if not video_path.exists():
    print(f"Video {video_path} no encontrado, usando fotograma sintético.")
    cap = None
else:
    cap = cv2.VideoCapture(str(video_path))
    print(f"Video cargado exitosamente: {video_path.name} (FPS: {cap.get(cv2.CAP_PROP_FPS)}, Total frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)})")

# Paso 1: Leer fotogramas en distintos momentos del video
frames_to_test = [30, 150, 300]
tested_detections = []

if cap and cap.isOpened():
    for f_idx in frames_to_test:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if ret:
            # Detección en modo cobertura
            b_cob = detector_cob.detect(frame)
            # Detección en modo precisión
            b_prec = detector_prec.detect(frame)
            tested_detections.append((f_idx, b_cob, b_prec))
            print(f"\n[Fotograma {f_idx}] Detecciones Modo Cobertura: {len(b_cob)} objetos | Modo Precisión: {len(b_prec)} objetos")
            for b in b_cob[:5]:
                print(f"  - {b['class']} ({b['score']:.2f}) en {b['xyxy']}")
    cap.release()

# Paso 2: Probar consultas del asistente sobre las detecciones reales obtenidas
print("\n" + "="*75)
print("PRUEBA DE PREGUNTAS SOBRE LAS DETECCIONES DEL VIDEO REAL")
print("="*75)

if tested_detections:
    f_idx, sample_cob, sample_prec = tested_detections[0]
    print(f"\nFotograma {f_idx} (Modo Cobertura con {len(sample_cob)} objetos):")
    
    queries = [
        "¿Qué residuos ves?",
        "¿Dónde lo boto?",
        "¿Hay algo orgánico?",
        "¿Qué hago si el plástico está sucio?",
        "¿Quién descubrió América?"
    ]
    
    for q in queries:
        ans = assistant.answer(q, sample_cob)
        print(f"\nPregunta : {q}")
        print(f"Intención: {ans['intent']} (confianza: {ans['confidence']:.2f})")
        print(f"Respuesta: {ans['text']}")
