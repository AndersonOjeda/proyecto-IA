import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from residuos.assistant import Assistant

ast = Assistant(str(ROOT / 'runs/assistant_v2/best.pt'))

test_scenarios = [
    {
        "name": "1. Escena vacía (0 objetos)",
        "detections": [],
        "query": "¿Qué residuos ves?"
    },
    {
        "name": "2. Escena vacía con pregunta de separación",
        "detections": [],
        "query": "¿Dónde boto esta botella?"
    },
    {
        "name": "3. Un solo residuo (1 plástico)",
        "detections": [{'class': 'plastico', 'score': 0.82, 'xyxy': [100, 100, 250, 300]}],
        "query": "¿Dónde boto esto?"
    },
    {
        "name": "4. Varios residuos y pregunta ambigua ('esto')",
        "detections": [
            {'class': 'plastico', 'score': 0.75, 'xyxy': [50, 50, 150, 200]},
            {'class': 'organico', 'score': 0.68, 'xyxy': [200, 200, 350, 400]},
            {'class': 'metal', 'score': 0.80, 'xyxy': [400, 100, 500, 250]}
        ],
        "query": "¿Dónde boto esto?"
    },
    {
        "name": "5. Varios residuos y pregunta específica ('lata')",
        "detections": [
            {'class': 'plastico', 'score': 0.75, 'xyxy': [50, 50, 150, 200]},
            {'class': 'organico', 'score': 0.68, 'xyxy': [200, 200, 350, 400]},
            {'class': 'metal', 'score': 0.80, 'xyxy': [400, 100, 500, 250]}
        ],
        "query": "¿Cómo separo esta lata?"
    },
    {
        "name": "6. Pregunta sobre residuo no visible ('vidrio' no presente)",
        "detections": [
            {'class': 'plastico', 'score': 0.75, 'xyxy': [50, 50, 150, 200]},
            {'class': 'organico', 'score': 0.68, 'xyxy': [200, 200, 350, 400]}
        ],
        "query": "¿En qué caneca va el vidrio?"
    },
    {
        "name": "7. Pregunta sobre suciedad/limpieza condicional",
        "detections": [{'class': 'carton', 'score': 0.70, 'xyxy': [100, 100, 300, 300]}],
        "query": "¿Qué hago si el cartón está sucio?"
    },
    {
        "name": "8. Pregunta fuera de alcance",
        "detections": [{'class': 'plastico', 'score': 0.80, 'xyxy': [100, 100, 200, 200]}],
        "query": "¿Quién ganó el partido de ayer?"
    }
]

print("=== PRUEBAS DE RESPUESTA CONTEXTUAL DEL ASISTENTE V2 ===")
for sc in test_scenarios:
    print(f"\n--- {sc['name']} ---")
    print(f"Detecciones: {[d['class'] for d in sc['detections']]}")
    print(f"Consulta   : {sc['query']}")
    resp = ast.answer(sc['query'], sc['detections'])
    print(f"Intención  : {resp['intent']} (confianza: {resp['confidence']:.3f})")
    print(f"Respuesta  : {resp['text']}")
