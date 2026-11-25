import json
from pathlib import Path
import sys
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from residuos.assistant import INTENTS, tokens, validate_corpus, train, evaluate, Assistant
from residuos.data import read_json, write_json

# 1. Definición del Corpus Enriquecido V2 con familias de paráfrasis sin fuga
# Mantenemos val (10) y test (20) idénticos al benchmark canónico para comparación rigurosa.
canonical = read_json(ROOT / 'data/intents.json')

val_split = canonical['val']
test_split = canonical['test']

# Enriquecemos train con preguntas variadas, expresiones naturales y modismos colombianos
# asegurando un balance de 25 preguntas por cada una de las 5 intenciones (125 preguntas en train)
train_data = [
    # --- INVENTARIO (25 preguntas) ---
    {"text": "¿Qué residuos ves?", "intent": "inventario"},
    {"text": "Dime los objetos detectados", "intent": "inventario"},
    {"text": "Cuántos residuos aparecen en la imagen", "intent": "inventario"},
    {"text": "Enumera los materiales de la escena", "intent": "inventario"},
    {"text": "Qué basura hay frente a la cámara", "intent": "inventario"},
    {"text": "Muéstrame el resultado de la detección", "intent": "inventario"},
    {"text": "Qué objetos reconociste en este video", "intent": "inventario"},
    {"text": "Lista lo que ves en la mesa", "intent": "inventario"},
    {"text": "Cuántas latas y botellas se ven", "intent": "inventario"},
    {"text": "Describe los residuos presentes", "intent": "inventario"},
    {"text": "Haz un conteo de la basura presente", "intent": "inventario"},
    {"text": "Qué elementos reciclables y desechos hay", "intent": "inventario"},
    {"text": "Dame el reporte de objetos en pantalla", "intent": "inventario"},
    {"text": "Cuántas cosas detectaste en total", "intent": "inventario"},
    {"text": "Qué residuos sólidos identificas en la toma", "intent": "inventario"},
    {"text": "Lista los materiales visibles en este cuadro", "intent": "inventario"},
    {"text": "Hay alguna basura reconocida en el suelo", "intent": "inventario"},
    {"text": "Dime el inventario completo de residuos", "intent": "inventario"},
    {"text": "Qué cosas hay botadas frente al lente", "intent": "inventario"},
    {"text": "Cuáles son los residuos que alcanzaste a ver", "intent": "inventario"},
    {"text": "Identifica y cuenta todo lo que hay en el video", "intent": "inventario"},
    {"text": "Qué tipos de desechos aparecen reconocidos", "intent": "inventario"},
    {"text": "Muestra el resumen de lo detectado aquí", "intent": "inventario"},
    {"text": "Qué materiales alcanzaste a clasificar en la escena", "intent": "inventario"},
    {"text": "Puedes decirme qué basura estás viendo", "intent": "inventario"},

    # --- SEPARAR (25 preguntas) ---
    # Criterio: Pregunta por acción de desecho, caneca, color o procedimiento de separación
    {"text": "¿Cómo separo esta lata?", "intent": "separar"},
    {"text": "Dónde pongo el plástico", "intent": "separar"},
    {"text": "Qué recipiente corresponde al vidrio", "intent": "separar"},
    {"text": "En qué caneca va el papel", "intent": "separar"},
    {"text": "Dónde debo depositar el cartón", "intent": "separar"},
    {"text": "Cómo reciclo el metal", "intent": "separar"},
    {"text": "¿Dónde lo boto?", "intent": "separar"},
    {"text": "Ayúdame a separar estos residuos", "intent": "separar"},
    {"text": "Qué color uso para este envase", "intent": "separar"},
    {"text": "Cómo clasifico esta cáscara para botarla", "intent": "separar"},
    {"text": "En cuál caneca tengo que echar la botella", "intent": "separar"},
    {"text": "A qué caneca va este plástico", "intent": "separar"},
    {"text": "Dónde tiro este frasco de vidrio", "intent": "separar"},
    {"text": "Dónde se bota una lata de gaseosa o pola", "intent": "separar"},
    {"text": "Cómo debo clasificar esta caja de cartón", "intent": "separar"},
    {"text": "En qué caneca echo las hojas de cuaderno", "intent": "separar"},
    {"text": "Dónde pongo los restos de comida y cáscaras", "intent": "separar"},
    {"text": "A cuál recipiente corresponde esta chuspa o bolsa", "intent": "separar"},
    {"text": "Cuál es la caneca adecuada para el aluminio", "intent": "separar"},
    {"text": "Cómo separo los residuos según el código de colores", "intent": "separar"},
    {"text": "Dónde desecho este envase de plástico", "intent": "separar"},
    {"text": "En qué caneca blanca negra o verde debo botar esto", "intent": "separar"},
    {"text": "A dónde mando este tarro para reciclar", "intent": "separar"},
    {"text": "Dónde boto la cáscara de banano o plátano", "intent": "separar"},
    {"text": "Cuál es el depósito correcto para este material", "intent": "separar"},

    # --- ORGANICOS (25 preguntas) ---
    # Criterio: Pregunta por presencia o existencia de restos orgánicos/comida en la escena
    {"text": "¿Hay algo orgánico?", "intent": "organicos"},
    {"text": "Ves restos de comida", "intent": "organicos"},
    {"text": "Cuántos orgánicos detectaste", "intent": "organicos"},
    {"text": "Existe basura orgánica en la escena", "intent": "organicos"},
    {"text": "Identificaste cáscaras de fruta", "intent": "organicos"},
    {"text": "Hay alimentos entre los residuos", "intent": "organicos"},
    {"text": "Se observan desechos orgánicos", "intent": "organicos"},
    {"text": "Qué restos podrían ir al compostaje", "intent": "organicos"},
    {"text": "Aparecen residuos de comida en el video", "intent": "organicos"},
    {"text": "La cámara encontró algo orgánico", "intent": "organicos"},
    {"text": "Detectaste sobras de comida o almuerzo", "intent": "organicos"},
    {"text": "Hay cáscaras o residuos de frutas y verduras", "intent": "organicos"},
    {"text": "Aparece algún desecho biológico en la toma", "intent": "organicos"},
    {"text": "Reconociste residuos biodegradables en la imagen", "intent": "organicos"},
    {"text": "Ves conchos de café o restos vegetales", "intent": "organicos"},
    {"text": "La cámara detectó restos de comida orgánica", "intent": "organicos"},
    {"text": "Hay basura verde o compostable presente", "intent": "organicos"},
    {"text": "Encontraste desechos de alimentos en la mesa", "intent": "organicos"},
    {"text": "Se identificó alguna fruta descompuesta o cáscara", "intent": "organicos"},
    {"text": "Cuántos residuos orgánicos vio el modelo", "intent": "organicos"},
    {"text": "Hay materia orgánica visible entre los objetos", "intent": "organicos"},
    {"text": "Ves alguna cáscara de huevo o fruta en el cuadro", "intent": "organicos"},
    {"text": "Aparece algún desecho de origen vegetal o animal", "intent": "organicos"},
    {"text": "Dime si la cámara captó restos orgánicos", "intent": "organicos"},
    {"text": "Se alcanzan a distinguir restos de comida en el plano", "intent": "organicos"},

    # --- SUCIEDAD (25 preguntas) ---
    # Criterio: Pregunta sobre estado de contaminación, grasa, aceite, salsa o lavado previo
    {"text": "¿Qué hago si el cartón está sucio?", "intent": "suciedad"},
    {"text": "El papel tiene grasa dónde va", "intent": "suciedad"},
    {"text": "Debo lavar esta botella antes de separarla", "intent": "suciedad"},
    {"text": "Esta lata está contaminada qué hago", "intent": "suciedad"},
    {"text": "Puedo reciclar cartón mojado", "intent": "suciedad"},
    {"text": "El envase tiene restos de comida", "intent": "suciedad"},
    {"text": "Cómo separo papel sucio", "intent": "suciedad"},
    {"text": "El vidrio debe estar limpio y seco", "intent": "suciedad"},
    {"text": "Tengo plástico contaminado con aceite", "intent": "suciedad"},
    {"text": "Qué pasa si la caja tiene salsa", "intent": "suciedad"},
    {"text": "La botella de jugo quedó untada por dentro hay que enjuagarla", "intent": "suciedad"},
    {"text": "El papel higiénico o servilleta usada se puede reciclar", "intent": "suciedad"},
    {"text": "Tengo una caja de pizza llena de queso y grasa", "intent": "suciedad"},
    {"text": "Si el envase plástico tiene comida pegada se va a la negra", "intent": "suciedad"},
    {"text": "El cartón se mojó con agua de lluvia todavía sirve", "intent": "suciedad"},
    {"text": "Es obligatorio enjuagar las latas antes de botarlas", "intent": "suciedad"},
    {"text": "El vaso desechable tiene restos de café con leche", "intent": "suciedad"},
    {"text": "Qué hago con este recipiente sucio y grasoso", "intent": "suciedad"},
    {"text": "Un plástico manchado con pintura o químicos se recicla", "intent": "suciedad"},
    {"text": "Cómo manejo residuos que están impregnados de comida", "intent": "suciedad"},
    {"text": "Si no lavo la botella me la rechazan en la caneca blanca", "intent": "suciedad"},
    {"text": "El papel con manchas de grasa va a la caneca negra", "intent": "suciedad"},
    {"text": "Qué tratamiento le doy a un envase sucio antes de separarlo", "intent": "suciedad"},
    {"text": "El cartón con mugre o tierra todavía se puede aprovechar", "intent": "suciedad"},
    {"text": "Debo secar los materiales reciclables antes de echarlos", "intent": "suciedad"},

    # --- FUERA DE ALCANCE (25 preguntas) ---
    # Criterio: Consultas no relacionadas con reciclaje, residuos ni su manejo
    {"text": "Cuál es el clima mañana", "intent": "fuera_alcance"},
    {"text": "Resuelve esta ecuación", "intent": "fuera_alcance"},
    {"text": "Quién ganó el partido", "intent": "fuera_alcance"},
    {"text": "Cuéntame un chiste", "intent": "fuera_alcance"},
    {"text": "Necesito una receta de cocina", "intent": "fuera_alcance"},
    {"text": "Qué medicamento debo tomar", "intent": "fuera_alcance"},
    {"text": "Cuánto cuesta un computador", "intent": "fuera_alcance"},
    {"text": "Abre mi correo electrónico", "intent": "fuera_alcance"},
    {"text": "Hola cómo estás", "intent": "fuera_alcance"},
    {"text": "Qué hora es", "intent": "fuera_alcance"},
    {"text": "Cómo se llega a Pasto", "intent": "fuera_alcance"},
    {"text": "Cuál es la capital de Colombia", "intent": "fuera_alcance"},
    {"text": "Explica la teoría de la relatividad", "intent": "fuera_alcance"},
    {"text": "Pon una canción de salsa", "intent": "fuera_alcance"},
    {"text": "Quién escribió Cien años de soledad", "intent": "fuera_alcance"},
    {"text": "Cuánto vale un pasaje de bus", "intent": "fuera_alcance"},
    {"text": "Dame el pronóstico del tiempo para hoy", "intent": "fuera_alcance"},
    {"text": "Escribe un poema sobre la vida", "intent": "fuera_alcance"},
    {"text": "Qué películas están dando en cine", "intent": "fuera_alcance"},
    {"text": "Cuánto es veinticinco por cuatro", "intent": "fuera_alcance"},
    {"text": "Cómo preparo un café con leche", "intent": "fuera_alcance"},
    {"text": "Cuál es el mejor celular del mercado", "intent": "fuera_alcance"},
    {"text": "Buenos días asistente", "intent": "fuera_alcance"},
    {"text": "Traduce esta frase al inglés", "intent": "fuera_alcance"},
    {"text": "Qué tareas tengo pendientes para hoy", "intent": "fuera_alcance"}
]

corpus_v2 = {
    "provenance": "Corpus enriquecido V2 con 25 preguntas por intención en train (125 en total) en español colombiano, agrupadas por familias de paráfrasis sin fuga hacia val (10) ni test (20).",
    "disambiguation_policy": {
        "separar_vs_organicos": "Si la consulta pregunta por la acción o caneca donde desechar ('dónde boto', 'en qué caneca va'), la intención es 'separar' (incluso si nombra cáscaras). Si pregunta por la presencia o existencia en la escena ('¿hay algo orgánico?', '¿ves restos de comida?'), la intención es 'organicos'.",
        "separar_vs_suciedad": "Si la consulta indaga sobre contaminación, grasa, aceite, comida pegada o necesidad de lavado previo, la intención es 'suciedad'. Si solo consulta la caneca o procedimiento de separación estándar de un material, es 'separar'."
    },
    "train": train_data,
    "val": val_split,
    "test": test_split
}

corpus_v2_path = ROOT / 'data/intents_v2.json'
write_json(corpus_v2_path, corpus_v2)
print(f"Corpus V2 guardado en: {corpus_v2_path}")

# Validar corpus contra colisiones y requisitos
validate_corpus(corpus_v2)
print(f"[OK] Corpus V2 validado sin colisiones de texto ni fuga de tokens.")
print(f"  Train: {len(train_data)} preguntas ({len(train_data)//5} por clase)")
print(f"  Val: {len(val_split)} preguntas ({len(val_split)//5} por clase)")
print(f"  Test: {len(test_split)} preguntas ({len(test_split)//5} por clase)")

# 2. Entrenar nuevo modelo desde cero en runs/assistant_v2
print("\n" + "="*75)
print("ENTRENANDO INTENT-TRANSFORMER V2 DESDE CERO (SIN PESOS EXTERNOS)")
print("="*75)
out_dir = ROOT / 'runs/assistant_v2'
history = train(corpus=str(corpus_v2_path), output=str(out_dir), epochs=100, seed=42)
print("Entrenamiento completado exitosamente.")

# 3. Evaluación comparativa: V1 vs V2 sobre la MISMA validación y el MISMO test
print("\n" + "="*75)
print("COMPARATIVA DE EXACTITUD Y F1: ASISTENTE V1 vs ASISTENTE V2")
print("="*75)

def evaluate_assistant_model(cp_path, split_name='val'):
    ast = Assistant(str(cp_path))
    data = read_json(corpus_v2_path)[split_name]
    matrix = [[0]*len(INTENTS) for _ in INTENTS]
    correct = 0
    results = []
    for r in data:
        pred, conf = ast.recognize(r['text'])
        t_idx = INTENTS.index(r['intent'])
        p_idx = INTENTS.index(pred)
        matrix[t_idx][p_idx] += 1
        is_ok = (pred == r['intent'])
        if is_ok:
            correct += 1
        results.append({'text': r['text'], 'intent': r['intent'], 'pred': pred, 'conf': conf, 'ok': is_ok})
    acc = correct / len(data)
    
    # Calcular macro F1
    f1s = []
    per_class = {}
    for i, c in enumerate(INTENTS):
        tp = matrix[i][i]
        fp = sum(matrix[k][i] for k in range(len(INTENTS)) if k != i)
        fn = sum(matrix[i][k] for k in range(len(INTENTS)) if k != i)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1s.append(f1)
        per_class[c] = {'tp': tp, 'fp': fp, 'fn': fn, 'precision': p, 'recall': r, 'f1': f1}
    macro_f1 = sum(f1s) / len(f1s)
    return {'acc': acc, 'macro_f1': macro_f1, 'matrix': matrix, 'per_class': per_class, 'results': results}

v1_val = evaluate_assistant_model(ROOT / 'runs/assistant/best.pt', 'val')
v2_val = evaluate_assistant_model(ROOT / 'runs/assistant_v2/best.pt', 'val')

v1_test = evaluate_assistant_model(ROOT / 'runs/assistant/best.pt', 'test')
v2_test = evaluate_assistant_model(ROOT / 'runs/assistant_v2/best.pt', 'test')

print("\n--- RESULTADOS EN VALIDACIÓN (10 preguntas canónicas) ---")
print(f"V1 (Anterior): Exactitud = {v1_val['acc']*100:.1f}%, Macro-F1 = {v1_val['macro_f1']*100:.1f}%")
print(f"V2 (Nuevo)   : Exactitud = {v2_val['acc']*100:.1f}%, Macro-F1 = {v2_val['macro_f1']*100:.1f}%")

print("\n--- RESULTADOS EN TEST HISTÓRICO (20 preguntas canónicas) ---")
print(f"V1 (Anterior): Exactitud = {v1_test['acc']*100:.1f}% ({sum(r['ok'] for r in v1_test['results'])}/20), Macro-F1 = {v1_test['macro_f1']*100:.1f}%")
print(f"V2 (Nuevo)   : Exactitud = {v2_test['acc']*100:.1f}% ({sum(r['ok'] for r in v2_test['results'])}/20), Macro-F1 = {v2_test['macro_f1']*100:.1f}%")

print("\nDesglose por Intención en Test:")
print(f"{'Intención':<14} | {'V1 Acc':>10} | {'V2 Acc':>10} | {'V1 F1':>10} | {'V2 F1':>10}")
print("-" * 65)
for c in INTENTS:
    v1_c = v1_test['per_class'][c]
    v2_c = v2_test['per_class'][c]
    print(f"{c:<14} | {v1_c['tp']}/4 ({v1_c['recall']*100:.0f}%) | {v2_c['tp']}/4 ({v2_c['recall']*100:.0f}%) | {v1_c['f1']*100:>9.1f}% | {v2_c['f1']*100:>9.1f}%")

# Guardar informe comparativo
comp_report = {
    'validation_canonical': {'v1': v1_val, 'v2': v2_val},
    'test_canonical': {'v1': v1_test, 'v2': v2_test}
}
write_json(out_dir / 'comparativa_v1_v2.json', comp_report)
