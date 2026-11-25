# Protocolo de Evaluación Humana y Pruebas a Ciegas con Compañeros

**Asignatura:** Inteligencia Artificial (Proyecto 4)  
**Institución:** Universidad Cooperativa de Colombia (UCC), Campus Pasto  
**Docente:** Ing. Cristian Camilo Ordoñez Quintero  
**Estudiante:** Anderson Ojeda  

---

## 1. Objetivo y Distinción Metodológica

Para mantener la máxima transparencia y rigor científico exigido por la asignatura, se distingue estrictamente entre:

1. **Pruebas Automatizadas Ejecutadas (Auditadas en el Informe):**
   - Validación interna sobre 10 preguntas canónicas (`accuracy = 100%`, `Macro-F1 = 100%`).
   - Test histórico congelado sobre 20 preguntas fijas (`accuracy = 75%`, `Macro-F1 = 71.8%`).
   - Pruebas sintéticas y de video real sobre 8 escenarios de interacción contextual.

2. **Evaluación Humana a Ciegas con Compañeros (Pendiente / En Vivo):**
   - Recolección de consultas en lenguaje natural redactadas por compañeros de clase durante la sustentación o actividades de laboratorio.
   - El objetivo es evaluar la robustez del modelo ante modismos, jerga local nariñense/colombiana y estructuras sintácticas no vistas en el entrenamiento.

---

## 2. Formato de Recolección de Datos

Las consultas se registran formalmente en el archivo [`data/formato_evaluacion_humana.json`](file:///c:/Users/Anderson%20Ojeda/Downloads/IA/proyecto%20final/data/formato_evaluacion_humana.json).

### Estructura de Registro por Consulta:
```json
{
  "id": "peer_001",
  "evaluador": "Nombre del Compañero",
  "fecha": "AAAA-MM-DD",
  "consulta": "Texto exacto de la pregunta formulada",
  "intencion_esperada": "separar | organicos | suciedad | inventario | fuera_alcance",
  "objetos_en_escena": ["material1", "material2"],
  "respuesta_esperada_humana": "Criterio de respuesta aceptable según el compañero",
  "resultado_modelo": {
    "ejecutado": false,
    "intencion_predicha": null,
    "confianza": null,
    "acierto_intencion": null,
    "respuesta_asistente": null,
    "claridad_calificacion_1_a_5": null,
    "observaciones": ""
  }
}
```

---

## 3. Criterios de Desambiguación Operativa

Para evitar sesgos y subjetividad al clasificar la intención esperada, se aplican las siguientes reglas:

1. **Acción o Destino (`separar`):**  
   Toda pregunta que indague *dónde*, *cómo*, o *en qué caneca/bolsa* botar o reciclar un residuo.  
   *Ejemplo:* «¿Dónde boto esta cáscara?», «¿Qué hago con este vaso de café?».

2. **Presencia de Materia Orgánica (`organicos`):**  
   Preguntas que indagan si hay comida, cáscaras, restos o material biodegradable presente en la escena.  
   *Ejemplo:* «¿Hay algo orgánico en la mesa?», «¿Ves restos de fruta?».

3. **Condición de Limpieza y Grasa (`suciedad`):**  
   Preguntas sobre la condición física del objeto (si está sucio, engrasado, manchado o si se debe lavar antes de reciclar).  
   *Ejemplo:* «¿Tengo que lavar la botella?», «El cartón tiene grasa de pizza, ¿sirve?».

4. **Recuento o Reconocimiento (`inventario`):**  
   Preguntas orientadas a saber cuántos o cuáles residuos detecta el sistema.  
   *Ejemplo:* «¿Qué basuras ves?», «¿Cuántos objetos hay en el fotograma?».

5. **Fuera de Alcance (`fuera_alcance`):**  
   Consultas no relacionadas con reciclaje, materiales o la escena actual.  
   *Ejemplo:* «¿Cuál es el clima hoy?», «¿Qué es la inteligencia artificial?».

---

## 4. Script de Evaluación Rápida en Vivo

Durante la demostración de 10 minutos, para procesar interactivamente las preguntas de los compañeros o del profesor:

```powershell
.\.venv\Scripts\python.exe scripts/test_assistant_scenarios.py
```
O directamente desde la interfaz gráfica:
```powershell
.\.venv\Scripts\python.exe app/main.py --mode precision
```
Escribiendo la consulta en la caja de texto inferior de la ventana.
