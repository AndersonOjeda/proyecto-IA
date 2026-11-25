"""Lanzador directo del editor de cajas para el lote manual de 90 fotos."""
import tkinter as tk
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from residuos.annotate import Annotator
from residuos import LABELS, CLASSES

def main():
    root = tk.Tk()
    app = Annotator(root, data='data/preparado/deteccion')
    
    # Pre-cargar automáticamente la primera imagen pendiente del lote manual
    manual_pending = [r for r in app.store.db['images'] if r.get('manual_batch') and not r.get('reviewed')]
    
    if manual_pending:
        first = manual_pending[0]
        app.open_record(first)
        hint = first.get('target_hint')
        if hint in CLASSES:
            app.class_name.set(LABELS[CLASSES.index(hint)])
        app.status.set(f"LOTE MANUAL ({len(manual_pending)} pendientes) | ID: {first['id']} | Clase sugerida: {hint}")
    else:
        app.status.set("¡Todas las fotos del lote manual han sido revisadas!")
        
    root.mainloop()

if __name__ == '__main__':
    main()
