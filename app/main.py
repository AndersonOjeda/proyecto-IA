"""Punto de entrada principal para la aplicación gráfica de detección de residuos."""
import argparse
import sys
from pathlib import Path
import tkinter as tk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from residuos.app import Application

def main():
    parser = argparse.ArgumentParser(description="Aplicación local de detección y separación de residuos.")
    parser.add_argument('--data', default='data/preparado/deteccion', help='Ruta al directorio de datos')
    parser.add_argument('--detector', default='runs/detector/best.pt', help='Ruta al checkpoint del detector (por defecto Experimento 5 canónico)')
    parser.add_argument('--assistant', default='runs/assistant/best.pt', help='Ruta al transformer del asistente')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--video', help='Video que se abre al iniciar; sin esta opción se muestra el selector en la app')
    parser.add_argument('--mode', default='cobertura', choices=['cobertura', 'precision'],
                        help='Modo operativo: cobertura (th=0.35, min_size=0) o precision (th=0.40, min_size=32px)')
    parser.add_argument('--min-box-size', type=int, default=None,
                        help='Filtro de tamaño mínimo de caja en píxeles (ancho y alto >= min_box_size)')
    args = parser.parse_args()
    for name in ('data', 'detector', 'assistant', 'video'):
        value = getattr(args, name)
        if value:
            path = Path(value)
            setattr(args, name, str(path.resolve() if path.is_absolute() else (ROOT/path).resolve()))
    
    root = tk.Tk()
    app = Application(
        root=root,
        data=args.data,
        detector=args.detector,
        assistant=args.assistant,
        device=args.device,
        mode=args.mode,
        min_box_size=args.min_box_size
    )
    if args.video:
        root.after(100, lambda: app.open_source(args.video))
    root.mainloop()

if __name__ == '__main__':
    main()
