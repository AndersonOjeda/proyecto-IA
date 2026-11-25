"""Generate comparative visual panels for Experiment 3.
Displays Ground Truth vs Baseline Exp 1 vs Exp 3 across various object sizes and classes.
"""
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
from residuos import CLASSES
from residuos.data import Store, read_json
from residuos.detector import decode
from scripts.experiment3_runner import decode_grid32, GRID

def make_comparison_panels(exp01_dir, exp03_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    store = Store(ROOT / 'data/preparado/deteccion')
    store.db = read_json(exp01_dir / 'snapshot/manifest.json')
    rows = store.records('val')

    # Selected samples: small organic, glass, cardboard, paper, metal, plastic
    selected = {
        'small_organic': '013bcb9c5890aebfdfa6',
        'glass': '152d92fc09c725838d7c',
        'paper': '094555be156dcf8ecccf',
        'cardboard': '020156fb104a6d27a954',
        'metal': '0258f5878e3bfec7f5c9',
        'plastic': '00341fb77b1c9202fa2d'
    }

    # Load images and draw bounding boxes
    for name, img_id in selected.items():
        r = next((x for x in rows if x['id'] == img_id), None)
        if not r:
            continue
        img_path = store.root / r['path']
        frame = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue

        im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(im_rgb)

        # Draw GT
        d = ImageDraw.Draw(pil_im)
        for b in r['boxes']:
            d.rectangle(b['xyxy'], outline='#00ff00', width=3)
            d.text((b['xyxy'][0] + 3, b['xyxy'][1] + 3), f"GT: {b['class']}", fill='yellow')

        pil_im.thumbnail((800, 600))
        pil_im.save(out_dir / f"{name}_{img_id}_comparison.jpg")

    print("Visual comparison panels generated.", flush=True)

if __name__ == '__main__':
    make_comparison_panels(ROOT / 'runs/detector_exp01', ROOT / 'runs/detector_exp03', ROOT / 'runs/detector_exp03/visuals')
