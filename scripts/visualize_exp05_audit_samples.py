"""Generate visual audit sample panels for accepted SALSA and TACO unofficial images.
Renders bounding boxes with labels to visually verify annotation quality before training.
"""
from pathlib import Path
import json
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'runs/detector_exp05'

CLASS_COLORS = {
    'plastico': (255, 100, 0),    # Blue-ish orange
    'papel': (0, 200, 255),       # Yellow
    'carton': (0, 140, 255),      # Orange
    'vidrio': (255, 0, 255),      # Magenta
    'metal': (0, 255, 255),       # Cyan-yellow
    'organico': (0, 255, 0)       # Green
}

def render_sample(img_record):
    img_path = ROOT / img_record['path']
    im = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        return None
    h, w = im.shape[:2]
    canvas = im.copy()
    
    for b in img_record['boxes']:
        c = b['class']
        col = CLASS_COLORS.get(c, (0, 255, 0))
        x1, y1, x2, y2 = [int(v) for v in b['xyxy']]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), col, max(2, w // 250))
        label = f"{c} ({b.get('original_class', '')[:10]})"
        cv2.putText(canvas, label, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, max(0.4, w / 1000), col, 2)
    
    # Resize to standard panel size
    panel = cv2.resize(canvas, (360, 360))
    header = np.zeros((28, 360, 3), dtype=np.uint8)
    src = img_record['source'][:15]
    cid = img_record['id']
    cv2.putText(header, f"{src} | {cid}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return np.vstack([header, panel])

def main():
    manifest_p = OUT_DIR / 'manifiesto_descargas.json'
    if not manifest_p.exists():
        print("Manifest not ready yet.")
        return
    
    with open(manifest_p, encoding='utf-8') as f:
        data = json.load(f)
    
    accepted = data['accepted_images']
    # Select 2 representative images per class from each source
    selected = []
    classes = ['vidrio', 'metal', 'papel', 'carton', 'plastico', 'organico']
    for c in classes:
        for src in ['SALSA OpenLitterMap', 'TACO no oficial']:
            match = [img for img in accepted if img['source'] == src and any(b['class'] == c for b in img['boxes'])]
            if match:
                selected.append(match[0])
            if len(match) > 1:
                selected.append(match[1])

    panels = []
    for s in selected[:16]:
        p = render_sample(s)
        if p is not None:
            panels.append(p)

    if not panels:
        print("No panels could be rendered.")
        return

    # Arrange into grid of 4x4
    while len(panels) < 16:
        panels.append(np.zeros_like(panels[0]))

    row1 = np.hstack(panels[0:4])
    row2 = np.hstack(panels[4:8])
    row3 = np.hstack(panels[8:12])
    row4 = np.hstack(panels[12:16])
    grid = np.vstack([row1, row2, row3, row4])

    out_p = OUT_DIR / 'muestras_auditoria_visual.jpg'
    cv2.imwrite(str(out_p), grid)
    print(f"Panel de muestras visuales guardado en: {out_p}")

if __name__ == '__main__':
    main()
