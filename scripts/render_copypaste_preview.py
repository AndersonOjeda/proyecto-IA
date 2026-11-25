"""Render visual verification panel of Verified Polygon Copy-Paste onto TACO oficial training backgrounds."""
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from residuos.data import Store

EXP04_DIR = ROOT / 'runs/detector_exp04'
MINORITY_4 = ('vidrio', 'papel', 'carton', 'metal')

def main():
    store = Store(ROOT / 'data/preparado/deteccion')
    orig_train = store.records('train')
    orig_train_pixels = np.load(ROOT / 'runs/detector_exp01/train.npy', mmap_mode='r')
    cutout_bank = torch.load(EXP04_DIR / 'verified_polygon_bank_4c.pt', weights_only=False)

    canvas = Image.new('RGB', (1600, 920), '#101d24')
    draw = ImageDraw.Draw(canvas)
    font_title = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 24)
    font_sub = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
    draw.text((20, 12), 'Validación Visual Fase 3 (Exp 4C) — Copy-Paste Poligonal Verificado sobre Fondos TACO Train', font=font_title, fill='white')

    sample_bg_indices = [i for i, r in enumerate(orig_train) if r['source'] == 'TACO oficial' and 1 <= len(r['boxes']) <= 2][:8]
    cutouts_by_c = defaultdict(list)
    for item in cutout_bank:
        cutouts_by_c[item['class']].append(item)

    rng = np.random.RandomState(42)
    for panel_idx, bg_i in enumerate(sample_bg_indices):
        r_bg = orig_train[bg_i]
        frame = np.array(orig_train_pixels[bg_i], copy=True)
        w_orig, h_orig = r_bg['width'], r_bg['height']
        orig_boxes_256 = [
            {'class': b['class'], 'xyxy': [b['xyxy'][0]*256/w_orig, b['xyxy'][1]*256/h_orig, b['xyxy'][2]*256/w_orig, b['xyxy'][3]*256/h_orig], 'kind': 'orig'}
            for b in r_bg['boxes']
        ]
        target_c = MINORITY_4[panel_idx % 4]
        # Show both TACO train polygon cutouts and External TrashNet polygon cutouts
        idx_c = (panel_idx * 17) % len(cutouts_by_c[target_c])
        item = cutouts_by_c[target_c][idx_c]
        rgb_p, alpha_p = item['rgb'].copy(), item['alpha'].copy()
        target_sz = rng.randint(38, 68)
        ph, pw = rgb_p.shape[:2]
        sc = target_sz / max(ph, pw)
        nw, nh = max(14, int(round(pw * sc))), max(14, int(round(ph * sc)))
        rgb_s = cv2.resize(rgb_p, (nw, nh), interpolation=cv2.INTER_LINEAR)
        alpha_s = cv2.resize(alpha_p, (nw, nh), interpolation=cv2.INTER_LINEAR)

        for _ in range(50):
            x0 = rng.randint(12, 256 - nw - 12)
            y0 = rng.randint(64, 256 - nh - 12)
            cand_xy = [x0, y0, x0 + nw, y0 + nh]
            bg_patch = frame[y0:y0+nh, x0:x0+nw].astype(np.float32)
            if bg_patch.mean(axis=(1, 2)).min() < 28.0 or bg_patch.mean(axis=(0, 2)).min() < 28.0 or bg_patch.std() < 10.0:
                continue
            if any(max(cand_xy[0], ob['xyxy'][0]-6) < min(cand_xy[2], ob['xyxy'][2]+6) and max(cand_xy[1], ob['xyxy'][1]-6) < min(cand_xy[3], ob['xyxy'][3]+6) for ob in orig_boxes_256):
                continue
            fg_patch = rgb_s.astype(np.float32)
            bg_lum = max(25.0, float(bg_patch.mean()))
            fg_lum = max(25.0, float(fg_patch[alpha_s > 127].mean())) if (alpha_s > 127).any() else 128.0
            lum_factor = np.clip((bg_lum / fg_lum) * 0.40 + 0.60, 0.78, 1.22)
            fg_harmonized = np.clip(fg_patch * lum_factor, 0, 255)
            a_norm = (alpha_s.astype(np.float32) / 255.0)[..., None]
            frame[y0:y0+nh, x0:x0+nw] = (a_norm * fg_harmonized + (1.0 - a_norm) * bg_patch).astype(np.uint8)
            ys_m, xs_m = np.nonzero(alpha_s > 32)
            orig_boxes_256.append({
                'class': target_c,
                'xyxy': [float(x0+xs_m.min()), float(y0+ys_m.min()), float(x0+xs_m.max()+1), float(y0+ys_m.max()+1)],
                'kind': 'PolyPaste',
                'origin': item['origin']
            })
            break

        pil_im = Image.fromarray(frame).resize((360, 360), Image.Resampling.BILINEAR)
        x_corner, y_corner = 20 + (panel_idx % 4) * 395, 55 + (panel_idx // 4) * 425
        draw.text((x_corner, y_corner), f"Paste: {target_c.upper()} ({item['origin'][:18]})", font=font_sub, fill='#4ee6b1')
        canvas.paste(pil_im, (x_corner, y_corner + 24))
        for b in orig_boxes_256:
            x1, y1, x2, y2 = [v * (360.0 / 256.0) for v in b['xyxy']]
            coords = [x_corner + x1, y_corner + 24 + y1, x_corner + x2, y_corner + 24 + y2]
            col = '#00ff66' if b['kind'] == 'orig' else '#00d8ff'
            draw.rectangle(coords, outline=col, width=2)
            draw.text((coords[0] + 2, max(y_corner + 24, coords[1] - 16)), f"{b['kind']}: {b['class']}", font=font_sub, fill=col)
        draw.text((x_corner, y_corner + 390), f"Fondo Train: {r_bg['source']} ({r_bg['id'][:10]})", font=font_sub, fill='#c5d5dc')

    out_p = EXP04_DIR / 'audit/vista_previa_copypaste_4c.jpg'
    canvas.save(out_p, quality=93)
    print(f"Saved updated preview to {out_p}", flush=True)

if __name__ == '__main__':
    main()
