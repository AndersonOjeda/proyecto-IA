"""Experiment 7 Runner: DetectorS8 Super-Unificado 100% desde cero.
Combina:
1. Base Train original: 3.645 imágenes
2. Ampliación Exterior SALSA + TACO: 1.505 imágenes
3. Lote Manual Humano revisado: 60 imágenes (vidrio, cartón, metal, papel)
4. DATOS_SEPARADOS limpios con escudo anti-fuga: 740 imágenes
Total entrenamiento: 5.950 imágenes reales.
Evaluación: sobre las 670 imágenes canónicas de validación bajo protocolo idéntico.
CERO modelos o pesos preentrenados (100% random weights).
"""
import os
import sys
import time
import json
import random
import hashlib
from pathlib import Path
from copy import deepcopy
from collections import Counter, defaultdict
from itertools import combinations

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from residuos import CLASSES
from residuos.data import Store
from scripts.experiment3_runner import (
    DetectorS8, CachedDataset32, encode_grid32, decode_grid32,
    GRID, SLOTS, seed_all, device_for
)
from scripts.experiment3c_runner import loss_fn_3c, CLASS_WEIGHTS_3C

EXP07_DIR = ROOT / 'runs/detector_exp07'
EXP07_DIR.mkdir(parents=True, exist_ok=True)


class ModelEMA:
    """Exponential Moving Average de los parámetros del modelo."""
    def __init__(self, model, decay=0.996):
        self.ema = deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(v * d + msd[k] * (1.0 - d))


def normalizar_ruta(p):
    """Normaliza una ruta de imagen para comparaciones anti-fuga."""
    if not p:
        return ''
    return os.path.normpath(str(p).strip()).replace('\\', '/').lower()


def extraer_identificadores(r):
    """Extrae rutas normalizadas declaradas en el registro (path, rel_path, image_path)."""
    paths = set()
    for field in ('path', 'rel_path', 'image_path'):
        val = r.get(field)
        if val:
            paths.add(normalizar_ruta(val))
    return paths


def validar_sin_fugas(orig_val, orig_train, ext_rows, manual_train, sep_rows):
    """Escudo anti-fuga en runtime:
    - Normaliza rutas de imagen (path, rel_path, image_path) y revisa hashes sha256.
    - Lanza RuntimeError si existe cualquier intersección entre orig_val y
      cualquiera de los 4 conjuntos de entrenamiento.
    - Retorna el dict leak_audit con los conteos y cualquier solapamiento detectado
      entre fuentes de entrenamiento sin detener el flujo salvo fuga hacia validación.
    """
    val_paths = set()
    val_hashes = {}
    for i, r in enumerate(orig_val):
        for p in extraer_identificadores(r):
            val_paths.add(p)
        sha = r.get('sha256')
        if sha:
            val_hashes[sha] = r.get('id', i)

    train_groups = {
        'orig_train': orig_train,
        'ext_rows': ext_rows,
        'manual_train': manual_train,
        'sep_rows': sep_rows
    }

    # 1. Comprobación estricta de fugas hacia validación canónica
    val_leaks = {}
    for name, rows in train_groups.items():
        leaks = []
        for r in rows:
            r_paths = extraer_identificadores(r)
            inter = r_paths.intersection(val_paths)
            if inter:
                leaks.append({
                    'reason': 'path_match',
                    'path': list(inter)[0],
                    'record_id': r.get('id', r.get('path'))
                })
            sha = r.get('sha256')
            if sha and sha in val_hashes:
                leaks.append({
                    'reason': 'sha256_match',
                    'sha256': sha,
                    'record_id': r.get('id', r.get('path')),
                    'val_id': val_hashes[sha]
                })
        if leaks:
            val_leaks[name] = leaks

    if val_leaks:
        detalles = []
        for name, leaks in val_leaks.items():
            detalles.append(f"  - {name}: {len(leaks)} filtraciones hacia validación (ej: {leaks[0]})")
        msg = (
            "RuntimeError: ¡ESCUDO ANTI-FUGA ACTIVADO! Se detectó fuga de datos de validación canónica:\n"
            + "\n".join(detalles)
        )
        raise RuntimeError(msg)

    # 2. Registrar solapamientos entre fuentes de entrenamiento sin detener el flujo
    train_overlaps = {}
    for (name1, rows1), (name2, rows2) in combinations(train_groups.items(), 2):
        paths1 = {p for r in rows1 for p in extraer_identificadores(r)}
        paths2 = {p for r in rows2 for p in extraer_identificadores(r)}
        inter_paths = list(paths1.intersection(paths2))

        hashes1 = {r['sha256'] for r in rows1 if r.get('sha256')}
        hashes2 = {r['sha256'] for r in rows2 if r.get('sha256')}
        inter_hashes = list(hashes1.intersection(hashes2))

        key = f"{name1}_vs_{name2}"
        train_overlaps[key] = {
            'path_overlaps': len(inter_paths),
            'sha256_overlaps': len(inter_hashes),
            'sample_path_overlaps': inter_paths[:5] if inter_paths else []
        }

    leak_audit = {
        'status': 'PASSED_CLEAN',
        'val_leakage_detected': False,
        'counts': {
            'orig_val': len(orig_val),
            'orig_train': len(orig_train),
            'ext_rows': len(ext_rows),
            'manual_train': len(manual_train),
            'sep_rows': len(sep_rows),
            'total_train': len(orig_train) + len(ext_rows) + len(manual_train) + len(sep_rows)
        },
        'validation_leakage_audit': {
            f"val_vs_{name}": 0 for name in train_groups
        },
        'train_overlaps': train_overlaps
    }
    print(f"[Escudo Anti-Fuga] Validación de integridad completada OK: 0 fugas hacia el split canónico.")
    return leak_audit


def sanear_cajas(boxes, width, height):
    """Saneamiento y chequeo de coordenadas:
    - Copia de forma segura cada caja sin mutar las estructuras originales.
    - Asegura nx1 = min(x1, x2), nx2 = max(x1, x2), ny1 = min(y1, y2), ny2 = max(y1, y2).
    - Recorta coordenadas dentro de [0, width] y [0, height].
    - Descarta cajas degeneradas (x2 <= x1 o y2 <= y1).
    """
    clean_boxes = []
    w_float = float(width)
    h_float = float(height)
    for b in boxes:
        x1, y1, x2, y2 = b['xyxy']
        nx1, nx2 = min(float(x1), float(x2)), max(float(x1), float(x2))
        ny1, ny2 = min(float(y1), float(y2)), max(float(y1), float(y2))
        cx1 = max(0.0, min(nx1, w_float))
        cy1 = max(0.0, min(ny1, h_float))
        cx2 = max(0.0, min(nx2, w_float))
        cy2 = max(0.0, min(ny2, h_float))
        if cx2 > cx1 and cy2 > cy1:
            b_clean = dict(b)
            b_clean['xyxy'] = [cx1, cy1, cx2, cy2]
            clean_boxes.append(b_clean)
    return clean_boxes


class DatasetExp07(Dataset):
    """Dataset integrado para Exp07: 3.645 base + 1.505 ext + 60 manuales + 740 datos_separados."""
    def __init__(self, orig_rows, orig_npy_path, ext_rows, ext_npy_path,
                 manual_rows, data_root, sep_rows, sep_npy_path, augment=True):
        self.orig_rows = orig_rows
        self.ext_rows = ext_rows
        self.manual_rows = manual_rows
        self.sep_rows = sep_rows

        self.orig_pixels = np.load(orig_npy_path, mmap_mode='r')
        self.ext_pixels = np.load(ext_npy_path, mmap_mode='r')
        self.sep_pixels = np.load(sep_npy_path, mmap_mode='r')
        self.augment = augment

        # Pre-cargar en memoria RAM las 60 imágenes manuales
        print(f"Pre-cargando {len(manual_rows)} imágenes manuales humanas en RAM...")
        manual_list = []
        for r in manual_rows:
            p = data_root / r['path']
            img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"No se pudo leer la imagen: {p}")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_res = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
            manual_list.append(img_res)
        self.manual_pixels = np.stack(manual_list, axis=0)  # [N, 256, 256, 3]

        self.samples = (
            [('orig', i) for i in range(len(self.orig_rows))] +
            [('ext', j) for j in range(len(self.ext_rows))] +
            [('manual', k) for k in range(len(self.manual_rows))] +
            [('sep', m) for m in range(len(self.sep_rows))]
        )
        print(f"DatasetExp07: Total muestras = {len(self.samples)} ({len(orig_rows)} base + {len(ext_rows)} ext + {len(manual_rows)} manuales + {len(sep_rows)} datos_separados)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        kind, real_idx = self.samples[idx]
        if kind == 'orig':
            r = self.orig_rows[real_idx]
            frame = np.array(self.orig_pixels[real_idx], copy=True)
        elif kind == 'ext':
            r = self.ext_rows[real_idx]
            frame = np.array(self.ext_pixels[real_idx], copy=True)
        elif kind == 'manual':
            r = self.manual_rows[real_idx]
            frame = np.array(self.manual_pixels[real_idx], copy=True)
        else:
            r = self.sep_rows[real_idx]
            frame = np.array(self.sep_pixels[real_idx], copy=True)

        w_orig, h_orig = r['width'], r['height']
        boxes = sanear_cajas(r.get('boxes', []), w_orig, h_orig)

        if self.augment:
            if random.random() < 0.5:
                frame = np.ascontiguousarray(frame[:, ::-1])
                flipped_boxes = []
                for b in boxes:
                    x1, y1, x2, y2 = b['xyxy']
                    nx1 = w_orig - x2
                    nx2 = w_orig - x1
                    b_flip = dict(b)
                    b_flip['xyxy'] = [min(nx1, nx2), y1, max(nx1, nx2), y2]
                    flipped_boxes.append(b_flip)
                boxes = sanear_cajas(flipped_boxes, w_orig, h_orig)
            if random.random() < 0.5:
                factor = random.uniform(0.85, 1.15)
                frame = np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        x_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        target_tensor = encode_grid32(boxes, w_orig, h_orig)
        return x_tensor, target_tensor


def evaluate_on_val_full(ckpt_path, val_rows, val_npy, device, out_json_path=None):
    """Evaluación completa exhaustiva en validación con barrido de umbrales y calibración.
    - Ordena siempre las predicciones decodificadas de mayor a menor score
      tanto al calcular AP@50 y mAP@50 (interpolación 101 puntos) como en el sweep de umbrales.
    - Sanea y verifica las cajas GT antes del emparejamiento 1-a-1 (IoU >= 0.5).
    """
    print(f"\nEjecutando evaluación completa en validación para {ckpt_path}...")
    saved = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = saved['model'] if 'model' in saved else saved
    model = DetectorS8()
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    valset = CachedDataset32(val_rows, val_npy, augment=False)
    loader = DataLoader(valset, batch_size=16, shuffle=False)

    outputs = []
    with torch.no_grad():
        for x, _ in loader:
            outputs.extend(model(x.to(device)).cpu())

    class_scores = {c: [] for c in CLASSES}
    class_hits = {c: [] for c in CLASSES}
    gt_counts = Counter()
    for r in val_rows:
        s_boxes = sanear_cajas(r.get('boxes', []), r['width'], r['height'])
        for b in s_boxes:
            gt_counts[b['class']] += 1

    for i, r in enumerate(val_rows):
        out = outputs[i]
        w, h = r['width'], r['height']
        gt_boxes = sanear_cajas(r.get('boxes', []), w, h)
        raw_preds = decode_grid32(out, w, h, threshold=0.01, nms=0.45)
        # Ordenamiento explícito de predicciones de mayor a menor score
        raw_preds = sorted(raw_preds, key=lambda p: p['score'], reverse=True)
        matched_gt = set()
        for p in raw_preds:
            cls = p['class']
            class_scores[cls].append(p['score'])
            best_iou = 0.0
            best_idx = -1
            for gi, gb in enumerate(gt_boxes):
                if gb['class'] == cls and gi not in matched_gt:
                    gx1, gy1, gx2, gy2 = gb['xyxy']
                    px1, py1, px2, py2 = p['xyxy']
                    ix1 = max(gx1, px1); iy1 = max(gy1, py1)
                    ix2 = min(gx2, px2); iy2 = min(gy2, py2)
                    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                    union = (gx2 - gx1) * (gy2 - gy1) + (px2 - px1) * (py2 - py1) - inter
                    iou = inter / max(union, 1e-6)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = gi
            if best_iou >= 0.5:
                class_hits[cls].append(1)
                matched_gt.add(best_idx)
            else:
                class_hits[cls].append(0)

    ap50 = {}
    for c in CLASSES:
        sc = class_scores[c]
        ht = class_hits[c]
        n_gt = gt_counts[c]
        if n_gt == 0 or len(sc) == 0:
            ap50[c] = 0.0
            continue
        order = np.argsort(-np.array(sc))
        sorted_hits = np.array(ht)[order]
        tp_cum = np.cumsum(sorted_hits)
        fp_cum = np.cumsum(1 - sorted_hits)
        rec = tp_cum / n_gt
        prec = tp_cum / (tp_cum + fp_cum)
        ap = 0.0
        for t in np.linspace(0.0, 1.0, 101):
            p_val = np.max(prec[rec >= t]) if np.any(rec >= t) else 0.0
            ap += p_val / 101.0
        ap50[c] = float(ap)

    mAP50 = float(np.mean(list(ap50.values())))

    thresholds = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
    sweep = {}
    best_f1 = -1.0
    best_op = {}

    for th in thresholds:
        th_key = f"{th:.2f}"
        metrics = {c: {'tp': 0, 'fp': 0, 'fn': 0} for c in CLASSES}
        for i, r in enumerate(val_rows):
            w, h = r['width'], r['height']
            gt_boxes = sanear_cajas(r.get('boxes', []), w, h)
            preds = decode_grid32(outputs[i], w, h, threshold=th, nms=0.45)
            # Ordenamiento explícito por score antes del matching 1-a-1
            preds = sorted(preds, key=lambda p: p['score'], reverse=True)
            matched = set()
            for p in preds:
                best_iou = 0.0
                best_gi = -1
                for gi, g in enumerate(gt_boxes):
                    if g['class'] == p['class'] and gi not in matched:
                        gx1, gy1, gx2, gy2 = g['xyxy']
                        px1, py1, px2, py2 = p['xyxy']
                        ix1 = max(gx1, px1); iy1 = max(gy1, py1)
                        ix2 = min(gx2, px2); iy2 = min(gy2, py2)
                        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                        union = (gx2 - gx1) * (gy2 - gy1) + (px2 - px1) * (py2 - py1) - inter
                        iou = inter / max(union, 1e-6)
                        if iou > best_iou:
                            best_iou = iou
                            best_gi = gi
                if best_iou >= 0.5:
                    metrics[p['class']]['tp'] += 1
                    matched.add(best_gi)
                else:
                    metrics[p['class']]['fp'] += 1
            for gi, g in enumerate(gt_boxes):
                if gi not in matched:
                    metrics[g['class']]['fn'] += 1

        tot_tp = sum(m['tp'] for m in metrics.values())
        tot_fp = sum(m['fp'] for m in metrics.values())
        tot_fn = sum(m['fn'] for m in metrics.values())
        prec = tot_tp / max(tot_tp + tot_fp, 1)
        rec = tot_tp / max(tot_tp + tot_fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-6)

        class_details = {}
        for c in CLASSES:
            c_tp = metrics[c]['tp']
            c_fp = metrics[c]['fp']
            c_fn = metrics[c]['fn']
            c_prec = c_tp / max(c_tp + c_fp, 1)
            c_rec = c_tp / max(c_tp + c_fn, 1)
            c_f1 = 2 * c_prec * c_rec / max(c_prec + c_rec, 1e-6)
            class_details[c] = {
                'tp': c_tp, 'fp': c_fp, 'fn': c_fn,
                'precision': c_prec, 'recall': c_rec, 'f1': c_f1
            }

        sweep[th_key] = {
            'threshold': th, 'tp': tot_tp, 'fp': tot_fp, 'fn': tot_fn,
            'precision': prec, 'recall': rec, 'micro_f1': f1,
            'classes': class_details
        }

        if f1 > best_f1:
            best_f1 = f1
            best_op = sweep[th_key]

    eval_summary = {
        'variant': 'Exp07_SuperUnified_DatosSeparados',
        'checkpoint': str(ckpt_path),
        'best_epoch': saved.get('epoch'),
        'val_loss': saved.get('val_loss'),
        'true_unthresholded_map50': mAP50,
        'true_unthresholded_ap50': ap50,
        'best_operating_point': best_op,
        'metrics_at_th_035': sweep.get('0.35', {}),
        'metrics_at_th_040': sweep.get('0.40', {}),
        'threshold_sweep': sweep
    }

    if out_json_path:
        with open(out_json_path, 'w', encoding='utf-8') as f:
            json.dump(eval_summary, f, indent=2, ensure_ascii=False)

    print(f"\n[Exp07 Evaluación]:")
    print(f"  mAP50: {mAP50*100:.2f}%")
    for c in CLASSES:
        print(f"    {c:10s}: AP50 = {ap50[c]*100:.2f}%")
    print(f"  Mejor Umbral: th={best_op['threshold']:.2f} -> Micro-F1 = {best_op['micro_f1']*100:.2f}% (TP={best_op['tp']}, FP={best_op['fp']})")
    return eval_summary


def export_threshold_json(eval_res, ckpt_path=EXP07_DIR / 'best.pt', out_path=EXP07_DIR / 'threshold.json'):
    """Genera artefacto threshold.json con modos cobertura y precision."""
    with open(ckpt_path, 'rb') as f:
        ckpt_sha = hashlib.sha256(f.read()).hexdigest()

    th_data = {
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "epoch": eval_res.get('best_epoch'),
        "threshold": eval_res['best_operating_point']['threshold'],
        "micro_f1": eval_res['best_operating_point']['micro_f1'],
        "modes": {
            "cobertura": {
                "threshold": 0.35,
                "min_box_size": 0,
                "description": "Modo máxima cobertura"
            },
            "precision": {
                "threshold": 0.40,
                "min_box_size": 32,
                "description": "Modo alta precisión"
            }
        }
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(th_data, f, indent=2)
    return th_data


def load_exp07_splits():
    """Carga los 5 conjuntos de datos del Experimento 7."""
    data_root = ROOT / 'data/preparado/deteccion'
    
    # 1. 670 registros canónicos de validación
    with open(ROOT / 'runs/detector_exp01/snapshot/manifest.json', encoding='utf-8') as f:
        m01 = json.load(f)
    orig_val = [r for r in m01['images'] if r.get('split') == 'val' and r.get('reviewed', True)]
    assert len(orig_val) == 670, f"Esperado 670 orig val, recibido {len(orig_val)}"

    # 2. Base train (3.645) y ext (1.505) de Exp05
    with open(ROOT / 'runs/detector_exp05/manifest_exp05.json', encoding='utf-8') as f:
        m05 = json.load(f)
    orig_train = m05['orig_train_rows']
    ext_rows = m05['ext_train_rows']

    # 3. 60 fotos manuales revisadas por el usuario en train
    with open(data_root / 'manifest.json', encoding='utf-8') as f:
        full_m = json.load(f)
    manual_train = [r for r in full_m['images'] if r.get('manual_batch') and r.get('reviewed') and r.get('split') == 'train']

    # 4. 740 registros limpios de DATOS_SEPARADOS
    with open(ROOT / 'data/datos_separados_limpios/valid_records.json', encoding='utf-8') as f:
        sep_rows = json.load(f)

    return orig_val, orig_train, ext_rows, manual_train, sep_rows, data_root


def train_exp07(epochs=25, batch_size=16, lr=0.001, seed=42, device_name='auto', dry_run=False):
    """Entrenamiento completo de Exp07 desde cero con pesos aleatorios."""
    seed_all(seed)
    device = device_for(device_name)
    save_dir = (EXP07_DIR / '_dry_run_temp') if dry_run else EXP07_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n================ INICIANDO ENTRENAMIENTO EXPERIMENTO 7 EN {device} ================", flush=True)
    if dry_run:
        print("[DRY RUN ACTIVADO] Se ejecutará 1 época rápida sobre 32 muestras sin modificar producción.", flush=True)
        epochs = 1

    orig_val, orig_train, ext_rows, manual_train, sep_rows, data_root = load_exp07_splits()

    print(f"Composición de entrenamiento Exp07:")
    print(f" - Base train original: {len(orig_train)}")
    print(f" - Exteriores SALSA+TACO: {len(ext_rows)}")
    print(f" - Manuales revisadas por el usuario: {len(manual_train)}")
    print(f" - DATOS_SEPARADOS limpios anti-fuga: {len(sep_rows)}")
    total_imgs = len(orig_train) + len(ext_rows) + len(manual_train) + len(sep_rows)
    print(f" -> TOTAL EXACTO: {total_imgs} imágenes de entrenamiento")

    # Ejecutar escudo anti-fuga en runtime
    leak_audit = validar_sin_fugas(orig_val, orig_train, ext_rows, manual_train, sep_rows)

    trainset = DatasetExp07(
        orig_rows=orig_train,
        orig_npy_path=ROOT / 'runs/detector_exp01/train.npy',
        ext_rows=ext_rows,
        ext_npy_path=ROOT / 'runs/detector_exp05/ext_train.npy',
        manual_rows=manual_train,
        data_root=data_root,
        sep_rows=sep_rows,
        sep_npy_path=ROOT / 'data/datos_separados_limpios/datos_separados_256.npy',
        augment=True
    )
    val_npy = ROOT / 'runs/detector_exp01/val.npy'
    valset = CachedDataset32(orig_val, val_npy, augment=False)

    if dry_run:
        trainset = torch.utils.data.Subset(trainset, list(range(min(32, len(trainset)))))
        valset = torch.utils.data.Subset(valset, list(range(min(32, len(valset)))))

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = DetectorS8().to(device)
    ema = ModelEMA(model, decay=0.996)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    config = {
        'variant': 'Exp07_SuperUnified_DatosSeparados',
        'architecture': 'DetectorS8',
        'weights_init': 'random_from_scratch',
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'seed': seed,
        'device': str(device),
        'ema_decay': 0.996,
        'dry_run': dry_run,
        'total_train_images': len(trainset),
        'base_train_images': len(orig_train),
        'external_verified_images': len(ext_rows),
        'manual_train_images': len(manual_train),
        'datos_separados_images': len(sep_rows),
        'val_images': len(orig_val),
        'leak_audit': leak_audit
    }
    with open(save_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    history = []
    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        model.train()
        train_loss = 0.0
        train_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn_3c(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ema.update(model)

            train_loss += loss.item()
            train_batches += 1

        train_loss /= max(train_batches, 1)

        # Validación
        ema.ema.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = ema.ema(x)
                loss = loss_fn_3c(pred, y)
                val_loss += loss.item()
                val_batches += 1

        val_loss /= max(val_batches, 1)
        scheduler.step()

        ep_time = time.time() - ep_start
        tot_time = time.time() - start_time
        curr_lr = opt.param_groups[0]['lr']

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': curr_lr,
            'epoch_seconds': ep_time,
            'total_seconds': tot_time
        })

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save({
                'model': ema.ema.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'config': config,
                'origin': 'residuos-from-scratch-v1'
            }, save_dir / 'best.pt')

        torch.save({
            'model': ema.ema.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
            'config': config,
            'origin': 'residuos-from-scratch-v1'
        }, save_dir / 'last.pt')

        print(f"Ep {epoch:02d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} {'[BEST]' if is_best else '      '} | LR: {curr_lr:.6f} | {ep_time:.1f}s", flush=True)

    with open(save_dir / 'history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    if dry_run:
        print("\n[DRY RUN] Evaluando subconjunto rápido en validación canónica...")
        eval_res = evaluate_on_val_full(
            save_dir / 'best.pt',
            orig_val[:32],
            val_npy,
            device,
            save_dir / 'evaluation.json'
        )
        export_threshold_json(eval_res, save_dir / 'best.pt', save_dir / 'threshold.json')
        print("\n[DRY RUN] Smoke test completado con éxito (el modelo de producción no fue alterado).")
        return eval_res
    else:
        # Evaluación completa final
        print("\nEvaluando modelo Exp07 en validación canónica...")
        eval_res = evaluate_on_val_full(
            save_dir / 'best.pt',
            orig_val,
            val_npy,
            device,
            save_dir / 'evaluation.json'
        )
        export_threshold_json(eval_res, save_dir / 'best.pt', save_dir / 'threshold.json')
        print("\n¡Entrenamiento y evaluación de Experimento 7 completados con éxito!")
        return eval_res


def evaluate_existing_exp07(device_name='auto'):
    """Ejecuta la auditoría anti-fuga y evaluación exhaustiva sobre el mejor checkpoint existente."""
    device = device_for(device_name)
    orig_val, orig_train, ext_rows, manual_train, sep_rows, _ = load_exp07_splits()
    
    # 1. Auditoría anti-fuga
    leak_audit = validar_sin_fugas(orig_val, orig_train, ext_rows, manual_train, sep_rows)

    # 2. Actualizar config.json si existe
    cfg_file = EXP07_DIR / 'config.json'
    if cfg_file.exists():
        with open(cfg_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = {
            'variant': 'Exp07_SuperUnified_DatosSeparados',
            'architecture': 'DetectorS8',
            'weights_init': 'random_from_scratch',
            'epochs': 25,
            'batch_size': 16,
            'lr': 0.001,
            'seed': 42,
            'device': str(device),
            'ema_decay': 0.996,
            'total_train_images': len(orig_train) + len(ext_rows) + len(manual_train) + len(sep_rows),
            'base_train_images': len(orig_train),
            'external_verified_images': len(ext_rows),
            'manual_train_images': len(manual_train),
            'datos_separados_images': len(sep_rows),
            'val_images': len(orig_val)
        }
    config['leak_audit'] = leak_audit
    with open(cfg_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    # 3. Evaluación exhaustiva en validación
    val_npy = ROOT / 'runs/detector_exp01/val.npy'
    eval_res = evaluate_on_val_full(
        EXP07_DIR / 'best.pt',
        orig_val,
        val_npy,
        device,
        EXP07_DIR / 'evaluation.json'
    )

    # 4. Generar threshold.json
    export_threshold_json(eval_res, EXP07_DIR / 'best.pt', EXP07_DIR / 'threshold.json')
    return eval_res


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Experimento 7: DetectorS8 Super-Unificado 100% desde cero.")
    parser.add_argument('--epochs', type=int, default=25, help='Número de épocas de entrenamiento (default: 25)')
    parser.add_argument('--batch-size', type=int, default=16, help='Tamaño del batch (default: 16)')
    parser.add_argument('--lr', type=float, default=0.001, help='Tasa de aprendizaje (default: 0.001)')
    parser.add_argument('--seed', type=int, default=42, help='Semilla aleatoria (default: 42)')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], default='auto', help='Dispositivo de cómputo (default: auto)')
    parser.add_argument('--dry-run', action='store_true', help='Ejecución rápida de prueba (1 época, 32 muestras) sin sobreescribir el modelo de producción')
    parser.add_argument('--eval-only', action='store_true', help='Ejecuta solo la auditoría anti-fuga y evaluación sobre el checkpoint existente')

    args = parser.parse_args()

    if args.eval_only:
        evaluate_existing_exp07(device_name=args.device)
    else:
        train_exp07(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device_name=args.device,
            dry_run=args.dry_run
        )


if __name__ == '__main__':
    main()
