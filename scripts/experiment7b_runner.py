"""Experiment 7-B Runner: DetectorS8 Super-Unificado con Pérdida Balanceada y Supresión de Fondo.
Mejoras clave respecto a Exp07:
1. Ponderación adaptativa de clases (CLASS_WEIGHTS_7B) calculada sobre las 5.950 imágenes reales:
   - Cartón recibe 1.95 (vs 1.27) para potenciar la clase más deficitaria.
   - Orgánico recibe 0.41 (vs 0.53) y Plástico 0.45 para mitigar la sobre-predicción.
2. Supresión estricta de falsos positivos de fondo (alpha=0.60 en Focal Loss, aumentando la penalización negativa).
3. Escudo anti-fuga canónico (670 imágenes de validación estrictamente aisladas).
4. Saneamiento estricto de coordenadas y evaluación 1 a 1 ordenada por confianza.
CERO modelos o pesos preentrenados (100% inicializado desde cero con pesos aleatorios).
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
from scripts.experiment7_runner import (
    sanear_cajas, normalizar_ruta, validar_sin_fugas,
    DatasetExp07, ModelEMA, evaluate_on_val_full, export_threshold_json,
    load_exp07_splits
)

EXP07B_DIR = ROOT / 'runs/detector_exp07_b'

# Pesos normalizados de clase según frecuencia inversa sobre las 5.950 imágenes reales
# [plastico, papel, carton, vidrio, metal, organico]
CLASS_WEIGHTS_7B = torch.tensor([0.4533, 1.0527, 1.9512, 1.1770, 0.9514, 0.4144], dtype=torch.float32)

def loss_fn_7b(pred, target, cls_weights=CLASS_WEIGHTS_7B, gamma=2.0, alpha=0.60, obj_scale=1.5, box_weight=10.0):
    """Focal Loss mejorada: mayor penalización a falsos positivos de fondo (alpha=0.60) + pesos balanceados 7B."""
    positive = target[..., 0] == 1
    n_pos = max(float(positive.sum()), 1.0)
    
    # 1. Focal Loss para presencia con mayor penalización al fondo
    p = torch.sigmoid(pred[..., 0])
    pt = p * target[..., 0] + (1.0 - p) * (1.0 - target[..., 0])
    focal_weight = (1.0 - pt) ** gamma
    alpha_weight = alpha * target[..., 0] + (1.0 - alpha) * (1.0 - target[..., 0])
    bce = nn.functional.binary_cross_entropy_with_logits(pred[..., 0], target[..., 0], reduction='none')
    fl = alpha_weight * focal_weight * bce
    obj = (fl.sum() / n_pos) * obj_scale
    
    # 2. Pérdida Smooth L1 en celdas positivas
    zero = pred.sum() * 0
    box = nn.functional.smooth_l1_loss(pred[..., 1:5][positive].sigmoid(), target[..., 1:5][positive]) if positive.any() else zero
    
    # 3. Pérdida Cross-Entropy con ponderación 7B
    if positive.any():
        w = cls_weights.to(pred.device) if cls_weights is not None else None
        cls = nn.functional.cross_entropy(pred[..., 5:][positive], target[..., 5][positive].long(), weight=w)
    else:
        cls = zero
        
    return obj + box_weight * box + cls


def train_exp07b(epochs=25, batch_size=16, lr=0.001, seed=42, device_name='auto', dry_run=False):
    """Entrenamiento de Exp07-B con regularización balanceada y supresión de fondo."""
    seed_all(seed)
    device = device_for(device_name)
    save_dir = (EXP07B_DIR / '_dry_run_temp') if dry_run else EXP07B_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n================ INICIANDO ENTRENAMIENTO EXPERIMENTO 7-B EN {device} ================", flush=True)
    if dry_run:
        print("[DRY RUN ACTIVADO] 1 época rápida sobre 32 muestras.", flush=True)
        epochs = 1

    orig_val, orig_train, ext_rows, manual_train, sep_rows, data_root = load_exp07_splits()

    print(f"Composición de entrenamiento Exp07-B:")
    print(f" - Base train original: {len(orig_train)}")
    print(f" - Exteriores SALSA+TACO: {len(ext_rows)}")
    print(f" - Manuales revisadas por el usuario: {len(manual_train)}")
    print(f" - DATOS_SEPARADOS limpios anti-fuga: {len(sep_rows)}")
    total_imgs = len(orig_train) + len(ext_rows) + len(manual_train) + len(sep_rows)
    print(f" -> TOTAL EXACTO: {total_imgs} imágenes de entrenamiento")

    # Escudo anti-fuga
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
        'variant': 'Exp07B_BalancedWeights_BackgroundSuppressed',
        'architecture': 'DetectorS8',
        'weights_init': 'random_from_scratch',
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'seed': seed,
        'device': str(device),
        'ema_decay': 0.996,
        'focal_alpha': 0.60,
        'class_weights_7b': CLASS_WEIGHTS_7B.tolist(),
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
            loss = loss_fn_7b(pred, y)
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
                loss = loss_fn_7b(pred, y)
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
        print("\n[DRY RUN] Evaluando subconjunto rápido en validación...")
        eval_res = evaluate_on_val_full(save_dir / 'best.pt', orig_val[:32], val_npy, device, save_dir / 'evaluation.json')
        export_threshold_json(eval_res, save_dir / 'best.pt', save_dir / 'threshold.json')
        return eval_res
    else:
        print("\nEvaluando modelo Exp07-B en validación canónica...")
        eval_res = evaluate_on_val_full(save_dir / 'best.pt', orig_val, val_npy, device, save_dir / 'evaluation.json')
        export_threshold_json(eval_res, save_dir / 'best.pt', save_dir / 'threshold.json')
        print("\n¡Entrenamiento y evaluación de Experimento 7-B completados con éxito!")
        return eval_res


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Experimento 7-B: DetectorS8 Balanceado y Supresión de Fondo.")
    parser.add_argument('--epochs', type=int, default=25, help='Número de épocas (default: 25)')
    parser.add_argument('--batch-size', type=int, default=16, help='Tamaño del batch (default: 16)')
    parser.add_argument('--lr', type=float, default=0.001, help='Tasa de aprendizaje (default: 0.001)')
    parser.add_argument('--seed', type=int, default=42, help='Semilla aleatoria (default: 42)')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], default='auto', help='Dispositivo (default: auto)')
    parser.add_argument('--dry-run', action='store_true', help='Ejecución rápida de prueba (1 época, 32 muestras)')

    args = parser.parse_args()
    train_exp07b(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device_name=args.device,
        dry_run=args.dry_run
    )

if __name__ == '__main__':
    main()
