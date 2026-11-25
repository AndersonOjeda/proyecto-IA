"""Script de entrenamiento optimizado para Recall del detector de residuos desde cero.

Garantiza:
1. Inicialización 100% desde cero (cero modelos ni pesos preentrenados).
2. Partición estricta anti-fuga por sesión.
3. Aumentación geométrica segura solo en entrenamiento.
4. Función de pérdida DetectorLoss con Focal Loss, GIoU y pesos de clase rebalanceados.
"""
import argparse
import json
import time
from pathlib import Path
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from residuos import CLASSES
from residuos.data import Store, write_json, digest
from residuos.detector import WasteDataset, seed_all, device_for, SIZE, GRID, SLOTS
from training.multi_waste_detector import DetectorLoss, create_detector, DEFAULT_CLASS_WEIGHTS

def train_detector(data_root='data/preparado/deteccion', output_dir='runs/detector_recall',
                   epochs=40, batch_size=16, lr=0.001, seed=42, device_name='auto',
                   model_type='standard', full_validate=False):
    seed_all(seed)
    device = device_for(device_name)
    store = Store(data_root)
    if full_validate:
        print("Ejecutando validación completa de hashes e integridad...")
        errors = store.validate()
        if errors:
            raise ValueError("Errores en el dataset:\n" + "\n".join(errors))
    else:
        # Validación rápida de taxonomía y particiones
        if store.db.get('classes') != list(CLASSES):
            raise ValueError("Taxonomía incompatible en store.")
        
    trainset = WasteDataset(store, 'train', augment=True)
    valset = WasteDataset(store, 'val', augment=False)
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    config = dict(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        device=str(device),
        architecture=model_type,
        size=SIZE,
        grid=GRID,
        slots=SLOTS,
        manifest_sha256=digest(store.path),
        data_root=str(store.root),
        classes=list(CLASSES),
        loss='DetectorLoss_Focal_GIoU_Weighted'
    )
    write_json(out_dir / 'config.json', config)
    
    loaders = {
        'train': DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0)
    }
    
    model = create_detector('standard').to(device)
    loss_fn = DetectorLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_loss = float('inf')
    history = []
    start_time = time.perf_counter()
    
    print(f"Iniciando entrenamiento del detector desde cero en {device}...")
    print(f"Muestras de entrenamiento: {len(trainset)}, Validación: {len(valset)}")
    
    for epoch in range(epochs):
        row = {'epoch': epoch + 1}
        for split, loader in loaders.items():
            is_train = (split == 'train')
            model.train(is_train)
            total_loss = 0.0
            
            for b_idx, (x, y) in enumerate(loader):
                x, y = x.to(device), y.to(device)
                with torch.set_grad_enabled(is_train):
                    pred = model(x)
                    loss = loss_fn(pred, y)
                    if not torch.isfinite(loss):
                        raise ValueError(f"Pérdida no finita en época {epoch+1}")
                    if is_train:
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        optimizer.step()
                total_loss += loss.item() * len(x)
                if is_train and ((b_idx + 1) % 75 == 0 or (b_idx + 1) == len(loader)):
                    print(f"  [Época {epoch+1:02d}/{epochs:02d}] Lote {b_idx+1}/{len(loader)} | Loss: {loss.item():.4f}", flush=True)
            
            row[f'{split}_loss'] = total_loss / len(loader.dataset)
            
        scheduler.step()
        row['lr'] = float(scheduler.get_last_lr()[0])
        row['seconds_elapsed'] = time.perf_counter() - start_time
        history.append(row)
        write_json(out_dir / 'history.json', history)
        
        print(f"Época {epoch+1:02d}/{epochs:02d} | Train Loss: {row['train_loss']:.4f} | Val Loss: {row['val_loss']:.4f} | LR: {row['lr']:.6f}", flush=True)
        
        if row['val_loss'] < best_loss:
            best_loss = row['val_loss']
            save_payload = dict(
                origin='residuos-from-scratch-v1',
                classes=list(CLASSES),
                state=model.state_dict(),
                config=config,
                epoch=epoch + 1,
                val_loss=best_loss
            )
            torch.save(save_payload, out_dir / 'best.pt')
            
    print(f"Entrenamiento completado en {time.perf_counter() - start_time:.1f}s. Mejor val_loss: {best_loss:.4f}", flush=True)
    return history

def main():
    p = argparse.ArgumentParser(description='Entrenamiento de detector de residuos desde cero.')
    p.add_argument('--data', default='data/preparado/deteccion')
    p.add_argument('--output', default='runs/detector_recall')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--model-type', default='standard', choices=['standard', 's8_residual'])
    p.add_argument('--full-validate', action='store_true', default=False)
    args = p.parse_args()
    
    train_detector(
        data_root=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        seed=args.seed,
        device_name=args.device,
        model_type=args.model_type,
        full_validate=args.full_validate
    )

if __name__ == '__main__':
    main()
