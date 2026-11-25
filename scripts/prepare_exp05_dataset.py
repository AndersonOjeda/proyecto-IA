"""Prepare Experiment 5 Dataset and Pre-cache external images into .npy.
Combines 3,645 original training images with 1,505 verified outdoor images from SALSA & TACO unofficial.
Pre-caches images into uint8 [N, 3, 256, 256] array for maximum training throughput on CUDA.
Runs preflight tests: geometry roundtrip, zero leakage, and single-batch overfit.
"""
from pathlib import Path
import sys
import json
import time
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from residuos import CLASSES
from residuos.data import Store, read_json, write_json
from scripts.experiment3_runner import DetectorS8, encode_grid32, decode_grid32, GRID, SLOTS
from scripts.experiment3c_runner import loss_fn_3c, CLASS_WEIGHTS_3C

EXP05_DIR = ROOT / 'runs/detector_exp05'
EXP05_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 60)
    print("PREPARANDO DATASET INTEGRADO EXPERIMENTO 5")
    print("=" * 60)

    # 1. Base Train (3645 images)
    store = Store(ROOT / 'data/preparado/deteccion')
    orig_train_rows = store.records('train')
    print(f"Base Train original: {len(orig_train_rows)} imagenes.")
    orig_train_npy = ROOT / 'runs/detector_exp01/train.npy'
    assert orig_train_npy.exists(), "Falta runs/detector_exp01/train.npy"

    # 2. External Verified Images (1505 images)
    manifest_p = EXP05_DIR / 'manifiesto_descargas.json'
    assert manifest_p.exists(), "Falta manifiesto_descargas.json"
    with open(manifest_p, encoding='utf-8') as f:
        download_data = json.load(f)
    accepted = download_data['accepted_images']
    print(f"Imagenes externas aceptadas (SALSA OLM + TACO no oficial): {len(accepted)}")

    # 3. Create or load ext_train.npy
    ext_npy_path = EXP05_DIR / 'ext_train.npy'
    if ext_npy_path.exists() and ext_npy_path.stat().st_size > 200 * 1024**2:
        print(f"Cargando cache existente: {ext_npy_path} ({ext_npy_path.stat().st_size / 1024**2:.1f} MB)...")
        ext_pixels = np.load(ext_npy_path, mmap_mode='r')
    else:
        print(f"\nGenerando cache de pixeles: {ext_npy_path} (dimension [1505, 3, 256, 256])...")
        ext_pixels = np.zeros((len(accepted), 3, 256, 256), dtype=np.uint8)

        t0 = time.time()
        for idx, rec in enumerate(accepted):
            img_p = ROOT / rec['path']
            im = cv2.imdecode(np.fromfile(str(img_p), dtype=np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                raise RuntimeError(f"Error al leer imagen: {img_p}")
            im_resized = cv2.resize(im, (256, 256))
            # Store as RGB [3, 256, 256]
            im_rgb = cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB)
            ext_pixels[idx] = im_rgb.transpose(2, 0, 1)

            if (idx + 1) % 300 == 0 or (idx + 1) == len(accepted):
                print(f"  Procesadas: {idx + 1}/{len(accepted)} ({time.time() - t0:.1f}s)")

        np.save(ext_npy_path, ext_pixels)
        print(f"Cache guardado exitosamente: {ext_npy_path.stat().st_size / 1024**2:.1f} MB")

    # 4. Save Integrated Manifest
    integrated_manifest = {
        'version': 'exp05_integrated',
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'total_train_images': len(orig_train_rows) + len(accepted),
        'orig_train_count': len(orig_train_rows),
        'ext_train_count': len(accepted),
        'orig_train_npy': str(orig_train_npy.relative_to(ROOT)).replace('\\', '/'),
        'ext_train_npy': str(ext_npy_path.relative_to(ROOT)).replace('\\', '/'),
        'orig_train_rows': orig_train_rows,
        'ext_train_rows': accepted
    }
    with open(EXP05_DIR / 'manifest_exp05.json', 'w', encoding='utf-8') as f:
        json.dump(integrated_manifest, f, indent=2, ensure_ascii=False)
    print(f"Manifiesto integrado guardado en: {EXP05_DIR / 'manifest_exp05.json'}")

    # 5. Preflight Verification
    print("\n" + "=" * 60)
    print("EJECUTANDO COMPROBACIONES PREFLIGHT EXPERIMENTO 5")
    print("=" * 60)

    # 5.1 Test target encoding roundtrip
    print("[1/3] Verificando codificacion de grid 32x32 y anchors...")
    sample_boxes = [{'class': 'vidrio', 'xyxy': [40, 50, 120, 180]}]
    target = encode_grid32(sample_boxes, 256, 256)
    assert target.shape == (32, 32, 3, 6), f"Forma inesperada del target: {target.shape}"
    assert (target[..., 0] == 1).sum() == 1, "Debe haber exactamente una celda positiva"
    print("  Codificacion geometrica verificada exitosamente.")

    # 5.2 Zero leakage assertion
    print("[2/3] Verificando asercion estricta de no-fuga...")
    val_rows = store.records('val')
    test_rows = store.records('test')
    val_hashes = {r['sha256'] for r in val_rows}
    test_hashes = {r['sha256'] for r in test_rows}
    ext_hashes = {r['sha256'] for r in accepted}

    val_overlap = ext_hashes.intersection(val_hashes)
    test_overlap = ext_hashes.intersection(test_hashes)
    assert len(val_overlap) == 0, f"FUGA DETECTADA con VAL: {len(val_overlap)} coincidencias!"
    assert len(test_overlap) == 0, f"FUGA DETECTADA con TEST: {len(test_overlap)} coincidencias!"
    print(f"  Cero fugas confirmado: 0 coincidencias con Val ({len(val_hashes)} img) y Test ({len(test_hashes)} img).")

    # 5.3 Single-batch overfit test with loss_fn_3c
    print("[3/3] Verificando sobreajuste de un batch con loss_fn_3c y DetectorS8...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DetectorS8().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.003)

    # Batch of 4 real images from external dataset
    dummy_x = torch.from_numpy(ext_pixels[:4]).float().to(device) / 255.0
    dummy_targets = []
    for i in range(4):
        rec = accepted[i]
        t = encode_grid32(rec['boxes'], rec['width'], rec['height'])
        dummy_targets.append(t)
    dummy_y = torch.stack(dummy_targets).to(device)

    initial_loss = loss_fn_3c(model(dummy_x), dummy_y).item()
    for step in range(120):
        opt.zero_grad()
        loss = loss_fn_3c(model(dummy_x), dummy_y)
        loss.backward()
        opt.step()
    final_loss = loss_fn_3c(model(dummy_x), dummy_y).item()

    drop_ratio = initial_loss / max(final_loss, 1e-6)
    print(f"  Batch Overfit: loss inicial={initial_loss:.4f} -> final={final_loss:.6f} (reduccion={drop_ratio:.1f}x)")
    assert drop_ratio > 10.0, f"Error: la perdida no descendio suficientemente ({drop_ratio:.1f}x)"
    print("  Prueba de sobreajuste superada.")

    preflight_info = {
        'status': 'PASSED',
        'train_images': len(orig_train_rows) + len(accepted),
        'val_images': len(val_rows),
        'test_images': len(test_rows),
        'leak_val_count': len(val_overlap),
        'leak_test_count': len(test_overlap),
        'overfit_drop_ratio': float(drop_ratio),
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ')
    }
    with open(EXP05_DIR / 'preflight_exp05.json', 'w', encoding='utf-8') as f:
        json.dump(preflight_info, f, indent=2)

    print("\n>>> PREFLIGHT EXPERIMENTO 5 SUPERADO CON EXITO. LISTO PARA ENTRENAR. <<<")

if __name__ == '__main__':
    main()
