"""Módulo de integración y escudo anti-fuga para GARBAGE CLASSIFICATION 3 (Roboflow).
Filtra por SHA-256, dHash perceptual y valida geometría estricta de cajas.
Vectorizado con NumPy para máxima velocidad de filtrado.
"""
import json
import hashlib
from pathlib import Path
import cv2
import numpy as np

CLASS_MAP = {
    'BIODEGRADABLE': 'organico',
    'CARDBOARD': 'carton',
    'GLASS': 'vidrio',
    'METAL': 'metal',
    'PAPER': 'papel',
    'PLASTIC': 'plastico'
}

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def compute_dhash(image, hash_size=8):
    """Calcula difference hash perceptual de una imagen OpenCV."""
    resized = cv2.resize(image, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.flatten()

def main():
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / 'data/preparado/deteccion/manifest.json'
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
        
    print("="*75, flush=True)
    print("      ESCUDO ANTI-FUGA Y FILTRADO GEOMÉTRICO (GARBAGE CLASSIFICATION 3)", flush=True)
    print("="*75, flush=True)
    
    val_images = [r for r in manifest['images'] if r['split'] == 'val']
    test_images = [r for r in manifest['images'] if r['split'] == 'test']
    train_images = [r for r in manifest['images'] if r['split'] == 'train']
    
    val_hashes = {r['sha256'] for r in val_images}
    test_hashes = {r['sha256'] for r in test_images}
    train_hashes = {r['sha256'] for r in train_images}
    
    print(f"Dataset actual: Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}", flush=True)
    print("Extrayendo dHash perceptual de Validación y Test para escudo anti-fuga...", flush=True)
    
    eval_dhashes = []
    for r in val_images + test_images:
        p = root / 'data/preparado/deteccion' / r['path']
        im = cv2.imread(str(p.resolve()), cv2.IMREAD_GRAYSCALE)
        if im is not None:
            eval_dhashes.append(compute_dhash(im))
            
    eval_matrix = np.array(eval_dhashes, dtype=bool) if eval_dhashes else np.empty((0, 64), dtype=bool)
    print(f"Perceptual dHashes de evaluación registrados: {len(eval_matrix)}", flush=True)
    
    # Procesar split 'train' de GC3
    gc3_train_dir = root / 'data/externos/GARBAGE CLASSIFICATION 3.v2-gc1.coco/train'
    with open(gc3_train_dir / '_annotations.coco.json', 'r', encoding='utf-8') as f:
        gc3_data = json.load(f)
        
    cats = {c['id']: c['name'] for c in gc3_data['categories']}
    anns_by_img = {}
    for a in gc3_data['annotations']:
        anns_by_img.setdefault(a['image_id'], []).append(a)
        
    print(f"\nAnalizando {len(gc3_data['images'])} imágenes de GC3 train...", flush=True)
    
    accepted_records = []
    rejected_reasons = {
        'missing_file': 0,
        'sha256_collision_eval': 0,
        'sha256_collision_train': 0,
        'perceptual_leakage_eval': 0,
        'unreadable_image': 0,
        'no_valid_boxes': 0
    }
    
    class_object_counts = {c: 0 for c in CLASS_MAP.values()}
    out_dir = root / 'runs/detector_gc3_integration'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, img_info in enumerate(gc3_data['images']):
        if (idx + 1) % 1500 == 0 or (idx + 1) == len(gc3_data['images']):
            print(f"  Progreso: {idx+1}/{len(gc3_data['images'])} evaluadas (Aceptadas: {len(accepted_records)})...", flush=True)
            
        img_id = img_info['id']
        file_name = img_info['file_name']
        img_path = gc3_train_dir / file_name
        
        if not img_path.exists():
            rejected_reasons['missing_file'] += 1
            continue
            
        sha = compute_sha256(img_path)
        if sha in val_hashes or sha in test_hashes:
            rejected_reasons['sha256_collision_eval'] += 1
            continue
            
        if sha in train_hashes:
            rejected_reasons['sha256_collision_train'] += 1
            continue
            
        im_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if im_gray is None:
            rejected_reasons['unreadable_image'] += 1
            continue
            
        # Escudo perceptual vectorizado contra val y test (Hamming <= 2)
        if len(eval_matrix) > 0:
            dh = compute_dhash(im_gray)
            diffs = np.count_nonzero(eval_matrix != dh, axis=1)
            if (diffs <= 2).any():
                rejected_reasons['perceptual_leakage_eval'] += 1
                continue
            
        w_img = img_info['width']
        h_img = img_info['height']
        raw_anns = anns_by_img.get(img_id, [])
        
        valid_boxes = []
        for ann in raw_anns:
            cat_name = cats.get(ann['category_id'])
            if cat_name not in CLASS_MAP:
                continue
            target_class = CLASS_MAP[cat_name]
            bx, by, bw, bh = ann['bbox']
            x1 = max(0.0, min(float(bx), float(w_img - 1)))
            y1 = max(0.0, min(float(by), float(h_img - 1)))
            x2 = max(0.0, min(float(bx + bw), float(w_img)))
            y2 = max(0.0, min(float(by + bh), float(h_img)))
            
            if (x2 - x1) < 4.0 or (y2 - y1) < 4.0:
                continue
                
            valid_boxes.append({
                'class': target_class,
                'xyxy': [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                'original_class': cat_name,
                'area': round((x2 - x1) * (y2 - y1), 2)
            })
            
        if not valid_boxes:
            rejected_reasons['no_valid_boxes'] += 1
            continue
            
        # Control de saturación de centros en celdas para estabilidad en rejilla 32x32
        cell_counts = {}
        kept_boxes = []
        for b in sorted(valid_boxes, key=lambda b: b['area'], reverse=True):
            cx = (b['xyxy'][0] + b['xyxy'][2]) / 2.0 / w_img
            cy = (b['xyxy'][1] + b['xyxy'][3]) / 2.0 / h_img
            gx = min(int(cx * 32), 31)
            gy = min(int(cy * 32), 31)
            if cell_counts.get((gy, gx), 0) < 3:
                cell_counts[(gy, gx)] = cell_counts.get((gy, gx), 0) + 1
                kept_boxes.append(b)
                class_object_counts[b['class']] += 1
                
        if not kept_boxes:
            continue
            
        rec = {
            'id': f"gc3_{img_id}_{sha[:8]}",
            'path': str(img_path.relative_to(root)).replace('\\', '/'),
            'source': 'Roboflow Garbage Classification 3',
            'split': 'train',
            'width': w_img,
            'height': h_img,
            'sha256': sha,
            'boxes': kept_boxes
        }
        accepted_records.append(rec)
        train_hashes.add(sha)
        
    print("\n" + "="*75, flush=True)
    print("RESULTADOS DE LA AUDITORÍA Y FILTRADO ANTI-FUGA:", flush=True)
    print("="*75, flush=True)
    print(f"Imágenes candidatas examinadas: {len(gc3_data['images'])}", flush=True)
    print(f"Imágenes aceptadas limpias para entrenamiento: {len(accepted_records)}", flush=True)
    print("\nMotivos de descarte:", flush=True)
    for reason, cnt in rejected_reasons.items():
        print(f"  - {reason:<28}: {cnt}", flush=True)
        
    print("\nObjetos válidos incorporados por clase:", flush=True)
    for c_name, cnt in class_object_counts.items():
        print(f"  - {c_name:<12}: +{cnt:5d} cajas", flush=True)
        
    summary = {
        'candidate_images': len(gc3_data['images']),
        'accepted_images': len(accepted_records),
        'rejected_reasons': rejected_reasons,
        'class_object_counts': class_object_counts
    }
    
    with open(out_dir / 'audit_leakage.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
        
    with open(out_dir / 'gc3_clean_train.json', 'w', encoding='utf-8') as f:
        json.dump(accepted_records, f, indent=2)
        
    print(f"\nManifiesto limpio guardado en: {out_dir / 'gc3_clean_train.json'}", flush=True)

if __name__ == '__main__':
    main()
