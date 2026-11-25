"""Downloader and validator for SALSA and TACO unofficial outdoor images for Experiment 5.
Applies multithreading, retry logic, image integrity checks, dimension validation,
exact hash (SHA-256) and perceptual hash (dHash) anti-leakage checks against val and test,
and produces a full audit manifest.
"""
import os
import sys
import json
import time
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import urllib.error

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from audit_external_datasets import source_mapping

DOWNLOAD_DIR = ROOT / 'data/externos/exp05_downloads'
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = ROOT / 'runs/detector_exp05'
OUT_DIR.mkdir(parents=True, exist_ok=True)

def compute_dhash(image_path, hash_size=8):
    """Compute difference hash (dHash) for perceptual visual similarity."""
    try:
        img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        resized = cv2.resize(img, (hash_size + 1, hash_size))
        diff = resized[:, 1:] > resized[:, :-1]
        return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
    except Exception:
        return None

def hamming_distance(h1, h2):
    return bin(h1 ^ h2).count('1')

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def download_one(task):
    """Download single image with retries and return status dict."""
    img_id, url, dest_path, expected_w, expected_h = task
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        # Check if readable
        try:
            im = cv2.imdecode(np.fromfile(str(dest_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if im is not None:
                h, w = im.shape[:2]
                return {
                    'img_id': img_id,
                    'status': 'cached',
                    'path': dest_path,
                    'width': w,
                    'height': h,
                    'bytes': dest_path.stat().st_size,
                    'error': None
                }
        except Exception:
            pass

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) ResiduosResearch/1.0'}
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read()
            if len(content) < 1000:
                raise ValueError(f"Content too small ({len(content)} bytes)")
            
            # Verify image decode
            im = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                raise ValueError("Corrupt image data, could not decode")
            
            with open(dest_path, 'wb') as f:
                f.write(content)
            
            h, w = im.shape[:2]
            return {
                'img_id': img_id,
                'status': 'downloaded',
                'path': dest_path,
                'width': w,
                'height': h,
                'bytes': len(content),
                'error': None
            }
        except Exception as e:
            time.sleep(0.5 * (attempt + 1))
            last_err = str(e)
    
    return {
        'img_id': img_id,
        'status': 'failed',
        'path': dest_path,
        'width': None,
        'height': None,
        'bytes': 0,
        'error': last_err
    }

def main():
    print("=" * 60)
    print("INICIANDO DESCARGA Y AUDITORIA EXPERIMENTO 5")
    print("=" * 60)

    # 1. Build anti-leakage reference database
    print("\n[1/5] Cargando base de huellas de validacion, test y lote manual...")
    with open(ROOT / 'data/preparado/deteccion/manifest.json', encoding='utf-8') as f:
        orig_manifest = json.load(f)

    leak_hashes = {}
    leak_dhashes = {}
    leak_stems = {}

    for img in orig_manifest['images']:
        split = img['split']
        # If val, test, or reviewed=False (quarantined manual)
        is_protected = (split in ('val', 'test')) or (not img.get('reviewed', True))
        if is_protected:
            tag = f"{split}_{'unreviewed' if not img.get('reviewed', True) else 'eval'}"
            h = img.get('sha256')
            if h:
                leak_hashes[h] = tag
            stem = Path(img['path']).stem
            leak_stems[stem] = tag
            # compute dhash of protected images if local file exists
            loc = ROOT / 'data/preparado/deteccion' / img['path']
            if loc.exists():
                dh = compute_dhash(loc)
                if dh is not None:
                    leak_dhashes[dh] = tag

    print(f"  Huellas protegidas cargadas: {len(leak_hashes)} SHA-256, {len(leak_stems)} stems, {len(leak_dhashes)} dHashes")

    # 2. Select SALSA candidates
    print("\n[2/5] Seleccionando candidatos limpios de SALSA...")
    salsa_path = ROOT / 'data/externos/salsa/src/salsa_utils/data/annotations.json'
    with open(salsa_path, encoding='utf-8') as f:
        salsa = json.load(f)

    salsa_cats = {c['id']: c['name'] for c in salsa['categories']}
    salsa_img_anns = defaultdict(list)
    for a in salsa['annotations']:
        salsa_img_anns[a['image_id']].append(a)

    AMBIGUOUS_SALSA = {'Cigarette', 'Clothes', 'Electronics', 'Garbage bag', 'Miscellaneous'}
    salsa_candidates = []
    
    # Mapping SALSA classes to 6 project classes
    SALSA_CLASS_MAP = {
        'Glass': 'vidrio',
        'Metal': 'metal',
        'Paper': 'papel',
        'Plastic': 'plastico',
        'Organic': 'organico'
    }

    for img in salsa['images']:
        fn = img['file_name'].split('.')[0]
        # Skip TACO batches from SALSA to avoid any overlap or repetition
        if fn.startswith('batch_'):
            continue
        
        anns = salsa_img_anns[img['id']]
        if not anns:
            continue
        
        cats = {salsa_cats[a['category_id']] for a in anns}
        if cats.intersection(AMBIGUOUS_SALSA):
            continue
        
        # Must have valid boxes
        valid_boxes = True
        boxes_list = []
        for a in anns:
            x, y, w, h = a['bbox']
            cname = salsa_cats[a['category_id']]
            if cname not in SALSA_CLASS_MAP:
                valid_boxes = False
                break
            if w < 3 or h < 3 or x < -5 or y < -5:
                valid_boxes = False
                break
            boxes_list.append({
                'class': SALSA_CLASS_MAP[cname],
                'xyxy': [x, y, x + w, y + h],
                'original_class': cname
            })
        
        if valid_boxes and boxes_list:
            salsa_candidates.append({
                'source': 'SALSA OpenLitterMap',
                'id': f"salsa_{img['id']}",
                'file_name': img['file_name'],
                'url': img['flickr_url'],
                'license': img.get('license', 'ODbL'),
                'width': img['width'],
                'height': img['height'],
                'boxes': boxes_list
            })

    print(f"  Candidatos limpios SALSA OLM: {len(salsa_candidates)}")

    # 3. Select TACO unofficial candidates
    print("\n[3/5] Seleccionando candidatos limpios de TACO no oficial...")
    taco_path = ROOT / 'data/externos/TACO-master/TACO-master/data/annotations_unofficial.json'
    with open(taco_path, encoding='utf-8') as f:
        taco = json.load(f)

    taco_cats = {c['id']: c['name'] for c in taco['categories']}
    taco_img_anns = defaultdict(list)
    for a in taco['annotations']:
        taco_img_anns[a['image_id']].append(a)

    taco_candidates = []
    for img in taco['images']:
        anns = taco_img_anns[img['id']]
        if not anns:
            continue
        
        all_mapped = True
        boxes_list = []
        target_classes = set()
        for a in anns:
            cname = taco_cats[a['category_id']]
            target, status, _ = source_mapping('TACO oficial', cname)
            if status not in ('propuesto', 'condicional') or not target:
                all_mapped = False
                break
            x, y, w, h = a['bbox']
            if w < 3 or h < 3 or x < -5 or y < -5:
                all_mapped = False
                break
            target_classes.add(target)
            boxes_list.append({
                'class': target,
                'xyxy': [x, y, x + w, y + h],
                'original_class': cname
            })
        
        if not all_mapped:
            continue
        
        # Prioritize images containing minority classes: papel, carton, vidrio, metal, organico
        # or if plastic, only if accompanied or balanced
        has_minority = target_classes.intersection({'papel', 'carton', 'vidrio', 'metal', 'organico'})
        if has_minority:
            url = img.get('flickr_url') or img.get('flickr_640_url')
            if url:
                taco_candidates.append({
                    'source': 'TACO no oficial',
                    'id': f"taco_unoff_{img['id']}",
                    'file_name': f"taco_unoff_{img['id']:06d}.jpg",
                    'url': url,
                    'license': img.get('license', 'CC/ODbL'),
                    'width': img['width'],
                    'height': img['height'],
                    'boxes': boxes_list
                })

    print(f"  Candidatos limpios TACO no oficial (prioridad minoritarias): {len(taco_candidates)}")

    all_candidates = salsa_candidates + taco_candidates
    print(f"\nTotal candidatos a descargar y auditar: {len(all_candidates)}")

    # 4. Multithreaded Download
    print("\n[4/5] Ejecutando descargas multihilo (8 hilos con reintentos)...")
    download_tasks = []
    cand_by_id = {}
    for c in all_candidates:
        cand_by_id[c['id']] = c
        ext = Path(c['file_name']).suffix or '.jpg'
        dest = DOWNLOAD_DIR / f"{c['id']}{ext}"
        download_tasks.append((c['id'], c['url'], dest, c['width'], c['height']))

    download_results = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(download_one, t): t[0] for t in download_tasks}
        completed = 0
        for f in as_completed(futures):
            res = f.result()
            download_results[res['img_id']] = res
            completed += 1
            if completed % 100 == 0 or completed == len(download_tasks):
                elapsed = time.time() - t0
                rate = completed / max(elapsed, 0.001)
                print(f"  Progreso: {completed}/{len(download_tasks)} ({completed/len(download_tasks)*100:.1f}%) | {rate:.1f} img/s")

    # 5. Full Validation and Leakage Audit
    print("\n[5/5] Realizando auditoria de integridad, dimensiones y no-fuga...")
    accepted_images = []
    rejected_images = []
    audit_summary = Counter()

    for cand in all_candidates:
        cid = cand['id']
        res = download_results.get(cid)
        if not res or res['status'] == 'failed':
            rejected_images.append({
                'id': cid,
                'source': cand['source'],
                'url': cand['url'],
                'reason': f"download_failed: {res.get('error') if res else 'no_result'}"
            })
            audit_summary['download_failed'] += 1
            continue
        
        filepath = res['path']
        # Compute SHA-256
        sha = sha256_file(filepath)
        if sha in leak_hashes:
            rejected_images.append({
                'id': cid,
                'source': cand['source'],
                'url': cand['url'],
                'sha256': sha,
                'reason': f"leak_sha256_match_{leak_hashes[sha]}"
            })
            audit_summary['leak_sha256_rejected'] += 1
            continue
        
        # Check dHash
        dh = compute_dhash(filepath)
        if dh is None:
            rejected_images.append({
                'id': cid,
                'source': cand['source'],
                'url': cand['url'],
                'reason': 'corrupt_image_decode_failed'
            })
            audit_summary['corrupt_decode_failed'] += 1
            continue
        
        is_visual_leak = False
        matched_tag = None
        for ref_dh, tag in leak_dhashes.items():
            if hamming_distance(dh, ref_dh) <= 4:
                is_visual_leak = True
                matched_tag = tag
                break
        
        if is_visual_leak:
            rejected_images.append({
                'id': cid,
                'source': cand['source'],
                'url': cand['url'],
                'reason': f"leak_visual_dhash_match_{matched_tag}"
            })
            audit_summary['leak_visual_dhash_rejected'] += 1
            continue
        
        # Dimension sanity check
        real_w, real_h = res['width'], res['height']
        exp_w, exp_h = cand['width'], cand['height']
        # Rescale bboxes if image dimensions differ (e.g. Flickr 640 vs full resolution)
        scale_x = real_w / max(exp_w, 1)
        scale_y = real_h / max(exp_h, 1)
        
        adjusted_boxes = []
        valid_geometry = True
        for b in cand['boxes']:
            x1, y1, x2, y2 = b['xyxy']
            adj_x1 = max(0.0, x1 * scale_x)
            adj_y1 = max(0.0, y1 * scale_y)
            adj_x2 = min(float(real_w), x2 * scale_x)
            adj_y2 = min(float(real_h), y2 * scale_y)
            if (adj_x2 - adj_x1) < 4 or (adj_y2 - adj_y1) < 4:
                valid_geometry = False
                break
            adjusted_boxes.append({
                'class': b['class'],
                'xyxy': [round(adj_x1, 2), round(adj_y1, 2), round(adj_x2, 2), round(adj_y2, 2)],
                'original_class': b['original_class']
            })
        
        if not valid_geometry or not adjusted_boxes:
            rejected_images.append({
                'id': cid,
                'source': cand['source'],
                'url': cand['url'],
                'reason': 'invalid_geometry_after_rescaling'
            })
            audit_summary['invalid_geometry'] += 1
            continue
        
        accepted_images.append({
            'id': cid,
            'source': cand['source'],
            'path': str(filepath.relative_to(ROOT)).replace('\\', '/'),
            'file_name': filepath.name,
            'sha256': sha,
            'dhash': str(dh),
            'license': cand['license'],
            'width': real_w,
            'height': real_h,
            'boxes': adjusted_boxes
        })
        audit_summary['accepted'] += 1

    # Class distribution of accepted images
    accepted_box_counts = Counter()
    accepted_img_counts = Counter()
    for img in accepted_images:
        classes = {b['class'] for b in img['boxes']}
        for c in classes:
            accepted_img_counts[c] += 1
        for b in img['boxes']:
            accepted_box_counts[b['class']] += 1

    print("\n" + "=" * 60)
    print("RESUMEN DE AUDITORIA Y DESCARGAS")
    print("=" * 60)
    print(f"Total candidatos evaluados: {len(all_candidates)}")
    print(f"Aceptados: {len(accepted_images)} ({len(accepted_images)/len(all_candidates)*100:.1f}%)")
    print(f"Rechazados: {len(rejected_images)}")
    print("Desglose de motivos de rechazo / aceptacion:", dict(audit_summary))
    print("\nDistribucion de clases ACEPTADAS:")
    print("  Objetos por clase:", dict(accepted_box_counts))
    print("  Imagenes con presencia de clase:", dict(accepted_img_counts))

    # Save outputs
    manifest_payload = {
        'version': 'exp05_outdoor_verified',
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'audit_summary': dict(audit_summary),
        'accepted_box_counts': dict(accepted_box_counts),
        'accepted_img_counts': dict(accepted_img_counts),
        'accepted_images': accepted_images,
        'rejected_images': rejected_images
    }
    with open(OUT_DIR / 'manifiesto_descargas.json', 'w', encoding='utf-8') as f:
        json.dump(manifest_payload, f, indent=2, ensure_ascii=False)

    report_payload = {
        'total_candidates': len(all_candidates),
        'accepted_count': len(accepted_images),
        'rejected_count': len(rejected_images),
        'rejection_breakdown': dict(audit_summary),
        'accepted_box_counts': dict(accepted_box_counts),
        'accepted_img_counts': dict(accepted_img_counts),
        'licenses': dict(Counter(img['license'] for img in accepted_images))
    }
    with open(OUT_DIR / 'auditoria_descargas_y_licencias.json', 'w', encoding='utf-8') as f:
        json.dump(report_payload, f, indent=2, ensure_ascii=False)

    print(f"\nManifiesto guardado en: {OUT_DIR / 'manifiesto_descargas.json'}")
    print(f"Auditoria guardada en: {OUT_DIR / 'auditoria_descargas_y_licencias.json'}")

if __name__ == '__main__':
    main()
