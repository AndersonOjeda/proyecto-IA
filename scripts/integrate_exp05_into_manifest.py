"""Integra las 1,505 imágenes de exteriores verificadas (SALSA OLM + TACO no oficial)
directamente en data/preparado/deteccion/manifest.json como imágenes revisadas de train.

Preserva:
- Respaldo de seguridad en manifest.json.pre_exp05.bak.
- Cero fugas contra validación y prueba (previamente auditadas con SHA-256).
- Exclusión total de datasets de estudio / fondo blanco (TrashNet crudo).
"""
import json
import shutil
from pathlib import Path

def main():
    manifest_path = Path('data/preparado/deteccion/manifest.json')
    exp05_path = Path('runs/detector_exp05/manifest_exp05.json')
    
    assert manifest_path.exists(), "Falta data/preparado/deteccion/manifest.json"
    assert exp05_path.exists(), "Falta runs/detector_exp05/manifest_exp05.json"
    
    # 1. Crear respaldo seguro
    backup_path = Path('data/preparado/deteccion/manifest.json.pre_exp05.bak')
    if not backup_path.exists():
        shutil.copy2(manifest_path, backup_path)
        print(f"Respaldo creado en: {backup_path}")
        
    manifest_data = json.loads(manifest_path.read_text(encoding='utf-8'))
    exp05_data = json.loads(exp05_path.read_text(encoding='utf-8'))
    
    existing_ids = {img['id'] for img in manifest_data['images']}
    existing_hashes = {img['sha256'] for img in manifest_data['images']}
    
    ext_rows = exp05_data.get('ext_train_rows', [])
    print(f"Candidatos externos en Exp05: {len(ext_rows)}")
    
    added = 0
    skipped_existing = 0
    
    for rec in ext_rows:
        img_id = str(rec['id'])
        if img_id in existing_ids or rec['sha256'] in existing_hashes:
            skipped_existing += 1
            continue
            
        file_name = rec.get('file_name', Path(rec['path']).name)
        rel_path = f"../../externos/exp05_downloads/{file_name}"
        
        # Validar existencia en disco
        disk_path = Path('data/preparado/deteccion') / rel_path
        if not disk_path.exists():
            print(f"Alerta: archivo no encontrado en disco: {disk_path}")
            continue
            
        boxes = []
        for b in rec['boxes']:
            boxes.append({
                'class': b['class'],
                'xyxy': [round(float(v), 2) for v in b['xyxy']]
            })
            
        new_entry = {
            'id': img_id,
            'path': rel_path,
            'original_path': rec['path'],
            'source': rec.get('source', 'SALSA / TACO Unofficial'),
            'source_image_id': img_id,
            'original_split': 'train',
            'original_group': f"ext_exp05:{img_id}",
            'group_evidence': "external_verified_outdoor",
            'group': f"g_exp05_{img_id}",
            'session': f"s_exp05_{img_id}",
            'split': 'train',
            'video': None,
            'frame': None,
            'width': int(rec['width']),
            'height': int(rec['height']),
            'sha256': rec['sha256'],
            'available': True,
            'provenance': {
                'image_url': None,
                'license': str(rec.get('license', 'CC / ODbL')),
                'license_note': 'Imagen exterior verificada de SALSA OpenLitterMap / TACO comunidad.'
            },
            'boxes': boxes,
            'pending_boxes': [],
            'pending_reasons': [],
            'reviewed': True,
            'eligible': True,
            'review_status': 'reviewed_complete',
            'manual_batch': False
        }
        
        manifest_data['images'].append(new_entry)
        existing_ids.add(img_id)
        existing_hashes.add(rec['sha256'])
        added += 1
        
    print(f"Imágenes añadidas al manifiesto activo: {added} (Omitidas existentes: {skipped_existing})")
    
    # 2. Guardar manifiesto atómicamente
    tmp_path = manifest_path.with_suffix('.tmp')
    tmp_path.write_text(json.dumps(manifest_data, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp_path.replace(manifest_path)
    print(f"Manifiesto actualizado guardado exitosamente en: {manifest_path}")
    
    # 3. Si existe data/real_waste/annotations.json como archivo o hardlink, sincronizarlo
    rw_ann = Path('data/real_waste/annotations.json')
    if rw_ann.exists():
        try:
            rw_ann.unlink()
            import os
            os.link(str(manifest_path.resolve()), str(rw_ann.resolve()))
            print(f"Hardlink sincronizado en: {rw_ann}")
        except Exception:
            shutil.copy2(manifest_path, rw_ann)
            print(f"Copia sincronizada en: {rw_ann}")

if __name__ == '__main__':
    main()
