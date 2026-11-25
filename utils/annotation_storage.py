"""Módulo de almacenamiento y manipulación de anotaciones.

Compatible con residuos.data.Store y preparado para fuentes en:
- data/preparado/deteccion/manifest.json (fuente activa real)
- data/real_waste/annotations.json
- data/manifest.json
"""
import hashlib
import json
from pathlib import Path
from residuos import CLASSES
from residuos.data import Store as BaseStore, digest, read_json, write_json, SPLITS

def resolve_manifest_path(explicit_path=None):
    if explicit_path and Path(explicit_path).exists():
        p = Path(explicit_path)
        return p if p.is_file() else p / 'manifest.json'
    
    candidates = [
        Path('data/preparado/deteccion/manifest.json'),
        Path('data/real_waste/annotations.json'),
        Path('data/manifest.json')
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return Path('data/manifest.json').resolve()

class AnnotationStorage(BaseStore):
    """Extensión de Store para resumen analítico de muestras reales."""
    def __init__(self, root=None):
        if root is None:
            manifest = resolve_manifest_path()
            super().__init__(root=manifest.parent if manifest.name == 'manifest.json' else manifest.parent)
            self.path = manifest
            self.db = read_json(self.path) if self.path.exists() else {
                'version': 2, 'classes': list(CLASSES), 'videos': {}, 'images': []
            }
        else:
            super().__init__(root=root)

    def detailed_summary(self):
        """Retorna conteos detallados por partición, clase, estado de revisión y tamaños."""
        summary = {
            'splits': {s: {'images_total': 0, 'images_reviewed': 0, 'images_pending': 0, 'objects': {c: 0 for c in CLASSES}} for s in SPLITS},
            'object_sizes': {c: {'small_lt32': 0, 'medium_32_96': 0, 'large_ge96': 0} for c in CLASSES},
            'manifest_path': str(self.path),
            'classes': list(CLASSES)
        }
        
        for img in self.db.get('images', []):
            sp = img.get('split', 'train')
            if sp not in summary['splits']:
                continue
            summary['splits'][sp]['images_total'] += 1
            if img.get('reviewed', False):
                summary['splits'][sp]['images_reviewed'] += 1
                for b in img.get('boxes', []):
                    cls_name = b.get('class')
                    if cls_name in summary['splits'][sp]['objects']:
                        summary['splits'][sp]['objects'][cls_name] += 1
                        
                        # Cálculo de tamaño de caja
                        x1, y1, x2, y2 = b.get('xyxy', [0, 0, 0, 0])
                        w = max(0, x2 - x1)
                        h = max(0, y2 - y1)
                        max_dim = max(w, h)
                        if max_dim < 32:
                            summary['object_sizes'][cls_name]['small_lt32'] += 1
                        elif max_dim < 96:
                            summary['object_sizes'][cls_name]['medium_32_96'] += 1
                        else:
                            summary['object_sizes'][cls_name]['large_ge96'] += 1
            else:
                summary['splits'][sp]['images_pending'] += 1
                
        return summary
