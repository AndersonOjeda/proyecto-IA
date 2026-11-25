"""Manifiesto único: procedencia, revisión humana y partición por sesión."""
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from . import CLASSES

SPLITS = ('train', 'val', 'test')

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

class Store:
    def __init__(self, root='data'):
        self.root = Path(root).resolve()
        self.path = self.root / 'manifest.json'
        if not self.path.exists() and (self.root / 'preparado/deteccion/manifest.json').exists():
            self.root = (self.root / 'preparado/deteccion').resolve()
            self.path = self.root / 'manifest.json'
        self.db = read_json(self.path) if self.path.exists() else {
            'version': 1, 'classes': list(CLASSES), 'videos': {}, 'images': []}

    def save(self):
        write_json(self.path, self.db)

    def register(self, video, session, split):
        import shutil
        if not session.strip() or split not in SPLITS:
            raise ValueError('Indique sesión y partición válidas.')
        for v in self.db['videos'].values():
            if v['session'] == session and v['split'] != split:
                raise ValueError('Toda la sesión debe pertenecer a una sola partición.')
        video = Path(video)
        key = digest(video)
        if key in self.db['videos']:
            previous = self.db['videos'][key]
            if (previous['session'], previous['split']) != (session, split):
                raise ValueError('Este video ya está registrado con otra procedencia.')
            return key
        dest = self.root / 'videos' / (key[:16] + video.suffix.lower())
        dest.parent.mkdir(parents=True, exist_ok=True)
        if video.resolve() != dest.resolve():
            shutil.copy2(video, dest)
        self.db['videos'][key] = dict(path=dest.relative_to(self.root).as_posix(),
                                     original_name=video.name, session=session, split=split)
        self.save()
        return key

    def capture(self, video_id, index, frame):
        import cv2
        uid = f'{video_id[:16]}_{index:08d}'
        found = next((r for r in self.db['images'] if r['id'] == uid), None)
        if found:
            return found
        v = self.db['videos'][video_id]
        path = self.root / 'images' / (uid + '.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode('.png', frame)
        if not ok:
            raise ValueError('No se pudo guardar el fotograma.')
        encoded.tofile(str(path))
        h, w = frame.shape[:2]
        record = dict(id=uid, path=path.relative_to(self.root).as_posix(), video=video_id,
                      session=v['session'], split=v['split'], frame=index, width=w, height=h,
                      reviewed=False, boxes=[], sha256=digest(path))
        self.db['images'].append(record)
        self.save()
        return record

    def annotate(self, uid, boxes, reviewed=True):
        record = next(r for r in self.db['images'] if r['id'] == uid)
        self.check_boxes(boxes, record['width'], record['height'])
        if record.get('source'):
            from datetime import datetime, timezone
            import copy
            record.setdefault('review_history', []).append(dict(
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                action='confirmar_revision_completa' if reviewed else 'guardar_pendiente',
                previous_boxes=copy.deepcopy(record['boxes']),
                previous_pending_boxes=copy.deepcopy(record.get('pending_boxes', [])),
                previous_reasons=list(record.get('pending_reasons', []))))
        record.update(boxes=boxes, reviewed=bool(reviewed))
        if reviewed:
            record['pending_boxes'] = []
            record['review_status'] = 'human_reviewed'
            record['pending_reasons'] = []
            record['eligible'] = not bool(record.get('duplicate_of'))
        self.save()

    @staticmethod
    def check_boxes(boxes, width, height):
        for b in boxes:
            if b['class'] not in CLASSES or len(b['xyxy']) != 4:
                raise ValueError('Clase o caja inválida.')
            x1, y1, x2, y2 = b['xyxy']
            if not all(math.isfinite(v) for v in b['xyxy']) or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError('Las cajas deben tener área positiva y estar dentro de la imagen.')

    def records(self, split):
        return [r for r in self.db['images'] if r['split'] == split and r['reviewed']
                and r.get('eligible', True) and not r.get('pending_boxes') and not r.get('duplicate_of')]

    def validate(self):
        import cv2
        import numpy as np
        errors, sessions, hashes, ids = [], {}, {}, set()
        if self.db.get('classes') != list(CLASSES):
            errors.append('Taxonomía incompatible.')
        for key, v in self.db['videos'].items():
            if v['split'] not in SPLITS or not v['session'].strip():
                errors.append(f'Procedencia inválida: {key}')
            if v['session'] in sessions and sessions[v['session']] != v['split']:
                errors.append(f'Fuga entre particiones: {v["session"]}')
            sessions[v['session']] = v['split']
            path = self.root / v['path']
            if not path.exists() or digest(path) != key:
                errors.append(f'Video ausente o modificado: {v["path"]}')
        for r in self.db['images']:
            try:
                if r['id'] in ids:
                    raise ValueError('ID repetido')
                ids.add(r['id'])
                if r.get('video'):
                    v = self.db['videos'][r['video']]
                    if (r['session'], r['split']) != (v['session'], v['split']):
                        raise ValueError('Procedencia inconsistente')
                else:
                    if not r.get('source') or not r.get('group') or r['split'] not in SPLITS:
                        raise ValueError('Falta procedencia de imagen externa')
                    group = r['group']
                    if group in sessions and sessions[group] != r['split']:
                        raise ValueError('Grupo externo en varias particiones')
                    sessions[group] = r['split']
                    if r.get('pending_boxes') and r['reviewed']:
                        raise ValueError('Una imagen con cajas ambiguas no puede estar revisada')
                self.check_boxes(r['boxes'], r['width'], r['height'])
                if not isinstance(r['reviewed'], bool):
                    raise ValueError('Estado de revisión inválido')
                path = self.root / r['path']
                checksum = digest(path)
                if checksum != r['sha256']:
                    raise ValueError('Imagen modificada')
                if checksum in hashes and hashes[checksum] != r['split']:
                    raise ValueError('Imagen idéntica en distintas particiones')
                hashes[checksum] = r['split']
                im = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if im is None or im.shape[:2] != (r['height'], r['width']):
                    raise ValueError('Dimensiones incorrectas o imagen ilegible')
            except (KeyError, ValueError, OSError, TypeError) as e:
                errors.append(f'{r.get("id")}: {e}')
        return errors

    def stats(self):
        result = {}
        for split in SPLITS:
            rows = [r for r in self.db['images'] if r['split'] == split]
            reviewed = self.records(split)
            result[split] = dict(
                videos=sum(v['split'] == split for v in self.db['videos'].values()),
                sessions=len({v['session'] for v in self.db['videos'].values() if v['split'] == split}),
                images=len(rows), pending=sum(not r['reviewed'] for r in rows),
                reviewed_empty=sum(not r['boxes'] for r in reviewed),
                objects={c:sum(b['class'] == c for r in reviewed for b in r['boxes']) for c in CLASSES},
                images_per_class={c:sum(any(b['class'] == c for b in r['boxes']) for r in reviewed) for c in CLASSES})
        return result
