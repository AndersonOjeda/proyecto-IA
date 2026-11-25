"""Ejecuta el descargador original de TACO en procesos aislados por imagen.

Los JSON de trabajo y las descargas quedan fuera de data/externos.
No descarga miniaturas ni el conjunto no oficial. Reanuda sin releer descargas
ya verificadas; cada intento queda registrado.
"""
import concurrent.futures
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from PIL import Image,ImageOps

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/externos/TACO-master/TACO-master'
DEST=ROOT/'data/preparado/taco_oficial'

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    official=json.loads((SOURCE/'data/annotations.json').read_text())
    (DEST/'annotations_original.json').write_bytes((SOURCE/'data/annotations.json').read_bytes())
    cached={}
    ledger=DEST/'descargas.json'
    if ledger.exists(): cached={r['file_name']:r for r in json.loads(ledger.read_text(encoding='utf-8'))}
    def task(im):
        if im['file_name'] in cached and cached[im['file_name']]['status']=='ok' and (DEST/im['file_name']).exists(): return cached[im['file_name']]
        queue=DEST/f'_download_{im["id"]}.json'
        queue.write_text(json.dumps({'images':[im]}),encoding='utf-8')
        attempts=[]; path=DEST/im['file_name']
        # Recuperar bytes descargados que la primera verificación apartó por EXIF.
        if not path.exists():
            for candidate in path.parent.glob(path.stem+'.invalid-attempt-*'):
                try:
                    with Image.open(candidate) as test:
                        test.load(); upright=ImageOps.exif_transpose(test)
                        if upright.size==(im['width'],im['height']):candidate.rename(path);break
                except Exception:pass
        for attempt in range(2):
            try:
                result=subprocess.run([sys.executable,str(ROOT/'scripts/taco_download_entry.py'),str(SOURCE/'download.py'),str(queue)],capture_output=True,text=True,timeout=240)
                if result.returncode: raise RuntimeError(result.stderr[-1400:])
                with Image.open(path) as image:
                    image.load();raw_width,raw_height=image.size
                    orientation=image.getexif().get(274,1)
                    upright=ImageOps.exif_transpose(image);width,height=upright.size
                if (width,height)!=(im['width'],im['height']): raise ValueError(f'Dimensiones {width}x{height}, esperadas {im["width"]}x{im["height"]}')
                with path.open('rb') as f: checksum=hashlib.file_digest(f,'sha256').hexdigest()
                return dict(**im,status='ok',sha256=checksum,actual_width=width,actual_height=height,
                            raw_width=raw_width,raw_height=raw_height,exif_orientation=orientation,bytes=path.stat().st_size,attempt_errors=attempts)
            except Exception as exc:
                attempts.append(str(exc))
                if path.exists():
                    # Sólo apartar la descarga propia inválida; nunca tocar fuentes.
                    path.rename(path.with_suffix(f'.invalid-attempt-{attempt}'))
        return dict(**im,status='failed',attempt_errors=attempts)
    completed=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        futures=[pool.submit(task,im) for im in official['images']]
        for future in concurrent.futures.as_completed(futures):
            completed.append(future.result())
            if len(completed)%25==0 or len(completed)==len(futures):
                tmp=ledger.with_suffix('.tmp'); tmp.write_text(json.dumps(sorted(completed,key=lambda r:r['id']),ensure_ascii=False,indent=2),encoding='utf-8'); tmp.replace(ledger)
                print(f'{len(completed)}/{len(futures)}; recuperadas={sum(r["status"]=="ok" for r in completed)}',flush=True)
    failures=[r for r in completed if r['status']!='ok']
    (DEST/'fallos.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Finalizado:',len(completed)-len(failures),'recuperadas;',len(failures),'fallidas',flush=True)

if __name__=='__main__': main()
