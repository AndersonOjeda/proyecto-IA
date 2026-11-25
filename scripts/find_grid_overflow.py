import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from residuos.data import Store
from residuos.detector import encode

s = Store('data/preparado/deteccion')
rows = s.records('train')
for r in rows:
    try:
        encode(r['boxes'], r['width'], r['height'])
    except Exception as e:
        print("OVERFLOW_FOUND:", r['id'], r['path'], e)
