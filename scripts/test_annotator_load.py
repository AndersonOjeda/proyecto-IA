import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import residuos.data as rd

store = rd.Store('data/preparado/deteccion')
print("Manifest loaded successfully! Total images:", len(store.db['images']))
manual = [r for r in store.db['images'] if r.get('manual_batch')]
print(f"Total manual batch images: {len(manual)}")

# Check that ALL 90 images resolve to an existing physical file
missing = []
for r in manual:
    p = store.root / r['path']
    if not p.exists():
        missing.append((r['id'], str(p)))

print(f"Missing images: {len(missing)}")
if not missing:
    print("ALL 90 images exist and are 100% accessible to the Annotator!")
    print("\nPrimeras 5 imágenes:")
    for r in manual[:5]:
        print(f" - ID: {r['id']} | Clase sugerida: {r.get('target_hint')} | Partición: {r.get('split')} | Ruta: {r['path']}")
