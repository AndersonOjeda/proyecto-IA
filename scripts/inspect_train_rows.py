import json

with open('runs/detector_exp05/manifest_exp05.json', 'r', encoding='utf-8') as f:
    m05 = json.load(f)

print("Keys in manifest_exp05.json:", list(m05.keys()) if isinstance(m05, dict) else len(m05))
if isinstance(m05, dict):
    for k in m05:
        if isinstance(m05[k], list):
            print(f"Key {k}: list of {len(m05[k])}")

# Also check how experiment5_runner.py loaded orig_train:
# store = Store(ROOT / 'data/preparado/deteccion')
# orig_train = store.records('train')
# In experiment5_runner.py, data/preparado/deteccion had NOT yet been updated with SALSA/TACO when exp05 was run!
# Now data/preparado/deteccion HAS been updated with SALSA/TACO (total 5150 train images).
# Let's check:
from residuos.data import Store
store = Store('data/preparado/deteccion')
train_recs = store.records('train')
print("Current store.records('train'):", len(train_recs))

# The first 3645 are the original base train records that match train.npy!
# Let's verify:
with open('runs/detector_exp05/manifiesto_descargas.json', 'r', encoding='utf-8') as f:
    d = json.load(f)
ext_rows = d['accepted_images']
print("ext_rows count:", len(ext_rows))
