"""Inspect the three new external datasets in data/externos/."""
from pathlib import Path
import json
import zipfile
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
EXT_ROOT = ROOT / 'data/externos'

def main():
    # 1. Garbage Classification 3
    gc_dir = EXT_ROOT / 'GARBAGE CLASSIFICATION 3.v2-gc1.coco'
    print('=== 1. GARBAGE CLASSIFICATION 3 ===', flush=True)
    for split in ('train', 'valid', 'test'):
        ann_file = gc_dir / split / '_annotations.coco.json'
        if ann_file.exists():
            data = json.loads(ann_file.read_text('utf-8'))
            cats = {c['id']: c['name'] for c in data['categories']}
            counts = Counter(cats[a['category_id']] for a in data['annotations'])
            print(f"  Split {split}: {len(data['images'])} images, {len(data['annotations'])} annotations", flush=True)
            print(f"    Categories: {cats}", flush=True)
            print(f"    Counts: {dict(counts)}", flush=True)

    # 2. Trash Detection Dataset
    td_dir = EXT_ROOT / 'Trash Detection Dataset/Trash Detection Dataset/CUSTOM_DATASET'
    print('\n=== 2. TRASH DETECTION DATASET ===', flush=True)
    if (td_dir / 'data.yaml').exists():
        print('  data.yaml:\n' + (td_dir / 'data.yaml').read_text('utf-8').strip(), flush=True)
    names = ['BIODEGRADABLE', 'CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']
    for split in ('train', 'valid', 'test'):
        im_dir = td_dir / split / 'images'
        lb_dir = td_dir / split / 'labels'
        if im_dir.exists():
            ims = list(im_dir.glob('*.*'))
            lbs = list(lb_dir.glob('*.txt'))
            c_counts = Counter()
            fmt_counts = Counter()
            for lb in lbs:
                for line in lb.read_text('utf-8').splitlines():
                    parts = line.split()
                    if parts:
                        cid = int(parts[0])
                        c_counts[names[cid]] += 1
                        fmt_counts[len(parts) - 1] += 1
            print(f"  Split {split}: {len(ims)} images, {len(lbs)} label files", flush=True)
            print(f"    Counts: {dict(c_counts)}", flush=True)
            print(f"    Coord lengths: {dict(fmt_counts)}", flush=True)

    # 3. Re-annotated TrashNet zip
    tn_dirs = [d for d in EXT_ROOT.iterdir() if 'TrashNet' in d.name]
    print('\n=== 3. RE-ANNOTATED TRASHNET ===', flush=True)
    for d in tn_dirs:
        for zf_path in d.rglob('*.zip'):
            print(f"  Zip file: {zf_path.name} | Size: {zf_path.stat().st_size} bytes", flush=True)
            with zipfile.ZipFile(zf_path, 'r') as zf:
                namelist = zf.namelist()
                print(f"  Total entries in zip: {len(namelist)}", flush=True)
                exts = Counter(Path(n).suffix.lower() for n in namelist if not n.endswith('/'))
                print(f"  Extensions in zip: {dict(exts)}", flush=True)
                top_dirs = Counter('/'.join(n.split('/')[:2]) for n in namelist if '/' in n)
                print(f"  Top dirs in zip: {dict(top_dirs)}", flush=True)
                special = [n for n in namelist if Path(n).suffix.lower() in ('.yaml', '.yml', '.json', '.md', '.csv') or 'readme' in n.lower() or 'classes' in n.lower()]
                print(f"  Special files: {special[:30]}", flush=True)
                print(f"  Sample entries: {namelist[:20]}", flush=True)
                for sp in special[:5]:
                    if not sp.endswith('/'):
                        try:
                            txt = zf.read(sp).decode('utf-8', errors='replace')[:1000]
                            print(f"  --- Content of {sp} ---\n{txt}\n  ----------------------", flush=True)
                        except Exception as e:
                            print(f"  Error reading {sp}: {e}", flush=True)

if __name__ == '__main__':
    main()
