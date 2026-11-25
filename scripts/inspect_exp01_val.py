import json
from pathlib import Path

# Check runs/detector_exp01 or runs/detector_exp05
with open('runs/detector_exp01/snapshot/manifest.json', 'r', encoding='utf-8') as f:
    m01 = json.load(f)

val01 = [r for r in m01['images'] if r.get('split') == 'val' and r.get('reviewed', True)]
print("val in exp01 snapshot:", len(val01))

# Check how CachedDataset32 was used in experiment3_runner.py
with open('scripts/experiment3_runner.py', 'r', encoding='utf-8') as f:
    code = f.read()
import re
print("Occurrences of val.npy in experiment3_runner:")
for m in re.finditer(r'val\.npy', code):
    start = max(0, m.start() - 100)
    end = min(len(code), m.end() + 100)
    print("---")
    print(code[start:end])
