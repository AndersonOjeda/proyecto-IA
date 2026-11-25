"""Experiment 2A: refine only confidence on the unchanged baseline checkpoint."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from residuos import CLASSES
from residuos.data import Store,read_json,write_json,digest
from residuos.evaluate import metrics
from scripts.diagnose_detector import candidates,nms_indices

base=ROOT/'runs/detector_exp01';audit=ROOT/'runs/detector_exp02/diagnosis_baseline';out=ROOT/'runs/detector_exp02/threshold_only'
out.mkdir(parents=True,exist_ok=True)
d=read_json(audit/'diagnosis.json');threshold=d['best_global_threshold']['threshold'];config=read_json(base/'config.json')
store=Store(config['data_root']);store.db=read_json(base/'snapshot/manifest.json');rows=store.records('val');raw=np.load(audit/'raw_validation.npz')['logits'];pairs=[]
for z,r in zip(raw,rows):
    xy,c,s=candidates(z,r);keep=nms_indices(xy,c,s)[:100];keep=keep[s[keep]>=threshold]
    pairs.append(([{'class':CLASSES[c[k]],'xyxy':xy[k].tolist(),'score':float(s[k])} for k in keep],r['boxes']))
result=metrics(pairs)
assert result['tp']==d['best_global_threshold']['tp'] and result['fp']==d['best_global_threshold']['fp']
write_json(out/'results.json',dict(**result,threshold=threshold,ap50=d['ap50'],map50=d['map50'],
    checkpoint=str(base/'best.pt'),checkpoint_sha256=digest(base/'best.pt'),trained=False,split='val',test_images_read=0,
    selection='exact confidence breakpoints, maximum validation micro-F1; NMS .45 and max100 unchanged',
    note='AP50 unchanged by chosen operating threshold; same ranked predictions'))
print(result,threshold,flush=True)
