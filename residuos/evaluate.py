"""Evaluación de localización con correspondencia uno a uno, IoU >= 0.5."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from . import CLASSES
from .data import Store, write_json, digest, read_json
from .detector import device_for, load_model, predict, iou, seed_all

def metrics(pairs):
    counts = {c:dict(tp=0,fp=0,fn=0,objects=0,images=0) for c in CLASSES}
    for predictions, truth in pairs:
        for c in CLASSES:
            gt = [b for b in truth if b['class']==c]
            pr = sorted([b for b in predictions if b['class']==c], key=lambda b:b['score'],reverse=True)
            counts[c]['objects'] += len(gt); counts[c]['images'] += bool(gt)
            matched = set()
            for p in pr:
                candidates = [(iou(p['xyxy'],g['xyxy']),j) for j,g in enumerate(gt) if j not in matched]
                score,j = max(candidates,default=(0,-1))
                if score >= .5: matched.add(j); counts[c]['tp'] += 1
                else: counts[c]['fp'] += 1
            counts[c]['fn'] += len(gt)-len(matched)
    for c, n in counts.items():
        n['precision'] = n['tp']/(n['tp']+n['fp']) if n['tp']+n['fp'] else None
        n['recall'] = n['tp']/n['objects'] if n['objects'] else None
        n['status'] = 'evaluada' if n['objects'] else 'sin positivos reales; exhaustividad no evaluable'
    tp,fp,fn = [sum(n[k] for n in counts.values()) for k in ('tp','fp','fn')]
    return dict(classes=counts,micro_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,tp=tp,fp=fp,fn=fn)

def panel(frame, boxes, title):
    im = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); draw = ImageDraw.Draw(im)
    for b in boxes:
        draw.rectangle(b['xyxy'],outline='#19e08c',width=3)
        label = b['class'] + (f' {b["score"]:.2f}' if 'score' in b else '')
        x,y = b['xyxy'][:2]; draw.text((x+3,y+3),label,fill='yellow',stroke_width=1,stroke_fill='black')
    result = Image.new('RGB',(im.width,im.height+30),'#182532'); result.paste(im,(0,30))
    ImageDraw.Draw(result).text((8,8),title,fill='white')
    return result

def comparison(frame, predictions, truth):
    left,right = panel(frame,predictions,'Predicciones del modelo'),panel(frame,truth,'Anotaciones humanas')
    im = Image.new('RGB',(left.width*2,left.height)); im.paste(left,(0,0)); im.paste(right,(left.width,0)); return im

def collect(store, model, device, split, threshold, min_box_size=0):
    pairs, visuals = [], []
    for r in store.records(split):
        frame = cv2.imdecode(np.fromfile(store.root/r['path'],dtype=np.uint8),cv2.IMREAD_COLOR)
        predictions = predict(model,frame,device,threshold)
        if min_box_size > 0:
            predictions = [
                b for b in predictions 
                if (b['xyxy'][2] - b['xyxy'][0]) >= min_box_size 
                and (b['xyxy'][3] - b['xyxy'][1]) >= min_box_size
            ]
        pairs.append((predictions,r['boxes'])); visuals.append((r,frame,predictions))
    if not pairs: raise ValueError(f'No hay imágenes revisadas en {split}.')
    return pairs,visuals

def calibrate(data='data', checkpoint='runs/detector/best.pt', device='auto'):
    seed_all(42); device=device_for(device); store=Store(data)
    errors=store.validate()
    if errors: raise ValueError('\n'.join(errors))
    if not any(r['boxes'] for r in store.records('val')): raise ValueError('Validación necesita objetos reales para elegir umbral.')
    model,_=load_model(checkpoint,device)
    results=[]
    for threshold in (.05,.1,.15,.2,.25,.35,.5,.65,.8):
        pairs,_=collect(store,model,device,'val',threshold)
        results.append(dict(threshold=threshold,**metrics(pairs)))
    best=max(results,key=lambda r:(r['micro_f1'],r['threshold']))
    result=dict(threshold=best['threshold'],criterion='micro F1; IoU >= 0.5; NMS 0.45',
                checkpoint_sha256=digest(checkpoint),manifest_sha256=digest(store.path),split='val',grid_search=results)
    write_json(Path(checkpoint).parent/'threshold.json',result); return result

def threshold_for(checkpoint):
    path=Path(checkpoint).parent/'threshold.json'
    if not path.exists(): raise ValueError('Falta calibración. Ejecute: python -m residuos calibrate')
    result=read_json(path)
    if result['checkpoint_sha256'] != digest(checkpoint): raise ValueError('El modelo cambió; repita calibrate.')
    value = float(result['threshold'])
    import math
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Umbral inválido en threshold.json.')
    return value

def evaluate(data='data', checkpoint='runs/detector/best.pt', output='runs/evaluation', device='auto', mode='cobertura', min_box_size=None, threshold=None):
    seed_all(42); device=device_for(device); store=Store(data)
    errors=store.validate()
    if errors: raise ValueError('\n'.join(errors))
    model,_=load_model(checkpoint,device); base_threshold=threshold_for(checkpoint)
    mode = (mode or 'cobertura').lower()
    if mode == 'precision':
        actual_threshold = 0.40 if threshold is None else float(threshold)
        actual_min_size = 32 if min_box_size is None else int(min_box_size)
    else:
        actual_threshold = base_threshold if threshold is None else float(threshold)
        actual_min_size = 0 if min_box_size is None else int(min_box_size)
    pairs,visuals=collect(store,model,device,'test',actual_threshold,min_box_size=actual_min_size)
    result=dict(**metrics(pairs),split='test',threshold=actual_threshold,min_box_size=actual_min_size,mode=mode,iou=.5,nms=.45,images=len(pairs),
                checkpoint_sha256=digest(checkpoint),manifest_sha256=digest(store.path))
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    errors_list=[]
    for r,frame,p in visuals:
        single=metrics([(p,r['boxes'])]); error=bool(single['fp'] or single['fn'])
        folder=out/('errors' if error else 'comparisons'); folder.mkdir(exist_ok=True)
        comparison(frame,p,r['boxes']).save(folder/(r['id']+'.png'))
        if error: errors_list.append(dict(id=r['id'],fp=single['fp'],fn=single['fn']))
    write_json(out/'metrics.json',result); write_json(out/'errors.json',errors_list)
    write_json(out/'predictions.json',[dict(id=r['id'],predictions=p,truth=r['boxes']) for r,_,p in visuals])
    return result
