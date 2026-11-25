"""Experiment 1: frozen manifest, real-data gates, scratch training, validation only."""
import argparse
import copy
import csv
import json
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from residuos import CLASSES
from residuos.data import Store, write_json, digest
from residuos.detector import Detector, SIZE, GRID, SLOTS, seed_all, device_for, encode, loss_fn, decode, iou, load_model
from residuos.evaluate import metrics, comparison


class Cached(Dataset):
    def __init__(self, rows, path, augment=False):
        self.rows, self.pixels, self.augment = rows, np.load(path, mmap_mode='r'), augment
        self.targets = [encode(r['boxes'], r['width'], r['height']) for r in rows]

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        pixels = self.pixels[i].copy()
        r = self.rows[i]
        target = self.targets[i]
        if self.augment and random.random() < .5:
            pixels = pixels[:, ::-1].copy()
            boxes = [dict(b, xyxy=[r['width']-b['xyxy'][2], b['xyxy'][1], r['width']-b['xyxy'][0], b['xyxy'][3]]) for b in r['boxes']]
            target = encode(boxes, r['width'], r['height'])
        x = torch.from_numpy(pixels).permute(2, 0, 1).float()/255
        if self.augment: x = (x*random.uniform(.85, 1.15)).clamp(0, 1)
        return x, target


def original(store, r):
    im = cv2.imdecode(np.fromfile(store.root/r['path'], dtype=np.uint8), cv2.IMREAD_COLOR)
    if im is None or im.shape[:2] != (r['height'], r['width']):
        raise ValueError('Unreadable or dimension mismatch: '+r['id'])
    return im


def freeze(out, store, args):
    snapshot = out/'snapshot'
    snapshot.mkdir(parents=True)
    shutil.copy2(store.path, snapshot/'manifest.json')
    shutil.copy2(ROOT/'docs/preparacion_dataset/grupos_particiones.json', snapshot/'grupos_particiones.json')
    for name in ('detector.py', 'data.py', 'evaluate.py'):
        shutil.copy2(ROOT/'residuos'/name, snapshot/name)
    shutil.copy2(__file__, snapshot/'experiment_detector.py')
    # The frozen original contains relative paths: retain its original root explicitly.
    store.db = json.loads((snapshot/'manifest.json').read_text(encoding='utf-8'))
    rows = {s: store.records(s) for s in ('train', 'val')}
    all_ids = set()
    groups, hashes = {}, {}
    for r in store.db['images']:
        assert r['id'] not in all_ids
        all_ids.add(r['id'])
        for key, dest in ((r['group'], groups), (r['sha256'], hashes)):
            assert key not in dest or dest[key] == r['split']
            dest[key] = r['split']
    for rr in rows.values():
        for r in rr:
            assert not r.get('manual_batch') and not r.get('pending_boxes')
            Store.check_boxes(r['boxes'], r['width'], r['height'])
            encode(r['boxes'], r['width'], r['height'])
    with (snapshot/'partitions.csv').open('w', newline='', encoding='utf-8-sig') as f:
        fields = ['id','source','group','split','original_split','sha256','eligible','review_status']
        w = csv.DictWriter(f, fields, extrasaction='ignore'); w.writeheader(); w.writerows(store.db['images'])
    write_json(snapshot/'eligible_ids.json', {s:[r['id'] for r in rr] for s, rr in rows.items()})
    config = dict(epochs=args.epochs,batch=args.batch,lr=.001,seed=args.seed,device=str(args.device),
                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  torch_version=str(torch.__version__),cuda_version=torch.version.cuda,python=sys.version,
                  platform=platform.platform(),size=SIZE,grid=GRID,slots=SLOTS,classes=list(CLASSES),
                  manifest_sha256=digest(snapshot/'manifest.json'),data_root=str(store.root),
                  started_utc=datetime.now(timezone.utc).isoformat(),initialization='random all layers including backbone; no checkpoint loaded',
                  optimizer='AdamW',weight_decay=.01,checkpoint_selection='minimum validation loss',
                  test_policy='No test image decoding or inference; only partition metadata snapshotted',
                  threshold_selection='maximum micro F1 on validation; ties prefer higher confidence',
                  thresholds=[.01,.025,.05,.075,.1,.15,.2,.25,.35,.5,.65,.8],
                  counts={s:dict(images=len(rr),objects=sum(len(r['boxes']) for r in rr)) for s,rr in rows.items()},
                  augmentation='horizontal flip p=0.5 and brightness multiplier U(0.85,1.15)',
                  threads=torch.get_num_threads(),deterministic_cudnn=True)
    write_json(out/'config.json',config)
    return rows, config


def cache(store, rows, out):
    start = time.perf_counter()
    for split, rr in rows.items():
        pixels = np.lib.format.open_memmap(out/(split+'.npy'),mode='w+',dtype=np.uint8,shape=(len(rr),SIZE,SIZE,3))
        for i,r in enumerate(rr):
            pixels[i] = cv2.cvtColor(cv2.resize(original(store,r),(SIZE,SIZE)),cv2.COLOR_BGR2RGB)
            if (i+1)%500 == 0: print('cache',split,i+1,flush=True)
        pixels.flush(); del pixels
    return time.perf_counter()-start


def preflight(store, rows, out, device, seed):
    started = time.perf_counter()
    dataset = Cached(rows['train'],out/'train.npy')
    chosen = []
    for c in CLASSES:
        candidates = [i for i,r in enumerate(rows['train']) if any(b['class']==c for b in r['boxes']) and i not in chosen]
        chosen.append(min(candidates,key=lambda i:(len(rows['train'][i]['boxes']),rows['train'][i]['id'])))
    canvas = Image.new('RGB',(1024,6*286),'#17212b')
    max_error = 0.
    for k,i in enumerate(chosen):
        r = rows['train'][i]
        frame = original(store,r)
        target = dataset.targets[i]
        reconstructed=[]
        for gy,gx,slot in (target[...,0]>0).nonzero().tolist():
            _,dx,dy,w,h,c = target[gy,gx,slot].tolist()
            cx,cy=(gx+dx)/GRID,(gy+dy)/GRID
            xy=[(cx-w/2)*SIZE,(cy-h/2)*SIZE,(cx+w/2)*SIZE,(cy+h/2)*SIZE]
            reconstructed.append(dict(**{'class':CLASSES[int(c)]},xyxy=xy))
        scaled=[dict(b,xyxy=[b['xyxy'][0]*SIZE/r['width'],b['xyxy'][1]*SIZE/r['height'],b['xyxy'][2]*SIZE/r['width'],b['xyxy'][3]*SIZE/r['height']]) for b in r['boxes']]
        for b in scaled:
            candidate=min((q for q in reconstructed if q['class']==b['class']),key=lambda q:np.max(np.abs(np.array(q['xyxy'])-b['xyxy'])))
            max_error=max(max_error,float(np.max(np.abs(np.array(candidate['xyxy'])-b['xyxy']))))
        from residuos.evaluate import panel
        left=panel(frame,r['boxes'],'Original: '+r['id']); left.thumbnail((768,286))
        right=panel(cv2.resize(frame,(SIZE,SIZE)),reconstructed,'256px: cajas reconstruidas del target')
        canvas.paste(left,(0,k*286));canvas.paste(right,(768,k*286))
    canvas.save(out/'preflight_resize.jpg')
    assert max_error < .001
    seed_all(seed)
    model=Detector().to(device)
    x=torch.stack([dataset[i][0] for i in chosen]).to(device)
    y=torch.stack([dataset[i][1] for i in chosen]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
    trace=[]
    for step in range(800):
        model.train();loss=loss_fn(model(x),y)
        assert torch.isfinite(loss)
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        if step%50==0 or step==799:
            model.eval()
            with torch.no_grad():
                z=model(x);value=float(loss_fn(z,y));pairs=[(decode(p,rows['train'][i]['width'],rows['train'][i]['height'],.1),rows['train'][i]['boxes']) for p,i in zip(z,chosen)]
            met=metrics(pairs);trace.append(dict(step=step+1,loss=value,tp=met['tp'],fp=met['fp'],fn=met['fn'],f1=met['micro_f1']))
            print('preflight',trace[-1],flush=True)
            if step>=150 and value<trace[0]['loss']*.2 and met['micro_f1']>=.9: break
    result=dict(resize_max_pixel_error=max_error,finite=True,trace=trace,images=[rows['train'][i]['id'] for i in chosen],
                seconds=time.perf_counter()-started,passed=trace[-1]['loss']<trace[0]['loss']*.2 and trace[-1]['f1']>=.9,
                threshold=.1,iou=.5,weights_discarded=True,device=str(device),torch_version=str(torch.__version__))
    write_json(out/'preflight.json',result)
    if not result['passed']: raise RuntimeError('Overfit gate failed; full training was not started')
    del optimizer,model,x,y
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result


def train(rows,out,config,device):
    seed_all(config['seed']) # New initialization: never reuse the tiny-batch model.
    shutil.copy2(__file__,out/'snapshot/experiment_detector_training.py')
    config['training_script_sha256']=digest(__file__)
    model=Detector().to(device)
    config['parameters']=sum(p.numel() for p in model.parameters())
    write_json(out/'config.json',config)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'])
    loaders={s:DataLoader(Cached(rr,out/(s+'.npy'),s=='train'),batch_size=config['batch'],shuffle=s=='train',num_workers=0,pin_memory=device.type=='cuda') for s,rr in rows.items()}
    best=float('inf');history=[];start=time.perf_counter()
    for epoch in range(config['epochs']):
        tick=time.perf_counter();entry={'epoch':epoch+1}
        for split,loader in loaders.items():
            model.train(split=='train');total=0.
            for x,y in loader:
                x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
                with torch.set_grad_enabled(split=='train'):
                    loss=loss_fn(model(x),y)
                    if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                    if split=='train':
                        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
                total+=loss.item()*len(x)
            entry[split+'_loss']=total/len(loader.dataset)
        entry.update(seconds=time.perf_counter()-tick,seconds_elapsed=time.perf_counter()-start)
        history.append(entry);write_json(out/'history.json',history);print('epoch',entry,flush=True)
        if entry['val_loss']<best:
            best=entry['val_loss']
            torch.save(dict(origin='residuos-from-scratch-v1',classes=list(CLASSES),state=model.state_dict(),config=config,epoch=epoch+1),out/'best.pt')
        torch.save(dict(epoch=epoch+1,state=model.state_dict(),optimizer=optimizer.state_dict(),config=config),out/'last_training.pt')
    return history


def validation(store,rows,out,device,config):
    start=time.perf_counter();model,saved=load_model(out/'best.pt',device)
    dataset=Cached(rows['val'],out/'val.npy')
    outputs=[]
    with torch.no_grad():
        for x,_ in DataLoader(dataset,batch_size=config['batch']): outputs.extend(model(x.to(device)).cpu())
    all_predictions=[decode(z,r['width'],r['height'],min(config['thresholds'])) for z,r in zip(outputs,rows['val'])]
    sweep=[]
    for threshold in config['thresholds']:
        sweep.append(dict(threshold=threshold,**metrics([([b for b in p if b['score']>=threshold],r['boxes']) for p,r in zip(all_predictions,rows['val'])])))
    best=max(sweep,key=lambda x:(x['micro_f1'],x['threshold']));threshold=best['threshold']
    write_json(out/'threshold.json',dict(threshold=threshold,criterion='validation micro F1; IoU >= 0.5; same-class one-to-one score-greedy; NMS .45',checkpoint_sha256=digest(out/'best.pt'),manifest_sha256=config['manifest_sha256'],split='val',grid_search=sweep))
    preds=[[b for b in pp if b['score']>=threshold] for pp in all_predictions]
    result=dict(**best,images=len(rows['val']),split='val',iou=.5,nms=.45,checkpoint_epoch=saved['epoch'],seconds=time.perf_counter()-start,
                note='Validation used for model and threshold selection; not an unbiased final test estimate.')
    write_json(out/'validation_metrics.json',result)
    write_json(out/'validation_predictions.json',[dict(id=r['id'],source=r['source'],original_path=r['original_path'],predictions=p,truth=r['boxes']) for p,r in zip(preds,rows['val'])])
    with (out/'validation_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,['class','objects','tp','fp','fn','precision','recall']);w.writeheader()
        for c,n in result['classes'].items():w.writerow(dict({'class':c},**{k:n[k] for k in w.fieldnames[1:]}))
    # Class-agnostic one-to-one localization for a separate, diagnostic confusion table.
    confusion={c:{d:0 for d in CLASSES} for c in CLASSES};examples=[]
    for r,pp in zip(rows['val'],preds):
        used=set()
        for p in sorted(pp,key=lambda b:b['score'],reverse=True):
            score,j=max([(iou(p['xyxy'],g['xyxy']),j) for j,g in enumerate(r['boxes']) if j not in used],default=(0,-1))
            if score>=.5:
                used.add(j);g=r['boxes'][j];confusion[g['class']][p['class']]+=1
                if g['class']!=p['class']:examples.append(dict(id=r['id'],true_class=g['class'],predicted_class=p['class'],iou=score,score=p['score']))
    write_json(out/'confusions.json',dict(rule='class-agnostic IoU>=0.5 one-to-one; separate diagnostic, not per-class TP matching',matrix=confusion,examples=examples))
    selected=[]
    for c in ('papel','carton','vidrio','plastico','metal','organico'):
        candidates=[i for i,r in enumerate(rows['val']) if any(b['class']==c for b in r['boxes']) and i not in selected]
        # Prefer a measured localized confusion, else deterministic first example; no test access.
        wrong={e['id'] for e in examples if e['true_class']==c}
        selected.append(min(candidates,key=lambda i:(rows['val'][i]['id'] not in wrong,rows['val'][i]['id'])))
    sheet=Image.new('RGB',(1200,6*400),'#182532');selected_info=[]
    for k,i in enumerate(selected):
        r=rows['val'][i];frame=original(store,r)
        factor=min(580/frame.shape[1],350/frame.shape[0]);small=cv2.resize(frame,(round(frame.shape[1]*factor),round(frame.shape[0]*factor)))
        scaled=lambda bb:[dict(b,xyxy=[v*factor for v in b['xyxy']]) for b in bb]
        pic=comparison(small,scaled(preds[i]),scaled(r['boxes']))
        sheet.paste(pic,(0,k*400+20));ImageDraw.Draw(sheet).text((5,k*400+3),r['id'],fill='white')
        selected_info.append(dict(id=r['id'],predictions=preds[i],truth=r['boxes']))
    sheet.save(out/'validation_preview.jpg');write_json(out/'preview_examples.json',selected_info)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='runs/detector_exp01');p.add_argument('--epochs',type=int,default=40);p.add_argument('--batch',type=int,default=16);p.add_argument('--seed',type=int,default=20261123)
    p.add_argument('--pause-after-preflight',action='store_true');p.add_argument('--continue-from-preflight',action='store_true')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--continue-prepared',action='store_true')
    args=p.parse_args();args.device=device_for('auto');seed_all(args.seed);cv2.setNumThreads(1)
    out=ROOT/args.output
    store=Store(ROOT/'data/preparado/deteccion');started=time.perf_counter()
    if args.continue_from_preflight or args.continue_prepared:
        config=json.loads((out/'config.json').read_text(encoding='utf-8'))
        if args.continue_from_preflight: assert json.loads((out/'preflight.json').read_text(encoding='utf-8'))['passed']
        assert not (out/'history.json').exists(), 'Training already started'
        assert digest(store.path)==config['manifest_sha256']
        store.db=json.loads((out/'snapshot/manifest.json').read_text(encoding='utf-8'))
        rows={s:store.records(s) for s in ('train','val')}
        if args.continue_prepared:
            config.update(device=str(args.device),gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,torch_version=str(torch.__version__),cuda_version=torch.version.cuda)
            write_json(out/'config.json',config)
            preflight(store,rows,out,args.device,config['seed'])
            if args.pause_after_preflight:return
        else:
            config.update(device=str(args.device),gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,torch_version=str(torch.__version__),cuda_version=torch.version.cuda)
            write_json(out/'config.json',config)
    else:
        if out.exists(): raise ValueError('Use a new experiment directory to preserve existing results')
        out.mkdir(parents=True)
        rows,config=freeze(out,store,args)
        config['cache_seconds']=cache(store,rows,out);write_json(out/'config.json',config)
        if args.prepare_only:return
        preflight(store,rows,out,args.device,args.seed)
        if args.pause_after_preflight: return
    train(rows,out,config,args.device)
    result=validation(store,rows,out,args.device,config)
    assert digest(store.path)==config['manifest_sha256'],'Original manifest changed during experiment'
    pre=json.loads((out/'preflight.json').read_text(encoding='utf-8'))
    history=json.loads((out/'history.json').read_text(encoding='utf-8'))
    write_json(out/'completed.json',dict(process_seconds=time.perf_counter()-started,
        measured_pipeline_seconds=config['cache_seconds']+pre['seconds']+history[-1]['seconds_elapsed']+result['seconds'],
        cache_seconds=config['cache_seconds'],preflight_seconds=pre['seconds'],training_seconds=history[-1]['seconds_elapsed'],validation_seconds=result['seconds'],
        finished_utc=datetime.now(timezone.utc).isoformat(),test_images_read=0,manual_batch_used=0,manifest_unchanged=True,metrics=result))
    print('COMPLETED',json.dumps(result),flush=True)


if __name__=='__main__':main()
