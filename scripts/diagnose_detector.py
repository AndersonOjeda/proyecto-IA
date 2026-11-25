"""Validation-only raw-candidate audit, AP50 and operating-point diagnostics."""
import argparse, csv, hashlib, json, sys, time
from collections import Counter, defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import torch
from PIL import Image,ImageDraw
from torch.utils.data import DataLoader
from residuos import CLASSES
from residuos.data import Store,read_json,write_json,digest
from residuos.detector import GRID,SIZE,encode,load_model,device_for,seed_all,loss_fn
from residuos.evaluate import metrics
from scripts.experiment_detector import Cached,original


def overlaps(a,b):
    a=np.asarray(a).reshape(-1,4);b=np.asarray(b).reshape(-1,4)
    lo=np.maximum(a[:,None,:2],b[None,:,:2]);hi=np.minimum(a[:,None,2:],b[None,:,2:])
    inter=np.maximum(hi-lo,0).prod(2)
    areas=lambda x:np.maximum(x[:,2:]-x[:,:2],0).prod(1)
    return inter/np.maximum(areas(a)[:,None]+areas(b)[None,:]-inter,1e-12)


def candidates(z,r):
    z=torch.as_tensor(z);wh=z[...,1:5].sigmoid().numpy().reshape(-1,4)
    gy,gx,slot=np.indices((GRID,GRID,3));gx=gx.ravel();gy=gy.ravel()
    cx=(gx+wh[:,0])/GRID;cy=(gy+wh[:,1])/GRID
    xy=np.stack([cx-wh[:,2]/2,cy-wh[:,3]/2,cx+wh[:,2]/2,cy+wh[:,3]/2],1).clip(0,1)
    xy*=np.array([r['width'],r['height']]*2)
    prob,cls=z[...,5:].softmax(-1).max(-1)
    scores=(z[...,0].sigmoid()*prob).numpy().ravel()
    return xy,cls.numpy().ravel(),scores


def nms_indices(xy,classes,scores,threshold=.45):
    order=np.argsort(-scores,kind='stable');kept=[]
    while len(order):
        k=order[0];kept.append(int(k));order=order[1:]
        if not len(order):break
        suppress=(classes[order]==classes[k])&(overlaps(xy[k:k+1],xy[order])[0]>threshold)
        order=order[~suppress]
    return np.array(kept,dtype=int)


def match(xy,cls,scores,truth):
    gt=np.array([b['xyxy'] for b in truth]).reshape(-1,4);gc=np.array([CLASSES.index(b['class']) for b in truth])
    ov=overlaps(xy,gt);used=set();hits=np.zeros(len(xy),bool);matched=np.full(len(xy),-1,int);reasons=[]
    for i in np.argsort(-scores,kind='stable'):
        eligible=[j for j in range(len(gt)) if gc[j]==cls[i] and j not in used]
        value,j=max([(ov[i,j],j) for j in eligible],default=(0,-1))
        if value>=.5:hits[i]=True;matched[i]=j;used.add(j)
    for i in range(len(xy)):
        if hits[i]:reason='tp'
        elif np.any((ov[i]>=.5)&(gc==cls[i])):reason='duplicate'
        elif np.any(ov[i]>=.5):reason='wrong_class'
        elif np.any(ov[i]>=.1):reason='localization'
        else:reason='background'
        reasons.append(reason)
    return hits,matched,reasons,ov


def ap50(scores,hits,objects):
    if not objects:return None
    order=np.argsort(-np.asarray(scores),kind='stable');tp=np.cumsum(np.asarray(hits,dtype=int)[order]);fp=np.arange(1,len(order)+1)-tp
    recall=tp/objects;precision=tp/np.maximum(tp+fp,1)
    # Explicit 101-point interpolated AP; absent recall bins contribute zero.
    envelope=np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.mean([envelope[np.searchsorted(recall,r)] if np.searchsorted(recall,r)<len(envelope) else 0 for r in np.linspace(0,1,101)]))


def operating_point(scores,hits,objects):
    scores=np.asarray(scores);hits=np.asarray(hits,dtype=int)
    if not len(scores) or not hits.any():return dict(threshold=1.,tp=0,fp=0,fn=objects,micro_f1=0.)
    order=np.argsort(-scores,kind='stable');s=scores[order];tp=np.cumsum(hits[order]);ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    f1=2*tp[ends]/(ends+1+objects)
    best=ends[np.argmax(f1)]
    return dict(threshold=float(s[best]),tp=int(tp[best]),fp=int(best+1-tp[best]),fn=int(objects-tp[best]),micro_f1=float(2*tp[best]/(best+1+objects)))


def size_bin(xy,r):
    area=(xy[2]-xy[0])*(xy[3]-xy[1])*SIZE*SIZE/(r['width']*r['height'])
    return 'small_<32px' if area<32**2 else 'medium_32-96px' if area<96**2 else 'large_>=96px'


def audit_pipeline(rows,store,base,out):
    dataset=Cached(rows,base/'val.npy');errors=[];maxerr=0.;objects=0;ids=Counter()
    for i,r in enumerate(rows):
        _,target=dataset[i]
        assert int(target[...,0].sum())==len(r['boxes'])
        for b in r['boxes']:
            c=CLASSES.index(b['class']);ids[c]+=1
        for gy,gx,sl in (target[...,0]>0).nonzero().tolist():
            _,dx,dy,w,h,c=target[gy,gx,sl].tolist();cx=(gx+dx)/GRID;cy=(gy+dy)/GRID
            xy=np.array([(cx-w/2)*r['width'],(cy-h/2)*r['height'],(cx+w/2)*r['width'],(cy+h/2)*r['height']])
            err=min(np.max(np.abs(xy-np.array(b['xyxy']))) for b in r['boxes'] if b['class']==CLASSES[int(c)])
            maxerr=max(maxerr,float(err));objects+=1
    # Pixel cache, dimensions and RGB ordering against the actual loader on seeded samples.
    sample=sorted(range(len(rows)),key=lambda i:hashlib.sha256(('20261123'+rows[i]['id']).encode()).hexdigest())[:12]
    for i in sample:
        r=rows[i];frame=original(store,r);want=cv2.cvtColor(cv2.resize(frame,(SIZE,SIZE)),cv2.COLOR_BGR2RGB)
        assert np.array_equal(dataset.pixels[i],want)
    # An oracle prediction exercises slot/class indexing, loss target and decode/evaluation.
    truth=[{'class':c,'xyxy':[15+30*i,20,35+30*i,60]} for i,c in enumerate(CLASSES)]
    t=encode(truth,256,256);z=torch.full((16,16,3,11),-20.)
    pos=t[...,0]>0;z[...,1:5][pos]=torch.logit(t[...,1:5][pos].clamp(1e-6,1-1e-6));z[...,0][pos]=20
    for gy,gx,sl in pos.nonzero().tolist():z[gy,gx,sl,5+int(t[gy,gx,sl,5])]=20
    xy,c,s=candidates(z,dict(width=256,height=256));keep=s>.5;h,_,_,_=match(xy[keep],c[keep],s[keep],truth)
    assert h.sum()==6 and np.isfinite(float(loss_fn(z[None],t[None]))) and maxerr<.01
    result=dict(objects_checked=objects,class_ids=dict(ids),max_coordinate_error_original_pixels=maxerr,
                cache_samples_checked=12,oracle_tp=int(h.sum()),oracle_loss=float(loss_fn(z[None],t[None])),
                found_index_or_coordinate_bug=False,classes=list(CLASSES),test_images_read=0)
    write_json(out/'pipeline_checks.json',result)
    return result


def analyze(base,out,checkpoint,threshold=None):
    started=time.perf_counter();out.mkdir(parents=True,exist_ok=True);seed_all(20261123);cv2.setNumThreads(1)
    config=read_json(base/'config.json');store=Store(config['data_root']);store.db=read_json(base/'snapshot/manifest.json')
    rows=store.records('val');ids=[r['id'] for r in rows]
    assert ids==read_json(base/'snapshot/eligible_ids.json')['val']
    ckhash=digest(checkpoint);rawpath=out/'raw_validation.npz'
    if rawpath.exists():
        saved=np.load(rawpath);assert saved['checkpoint_sha256'].item()==ckhash and saved['ids'].tolist()==ids;raw=saved['logits']
    else:
        model,_=load_model(checkpoint,device_for('auto'));outputs=[]
        with torch.no_grad():
            for x,_ in DataLoader(Cached(rows,base/'val.npy'),batch_size=16):outputs.append(model(x.to(device_for('auto'))).cpu().numpy())
        raw=np.concatenate(outputs);np.savez_compressed(rawpath,logits=raw,ids=np.array(ids),checkpoint_sha256=ckhash)
    if threshold is None:threshold=read_json(checkpoint.parent/'threshold.json')['threshold']
    stage_names=['raw_any_class','raw_correct_class','threshold_correct','nms_correct','capped_correct']
    counts={c:dict(objects=0,**{k:0 for k in stage_names}) for c in CLASSES}
    objects=[];pred_details=[];all_scores=[];all_hits=[];all_cls=[];finalpairs=[];visual_pool=defaultdict(list)
    source_size=defaultdict(lambda:Counter());loss_terms=[]
    for i,r in enumerate(rows):
        xy,cl,sc=candidates(raw[i],r);gt=r['boxes'];gc=np.array([CLASSES.index(b['class']) for b in gt]);ov=overlaps(xy,[b['xyxy'] for b in gt])
        nk=nms_indices(xy,cl,sc);capped=nk[:100];final=capped[sc[capped]>=threshold]
        hits,matched,reasons,_=match(xy[capped],cl[capped],sc[capped],gt)
        all_scores.extend(sc[capped].tolist());all_hits.extend(hits.tolist());all_cls.extend(cl[capped].tolist())
        filtered=sc[capped]>=threshold
        used=set(matched[filtered&hits].tolist())
        finalpairs.append(([{'class':CLASSES[cl[k]],'xyxy':xy[k].tolist(),'score':float(sc[k])} for k in final],gt))
        for j,b in enumerate(gt):
            c=b['class'];correct=cl==gc[j];good=ov[:,j]>=.5
            flags=[bool(good.any()),bool((good&correct).any()),bool((good&correct&(sc>=threshold)).any()),bool(((ov[nk,j]>=.5)&(cl[nk]==gc[j])&(sc[nk]>=threshold)).any()),bool(((ov[final,j]>=.5)&(cl[final]==gc[j])).any())]
            counts[c]['objects']+=1
            for k,v in zip(stage_names,flags):counts[c][k]+=int(v)
            row=dict(id=r['id'],source=r['source'],object_index=j,original_class=c,size=size_bin(b['xyxy'],r),
                     width_256=(b['xyxy'][2]-b['xyxy'][0])*256/r['width'],height_256=(b['xyxy'][3]-b['xyxy'][1])*256/r['height'],
                     max_iou_any=float(ov[:,j].max()),max_iou_correct=float(ov[correct,j].max()) if correct.any() else 0,
                     best_correct_score=float(sc[good&correct].max()) if (good&correct).any() else None,
                     matched_final=j in used,**dict(zip(stage_names,flags)))
            objects.append(row)
            if c=='organico':
                key=(r['source'],row['size']);source_size[key]['objects']+=1;source_size[key]['tp']+=j in used;source_size[key]['fn']+=j not in used
                for k,v in zip(stage_names,flags):source_size[key][k]+=v
                if j not in used:visual_pool['fn'].append(dict(image_index=i,object_index=j,box=b['xyxy'],kind='fn'))
        for q in np.flatnonzero(filtered):
            k=capped[q];row=dict(id=r['id'],source=r['source'],candidate=int(k),predicted_class=CLASSES[cl[k]],score=float(sc[k]),reason=reasons[q],matched_object=int(matched[q]),size=size_bin(xy[k],r),xyxy=xy[k].tolist())
            pred_details.append(row)
            if cl[k]==5:
                if not hits[q]:source_size[(r['source'],row['size'])]['fp_predicted_size']+=1
                kind='tp' if hits[q] else 'fp';visual_pool[kind].append(dict(image_index=i,candidate=int(k),object_index=int(matched[q]),box=xy[k].tolist(),kind=kind,reason=reasons[q]))
        t=encode(gt,r['width'],r['height']);z=torch.tensor(raw[i]);positive=t[...,0]==1
        bce=torch.nn.functional.binary_cross_entropy_with_logits(z[...,0],t[...,0],reduction='none')
        loss_terms.append(dict(obj=float((bce*torch.where(positive,5.,.2)).mean()),box8=float(8*torch.nn.functional.smooth_l1_loss(z[...,1:5][positive].sigmoid(),t[...,1:5][positive])),cls=float(torch.nn.functional.cross_entropy(z[...,5:][positive],t[...,5][positive].long()))))
    scores=np.array(all_scores);hits=np.array(all_hits);classes=np.array(all_cls);ap={};per_class_threshold={}
    for j,c in enumerate(CLASSES):
        select=classes==j;ap[c]=ap50(scores[select],hits[select],counts[c]['objects']);per_class_threshold[c]=operating_point(scores[select],hits[select],counts[c]['objects'])
    best=operating_point(scores,hits,len(objects));baseline_metrics=metrics(finalpairs)
    result=dict(checkpoint_sha256=ckhash,threshold=threshold,stages=counts,metrics=baseline_metrics,ap50=ap,map50=float(np.mean(list(ap.values()))),
                ap_protocol='101-point interpolated AP50; same-class greedy matching; all scores including below .01; classwise NMS .45; max 100 detections/image; six-class macro mean',
                stage_protocol='GT coverage (not one-to-one TP): any matching candidate IoU>=.5; raw uses all 768 top-1-class candidates without threshold/NMS/cap',
                duplicates=dict(Counter(x['predicted_class'] for x in pred_details if x['reason']=='duplicate')),
                fp_reasons=dict(Counter(x['reason'] for x in pred_details if x['reason']!='tp')),
                best_global_threshold=best,best_per_class_threshold=per_class_threshold,
                mean_loss_components={k:float(np.mean([x[k] for x in loss_terms])) for k in ('obj','box8','cls')},
                seconds=time.perf_counter()-started,test_images_read=0)
    write_json(out/'diagnosis.json',result);write_json(out/'organic_by_source_size.json',[dict(source=s,size=b,**dict(v)) for (s,b),v in sorted(source_size.items())])
    write_json(out/'final_predictions_diagnosed.json',pred_details)
    with (out/'objects.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,list(objects[0]));w.writeheader();w.writerows(objects)
    audit_pipeline(rows,store,base,out)
    canvas=Image.new('RGB',(1500,1200),'#182532');chosen=[]
    for col,kind in enumerate(('tp','fp','fn')):
        pool=sorted(visual_pool[kind],key=lambda q:hashlib.sha256((str(20261123)+rows[q['image_index']]['id']+str(q.get('candidate',q['object_index']))).encode()).hexdigest())
        used=set();selected=[]
        for q in pool:
            if q['image_index'] in used:continue
            selected.append(q);used.add(q['image_index'])
            if len(selected)==3:break
        for row,q in enumerate(selected):
            r=rows[q['image_index']];frame=original(store,r);im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(im)
            for b in r['boxes']:
                if b['class']=='organico':d.rectangle(b['xyxy'],outline='#00ff80',width=3)
            if kind!='fn':d.rectangle(q['box'],outline='#ffb000',width=4)
            box=q['box'];cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2;rad=max(96,(box[2]-box[0]),(box[3]-box[1]))
            crop=im.crop((max(0,cx-rad),max(0,cy-rad),min(im.width,cx+rad),min(im.height,cy+rad)));crop.thumbnail((250,320));im.thumbnail((245,320))
            x,y=col*500,row*400;canvas.paste(im,(x,y+60));canvas.paste(crop,(x+250,y+60));ImageDraw.Draw(canvas).text((x+5,y+5),kind.upper()+' '+r['id']+'\n'+q.get('reason','omitted')+' | green=GT orange=prediction',fill='white')
            chosen.append(dict(q,id=r['id'],source=r['source']))
    canvas.save(out/'organic_review.jpg');write_json(out/'organic_review_samples.json',dict(seed=20261123,samples=chosen))
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',default='runs/detector_exp01');p.add_argument('--output',default='runs/detector_exp02/diagnosis_baseline');p.add_argument('--checkpoint',default='runs/detector_exp01/best.pt');p.add_argument('--threshold',type=float)
    a=p.parse_args();analyze(ROOT/a.base,ROOT/a.output,ROOT/a.checkpoint,a.threshold)
