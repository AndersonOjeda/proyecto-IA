"""Reproduce Exp03C/Exp05 on identical validation and compare the real app worker."""
import csv,json,queue,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from residuos.data import Store,read_json,write_json,digest
from residuos.detector import load_model,seed_all,device_for
from residuos.evaluate import metrics
from residuos.app import Application
from scripts.experiment3_runner import decode_grid32,CachedDataset32
from scripts.experiment_detector import original

OUT=ROOT/'runs/verificacion_app_exp03_exp05'
SPECS=[('Exp03',ROOT/'runs/detector_exp03/corrected/best.pt',.40),('Exp05',ROOT/'runs/detector_exp05/best.pt',.35)]


def same_predictions(a,b):
    if len(a)!=len(b):return False,None,None
    ordered=lambda pp:sorted(pp,key=lambda p:(p['class'],*p['xyxy']))
    a,b=ordered(a),ordered(b);boxerr=scoreerr=0.
    for x,y in zip(a,b):
        if x['class']!=y['class']:return False,None,None
        boxerr=max(boxerr,max(abs(v-w) for v,w in zip(x['xyxy'],y['xyxy'])))
        scoreerr=max(scoreerr,abs(x['score']-y['score']))
    return boxerr<=.02 and scoreerr<=1e-5,boxerr,scoreerr


def main():
    start=time.perf_counter();OUT.mkdir(exist_ok=True);seed_all(42);cv2.setNumThreads(1);device=device_for('auto')
    store=Store(ROOT/'data/preparado/deteccion');rows=store.records('val')
    # Assert exactly the same IDs, labels and ordering as the frozen validation.
    frozen=read_json(ROOT/'runs/detector_exp01/snapshot/manifest.json')
    previous=Store(store.root);previous.db=frozen
    fields=lambda rr:[{k:r[k] for k in ('id','sha256','width','height','boxes','split','group')} for r in rr]
    assert fields(rows)==fields(previous.records('val'))
    exp3=Store(store.root);exp3.db=read_json(ROOT/'runs/detector_exp03/corrected/snapshot/manifest.json')
    assert fields(rows)==fields(exp3.records('val'))
    import hashlib
    fingerprint=hashlib.sha256(json.dumps(fields(rows),sort_keys=True).encode()).hexdigest()
    write_json(OUT/'validation_identity.json',dict(images=len(rows),objects=sum(len(r['boxes']) for r in rows),fingerprint=fingerprint,ids=[r['id'] for r in rows],test_images_read=0))
    dataset=CachedDataset32(rows,ROOT/'runs/detector_exp01/val.npy',augment=False)
    models={};results={};allpreds={};baseline_files={}
    for name,cp,th in SPECS:
        baseline_files[str(cp)]=digest(cp)
        model,meta=load_model(cp,device);models[name]=model
        cache=OUT/(name+'_predictions.json')
        key=dict(checkpoint_sha256=digest(cp),validation_fingerprint=fingerprint,threshold=th,nms=.45,max_detections=100,inference_batch=1,decoder_sha256=digest(ROOT/'scripts/experiment3_runner.py'))
        if cache.exists() and read_json(cache)['compatibility']==key:
            saved=read_json(cache);preds=[r['predictions'] for r in saved['images']];assert [r['id'] for r in saved['images']]==[r['id'] for r in rows]
            reused=True
        else:
            preds=[];idx=0
            with torch.no_grad():
                for x,_ in DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0):
                    for z in model(x.to(device)).cpu():
                        r=rows[idx];preds.append(decode_grid32(z,r['width'],r['height'],threshold=th,nms=.45));idx+=1
            write_json(cache,dict(compatibility=key,images=[dict(id=r['id'],predictions=p) for r,p in zip(rows,preds)]));reused=False
        met=metrics([(p,r['boxes']) for p,r in zip(preds,rows)])
        met.update(precision=met['tp']/(met['tp']+met['fp']),recall=met['tp']/(met['tp']+met['fn']))
        assert sum(n['objects'] for n in met['classes'].values())==1564
        assert all(sum(n[k] for n in met['classes'].values())==met[k] for k in ('tp','fp','fn'))
        results[name]=dict(**met,threshold=th,checkpoint=str(cp),checkpoint_sha256=key['checkpoint_sha256'],epoch=meta['epoch'],saved_predictions_reused=reused)
        allpreds[name]=preds
        # A sidecar is bound to the actual checkpoint; never borrow another model's calibration.
        sidecar=cp.parent/'threshold.json'
        if sidecar.exists():
            old=read_json(sidecar)
            if old.get('checkpoint_sha256')!=key['checkpoint_sha256'] or old.get('threshold')!=th:
                write_json(OUT/(name+'_threshold_before.json'),old)
        write_json(sidecar,dict(threshold=th,checkpoint_sha256=key['checkpoint_sha256'],manifest_sha256=digest(store.path),
            validation_fingerprint=fingerprint,split='val',criterion='User-requested operating point reproduced on original validation; no recalibration or training',
            nms=.45,iou=.5,max_detections=100,verified_metrics=str(OUT/'results.json')))
        print(name,json.dumps(results[name]),flush=True)
    # Run the app's actual worker over all identical frames, including its preprocessing and decoder.
    apps={};checks={n:dict(frames=0,mismatches=[],max_box_error_pixels=0.,max_score_error=0.) for n,_,_ in SPECS}
    for name,cp,th in SPECS:
        app=Application.__new__(Application);app.model=models[name];app.device=device;app.threshold=th;app.results=queue.Queue();apps[name]=app
    for i,r in enumerate(rows):
        frame=original(store,r)
        expected_rgb=cv2.cvtColor(cv2.resize(frame,(256,256)),cv2.COLOR_BGR2RGB)
        assert np.array_equal(dataset.pixels[i],expected_rgb),'Original and cached pixels differ: '+r['id']
        for name,cp,th in SPECS:
            app=apps[name];app.worker(frame,None,0);_,pred,view,error=app.results.get()
            assert error is None,error
            ok,be,se=same_predictions(pred,allpreds[name][i]);q=checks[name];q['frames']+=1
            if not ok:q['mismatches'].append(dict(id=r['id'],reference_count=len(allpreds[name][i]),app_count=len(pred),box_error=be,score_error=se))
            if be is not None:q['max_box_error_pixels']=max(q['max_box_error_pixels'],be)
            if se is not None:q['max_score_error']=max(q['max_score_error'],se)
            del view
        if (i+1)%100==0:print('app parity frames',i+1,flush=True)
    write_json(OUT/'app_parity.json',checks)
    assert all(not c['mismatches'] for c in checks.values()),'App/evaluator detection mismatch; inspect parity file'
    report5=read_json(ROOT/'runs/detector_exp05/evaluation.json')['threshold_sweep']['0.35']
    report_compare=dict(Exp05={k:dict(report=report5[k],reproduced=results['Exp05'][k],matches=report5[k]==results['Exp05'][k]) for k in ('tp','fp','fn')},
        Exp03_report_at_040=dict(reported_precision_percent=22.61,reported_fp=1133,reported_micro_f1_percent=21.86,
            reproduced_fp=results['Exp03']['fp'],reproduced_precision_percent=results['Exp03']['precision']*100,reproduced_micro_f1_percent=results['Exp03']['micro_f1']*100,
            note='Exp03 corrected/evaluation.json only swept .35 and .50 around .40; the .40 claim is in Exp05/RESULTADOS.md'))
    write_json(OUT/'report_comparison.json',report_compare)
    for p,h in baseline_files.items():assert digest(p)==h,'Checkpoint modified'
    write_json(OUT/'results.json',dict(models=results,protocol=dict(split='val',images=670,objects=1564,iou=.5,matching='score-descending greedy, same class, one-to-one',nms=.45,max_detections=100),
        device=str(device),seconds=time.perf_counter()-start,test_images_read=0,training_performed=False,checkpoint_files_unchanged=True))
    with (OUT/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,['model','threshold','class','objects','tp','fp','fn','precision','recall']);w.writeheader()
        for name,res in results.items():
            for c,n in res['classes'].items():w.writerow(dict(model=name,threshold=res['threshold'],**{'class':c},**{k:n[k] for k in w.fieldnames[3:]}))
    print('COMPLETED',json.dumps(report_compare),flush=True)


if __name__=='__main__':main()
