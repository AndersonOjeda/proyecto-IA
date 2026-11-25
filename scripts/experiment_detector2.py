"""Experiment 2B: scratch training with only the localization loss changed."""
import json,shutil,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import torch
from residuos.data import Store,read_json,write_json,digest
from residuos.detector import device_for,seed_all
from residuos.localization_loss import loss_fn_giou
from scripts import experiment_detector as baseline_runner
from scripts.diagnose_detector import analyze


def main():
    base=ROOT/'runs/detector_exp01';out=ROOT/'runs/detector_exp02/giou'
    if out.exists():raise ValueError('Never overwrite an experiment')
    out.mkdir(parents=True);snap=out/'snapshot';snap.mkdir()
    for name in ('manifest.json','partitions.csv','eligible_ids.json','grupos_particiones.json'):
        shutil.copy2(base/'snapshot'/name,snap/name)
    for source in (Path(__file__),ROOT/'residuos/localization_loss.py',ROOT/'scripts/diagnose_detector.py',ROOT/'residuos/detector.py'):
        shutil.copy2(source,snap/source.name)
    config=read_json(base/'config.json');config.update(
        experiment='02B',main_change='box loss: mean(1-GIoU) replacing mean SmoothL1; coefficient remains 8',
        started_utc=datetime.now(timezone.utc).isoformat(),baseline_checkpoint_sha256=digest(base/'best.pt'),
        evidence='796/1564 GT lack any IoU>=.5 candidate; organic 429/757, mostly small. Threshold-only micro-F1 .11098 to .11238.',
        initialization='fresh random Detector including backbone; no checkpoint loaded for training',
        checkpoint_selection='minimum validation loss with GIoU; loss values not directly comparable to baseline SmoothL1',
        loss_implementation_sha256=digest(ROOT/'residuos/localization_loss.py'))
    write_json(out/'config.json',config)
    # Reuse exact cached pixels from baseline; no decoding of test or re-partitioning.
    for split in ('train','val'):(out/(split+'.npy')).hardlink_to(base/(split+'.npy'))
    store=Store(config['data_root']);store.db=read_json(snap/'manifest.json');rows={s:store.records(s) for s in ('train','val')}
    assert {s:[r['id'] for r in rr] for s,rr in rows.items()}==read_json(snap/'eligible_ids.json')
    device=device_for('auto');seed_all(config['seed']);cv2.setNumThreads(1)
    assert str(device)==config['device']
    # Injection is scoped to this runner process; the baseline implementation/file is untouched.
    baseline_runner.loss_fn=loss_fn_giou
    start=time.perf_counter();baseline_runner.preflight(store,rows,out,device,config['seed'])
    baseline_runner.train(rows,out,config,device)
    metrics=baseline_runner.validation(store,rows,out,device,config)
    diag=analyze(out,out/'diagnosis',out/'best.pt')
    write_json(out/'completed.json',dict(seconds=time.perf_counter()-start,metrics=metrics,ap50=diag['ap50'],map50=diag['map50'],test_images_read=0,manual_batch_used=0,
        finished_utc=datetime.now(timezone.utc).isoformat(),manifest_sha256=digest(snap/'manifest.json')))
    print('COMPLETED',json.dumps(read_json(out/'completed.json')),flush=True)


if __name__=='__main__':main()
