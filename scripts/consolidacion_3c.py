"""Consolidation: checkpoint SHA-256/provenance, final experiment table, minority-class data audit.
Test split is NOT evaluated here (only counted)."""
from pathlib import Path
import sys, json, hashlib, collections
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from residuos import CLASSES
from residuos.data import Store, read_json, write_json

OUT = ROOT / 'runs/consolidacion'
OUT.mkdir(parents=True, exist_ok=True)

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

# experiments: checkpoint -> evaluation produced by the shared evaluator (measured, not typed)
EXPS = [
    ('Exp1 baseline', 'runs/detector_exp01/best.pt', None),
    ('Exp3C corrected (REFERENCIA)', 'runs/detector_exp03/corrected/best.pt', 'runs/detector_exp04/eval_3c_homogeneous.json'),
    ('Exp4A broad ep17', 'runs/detector_exp04/variant_4a_broad/best_val_loss.pt', 'runs/detector_exp04/variant_4a_broad/evaluation_best_val_loss.json'),
    ('Exp4A broad ep40', 'runs/detector_exp04/variant_4a_broad/best.pt', 'runs/detector_exp04/variant_4a_broad/evaluation.json'),
    ('Exp4B domain aligned', 'runs/detector_exp04/best.pt', 'runs/detector_exp04/evaluation.json'),
    ('Exp4C pure copy-paste', 'runs/detector_exp04/variant_4c_copypaste/best.pt', 'runs/detector_exp04/variant_4c_copypaste/evaluation.json'),
    ('Exp4C pure best_val_loss', 'runs/detector_exp04/variant_4c_copypaste/best_val_loss.pt', 'runs/detector_exp04/variant_4c_copypaste/evaluation_best_val_loss.json'),
    ('Exp4C1 hybrid (INVALIDO: abortado 1 epoca)', 'runs/detector_exp04/variant_4c1_hybrid_copypaste/best.pt', 'runs/detector_exp04/variant_4c1_hybrid_copypaste/evaluation.json'),
]
table = []
for name, ck, ev in EXPS:
    p = ROOT / ck
    c = torch.load(p, map_location='cpu', weights_only=False)
    row = {'experiment': name, 'checkpoint': ck, 'sha256': sha(p), 'bytes': p.stat().st_size,
           'origin': c.get('origin'), 'epoch': c.get('epoch'), 'weights_type': c.get('weights_type', 'RAW'),
           'params': sum(v.numel() for v in c['state'].values()) if 'state' in c else None}
    if ev and (ROOT / ev).exists():
        e = read_json(ROOT / ev)
        ev_ck = str(e.get('checkpoint', '')).replace('\\', '/')
        op = e['best_operating_point']
        row.update({'evaluation': ev, 'eval_checkpoint_matches': ev_ck.endswith(ck.split('runs/')[-1]) or ck.split('/')[-2] in ev_ck,
                    'eval_epoch': e.get('best_epoch'), 'epoch_matches': e.get('best_epoch') == c.get('epoch'),
                    'mAP50': e['true_unthresholded_map50'], 'ap50': e['true_unthresholded_ap50'],
                    'op_threshold': op['threshold'], 'micro_f1': op['micro_f1'], 'precision': op['precision'],
                    'recall': op['recall'], 'tp': op['tp'], 'fp': op['fp'], 'fn': op['fn']})
        sw = e.get('threshold_sweep', {}).get('0.40') or e.get('threshold_sweep', {}).get('0.4')
        if sw: row['at_040'] = {k: sw.get(k) for k in ('micro_f1', 'precision', 'recall', 'tp', 'fp', 'fn')}
    elif ck.endswith('exp01/best.pt'):
        r = ROOT / 'runs/detector_exp01/evaluation.json'
        row['note'] = 'arquitectura distinta (grid32); metricas de su propio informe, no del evaluador DetectorS8'
    table.append(row)

# inference config of the frozen candidate
ref = table[1]
infer = {'model': 'DetectorS8', 'checkpoint': ref['checkpoint'], 'sha256': ref['sha256'], 'epoch': ref['epoch'],
         'input_size': [256, 256], 'color': 'RGB', 'normalization': 'x/255', 'decoder': 'scripts.experiment3_runner.decode_grid32',
         'score_threshold': 0.40, 'nms_iou': 0.45, 'classes': list(CLASSES),
         'selection_basis': 'max Micro-F1 en barrido homogeneo sobre val (670 img)', 'test_evaluated': False}
write_json(OUT / 'inferencia_3c_congelada.json', infer)

# minority-class data audit
store = Store(ROOT / 'data/preparado/deteccion')
splits = {s: store.records(s) for s in ('train', 'val', 'test')}
sample = splits['train'][0]
src_key = next((k for k in ('source', 'provenance', 'origin', 'dataset') if k in sample), None)
def size_bin(b, r):
    x1, y1, x2, y2 = b['xyxy']; s = max(x2 - x1, y2 - y1) * 256 / max(r.get('width') or 1, r.get('height') or 1)
    return 'small<32' if s < 32 else ('medium32-96' if s < 96 else 'large>=96')
audit = {'record_keys': sorted(sample.keys()), 'source_key': src_key, 'splits': {}}
for s, rows in splits.items():
    d = {c: {'objects': 0, 'images': set(), 'by_source': collections.Counter(), 'by_size': collections.Counter(),
             'tiny_boxes_lt4px': 0, 'boxes_out_of_bounds': 0, 'images_with_multiclass': 0} for c in CLASSES}
    for r in rows:
        cls_in = {b['class'] for b in r['boxes']}
        for b in r['boxes']:
            c = b['class']; x1, y1, x2, y2 = b['xyxy']
            d[c]['objects'] += 1; d[c]['images'].add(r['id'])
            d[c]['by_source'][str(r.get(src_key, '?'))] += 1
            d[c]['by_size'][size_bin(b, r)] += 1
            if min(x2 - x1, y2 - y1) < 4: d[c]['tiny_boxes_lt4px'] += 1
            if x1 < -1 or y1 < -1 or x2 > (r.get('width') or 1e9) + 1 or y2 > (r.get('height') or 1e9) + 1: d[c]['boxes_out_of_bounds'] += 1
        for c in cls_in:
            if len(cls_in) > 1: d[c]['images_with_multiclass'] += 1
    audit['splits'][s] = {c: {**{k: v for k, v in x.items() if k != 'images'}, 'images': len(x['images']),
                              'by_source': dict(x['by_source']), 'by_size': dict(x['by_size'])} for c, x in d.items()}
    if s == 'test':  # only counts, no predictions
        audit['splits'][s] = {c: {'objects': v['objects'], 'images': v['images']} for c, v in audit['splits'][s].items()}
# per-class operating errors of 3C on val (from measured eval)
e3 = read_json(ROOT / 'runs/detector_exp04/eval_3c_homogeneous.json')
audit['val_3c_errors'] = {'confusion_op': e3.get('confusion_matrix_at_op_th'), 'stages': e3.get('stages'),
                          'recall_by_class_and_size': e3.get('recall_by_class_and_size')}
write_json(OUT / 'auditoria_clases_minoritarias.json', audit)
write_json(OUT / 'tabla_experimentos.json', table)

for r in table:
    print(r['experiment'], r['epoch'], r['sha256'][:16], r.get('mAP50'), r.get('micro_f1'), r.get('op_threshold'), r.get('fp'), r.get('eval_checkpoint_matches'), r.get('epoch_matches'))
print('source_key', src_key, audit['record_keys'])
for s in ('train', 'val', 'test'):
    for c in ('papel', 'carton', 'vidrio', 'metal'):
        print(s, c, json.dumps(audit['splits'][s][c]))
print(json.dumps(audit['val_3c_errors']['stages'])[:1500])
