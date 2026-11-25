"""Revisa sólo grupos sospechosos ya identificados; usa huellas del CSV."""
import csv,json,re,itertools
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/preparacion_dataset'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    audit=json.loads((ROOT/'docs/auditoria_externos/auditoria.json').read_text(encoding='utf-8'))
    rows=list(csv.DictReader((ROOT/'docs/auditoria_externos/verificacion_imagenes.csv').open(encoding='utf-8-sig')))
    index={r['path']:r for r in rows}; results=[]; samples=[]
    for group in audit['roboflow_base_cross_split']:
        if 'canonical' not in group['source']: continue
        paths=group['images']; a=paths[0]; b=next(p for p in paths[1:] if index[p]['split']!=index[a]['split'])
        with Image.open(ROOT/a) as im: aa=np.asarray(im.convert('RGB').resize((96,96)),dtype=np.float32)
        with Image.open(ROOT/b) as im: bb=np.asarray(im.convert('RGB').resize((96,96)),dtype=np.float32)
        options=[(f'rot{k*90}',np.rot90(bb,k)) for k in range(4)]+[(f'flip_rot{k*90}',np.rot90(bb[:,::-1],k)) for k in range(4)]
        error,transform=min((float(np.mean(np.abs(aa-v)))/255,n) for n,v in options)
        result=dict(base=group['base'],paths=paths,sha256_equal=index[a]['sha256']==index[b]['sha256'],
                    rgb_mae_96=error,best_transform=transform,decision='same_origin_name_group',
                    rationale='Nombre anterior a .rf coincide; diferencias de bytes no demuestran independencia.')
        results.append(result)
        if len(samples)<4: samples.append((a,b,f'Bangladesh: mismo origen; MAE={error:.4f}'))
    for group in audit['inferred_recording_cross_split'][:2]:
        a=group['images'][0];b=next(p for p in group['images'][1:] if index[p]['split']!=index[a]['split'])
        samples.append((a,b,'Food: mismo identificador de grabación'))
    canvas=Image.new('RGB',(1000,6*260),'#172630');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    for i,(a,b,label) in enumerate(samples):
        draw.text((8,i*260+4),label,font=font,fill='white')
        for j,path in enumerate((a,b)):
            with Image.open(ROOT/path) as im:
                im=im.convert('RGB');im.thumbnail((480,210));canvas.paste(im,(j*500,i*260+30))
            draw.text((j*500,i*260+242),index[path]['split']+' · '+Path(path).name[:48],font=font,fill='white')
    canvas.save(OUT/'evidencia_cruces.jpg',quality=90)
    report=dict(bangladesh_canonical=results,food_recording_groups=audit['inferred_recording_cross_split'],
                visual_samples=[dict(left=a,right=b,note=n) for a,b,n in samples],
                note='MAE es evidencia auxiliar, no un modelo ni detector de duplicados infalible. Agrupacion conservadora por procedencia.')
    (OUT/'evidencia_grupos.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('canonical_groups',len(results),'MAE_minmax',min(r['rgb_mae_96'] for r in results),max(r['rgb_mae_96'] for r in results))

if __name__=='__main__':main()
