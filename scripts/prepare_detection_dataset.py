"""Preparación conservadora a partir de la auditoría; no entrena modelos.

Reutiliza dimensiones/huellas verificadas del CSV. Sólo lee píxeles para
seleccionar una tanda diversa y generar la vista previa. Las ambigüedades
bloquean la escena completa. El resultado es compatible con el anotador.
"""
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_external_datasets import source_mapping
CLASSES=['plastico','papel','carton','vidrio','metal','organico']
SEED=20261123
BASE=ROOT/'data/preparado'
DEST=BASE/'deteccion'
REPORT=ROOT/'docs/preparacion_dataset'
TACO=ROOT/'data/externos/TACO-master/TACO-master/data'
REMAINS={'Apple-core','Apple-peel','Bone','Bone-fish','Egg-shell','Mussel-shell','Orange-peel','Pear-core','Pear-peel','Shrimp-shell'}

def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
def rel(path):return Path(path).relative_to(ROOT).as_posix()
def key(text):return hashlib.sha256(text.encode()).hexdigest()[:20]

def bbox_valid(b,w,h):return len(b)==4 and all(math.isfinite(v) for v in b) and 0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h

def origin_group(source,name,photo_url=None):
    raw=Path(name).name.split('.rf.')[0]
    if source=='Food Waste':
        recording=re.search(r'(202\d{11}_\d{6})',raw)
        if recording:return 'food_recording:'+recording[1], 'recording_id_in_filename'
    if source=='Bangladesh canonical':
        raw=re.sub(r'[-_]copy(?:[-_]\d+)?','',raw,flags=re.I).lower()
    if source=='TACO oficial' and photo_url:return 'taco_photo:'+photo_url.rsplit('/',1)[-1], 'photo_url'
    return source+':'+raw,'raw_filename_before_augmentation'

def capacity(boxes,w,h):
    counts=Counter()
    for b in boxes:
        x1,y1,x2,y2=b['xyxy'];counts[(min(15,int((x1+x2)/2/w*16)),min(15,int((y1+y2)/2/h*16)))]+=1
    return max(counts.values(),default=0)<=3

def repair_taco():
    """Sólo clipping inequívoco de <=2 px y desambiguación de IDs en copia."""
    official=load(TACO/'annotations.json'); unofficial=load(TACO/'annotations_unofficial.json')
    decisions=[];counts={}
    for filename,doc in [('annotations.json',official),('annotations_unofficial.json',unofficial)]:
        ims={r['id']:r for r in doc['images']};bad=[];seen=set();duplicates=[]
        for n,a in enumerate(doc['annotations']):
            if a['id'] in seen:duplicates.append(dict(index=n,id=a['id'],image_id=a['image_id']))
            seen.add(a['id']);im=ims[a['image_id']];x,y,w,h=a['bbox'];box=[x,y,x+w,y+h]
            if not bbox_valid(box,im['width'],im['height']):bad.append(dict(index=n,id=a['id'],image_id=a['image_id'],bbox=a['bbox']))
        counts[filename]=dict(invalid_boxes=bad,duplicate_ids=duplicates)
    maxid=max(a['id'] for a in official['annotations']);seen=set();ims={im['id']:im for im in official['images']}
    for index,a in enumerate(official['annotations']):
        a['original_annotation_id']=a['id'];a['original_annotation_index']=index
        if a['id'] in seen:
            maxid+=1;old=a['id'];a['id']=maxid
            decisions.append(dict(file='annotations.json',index=index,old_id=old,new_id=maxid,action='reasignar_id_duplicado',reason='Objeto diferente (imagen/clase/caja); conservar ambos registros.'))
        seen.add(a['id']);im=ims[a['image_id']];x,y,w,h=a['bbox'];box=[x,y,x+w,y+h]
        if not bbox_valid(box,im['width'],im['height']):
            clipped=[max(0,box[0]),max(0,box[1]),min(im['width'],box[2]),min(im['height'],box[3])]
            if bbox_valid(clipped,im['width'],im['height']) and max(abs(c-b) for c,b in zip(clipped,box))<=2:
                a['original_bbox']=a['bbox'][:];a['bbox']=[clipped[0],clipped[1],clipped[2]-clipped[0],clipped[3]-clipped[1]]
                decisions.append(dict(file='annotations.json',index=index,id=a['id'],image_id=a['image_id'],action='interseccion_caja_con_limites',before=a['original_bbox'],after=a['bbox'],reason='Desborde marginal <=2 px; no se infiere geometria oculta. Segmentacion original conservada, no usada para deteccion.'))
            else:decisions.append(dict(file='annotations.json',index=index,id=a['id'],action='pendiente_no_inequivoco'))
    save(BASE/'taco_oficial/annotations_corregidas.json',official)
    save(REPORT/'correcciones_taco.json',dict(files=counts,decisions=decisions,unofficial_policy='41 cajas se documentan, no se descargan ni corrigen datos no oficiales.'))
    return official

class Preparation:
    def __init__(self):
        with (ROOT/'docs/auditoria_externos/verificacion_imagenes.csv').open(encoding='utf-8-sig') as f:self.audit=list(csv.DictReader(f))
        self.cached={r['path']:r for r in self.audit};self.rows=[];self.missing=[];self.source_class_totals=Counter()

    def add_image(self,source,path,cached,original_split,source_id,license_value,photo_url=None):
        group,evidence=origin_group(source,path.name,photo_url)
        row=dict(id=key(source+'|'+str(source_id)),path=os.path.relpath(path,DEST).replace('\\','/'),
                 original_path=rel(path),source=source,source_image_id=source_id,original_split=original_split,
                 original_group=group,group_evidence=evidence,group=group,session=group,split=None,video=None,frame=None,
                 width=int(cached['width']),height=int(cached['height']),sha256=cached['sha256'],available=True,
                 provenance=dict(image_url=photo_url,license=license_value,license_note='Metadato conservado; CC sin version y null no implican licencia aclarada.' if source=='TACO oficial' else 'Ver fuentes locales de auditoria.'),
                 boxes=[],pending_boxes=[],source_annotations=[],pending_reasons=[],reviewed=False,eligible=False,
                 review_status='pending',manual_batch=False)
        self.rows.append(row);return row

    def add_box(self,row,cid,name,b,annotation_id,fmt,force_reason=None,source_geometry=None):
        b=[round(v,6) for v in b]
        mapping,status,note=source_mapping(row['source'],name)
        if row['source']=='Food Waste':
            if name in REMAINS:mapping,status,note='organico','propuesto','Resto alimentario reconocible por clase; convencion autorizada.'
            else:mapping,status,note='organico' if name!='Other-waste' else '', 'ambiguo','Revisar visualmente alimento entero/condicion de residuo o material incierto.'
        row['source_annotations'].append(dict(id=annotation_id,class_id=cid,original_class=name,xyxy=b,format=fmt,target_proposal=mapping,geometry=source_geometry))
        if not bbox_valid(b,row['width'],row['height']):
            row['pending_reasons'].append('geometria_invalida');return
        if status in ('propuesto','condicional') and mapping in CLASSES and not force_reason:
            row['boxes'].append(dict(**{'class':mapping},xyxy=b,original_class=name,annotation_id=annotation_id))
        else:
            row['pending_boxes'].append(dict(**{'class':None},xyxy=b,original_class=name,annotation_id=annotation_id,target_proposal=mapping))
            row['pending_reasons'].append(force_reason or 'clase_ambigua:'+name)

    def bangladesh(self):
        for item in self.audit:
            if item['source']!='Bangladesh canonical_75-15-10':continue
            image=ROOT/item['path'];label=Path(item['annotation']);label=ROOT/label
            row=self.add_image('Bangladesh canonical',image,item,item['split'],item['image_id'],'CC BY 4.0 (declarada)')
            row['provenance']['annotation_file']=rel(label)
            for n,line in enumerate(label.read_text(encoding='utf-8').splitlines()):
                if not line.strip():continue
                values=list(map(float,line.split()));cid=int(values[0]);v=values[1:]
                name=['Paper','Plastic Bottle','Plastic bag','Plastic waste'][cid]
                if len(v)==4:
                    cx,cy,w,h=v;b=[(cx-w/2)*row['width'],(cy-h/2)*row['height'],(cx+w/2)*row['width'],(cy+h/2)*row['height']];fmt='yolo_bbox'
                else:
                    b=[min(v[::2])*row['width'],min(v[1::2])*row['height'],max(v[::2])*row['width'],max(v[1::2])*row['height']];fmt='bbox_de_poligono_humano'
                self.add_box(row,cid,name,b,n,fmt,source_geometry=v)

    def food(self):
        for path in sorted((ROOT/'data/externos').glob('Food Waste*/*/_annotations.coco.json')):
            doc=load(path);cats={r['id']:r['name'] for r in doc['categories']};byim=defaultdict(list)
            for a in doc['annotations']:byim[a['image_id']].append(a)
            for im in doc['images']:
                image=path.parent/im['file_name'];item=self.cached[rel(image)]
                row=self.add_image('Food Waste',image,item,path.parent.name,str(path.parent.name)+':'+str(im['id']),'CC BY 4.0 (declarada)')
                row['provenance']['annotation_file']=rel(path)
                for a in byim[im['id']]:
                    x,y,w,h=a['bbox'];self.add_box(row,a['category_id'],cats[a['category_id']],[x,y,x+w,y+h],a['id'],'coco_bbox')
                if not byim[im['id']]:row['pending_reasons'].append('sin_cajas_no_confirmado_negativo')

    def taco(self,doc):
        ledger={r['id']:r for r in load(BASE/'taco_oficial/descargas.json')}
        if len(ledger)!=1500:raise RuntimeError('Esperar a que terminen los 1500 intentos de TACO.')
        cats={r['id']:r['name'] for r in doc['categories']};byim=defaultdict(list)
        for a in doc['annotations']:byim[a['image_id']].append(a)
        for im in doc['images']:
            downloaded=ledger[im['id']]
            if downloaded['status']!='ok':self.missing.append(downloaded);continue
            image=BASE/'taco_oficial'/im['file_name']
            row=self.add_image('TACO oficial',image,dict(width=downloaded['actual_width'],height=downloaded['actual_height'],sha256=downloaded['sha256']),
                               'sin_particion',im['id'],im.get('license'),im.get('flickr_url'))
            row['provenance']['original_image_metadata']=im
            row['provenance']['annotation_file']=rel(TACO/'annotations.json')
            row['provenance']['corrected_annotation_file']=rel(BASE/'taco_oficial/annotations_corregidas.json')
            for a in byim[im['id']]:
                x,y,w,h=a['bbox'];self.add_box(row,a['category_id'],cats[a['category_id']],[x,y,x+w,y+h],a['id'],'coco_bbox_corregida' if 'original_bbox' in a else 'coco_bbox')

    def reserve(self):
        for item in self.audit:
            if item['source']!='dataset-resized':continue
            image=ROOT/item['path'];row=self.add_image('dataset-resized',image,item,'sin_particion',item['image_id'],'no_documentada_en_extraccion')
            row['label_hint']=image.parent.name
            row['target_hint']=dict(plastic='plastico',paper='papel',cardboard='carton',glass='vidrio',metal='metal').get(image.parent.name)
            row['pending_reasons'].append('trash_ambiguo' if image.parent.name=='trash' else 'reserva_sin_cajas')

    def finalize_states(self):
        for row in self.rows:
            if not row['boxes'] and not row['pending_boxes'] and not row['pending_reasons']:row['pending_reasons'].append('sin_cajas_no_confirmado_negativo')
            if row['boxes'] and not capacity(row['boxes']+row['pending_boxes'],row['width'],row['height']):row['pending_reasons'].append('capacidad_detector_mas_de_3_centros_por_celda')
            row['pending_reasons']=sorted(set(row['pending_reasons']))
            row['reviewed']=not row['pending_reasons'] and not row['pending_boxes']
            row['eligible']=row['reviewed'];row['review_status']='accepted_source_annotations' if row['reviewed'] else 'pending_human_review'

    def visual_review_overrides(self):
        decisions=REPORT/'revisiones_visuales.json'
        if not decisions.exists():return
        for decision in load(decisions):
            row=next(r for r in self.rows if r['id']==decision['image_id'])
            if decision['action']=='dejar_escena_pendiente':
                moved=[dict(b,**{'class':None},target_proposal=decision['target_proposal']) for b in row['boxes'] if b['annotation_id'] in decision['annotation_ids']]
                row['boxes']=[b for b in row['boxes'] if b['annotation_id'] not in decision['annotation_ids']]
                row['pending_boxes'].extend(moved)
                if decision['reason'] not in row['pending_reasons']:row['pending_reasons'].append(decision['reason'])
                row['reviewed']=False;row['eligible']=False;row['review_status']='pending_human_review'

    def groups_and_splits(self):
        # Unión transitiva de procedencia y SHA: nunca repartir variantes conocidas.
        parent=list(range(len(self.rows)))
        def find(i):
            while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
            return i
        def union(a,b):parent[find(b)]=find(a)
        group_seen={};hash_seen={}
        for i,r in enumerate(self.rows):
            for field,seen in [('original_group',group_seen),('sha256',hash_seen)]:
                value=r[field]
                if value in seen:union(i,seen[value])
                else:seen[value]=i
        components=defaultdict(list)
        for i in range(len(self.rows)):components[find(i)].append(self.rows[i])
        groups=[]
        for members in components.values():
            gid='g_'+key('|'.join(sorted({r['original_group'] for r in members})))
            sources=sorted({r['source'] for r in members})
            for r in members:r['group']=gid;r['session']=gid
            groups.append(dict(id=gid,members=members,sources=sources,ready_count=sum(r['eligible'] for r in members)))
        # Reparto por grupo dentro de cada fuente. Hash de semilla estable, sin usar test/modelos.
        buckets=defaultdict(list)
        for g in groups:buckets['|'.join(g['sources'])].append(g)
        for source,bucket in buckets.items():
            bucket.sort(key=lambda g:hashlib.sha256(f'{SEED}:{g["id"]}'.encode()).hexdigest())
            n=len(bucket);nval=max(1,round(n*.15)) if n>=3 else 0;ntest=max(1,round(n*.15)) if n>=3 else 0
            for i,g in enumerate(bucket):
                split='val' if i<nval else 'test' if i<nval+ntest else 'train'
                for r in g['members']:r['split']=split
        # Una copia con etiquetas incompletas no puede rehabilitar otra copia
        # idéntica cuyo contenido exige revisión.
        exact=defaultdict(list)
        for r in self.rows:exact[r['sha256']].append(r)
        for members in exact.values():
            if len(members)<2:continue
            signatures={tuple(sorted((b['class'],tuple(b['xyxy'])) for b in r['boxes'])) for r in members}
            reason='anotaciones_duplicadas_discrepantes' if len(signatures)>1 else 'copia_identica_con_revision_pendiente'
            if len(signatures)>1 or any(not r['reviewed'] for r in members):
                for r in members:
                    r['eligible']=False;r['reviewed']=False;r['review_status']='pending_human_review'
                    r.setdefault('pending_reasons',[]).append(reason)
        # Marcar copias binarias redundantes para no inflar imágenes/objetos efectivos.
        sha_seen={}
        for r in sorted(self.rows,key=lambda r:(not r['eligible'],r['id'])):
            if r['sha256'] in sha_seen:
                r['duplicate_of']=sha_seen[r['sha256']];r['eligible']=False;r['review_status']='duplicate_excluded'
            else:sha_seen[r['sha256']]=r['id']
        self.groups=groups

    def manual_batch(self):
        target_classes=['papel','carton','vidrio','metal']
        counts={c:sum(any(b['class']==c for b in r['boxes']) for r in self.rows if r['eligible']) for c in target_classes}
        order=sorted(target_classes,key=lambda c:(counts[c],c));quota=dict(zip(order,[30,25,20,15]))
        rng=random.Random(SEED);selected=[]
        for c in order:
            candidates=sorted([r for r in self.rows if r['source']=='dataset-resized' and r.get('target_hint')==c and not r.get('duplicate_of')],key=lambda r:r['id'])
            rng.shuffle(candidates);candidates=candidates[:min(100,len(candidates))]
            features=[]
            for r in candidates:
                # Selección por diversidad visual simple, nunca localización automática.
                with Image.open(ROOT/r['original_path']) as image:
                    thumb=np.asarray(image.convert('RGB').resize((16,16)),dtype=np.float32)/255
                gray=thumb.mean(2);feature=np.concatenate([thumb.mean((0,1)),thumb.std((0,1)),np.histogram(gray,bins=8,range=(0,1),density=False)[0]/256,
                    [float(np.abs(np.diff(gray,axis=0)).mean()),float(np.abs(np.diff(gray,axis=1)).mean()),r['width']/r['height']]])
                features.append(feature)
            features=np.stack(features);features=(features-features.mean(0))/(features.std(0)+1e-6)
            chosen=[0]
            while len(chosen)<min(quota[c],len(candidates)):
                distances=((features[:,None]-features[chosen][None])**2).sum(2).min(1);distances[chosen]=-1;chosen.append(int(distances.argmax()))
            for i in chosen:
                r=candidates[i];dest=DEST/'images/lote_manual'/c/Path(r['original_path']).name;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(ROOT/r['original_path'],dest)
                # Copia nueva: verificar sólo esta copia, no releer la colección.
                with dest.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==r['sha256']
                r['path']=os.path.relpath(dest,DEST).replace('\\','/');r['manual_batch']=True;selected.append(r['id'])
        save(REPORT/'lote_manual.json',dict(seed=SEED,ready_unique_images_before_batch=counts,priority=order,selected_per_class=quota,selected_ids=selected,
                                         method='Muestreo semilla fija de hasta 100 candidatos/clase y seleccion farthest-point con color/textura/aspecto; sin modelos ni cajas inventadas.'))

    def report(self):
        summary=[]
        for c in CLASSES:
            row={'clase':c}
            for split in ('train','val','test'):
                active=[r for r in self.rows if r['eligible'] and r['split']==split and any(b['class']==c for b in r['boxes'])]
                row[split+'_imagenes_unicas']=len({r['sha256'] for r in active})
                row[split+'_cajas_validas']=sum(sum(b['class']==c for b in r['boxes']) for r in active)
                row[split+'_grupos']=len({r['group'] for r in active})
            pending=[]
            for r in self.rows:
                proposals={b['class'] for b in r['boxes']}|{t for b in r['pending_boxes'] for t in b.get('target_proposal','').split('|')}|{r.get('target_hint')}
                if not r['eligible'] and not r.get('duplicate_of') and c in proposals:pending.append(r)
            row['pendientes_imagenes_asociadas']=len({r['sha256'] for r in pending})
            row['lote_manual']=sum(r['manual_batch'] and r.get('target_hint')==c for r in self.rows)
            summary.append(row)
        REPORT.mkdir(parents=True,exist_ok=True)
        with (REPORT/'resumen_seis_clases.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
        reasoncounts=Counter(reason for r in self.rows if not r.get('duplicate_of') for reason in r['pending_reasons'])
        save(REPORT/'resumen.json',dict(seed=SEED,classes=summary,total_images=len(self.rows),
             ready_images=sum(r['eligible'] for r in self.rows),
             pending_images=len({r['sha256'] for r in self.rows if not r['reviewed'] and not r.get('duplicate_of')}),
             pending_records_including_duplicates=sum(not r['reviewed'] for r in self.rows),
             duplicate_images=sum(bool(r.get('duplicate_of')) for r in self.rows),pending_reasons=dict(reasoncounts),
             taco_recovered=sum(r['source']=='TACO oficial' for r in self.rows),taco_failed=len(self.missing),
             note='Conteos efectivos excluyen escenas pendientes completas y duplicados binarios. Pendientes por clase pueden solaparse; cajas validas en escena pendiente NO cuentan como listas. Los 888 vacios por confirmar y las escenas sin material asignable se informan en motivos, no se adjudican artificialmente a una clase. Grupos de origen conocidos, no sesiones de captura independientes verificadas para fotos sin metadatos.'))
        save(REPORT/'grupos_particiones.json',dict(seed=SEED,policy='70/15/15 de grupos por fuente; cuota redondeada, orden SHA256(semilla:id). Ningun grupo cruza conjuntos.',
                groups=[dict(id=g['id'],sources=g['sources'],split=g['members'][0]['split'],images=[r['id'] for r in g['members']],
                original_splits=sorted({r['original_split'] for r in g['members']}),ready_images=sum(r['eligible'] for r in g['members'])) for g in self.groups]))
        with (REPORT/'manifiesto_imagenes.csv').open('w',encoding='utf-8-sig',newline='') as f:
            fields=['id','source','original_path','path','sha256','group','group_evidence','original_split','split','review_status','eligible','pending_reasons','boxes','pending_boxes','manual_batch']
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for r in self.rows:writer.writerow({k:json.dumps(r[k],ensure_ascii=False) if isinstance(r[k],(list,dict)) else r[k] for k in fields})
        return summary

    def check(self):
        groups=defaultdict(set);hashes=defaultdict(set);ids=set()
        for r in self.rows:
            assert r['id'] not in ids;ids.add(r['id']);groups[r['group']].add(r['split']);hashes[r['sha256']].add(r['split'])
            assert not (r['eligible'] and (r['pending_boxes'] or r['pending_reasons'] or not r['reviewed']))
            for b in r['boxes']+r['pending_boxes']:assert bbox_valid(b['xyxy'],r['width'],r['height'])
            if r['eligible']:assert capacity(r['boxes'],r['width'],r['height'])
        assert all(len(s)==1 for s in groups.values());assert all(len(s)==1 for s in hashes.values())
        assert sum('sin_cajas_no_confirmado_negativo' in r['pending_reasons'] for r in self.rows if r['source']=='Food Waste')==888
        assert all(not r['reviewed'] for r in self.rows if r['source']=='dataset-resized')
        save(REPORT/'validacion_preparacion.json',dict(group_split_leaks=0,exact_hash_split_leaks=0,ready_with_ambiguity=0,
            food_empty_kept_pending=888,all_boxes_geometrically_valid=True,training_started=False,
            reused_audit='docs/auditoria_externos/verificacion_imagenes.csv',new_images_verified_by_downloader=True))

    def preview(self):
        font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
        canvas=Image.new('RGB',(1500,1020),'#13252e');draw=ImageDraw.Draw(canvas)
        draw.text((20,12),'Dataset preparado · cajas humanas de origen / polígonos convertidos · sin predicciones',font=font,fill='white')
        examples=[]
        for i,c in enumerate(CLASSES):
            candidates=[r for r in self.rows if r['eligible'] and any(b['class']==c for b in r['boxes'])]
            preferred={'carton':'Corrugated carton','organico':'Apple-core','metal':'Drink can'}.get(c)
            candidates.sort(key=lambda r:(r['split']!='train',preferred is not None and not any(b.get('original_class')==preferred for b in r['boxes']),len(r['boxes']),-max((b['xyxy'][2]-b['xyxy'][0])*(b['xyxy'][3]-b['xyxy'][1])/r['width']/r['height'] for b in r['boxes'] if b['class']==c),r['id']))
            if not candidates:continue
            r=candidates[0];im=ImageOps.exif_transpose(Image.open(DEST/r['path'])).convert('RGB');im.thumbnail((460,350))
            x=20+i%3*500;y=60+i//3*475;draw.text((x,y),c.upper(),font=font,fill='#58e6b1')
            ox=x+(460-im.width)//2;oy=y+35+(350-im.height)//2;canvas.paste(im,(ox,oy))
            for b in r['boxes']:
                xy=[ox+b['xyxy'][0]*im.width/r['width'],oy+b['xyxy'][1]*im.height/r['height'],ox+b['xyxy'][2]*im.width/r['width'],oy+b['xyxy'][3]*im.height/r['height']]
                draw.rectangle(xy,outline='#ffff55',width=3);draw.text((xy[0],xy[1]),b['class'],font=small,fill='#ffff55',stroke_width=1,stroke_fill='black')
            draw.text((x,y+395),r['source']+' · '+r['split'],font=small,fill='white');draw.text((x,y+420),r['id'],font=small,fill='#c2d9df')
            examples.append(dict(class_name=c,image_id=r['id'],original=r['original_path'],group=r['group']))
        canvas.save(REPORT/'vista_previa_seis_clases.jpg',quality=93)
        save(REPORT/'ejemplos_vista_previa.json',examples)

def main():
    if (DEST/'manifest.json').exists():raise SystemExit('El manifiesto ya existe. No sobrescribir revisiones humanas: use una nueva carpeta/version.')
    DEST.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    taco=repair_taco();p=Preparation()
    cached=REPORT/'metadata_sin_taco.json'
    if cached.exists():
        p.rows=load(cached)
        for row in p.rows:
            row['original_group'],row['group_evidence']=origin_group(row['source'],Path(row['original_path']).name)
            cached_image=p.cached.get(row['original_path'])
            if cached_image and cached_image['annotation']:row['provenance']['annotation_file']=cached_image['annotation']
    else:p.bangladesh();p.food();p.reserve()
    p.taco(taco);p.finalize_states();p.visual_review_overrides();p.groups_and_splits();p.manual_batch();p.check()
    save(DEST/'manifest.json',dict(version=2,classes=CLASSES,videos={},images=p.rows,unavailable_images=p.missing,
        preparation=dict(seed=SEED,mode='scene_pending_on_ambiguity',source_audit='docs/auditoria_externos',train_from_scratch_next_stage=True)))
    (DEST/'abrir_anotador.cmd').write_text('@echo off\npushd "%~dp0..\\..\\.."\n".venv\\Scripts\\python.exe" -m residuos annotate --data "data\\preparado\\deteccion"\npopd\n',encoding='ascii')
    summary=p.report();p.preview();print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
