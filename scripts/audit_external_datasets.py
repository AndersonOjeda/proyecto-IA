"""Auditoría de archivos locales, sin redes, modelos ni cambios en las fuentes.

Ejecutar: .venv/Scripts/python.exe scripts/audit_external_datasets.py
Las cajas de polígonos se calculan sólo en memoria para análisis/vista previa.
"""
import csv
import hashlib
import json
import math
import re
from itertools import combinations
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'data/externos'
OUTPUT = ROOT / 'docs/auditoria_externos'
EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
TARGETS = ['plastico', 'papel', 'carton', 'vidrio', 'metal', 'organico']

def relative(path):
    return Path(path).relative_to(ROOT).as_posix()

def source_mapping(source, name):
    if source.startswith('Bangladesh'):
        if name == 'Paper': return 'papel|carton', 'ambiguo', 'README incluye cajas de carton, vasos y papel bajo Paper; revisar objetos, no fusionar automaticamente.'
        return 'plastico', 'propuesto', 'Material explicito en nombre de clase.'
    if source == 'dataset-resized':
        maps = dict(plastic='plastico', paper='papel', cardboard='carton', glass='vidrio', metal='metal')
        if name in maps: return maps[name], 'propuesto', 'Etiqueta por carpeta; cantidad de objetos desconocida, sin localizacion.'
        return '', 'ambiguo', 'trash no demuestra material ni implica organico; revisar o excluir.'
    if source == 'Food Waste':
        if name in ('Other-waste','Food'): return '', 'ambiguo', 'Clase generica; inspeccion humana antes de asignar material.'
        if name in ('Bone','Bone-fish','Egg-shell','Mussel-shell','Shrimp-shell'):
            return 'organico', 'condicional', 'Propuesta como resto alimentario; confirmar alcance de huesos/conchas/cascaras y gestion local.'
        return 'organico', 'propuesto', 'Alimento o resto alimentario; el material no demuestra que ya sea un residuo.'
    if name in ('Battery','Cigarette','Shoe','Unlabeled litter','Rope & strings','Squeezable tube'):
        return '', 'ambiguo', 'Material mixto, desconocido o fuera de taxonomia; revisar/excluir.'
    if name in ('Aluminium blister pack','Carded blister pack','Drink carton','Meal carton','Paper cup','Plastified paper bag','Crisp packet'):
        candidates={'Aluminium blister pack':'metal|plastico','Carded blister pack':'carton|plastico','Drink carton':'carton|plastico|metal','Meal carton':'carton|plastico','Paper cup':'papel|plastico','Plastified paper bag':'papel|plastico','Crisp packet':'plastico|metal'}
        return candidates[name], 'ambiguo', 'Posible multicapa o mezcla; adoptar convencion o reanotar antes de integrar.'
    if name in ('Glass bottle','Broken glass','Glass cup','Glass jar'): target='vidrio'
    elif name in ('Aluminium foil','Metal bottle cap','Food Can','Aerosol','Drink can','Metal lid','Pop tab','Scrap metal'): target='metal'
    elif name in ('Toilet tube','Other carton','Egg carton','Corrugated carton','Pizza box'): target='carton'
    elif name in ('Magazine paper','Tissues','Wrapping paper','Normal paper','Paper bag','Paper straw'): target='papel'
    elif name == 'Food waste': target='organico'
    else: target='plastico'
    return target, 'propuesto', ('Material metalico; aerosol requiere gestion especifica, no inferir aprovechabilidad.' if name=='Aerosol' else 'Mapeo de material; limpieza/aprovechabilidad no se deduce de la etiqueta.')

class Audit:
    def __init__(self):
        self.images=[]; self.objects=[]; self.taxonomy={}; self.meta={}; self.issues=[]; self.cache={}
        self.annotation_files=set(); self.formats=Counter()

    def inspect(self,path):
        key=str(path)
        if key in self.cache: return self.cache[key]
        result=dict(exists=path.is_file(),readable=False,width=None,height=None,sha256='',error='')
        if result['exists']:
            try:
                with path.open('rb') as f: result['sha256']=hashlib.file_digest(f,'sha256').hexdigest()
                with Image.open(path) as im:
                    im.load(); result.update(readable=True,width=im.width,height=im.height)
            except Exception as exc: result['error']=str(exc)
        self.cache[key]=result
        return result

    def image(self,source,split,path,iid,annotation,width=None,height=None,label_status='anotada'):
        state=self.inspect(path)
        record=dict(source=source,split=split,path=relative(path),image_id=str(iid),annotation=relative(annotation) if annotation else '',
                    label_status=label_status,declared_width=width,declared_height=height,**state)
        record['key']=len(self.images)
        if state['readable'] and width is not None and (width,height)!=(state['width'],state['height']):
            self.issues.append(dict(type='dimension_mismatch',path=record['path'],declared=[width,height],actual=[state['width'],state['height']]))
        self.images.append(record)
        return record

    def object(self,im,name,cid,oid,box,fmt,polygon=None):
        valid=True; problems=[]
        if box is not None:
            w,h=im['width'] or im['declared_width'],im['height'] or im['declared_height']
            if len(box)!=4 or not all(math.isfinite(v) for v in box): valid=False; problems.append('nonfinite_box')
            elif box[2]<=box[0] or box[3]<=box[1]: valid=False; problems.append('nonpositive_area')
            elif w and h and (box[0]<-1e-5 or box[1]<-1e-5 or box[2]>w+1e-5 or box[3]>h+1e-5):
                valid=False; problems.append('box_out_of_image')
        target,status,note=source_mapping(im['source'],name)
        obj=dict(image_key=im['key'],class_id=cid,original_class=name,object_id=oid,box=box,format=fmt,
                 target=target,mapping_status=status,mapping_note=note,valid=valid,problems=problems)
        if polygon and (not all(math.isfinite(v) for v in polygon) or min(polygon)<-1e-8 or max(polygon)>1+1e-8):
            obj['valid']=False; obj['problems'].append('polygon_out_of_range')
        self.objects.append(obj); self.formats[fmt]+=1

    def coco(self,path,source,split):
        self.annotation_files.add(path)
        content=json.loads(path.read_text(encoding='utf-8'))
        classes={c['id']:c['name'] for c in content['categories']}
        self.taxonomy[source]=classes
        images={}
        for row in content['images']:
            if row['id'] in images: self.issues.append(dict(type='duplicate_image_id',annotation=relative(path),id=row['id']))
            images[row['id']]=self.image(source,split,path.parent/row['file_name'],row['id'],path,row.get('width'),row.get('height'))
        ids=set()
        for row in content['annotations']:
            if row['id'] in ids: self.issues.append(dict(type='duplicate_annotation_id',annotation=relative(path),id=row['id']))
            ids.add(row['id'])
            if row['image_id'] not in images:
                self.issues.append(dict(type='orphan_annotation',annotation=relative(path),id=row['id'])); continue
            if row['category_id'] not in classes:
                self.issues.append(dict(type='unknown_category',annotation=relative(path),id=row['id'])); continue
            b=row.get('bbox'); box=[b[0],b[1],b[0]+b[2],b[1]+b[3]] if b and len(b)==4 else None
            if box is None: self.issues.append(dict(type='missing_bbox',annotation=relative(path),id=row['id']))
            self.object(images[row['image_id']],classes[row['category_id']],row['category_id'],row['id'],box,
                        'COCO bbox+segmentacion' if row.get('segmentation') else 'COCO bbox')
        if source=='Food Waste':
            referenced={row['path'] for row in images.values()}
            for file in path.parent.iterdir():
                if file.suffix.lower() in EXT and relative(file) not in referenced:
                    self.image(source,split,file,file.name,path,label_status='imagen_sin_registro_COCO')

    def yolo(self,folder,source):
        yaml=folder/'data.yaml'; self.annotation_files.add(yaml)
        text=yaml.read_text(encoding='utf-8')
        # Exportes encontrados: names es una lista YAML simple, sin dependencias adicionales.
        block=text.split('names:',1)[1].split('nc:',1)[0]
        names=[line.strip()[2:].strip().strip('\"\'') for line in block.splitlines() if line.strip().startswith('- ')]
        if not names: raise ValueError(f'Lista names no reconocida: {yaml}')
        self.taxonomy[source]=dict(enumerate(names))
        for split in ('train','valid','test'):
            images_dir,labels_dir=folder/split/'images',folder/split/'labels'
            images={p.stem:p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in EXT}
            labels={p.stem:p for p in labels_dir.glob('*.txt')}
            for stem in sorted(set(images)|set(labels)):
                label=labels.get(stem); path=images.get(stem,images_dir/(stem+'.jpg'))
                row=self.image(source,split,path,stem,label,label_status='anotada' if label else 'sin_archivo_etiqueta')
                if label is None:
                    self.issues.append(dict(type='missing_label',path=row['path']));continue
                self.annotation_files.add(label)
                lines=[line.split() for line in label.read_text(encoding='utf-8').splitlines() if line.strip()]
                if not lines: row['label_status']='etiqueta_vacia_no_confirma_revision'
                for n,parts in enumerate(lines,1):
                    try:
                        cid=int(parts[0]); name=names[cid] if 0<=cid<len(names) else None
                        if name is None: raise ValueError('class id fuera de rango')
                        values=list(map(float,parts[1:])); w,h=row['width'],row['height']
                        if w is None: raise ValueError('No hay imagen decodificable para convertir coordenadas')
                        polygon=None
                        if len(values)==4:
                            cx,cy,bw,bh=values; box=[(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]; fmt='YOLO bbox'
                        elif len(values)>=6 and len(values)%2==0:
                            xs,ys=values[::2],values[1::2]
                            box=[min(xs)*w,min(ys)*h,max(xs)*w,max(ys)*h]; fmt='YOLO poligono; bbox derivada'; polygon=values
                        else: raise ValueError(f'Numero de coordenadas incompatible: {len(values)}')
                        self.object(row,name,cid,n,box,fmt,polygon)
                    except (ValueError,IndexError) as exc:
                        self.issues.append(dict(type='invalid_yolo_line',path=relative(label),line=n,error=str(exc)))
            manifest=folder.parent/'manifests'/f'{folder.name.removeprefix("data_")}_{split}.txt'
            if manifest.exists():
                expected=set(manifest.read_text().splitlines());actual={p.name for p in images.values()}
                if expected!=actual: self.issues.append(dict(type='manifest_mismatch',path=relative(manifest),missing=sorted(expected-actual),extra=sorted(actual-expected)))

    def classification(self,folder):
        source='dataset-resized'; names=sorted(p.name for p in folder.iterdir() if p.is_dir())
        self.taxonomy[source]=dict(enumerate(names))
        for cid,name in enumerate(names):
            for path in sorted((folder/name).iterdir()):
                if path.suffix.lower() not in EXT: continue
                im=self.image(source,'sin_particion',path,path.name,None,label_status='etiqueta_por_carpeta')
                self.object(im,name,cid,'',None,'Etiqueta de imagen')

    def save(self):
        OUTPUT.mkdir(parents=True,exist_ok=True)
        byimage=defaultdict(list)
        for obj in self.objects: byimage[obj['image_key']].append(obj)
        groups=defaultdict(list)
        for im in self.images: groups[(im['source'],im['split'])].append(im)
        summary=[]
        for (source,split),ims in sorted(groups.items()):
            for cid,name in [('', '__TOTAL__')]+list(self.taxonomy[source].items()):
                objs=[o for im in ims for o in byimage[im['key']] if name=='__TOTAL__' or o['original_class']==name]
                keys={o['image_key'] for o in objs}
                selected=ims if name=='__TOTAL__' else [im for im in ims if im['key'] in keys]
                boxobjs=[o for o in objs if o['box'] is not None]
                mapping=source_mapping(source,name) if name!='__TOTAL__' else ('','','No sumar imagenes por clase: una imagen puede contener varias clases.')
                summary.append(dict(fuente=source,particion=split,clase_id=cid,clase_original=name,
                    formato='; '.join(sorted({o['format'] for o in objs})) or 'sin instancias',
                    imagenes_referenciadas=len(selected),imagenes_en_disco=sum(im['exists'] for im in selected),
                    imagenes_legibles=sum(im['readable'] for im in selected),imagenes_faltantes=sum(not im['exists'] for im in selected),
                    imagenes_ilegibles=sum(im['exists'] and not im['readable'] for im in selected),
                    objetos_anotados=len(boxobjs) if source!='dataset-resized' else '',
                    objetos_con_imagen=len([o for o in boxobjs if self.images[o['image_key']]['exists']]) if source!='dataset-resized' else '',
                    cajas_invalidas=sum(not o['valid'] for o in boxobjs),
                    etiquetas_de_imagen=len(objs) if source=='dataset-resized' else 0,
                    imagenes_sin_objetos=sum(not byimage[im['key']] for im in selected),
                    mapeo_propuesto=mapping[0],estado_mapeo=mapping[1],observacion=mapping[2],
                    archivos_anotacion='; '.join(sorted({im['annotation'] for im in selected if im['annotation']})) if not source.startswith('Bangladesh') else relative(next(INPUT.rglob('data_canonical_75-15-10' if 'canonical' in source else 'data_robustness_70-15-15'))/split/'labels'),
                    licencia_local=self.meta[source]['license'],procedencia_local=self.meta[source]['provenance']))
        with (OUTPUT/'detalle_clases_particiones.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
        hashes=defaultdict(list);bases=defaultdict(list);sessions=defaultdict(list)
        for im in self.images:
            if im['sha256']: hashes[im['sha256']].append(im['key'])
            name=Path(im['path']).name
            base=name.split('.rf.')[0]
            bases[(im['source'],base)].append(im['key'])
            if im['source']=='Food Waste':
                m=re.search(r'(202\d{11}_\d{6})',name)
                if m: sessions[m[1]].append(im['key'])
        def across(keys): return len({self.images[k]['split'] for k in keys})>1
        leaks=[dict(sha256=h,images=[self.images[k]['path'] for k in ks],sources=sorted({self.images[k]['source'] for k in ks}),splits=sorted({self.images[k]['split'] for k in ks})) for h,ks in hashes.items() if across(ks)]
        same_source_leaks=[dict(sha256=h,source=s,images=[self.images[k]['path'] for k in ks if self.images[k]['source']==s]) for h,ks in hashes.items() for s in {self.images[k]['source'] for k in ks} if across([k for k in ks if self.images[k]['source']==s])]
        baseleaks=[dict(source=s,base=b,images=[self.images[k]['path'] for k in ks]) for (s,b),ks in bases.items() if across(ks)]
        sessionleaks=[dict(inferred_recording=s,splits=dict(Counter(self.images[k]['split'] for k in ks)),images=[self.images[k]['path'] for k in ks]) for s,ks in sessions.items() if across(ks)]
        invalid=[dict(image=self.images[o['image_key']]['path'],object_id=o['object_id'],box=o['box'],issues=o['problems']) for o in self.objects if not o['valid']]
        shared=Counter(pair for keys in hashes.values() for pair in combinations(sorted({self.images[k]['source'] for k in keys}),2))
        report=dict(input_actual=relative(INPUT),input_requested='datos/externos (no existe)',sources=self.meta,
                    totals=[r for r in summary if r['clase_original']=='__TOTAL__'],
                    annotation_formats=dict(self.formats),issues=self.issues,invalid_boxes=invalid,
                    missing_images=[im['path'] for im in self.images if not im['exists']],
                    unreadable_images=[im for im in self.images if im['exists'] and not im['readable']],
                    exact_cross_split_same_source=same_source_leaks,exact_cross_split_all_sources=leaks,
                    roboflow_base_cross_split=baseleaks,inferred_recording_cross_split=sessionleaks,
                    exact_shared_images_between_sources=[dict(sources=list(pair),unique_hashes=count) for pair,count in shared.items()],
                    inference_note='Origen de grabacion inferido solo de nombres; no equivale a metadatos de sesion verificados.',
                    source_integrity='Lectura exclusiva de originales. Cajas derivadas solo en memoria; no se importo al manifiesto ni se entreno.')
        (OUTPUT/'auditoria.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        with (OUTPUT/'verificacion_imagenes.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(self.images[0]));writer.writeheader();writer.writerows(self.images)
        self.preview(byimage)
        print(json.dumps(dict(totals=report['totals'],formats=report['annotation_formats'],issues=Counter(i['type'] for i in self.issues),
                             invalid_boxes=len(invalid),exact_leaks_same_source=len(same_source_leaks),base_leaks=len(baseleaks),
                             inferred_recording_leaks=[dict(recording=x['inferred_recording'],splits=x['splits']) for x in sessionleaks]),ensure_ascii=False,indent=2))

    def preview(self,byimage):
        canvas=Image.new('RGB',(1500,1040),'#101f28'); draw=ImageDraw.Draw(canvas)
        font_path=Path('C:/Windows/Fonts/arial.ttf')
        title=ImageFont.truetype(str(font_path),30); small=ImageFont.truetype(str(font_path),16); label=ImageFont.truetype(str(font_path),20)
        draw.text((22,16),'Datos locales: seis clases propuestas · anotaciones de origen',font=title,fill='white')
        choices=[]
        for i,target in enumerate(TARGETS):
            candidates=[o for o in self.objects if o['target']==target and o['mapping_status']=='propuesto' and o['valid'] and self.images[o['image_key']]['readable']]
            # Preferir localizacion existente y pocas cajas para una previsualizacion legible.
            candidates.sort(key=lambda o:(o['box'] is None, len(byimage[o['image_key']]), self.images[o['image_key']]['path']))
            obj=candidates[0]; im=self.images[obj['image_key']]
            image=Image.open(ROOT/im['path']).convert('RGB'); image.thumbnail((466,330))
            x=20+(i%3)*500; y=75+(i//3)*470
            draw.text((x,y),target.upper(),font=title,fill='#58e6b1')
            ox=x+(466-image.width)//2;oy=y+46+(330-image.height)//2
            canvas.paste(image,(ox,oy))
            sx=image.width/im['width']; sy=image.height/im['height']
            for b in byimage[im['key']]:
                if b['box'] is None or not b['valid']: continue
                coords=[ox+b['box'][0]*sx,oy+b['box'][1]*sy,ox+b['box'][2]*sx,oy+b['box'][3]*sy]
                draw.rectangle(coords,outline='#ffff44',width=3)
                text=b['original_class']; tx,ty=coords[:2]
                bb=draw.textbbox((tx,ty),text,font=small); draw.rectangle(bb,fill='#101f28'); draw.text((tx,ty),text,font=small,fill='#ffff44')
            detail=('Caja derivada del polígono' if 'poligono' in obj['format'] else 'Caja COCO de origen') if obj['box'] else 'Solo etiqueta de imagen · SIN CAJAS'
            draw.text((x,y+385),f'{im["source"]} · {im["split"]}',font=label,fill='white')
            draw.text((x,y+413),f'{obj["original_class"]} → {target}',font=small,fill='#c7d7de')
            draw.text((x,y+438),detail,font=small,fill='#ffd18a')
            choices.append(dict(target=target,image=im['path'],source=im['source'],original_class=obj['original_class'],format=obj['format']))
        canvas.save(OUTPUT/'vista_previa_seis_clases.jpg',quality=94)
        (OUTPUT/'ejemplos_vista_previa.json').write_text(json.dumps(choices,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    if not INPUT.exists(): raise SystemExit(f'No existe {INPUT}')
    before={relative(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in INPUT.rglob('*') if p.is_file()}
    audit=Audit()
    food=next(INPUT.glob('Food Waste*'))
    audit.meta['Food Waste']=dict(license='CC BY 4.0 declarada en README local',provenance='https://universe.roboflow.com/food-1b74y/food-waste-detection-jghxg',notes='Aumentos de origen documentados: tres versiones. COCO; no deducir que un alimento visible ya es basura.')
    for path in sorted(food.rglob('_annotations.coco.json')): audit.coco(path,'Food Waste',path.parent.name)
    for folder in sorted(INPUT.rglob('data_*')):
        if not (folder/'data.yaml').exists(): continue
        name='Bangladesh '+folder.name.removeprefix('data_')
        audit.meta[name]=dict(license='CC BY 4.0 declarada en LICENSE/data.yaml',provenance='https://universe.roboflow.com/waste-detections-by-fu/waste-detections-c0mkz',notes='Usar ids de data.yaml, no tabla del README: orden diferente. Exportaciones alternativas de la misma coleccion; no unir como datos independientes.')
        audit.yolo(folder,name)
    for path in sorted(INPUT.rglob('annotations*.json')):
        name='TACO no oficial' if 'unofficial' in path.name else 'TACO oficial'
        audit.meta[name]=dict(license='LICENSE MIT corresponde al software; licencia de imagenes no verificada (imagenes ausentes)',provenance='README local TACO; https://github.com/pedropro/TACO',notes='No oficial sin revision segun README.' if 'unofficial' in path.name else 'COCO con cajas y segmentaciones; sin particion local.')
        audit.coco(path,name,'sin_particion')
    folder=INPUT/'dataset-resized/dataset-resized'
    audit.meta['dataset-resized']=dict(license='Sin licencia/procedencia documental en la extraccion',provenance='Carpetas locales; estructura compatible con TrashNet, identidad no confirmada',notes='__MACOSX y ._* son metadatos, no fotografias. Sin cajas ni particiones.')
    audit.classification(folder)
    ignored=[relative(p) for p in INPUT.rglob('*') if p.is_file() and ('__MACOSX' in p.parts or p.name.startswith('._') or p.name=='.DS_Store')]
    audit.issues.append(dict(type='excluded_os_metadata',paths=ignored))
    audit.save()
    after={relative(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in INPUT.rglob('*') if p.is_file()}
    if before!=after: raise RuntimeError('El inventario fuente cambio durante el analisis; revisar concurrencia.')
    rows=list(csv.DictReader((OUTPUT/'detalle_clases_particiones.csv').open(encoding='utf-8-sig')))
    for source,split in {(r['fuente'],r['particion']) for r in rows}:
        group=[r for r in rows if (r['fuente'],r['particion'])==(source,split)]
        total=next(r for r in group if r['clase_original']=='__TOTAL__')
        if source!='dataset-resized':
            assert sum(int(r['objetos_anotados']) for r in group if r['clase_original']!='__TOTAL__')==int(total['objetos_anotados'])
        else:
            assert sum(int(r['etiquetas_de_imagen']) for r in group if r['clase_original']!='__TOTAL__')==int(total['etiquetas_de_imagen'])
    report_path=OUTPUT/'auditoria.json'; report=json.loads(report_path.read_text(encoding='utf-8'))
    report['verification']=dict(all_available_images_decoded=not report['unreadable_images'],source_inventory_unchanged=True,
                               training_performed=False,pretrained_models_used=False,class_count_sum_checks_passed=True)
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Originales intactos: inventario, tamanos y fechas sin cambios. No se entreno ningun modelo.')

if __name__=='__main__': main()
