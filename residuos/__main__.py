import argparse
import json

def main():
    parser=argparse.ArgumentParser(description='Proyecto 4: detección de residuos desde cero')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('annotate','stats','validate','train-detector','calibrate','evaluate','app'):
        p=sub.add_parser(name); p.add_argument('--data',default='data')
        if name in ('train-detector','calibrate','evaluate','app'): p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
        if name in ('calibrate','evaluate'): p.add_argument('--checkpoint',default='runs/detector/best.pt')
        if name=='train-detector':
            p.add_argument('--output',default='runs/detector'); p.add_argument('--epochs',type=int,default=40)
            p.add_argument('--batch',type=int,default=8); p.add_argument('--lr',type=float,default=.001); p.add_argument('--seed',type=int,default=42)
        if name=='evaluate':
            p.add_argument('--output',default='runs/evaluation')
            p.add_argument('--mode',default='cobertura',choices=['cobertura','precision'])
            p.add_argument('--min-box-size',type=int,default=None)
        if name=='app':
            p.add_argument('--detector',default='runs/detector/best.pt'); p.add_argument('--assistant',default='runs/assistant/best.pt')
    for name in ('train-assistant','evaluate-assistant'):
        p=sub.add_parser(name); p.add_argument('--corpus',default='data/intents.json')
        if name=='train-assistant':
            p.add_argument('--output',default='runs/assistant'); p.add_argument('--epochs',type=int,default=100)
            p.add_argument('--seed',type=int,default=42); p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
        else:
            p.add_argument('--checkpoint',default='runs/assistant/best.pt'); p.add_argument('--output',default='runs/assistant/evaluation.json')
    args=vars(parser.parse_args()); command=args.pop('command')
    if 'epochs' in args and args['epochs']<1: parser.error('--epochs debe ser positivo')
    if 'batch' in args and args['batch']<1: parser.error('--batch debe ser positivo')
    try:
        if command in ('stats','validate'):
            from .data import Store
            store=Store(args['data'])
            if command=='stats': result=store.stats()
            else:
                errors=store.validate()
                result=dict(valid=not errors,errors=errors,images=len(store.db['images']),note='Sin imágenes no se puede entrenar ni evaluar.' if not store.db['images'] else '')
                print(json.dumps(result,ensure_ascii=False,indent=2))
                return 1 if errors else 0
        elif command=='annotate':
            from .annotate import run
            run(**args); return 0
        elif command=='app':
            from .app import run
            run(**args); return 0
        elif command=='train-detector':
            from .detector import train
            result=train(**args)
        elif command in ('calibrate','evaluate'):
            from .evaluate import calibrate,evaluate
            result=(calibrate if command=='calibrate' else evaluate)(**args)
        else:
            from .assistant import train,evaluate
            result=(train if command=='train-assistant' else evaluate)(**args)
        print(json.dumps(result,ensure_ascii=False,indent=2)); return 0
    except (ValueError,FileNotFoundError) as e:
        print(f'No se pudo continuar: {e}'); return 1

if __name__=='__main__':
    raise SystemExit(main())
