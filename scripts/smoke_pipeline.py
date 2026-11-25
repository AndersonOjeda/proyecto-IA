"""Pipeline sintético aislado. Sus pesos y métricas NO son resultados del proyecto."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from residuos.data import Store,write_json
from residuos.detector import train
from residuos.evaluate import calibrate,evaluate

def main():
    root=Path('tmp/synthetic-smoke'); root.mkdir(parents=True,exist_ok=True)
    store=Store(root/'data')
    for i,split in enumerate(('train','val','test')):
        path=root/(split+'.avi')
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),5,(160,120))
        frame=np.full((120,160,3),20+i*40,np.uint8)
        for x,color in [(5,(255,0,0)),(55,(0,255,0)),(105,(0,0,255))]: cv2.rectangle(frame,(x,20),(x+30,80),color,-1)
        writer.write(frame); writer.release()
        key=store.register(path,'SINTETICO-'+split,split)
        record=store.capture(key,0,frame)
        store.annotate(record['id'],[{'class':c,'xyxy':[x,20,x+30,80]} for c,x in [('plastico',5),('metal',55),('organico',105)]])
    train(str(root/'data'),str(root/'detector'),epochs=1,batch=1,device='cpu')
    calibrate(str(root/'data'),str(root/'detector/best.pt'),device='cpu')
    evaluate(str(root/'data'),str(root/'detector/best.pt'),str(root/'evaluation'),device='cpu')
    write_json(root/'NOTICE.json',{'technical_test_only':True,'real_world_metrics':False,'purpose':'Verificar video, anotación, train, calibrate, evaluate y PNG; no entrenar un detector utilizable.'})
    print('Pipeline sintético completado. NO usar estos pesos en la demostración ni sus métricas en el informe.')

if __name__=='__main__': main()
