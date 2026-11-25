"""Run the project's own detector on a video, without the intent transformer."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
from residuos.detector import seed_all, device_for, load_model, predict
from residuos.evaluate import threshold_for

def main():
    p=argparse.ArgumentParser();p.add_argument('--video',required=True);p.add_argument('--checkpoint',default='runs/detector_exp01/best.pt');p.add_argument('--output',default='runs/detector_exp01/video_predicciones.mp4');p.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    a=p.parse_args();seed_all(20261123);device=device_for(a.device)
    model,_=load_model(a.checkpoint,device);threshold=threshold_for(a.checkpoint)
    source=Path(a.video).resolve();dest=Path(a.output).resolve()
    if not source.is_file():raise ValueError('Video inexistente: '+str(source))
    if source==dest or dest.exists():raise ValueError('Elija una salida nueva para conservar los archivos existentes')
    cap=cv2.VideoCapture(str(source))
    if not cap.isOpened():raise ValueError('No se pudo abrir el video')
    fps=cap.get(cv2.CAP_PROP_FPS) or 25
    width,height=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dest.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(dest),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
    if not writer.isOpened():raise ValueError('No se pudo crear el video de salida')
    frames=0;started=time.perf_counter()
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            for b in predict(model,frame,device,threshold):
                x1,y1,x2,y2=[round(v) for v in b['xyxy']]
                cv2.rectangle(frame,(x1,y1),(x2,y2),(30,220,100),2)
                cv2.putText(frame,f"{b['class']} {b['score']:.2f}",(x1,max(20,y1)),cv2.FONT_HERSHEY_SIMPLEX,.6,(30,220,100),2)
            writer.write(frame);frames+=1
    finally:cap.release();writer.release()
    print(json.dumps(dict(output=str(dest),frames=frames,seconds=time.perf_counter()-started,device=str(device),threshold=threshold)))

if __name__=='__main__':main()
