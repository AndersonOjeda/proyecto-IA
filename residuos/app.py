"""Interfaz local; inferencia en trabajador y actualización Tk en hilo principal."""
import queue
import threading
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import ImageTk
from . import group
from .data import Store, digest
from .detector import load_model, device_for, predict, seed_all
from .evaluate import panel, comparison, threshold_for
from .assistant import Assistant

class Application:
    def __init__(self, root, data='data', detector='runs/detector/best.pt', assistant='runs/assistant/best.pt', device='auto', mode='cobertura', min_box_size=None):
        self.root=root; self.store=Store(data); self.cap=None; self.running=False; self.busy=False
        self.detections=[]; self.results=queue.Queue(); self.generation=0; self.model=None; self.assistant=None
        self.mode = (mode or 'cobertura').lower()
        if self.mode == 'precision':
            self.min_box_size = 32 if min_box_size is None else int(min_box_size)
            default_th = 0.40
        else:
            self.min_box_size = 0 if min_box_size is None else int(min_box_size)
            default_th = None
        root.title('Residuos · CNN y transformer desde cero')
        seed_all(42); self.device=device_for(device)
        bar=ttk.Frame(root); bar.pack(fill='x',padx=10,pady=10)
        ttk.Button(bar,text='Abrir video',command=self.open_video).pack(side='left')
        ttk.Button(bar,text='Cámara web',command=lambda:self.open_source(0)).pack(side='left')
        ttk.Button(bar,text='Pausar / continuar',command=self.toggle).pack(side='left')
        ttk.Button(bar,text='Comparar validación',command=self.compare).pack(side='left')
        ttk.Button(bar,text='Guardar captura',command=self.capture).pack(side='left')
        self.status=tk.StringVar(value='Cargue un video o cámara.'); ttk.Label(root,textvariable=self.status,wraplength=1000).pack(fill='x',padx=10)
        self.preview=ttk.Label(root); self.preview.pack(padx=10,pady=10)
        self.objects=tk.StringVar(); ttk.Label(root,textvariable=self.objects,wraplength=1000).pack(fill='x',padx=10)
        ttk.Label(root,text='Consulta sobre el fotograma mostrado (se pausa al consultar):').pack(anchor='w',padx=10)
        self.question=tk.StringVar(); entry=ttk.Entry(root,textvariable=self.question,width=100); entry.pack(fill='x',padx=10)
        entry.bind('<Return>',lambda e:self.ask())
        ttk.Button(root,text='Consultar asistente',command=self.ask).pack(anchor='w',padx=10,pady=5)
        self.answer=tk.StringVar(value='La confianza es una puntuación del modelo, no una garantía de acierto.')
        ttk.Label(root,textvariable=self.answer,wraplength=1000).pack(fill='x',padx=10,pady=10)
        problems=[]
        self.detector_identity = None
        try:
            self.model,metadata=load_model(detector,self.device)
            base_threshold=threshold_for(detector)
            self.threshold = default_th if (default_th is not None and self.mode == 'precision') else base_threshold
            self.detector_identity = dict(checkpoint=str(Path(detector).resolve()),checkpoint_sha256=digest(detector),
                threshold=self.threshold,threshold_file=str(Path(detector).resolve().parent/'threshold.json'),
                origin=metadata['origin'],epoch=metadata.get('epoch'),device=str(self.device),
                mode=self.mode,min_box_size=self.min_box_size)
            print(json.dumps(self.detector_identity,ensure_ascii=False),flush=True)
            identity_text=f"Modelo: {self.detector_identity['checkpoint']}\nSHA256: {self.detector_identity['checkpoint_sha256']}\nUmbral: {self.threshold:.2f} (modo: {self.mode}, min_size: {self.min_box_size}px) | época {metadata.get('epoch')} | {self.device}"
            ttk.Label(root,text=identity_text,wraplength=1000).pack(fill='x',padx=10,pady=4)
        except ValueError as e: self.model=None; problems.append(str(e))
        try: self.assistant=Assistant(assistant,self.device)
        except ValueError as e: problems.append(str(e))
        if problems: self.status.set(' | '.join(problems))
        root.protocol('WM_DELETE_WINDOW',self.close)
        root.after(50,self.tick)

    def reset_source(self):
        self.generation+=1; self.running=False; self.detections=[]
        if self.cap: self.cap.release(); self.cap=None
        self.preview.configure(image=''); self.display=None; self.objects.set(''); self.answer.set('')

    def open_video(self):
        path=filedialog.askopenfilename(filetypes=[('Videos','*.mp4 *.avi *.mkv *.mov'),('Todos','*.*')])
        if path: self.open_source(path)

    def open_source(self,source):
        if self.model is None:
            messagebox.showerror('Detector no disponible',self.status.get()); return
        self.reset_source(); self.cap=cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.status.set('No se pudo abrir el video o cámara.'); self.cap.release(); self.cap=None; return
        self.running=True; self.status.set(f'Inferencia local · umbral validado {self.threshold:.2f} · {self.device}')

    def toggle(self):
        if self.cap:
            self.running=not self.running
            if not self.running: self.generation+=1

    def worker(self,frame,truth,generation):
        try:
            detections=predict(self.model,frame,self.device,self.threshold)
            if self.min_box_size > 0:
                detections=[b for b in detections if (b['xyxy'][2]-b['xyxy'][0])>=self.min_box_size and (b['xyxy'][3]-b['xyxy'][1])>=self.min_box_size]
            view=comparison(frame,detections,truth) if truth is not None else panel(frame,detections,'Detecciones del modelo')
            self.results.put((generation,detections,view,None))
        except Exception as e: self.results.put((generation,[],None,str(e)))

    def tick(self):
        try:
            generation,detections,view,error=self.results.get_nowait(); self.busy=False
            if generation==self.generation:
                if error: self.status.set(error); self.running=False
                else:
                    self.detections=detections; self.display=view
                    shown=view.copy(); shown.thumbnail((1050,540)); self.photo=ImageTk.PhotoImage(shown)
                    self.preview.configure(image=self.photo)
                    self.objects.set(' | '.join(f'{b["class"]}: {b["score"]:.2f} · {group(b["class"])}' for b in detections) or 'No hay una detección confiable en este fotograma.')
        except queue.Empty: pass
        if self.running and not self.busy and self.cap:
            ok,frame=self.cap.read()
            if not ok:
                self.running=False; self.status.set('Fin del video o lectura interrumpida. Se conserva el último fotograma.')
            else: self.launch(frame,None)
        self.root.after(30,self.tick)

    def launch(self,frame,truth):
        self.busy=True
        threading.Thread(target=self.worker,args=(frame,truth,self.generation),daemon=True).start()

    def compare(self):
        if self.model is None: messagebox.showerror('Detector',self.status.get()); return
        if self.busy: self.running=False; self.status.set('Espere a que termine el fotograma y pulse Comparar otra vez.'); return
        records=self.store.records('val')
        if not records: messagebox.showinfo('Validación','No hay fotogramas de validación revisados.'); return
        window=tk.Toplevel(self.root); window.title('Fotogramas de validación')
        listing=tk.Listbox(window,width=65,height=15); listing.pack()
        for r in records: listing.insert('end',f'{r["id"]} · {len(r["boxes"])} objetos')
        def choose():
            if not listing.curselection(): return
            r=records[listing.curselection()[0]]
            frame=cv2.imdecode(np.fromfile(self.store.root/r['path'],dtype=np.uint8),cv2.IMREAD_COLOR)
            if frame is None: messagebox.showerror('Imagen','No se pudo leer el fotograma.'); return
            self.reset_source(); self.launch(frame,r['boxes']); self.status.set(f'Comparación sobre validación: {r["id"]}'); window.destroy()
        ttk.Button(window,text='Comparar',command=choose).pack()

    def ask(self):
        self.running=False; self.generation+=1
        if not self.assistant:
            self.answer.set('Faltan pesos propios: python -m residuos train-assistant'); return
        response=self.assistant.answer(self.question.get(),self.detections)
        self.answer.set(f'{response["text"]}\nIntención: {response["intent"]} · puntuación: {response["confidence"]:.2f}')

    def capture(self):
        if getattr(self,'display',None) is None: return
        path=filedialog.asksaveasfilename(defaultextension='.png',filetypes=[('PNG','*.png')])
        if path: self.display.save(path)

    def close(self):
        self.running=False
        if self.cap: self.cap.release()
        self.root.destroy()

def run(**kwargs):
    root=tk.Tk(); Application(root,**kwargs); root.mainloop()
