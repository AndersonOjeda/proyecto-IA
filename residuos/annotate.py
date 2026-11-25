"""Editor de cajas en coordenadas originales; Tkinter + OpenCV."""
import copy
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
from . import CLASSES, LABELS
from .data import Store

class Annotator:
    def __init__(self, root, data='data'):
        self.root, self.store = root, Store(data)
        self.cap = self.record = self.frame = self.video_id = None
        self.boxes, self.dirty, self.drag = [], False, None
        root.title('Proyecto 4 · Anotación humana')
        bar = ttk.Frame(root); bar.pack(fill='x', padx=8, pady=8)
        ttk.Button(bar, text='Abrir video', command=self.open_video).pack(side='left')
        ttk.Button(bar, text='Reabrir imagen', command=self.reopen).pack(side='left')
        ttk.Button(bar, text='Pendientes', command=self.pending).pack(side='left')
        self.frame_no = tk.IntVar(value=0)
        ttk.Label(bar, text=' Fotograma:').pack(side='left')
        ttk.Entry(bar, textvariable=self.frame_no, width=9).pack(side='left')
        ttk.Button(bar, text='Ir / capturar', command=self.seek).pack(side='left')
        ttk.Button(bar, text='+30', command=self.next_frame).pack(side='left')
        self.class_name = tk.StringVar(value=LABELS[0])
        ttk.Combobox(bar, values=LABELS, textvariable=self.class_name, state='readonly', width=17).pack(side='left', padx=8)
        main = ttk.Frame(root); main.pack(fill='both', expand=True)
        self.canvas = tk.Canvas(main, width=900, height=580, bg='#18202a', highlightthickness=0)
        self.canvas.pack(side='left')
        side = ttk.Frame(main); side.pack(side='left', fill='y', padx=8)
        self.listbox = tk.Listbox(side, width=30, height=22); self.listbox.pack()
        ttk.Button(side, text='Aplicar clase seleccionada', command=self.reclass).pack(fill='x')
        ttk.Button(side, text='Redibujar caja seleccionada', command=self.redraw_box).pack(fill='x')
        ttk.Button(side, text='Eliminar caja', command=self.delete).pack(fill='x')
        ttk.Button(side, text='Guardar como revisada', command=self.save).pack(fill='x', pady=12)
        ttk.Label(side, text='Sin cajas + guardar = vacío revisado.\nArrastre el mouse para cada objeto.\nSeleccione una caja para corregirla.').pack()
        self.status = tk.StringVar(value='Abra un video y registre su sesión y partición.')
        ttk.Label(root, textvariable=self.status).pack(fill='x', padx=8, pady=8)
        self.canvas.bind('<ButtonPress-1>', self.press)
        self.canvas.bind('<B1-Motion>', self.motion)
        self.canvas.bind('<ButtonRelease-1>', self.release)
        self.replace_index = None
        root.protocol('WM_DELETE_WINDOW', self.close)

    def can_leave(self):
        return not self.dirty or messagebox.askyesno('Cambios sin guardar', '¿Descartar los cambios sin guardar?')

    def open_video(self):
        if not self.can_leave(): return
        path = filedialog.askopenfilename(filetypes=[('Videos', '*.mp4 *.avi *.mov *.mkv'), ('Todos', '*.*')])
        if not path: return
        session = simpledialog.askstring('Procedencia', 'Sesión (misma jornada/escena = misma sesión):')
        if not session: return
        split = simpledialog.askstring('Partición', 'Escriba train, val o test. Toda la sesión queda junta:')
        if not split: return
        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened(): raise ValueError('No se pudo abrir el video.')
            key = self.store.register(path, session.strip(), split.strip())
            if self.cap: self.cap.release()
            self.cap, self.video_id, self.dirty = cap, key, False
            self.frame_no.set(0); self.seek()
        except Exception as e: messagebox.showerror('Video', str(e))

    def seek(self):
        if self.cap is None or not self.can_leave(): return
        try:
            index = self.frame_no.get()
            if index < 0: raise ValueError('El índice debe ser no negativo.')
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = self.cap.read()
            if not ok: raise ValueError('Fotograma no disponible (fin del video o error de lectura).')
            record = self.store.capture(self.video_id, index, frame)
            self.load(record, frame)
        except Exception as e: messagebox.showerror('Fotograma', str(e))

    def next_frame(self):
        self.frame_no.set(self.frame_no.get() + 30); self.seek()

    def reopen(self):
        if not self.can_leave(): return
        path = filedialog.askopenfilename(initialdir=self.store.root / 'images', filetypes=[('Imágenes', '*.png *.jpg *.jpeg')])
        if not path: return
        from pathlib import Path
        import numpy as np
        r = next((r for r in self.store.db['images'] if (self.store.root / r['path']).resolve() == Path(path).resolve()), None)
        if r is None:
            messagebox.showerror('Imagen', 'Esta imagen no está en el manifiesto.'); return
        frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            messagebox.showerror('Imagen', 'No se pudo leer la imagen.'); return
        self.open_record(r, frame)

    def open_record(self, r, frame=None):
        import numpy as np
        if frame is None:
            frame = cv2.imdecode(np.fromfile(self.store.root/r['path'],dtype=np.uint8),cv2.IMREAD_COLOR)
        if frame is None:
            messagebox.showerror('Imagen','Imagen ausente o ilegible.'); return
        if self.cap: self.cap.release()
        self.cap = None; self.video_id = r.get('video')
        if self.video_id:
            self.cap = cv2.VideoCapture(str(self.store.root / self.store.db['videos'][self.video_id]['path']))
        self.frame_no.set(r.get('frame') or 0); self.load(r, frame)

    def pending(self):
        if not self.can_leave(): return
        rows = [r for r in self.store.db['images'] if not r['reviewed'] and r.get('available',True)]
        rows.sort(key=lambda r:(not r.get('manual_batch',False),r['id']))
        window=tk.Toplevel(self.root); window.title('Escenas pendientes: revisar TODOS los objetos')
        listing=tk.Listbox(window,width=100,height=20); listing.pack()
        for r in rows:
            listing.insert('end',f'{r["id"]} | {r.get("source","video")} | '+', '.join(r.get('pending_reasons',[])))
        def choose():
            if listing.curselection(): self.open_record(rows[listing.curselection()[0]]); window.destroy()
        ttk.Button(window,text='Abrir para revisión',command=choose).pack()

    def load(self, record, frame):
        self.record, self.frame = record, frame
        self.boxes = copy.deepcopy(record['boxes'] + record.get('pending_boxes', []))
        self.dirty = False
        self.replace_index = None
        self.scale = min(900 / frame.shape[1], 580 / frame.shape[0])
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb).resize((round(frame.shape[1]*self.scale), round(frame.shape[0]*self.scale)))
        self.photo = ImageTk.PhotoImage(im)
        self.render()

    def render(self):
        self.canvas.delete('all'); self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.listbox.delete(0, 'end')
        for i, b in enumerate(self.boxes):
            coords = [v*self.scale for v in b['xyxy']]
            known = b.get('class') in CLASSES
            label = LABELS[CLASSES.index(b['class'])] if known else 'REVISAR: '+b.get('original_class','material')
            self.canvas.create_rectangle(*coords, outline='#58ef9a' if known else '#ffa640', width=2)
            self.canvas.create_text(coords[0]+3, coords[1]+3, text=f'{i+1}: {label}', fill='#ffff66', anchor='nw')
            self.listbox.insert('end', f'{i+1}. {label} {b["xyxy"]}')
        state = 'cambios sin guardar' if self.dirty else ('revisada' if self.record['reviewed'] else 'pendiente')
        self.status.set(f'{self.record["id"]} | {self.record["split"]} | {state} | {len(self.boxes)} objetos')

    def point(self, event):
        return (min(max(event.x/self.scale, 0), self.record['width']), min(max(event.y/self.scale, 0), self.record['height']))

    def press(self, event):
        if self.record: self.drag = self.point(event)

    def motion(self, event):
        if self.drag is None: return
        self.canvas.delete('preview')
        self.canvas.create_rectangle(*[v*self.scale for v in (*self.drag, *self.point(event))], outline='orange', width=2, tags='preview')

    def release(self, event):
        if self.drag is None: return
        x, y = self.point(event); a, b = self.drag; self.drag = None
        self.canvas.delete('preview')
        if abs(x-a) < 2 or abs(y-b) < 2: return
        box = dict(xyxy=[round(min(x,a),2), round(min(y,b),2), round(max(x,a),2), round(max(y,b),2)],
                   **{'class': CLASSES[LABELS.index(self.class_name.get())]})
        if self.replace_index is None: self.boxes.append(box)
        else: self.boxes[self.replace_index] = box
        self.replace_index = None; self.dirty = True; self.render()

    def reclass(self):
        if self.listbox.curselection():
            self.boxes[self.listbox.curselection()[0]]['class'] = CLASSES[LABELS.index(self.class_name.get())]
            self.dirty = True; self.render()

    def redraw_box(self):
        if self.listbox.curselection():
            self.replace_index = self.listbox.curselection()[0]
            self.status.set('Arrastre el recuadro corregido; reemplazará la caja seleccionada.')

    def delete(self):
        if self.listbox.curselection():
            self.boxes.pop(self.listbox.curselection()[0]); self.replace_index = None
            self.dirty = True; self.render()

    def save(self):
        if self.record:
            try:
                if any(b.get('class') not in CLASSES for b in self.boxes):
                    raise ValueError('Asigne una clase a cada caja REVISAR antes de confirmar. No elimine un residuo real para tratarlo como fondo.')
                if self.record.get('source') and not messagebox.askyesno('Revisión completa',
                    '¿Revisó todos los residuos de la escena, incluyendo los no anotados?\nSi hay material incierto, cancele y deje la imagen pendiente. Guardar sin cajas confirma un negativo real.'):
                    return
                if self.record.get('source'):
                    from .detector import encode
                    encode(self.boxes,self.record['width'],self.record['height'])
                self.store.annotate(self.record['id'], copy.deepcopy(self.boxes))
                self.dirty = False; self.render()
            except Exception as e: messagebox.showerror('Guardar', str(e))

    def close(self):
        if self.can_leave():
            if self.cap: self.cap.release()
            self.root.destroy()

def run(data='data'):
    root = tk.Tk(); Annotator(root, data); root.mainloop()
