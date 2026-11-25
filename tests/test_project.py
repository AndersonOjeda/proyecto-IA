"""Pruebas técnicas sintéticas: nunca métricas de residuos reales."""
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch
from residuos.data import Store
from residuos.detector import Detector, encode, loss_fn, decode, iou, seed_all
from residuos.evaluate import metrics
from residuos.assistant import IntentTransformer, vocabulary, encode_text, respond, validate_corpus
from residuos.data import read_json

BOXES = [dict(**{'class':c},xyxy=xy) for c,xy in [
    ('plastico',[5,5,30,45]),('metal',[40,10,70,50]),('organico',[80,20,120,80])]]

def make_video(path, offset=0):
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),5,(160,120))
    if not writer.isOpened(): raise RuntimeError('Codec MJPG no disponible.')
    for i in range(3):
        frame=np.full((120,160,3),30+offset+i*10,dtype=np.uint8)
        cv2.rectangle(frame,(5,5),(30,45),(0,255,0),-1); writer.write(frame)
    writer.release()

class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.video=self.root/'source.avi'; make_video(self.video)
        self.store=Store(self.root/'data'); self.key=self.store.register(self.video,'sesion-1','train')
        cap=cv2.VideoCapture(str(self.video)); ok,self.frame=cap.read(); cap.release(); self.assertTrue(ok)

    def tearDown(self): self.temp.cleanup()

    def test_three_objects_save_reopen(self):
        record=self.store.capture(self.key,0,self.frame)
        self.store.annotate(record['id'],BOXES)
        reopened=Store(self.root/'data'); r=reopened.db['images'][0]
        self.assertEqual(r['boxes'],BOXES); self.assertTrue(r['reviewed'])
        self.assertEqual(reopened.validate(),[])

    def test_pending_versus_reviewed_empty(self):
        first=self.store.capture(self.key,0,self.frame)
        second=self.store.capture(self.key,1,self.frame+1)
        self.store.annotate(second['id'],[])
        stats=self.store.stats()['train']
        self.assertEqual(stats['pending'],1); self.assertEqual(stats['reviewed_empty'],1)
        self.assertFalse(first['reviewed']); self.assertEqual(len(self.store.records('train')),1)

    def test_session_and_duplicate_video_leakage(self):
        other=self.root/'other.avi'; make_video(other,20)
        with self.assertRaises(ValueError): self.store.register(other,'sesion-1','test')
        with self.assertRaises(ValueError): self.store.register(self.video,'otra-sesion','test')

    def test_invalid_box_rejected(self):
        with self.assertRaises(ValueError): self.store.check_boxes([{'class':'metal','xyxy':[4,4,2,2]}],160,120)

    def test_edit_and_delete_persist(self):
        r=self.store.capture(self.key,0,self.frame)
        edited=[dict(BOXES[0],**{'class':'papel'}),BOXES[2]]
        self.store.annotate(r['id'],edited)
        self.assertEqual(Store(self.root/'data').db['images'][0]['boxes'],edited)

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): seed_all(42)

    def test_forward_backward_empty_and_multiple(self):
        model=Detector(); x=torch.rand(2,3,256,256)
        y=torch.stack([encode(BOXES,160,120),encode([],160,120)])
        result=model(x); loss=loss_fn(result,y); loss.backward()
        self.assertEqual(tuple(result.shape),(2,16,16,3,11)); self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.features[0].weight.grad)

    def test_same_class_two_predictions_survive(self):
        raw=torch.full((16,16,3,11),-20.)
        for gy,gx in [(2,2),(12,12)]:
            raw[gy,gx,0,0]=10; raw[gy,gx,0,1:3]=0; raw[gy,gx,0,3:5]=-3
            raw[gy,gx,0,5]=10
        predictions=decode(raw,160,120,.5)
        self.assertEqual(len(predictions),2); self.assertTrue(all(p['class']=='plastico' for p in predictions))

    def test_same_cell_capacity_not_silent(self):
        boxes=[{'class':'metal','xyxy':[1,1,5,5]}]*4
        with self.assertRaises(ValueError): encode(boxes,160,120)
        self.assertEqual(encode(boxes[:2],160,120)[...,0].sum().item(),2)

    def test_one_to_one_metrics(self):
        truth=[{'class':'metal','xyxy':[0,0,10,10]}]
        p=[dict(truth[0],score=.9),dict(truth[0],score=.8)]
        score=metrics([(p,truth)])
        self.assertEqual((score['tp'],score['fp'],score['fn']),(1,1,0))
        self.assertEqual(score['classes']['metal']['precision'],.5)
        self.assertIsNone(score['classes']['vidrio']['recall'])
        self.assertEqual(iou([0,0,10,10],[0,0,10,10]),1)

    def test_vocabulary_train_only_and_transformer(self):
        corpus=read_json('data/intents.json'); validate_corpus(corpus)
        vocab=vocabulary(corpus['train']); self.assertNotIn('fotograma',vocab)
        model=IntentTransformer(len(vocab))
        ids=torch.tensor([encode_text('qué residuos ves',vocab),encode_text('',vocab)])
        logits=model(ids); logits.sum().backward()
        self.assertEqual(tuple(logits.shape),(2,5)); self.assertTrue(torch.isfinite(logits).all())

    def test_context_and_ambiguity(self):
        self.assertIn('varios',respond('separar','dónde lo boto',BOXES))
        self.assertIn('No hay detecciones',respond('inventario','qué ves',[]))
        self.assertIn('No hay una detección',respond('separar','separo el vidrio',BOXES))
        self.assertIn('blanco',respond('separar','separo esta lata',BOXES))
        self.assertIn('no determina limpieza',respond('suciedad','lata sucia',BOXES))
        mixed=[{'class':'plastico'},{'class':'vidrio'}]
        self.assertIn('no identifica',respond('separar','esta botella',mixed))
        self.assertTrue(respond('separar','esta botella',[{'class':'vidrio'}]).startswith('vidrio:'))

class GuiTests(unittest.TestCase):
    def test_mouse_save_reopen_three_boxes(self):
        import tkinter as tk
        from residuos.annotate import Annotator
        with tempfile.TemporaryDirectory() as td:
            root=tk.Tk(); root.withdraw()
            try:
                app=Annotator(root,Path(td)/'data')
                video=Path(td)/'video.avi'; make_video(video)
                key=app.store.register(video,'sesion','train')
                frame=np.full((120,160,3),80,np.uint8)
                record=app.store.capture(key,0,frame); app.load(record,frame)
                from residuos import CLASSES,LABELS
                for box in BOXES:
                    app.class_name.set(LABELS[CLASSES.index(box['class'])])
                    x1,y1,x2,y2=box['xyxy']
                    app.press(SimpleNamespace(x=x1*app.scale,y=y1*app.scale))
                    app.motion(SimpleNamespace(x=x2*app.scale,y=y2*app.scale))
                    app.release(SimpleNamespace(x=x2*app.scale,y=y2*app.scale))
                app.save()
                reopened=Store(Path(td)/'data').db['images'][0]
                with patch('residuos.annotate.filedialog.askopenfilename',return_value=str(app.store.root/reopened['path'])):
                    app.reopen()
                self.assertEqual(app.boxes,BOXES); self.assertEqual(app.listbox.size(),3)
                app.listbox.selection_set(1); app.class_name.set('Papel'); app.reclass()
                self.assertEqual(app.boxes[1]['class'],'papel')
                app.listbox.selection_set(1); app.redraw_box()
                app.press(SimpleNamespace(x=42*app.scale,y=12*app.scale))
                app.release(SimpleNamespace(x=72*app.scale,y=52*app.scale))
                self.assertEqual(app.boxes[1]['xyxy'],[42,12,72,52])
                app.listbox.selection_set(0); app.delete(); app.save()
                self.assertEqual(len(Store(Path(td)/'data').db['images'][0]['boxes']),2)
            finally:
                if app.cap: app.cap.release()
                root.destroy()

    def test_app_missing_weights_is_explicit(self):
        import tkinter as tk
        from residuos.app import Application
        with tempfile.TemporaryDirectory() as td:
            root=tk.Tk(); root.withdraw()
            try:
                app=Application(root,data=td,detector=td+'/missing.pt',assistant=td+'/missing-assistant.pt',device='cpu')
                self.assertIsNone(app.model); self.assertIn('train-detector',app.status.get())
                self.assertIn('train-assistant',app.status.get())
            finally: root.destroy()

    def test_video_inference_and_comparison_worker(self):
        import tkinter as tk
        import time
        from residuos.app import Application
        from residuos.data import write_json,digest
        from residuos import CLASSES
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td); checkpoint=folder/'best.pt'
            torch.save(dict(origin='residuos-from-scratch-v1',classes=list(CLASSES),state=Detector().state_dict()),checkpoint)
            write_json(folder/'threshold.json',dict(threshold=.99,checkpoint_sha256=digest(checkpoint)))
            video=folder/'video.avi'; make_video(video)
            root=tk.Tk(); root.withdraw()
            try:
                app=Application(root,data=folder/'data',detector=checkpoint,assistant=folder/'absent.pt',device='cpu')
                app.open_source(str(video))
                deadline=time.monotonic()+10
                while (app.running or app.busy) and time.monotonic()<deadline:
                    root.update(); time.sleep(.02)
                self.assertFalse(app.busy); self.assertIsNotNone(app.display)
                self.assertEqual(app.display.size,(160,150))
                self.assertEqual(app.detections,[])
                self.assertIn('No hay una detección confiable',app.objects.get())
                app.launch(np.zeros((120,160,3),np.uint8),BOXES)
                deadline=time.monotonic()+10
                while app.busy and time.monotonic()<deadline:
                    root.update(); time.sleep(.02)
                self.assertFalse(app.busy); self.assertEqual(app.display.size,(320,150))
            finally:
                if app.cap: app.cap.release()
                root.destroy()

if __name__=='__main__': unittest.main()
