"""Preparación y revisión: pruebas sin entrenamiento de modelos."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import cv2
from scripts.prepare_detection_dataset import origin_group,capacity,bbox_valid,Preparation
from residuos.data import Store,write_json,digest

class PreparationTests(unittest.TestCase):
    def test_groups_join_augmentation_copy_and_recording(self):
        self.assertEqual(origin_group('Bangladesh canonical','IMG_1-copy-2_jpg.rf.abc.jpg')[0],
                         origin_group('Bangladesh canonical','IMG_1_jpg.rf.def.jpg')[0])
        self.assertEqual(origin_group('Food Waste','image_20220209124902_000421_0149_jpg.rf.x.jpg')[0],
                         origin_group('Food Waste','20220209124902_000421-00_01_00-Still3.rf.y.jpg')[0])

    def test_mixed_scene_is_wholly_pending(self):
        p=Preparation.__new__(Preparation)
        row=dict(boxes=[{'class':'plastico','xyxy':[1,1,20,20]}],pending_boxes=[{'class':None,'xyxy':[25,25,45,45]}],
                 pending_reasons=['clase_ambigua:Paper'],width=100,height=100)
        p.rows=[row];p.finalize_states()
        self.assertFalse(row['eligible']);self.assertFalse(row['reviewed']);self.assertEqual(len(row['boxes']),1)

    def test_no_bbox_for_manual_reserve(self):
        self.assertFalse(bbox_valid([0,0,0,1],10,10))
        self.assertFalse(capacity([{'xyxy':[1,1,3,3]}]*4,100,100))

    def test_split_determinism_and_exact_duplicate_union(self):
        def row(i,group,sha):
            return dict(id=str(i),original_group=group,sha256=sha,source='test',eligible=True,reviewed=True,boxes=[])
        a=Preparation.__new__(Preparation);a.rows=[row(0,'groupA','same'),row(1,'groupB','same')]+[row(i,f'g{i}',f'h{i}') for i in range(2,25)]
        b=Preparation.__new__(Preparation);b.rows=copy.deepcopy(a.rows)
        a.groups_and_splits();b.groups_and_splits()
        self.assertEqual([(r['group'],r['split']) for r in a.rows],[(r['group'],r['split']) for r in b.rows])
        self.assertEqual(a.rows[0]['group'],a.rows[1]['group']);self.assertFalse(a.rows[1]['eligible'])

    def test_external_pending_editor_retains_ambiguity(self):
        import tkinter as tk
        from residuos.annotate import Annotator
        from residuos import CLASSES
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);path=folder/'image.jpg'
            cv2.imencode('.jpg',np.zeros((80,100,3),np.uint8))[1].tofile(path)
            r=dict(id='external',path='image.jpg',source='Bangladesh canonical',original_path='original.jpg',group='group',session='group',
                   split='train',video=None,frame=None,width=100,height=80,sha256=digest(path),reviewed=False,eligible=False,
                   boxes=[{'class':'plastico','xyxy':[1,1,20,20]}],
                   pending_boxes=[{'class':None,'original_class':'Paper','xyxy':[30,20,60,50]}],pending_reasons=['clase_ambigua:Paper'])
            write_json(folder/'manifest.json',dict(version=2,classes=list(CLASSES),videos={},images=[r]))
            root=tk.Tk();root.withdraw()
            try:
                app=Annotator(root,folder);app.open_record(app.store.db['images'][0])
                self.assertEqual(len(app.boxes),2);self.assertIsNone(app.cap)
                with patch('residuos.annotate.messagebox.showerror') as error:
                    app.save();self.assertTrue(error.called)
                self.assertFalse(Store(folder).db['images'][0]['reviewed'])
                app.listbox.selection_set(1);app.class_name.set('Cartón');app.reclass()
                with patch('residuos.annotate.messagebox.askyesno',return_value=True):app.save()
                store=Store(folder);self.assertEqual(len(store.records('train')),1)
                self.assertEqual(store.db['images'][0]['pending_boxes'],[])
                self.assertEqual(store.validate(),[])
            finally:root.destroy()

if __name__=='__main__':unittest.main()
