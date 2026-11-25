import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
import torch
from residuos import CLASSES
from residuos.detector import decode,load_model,seed_all
from residuos.evaluate import threshold_for
from residuos.data import digest,write_json
from scripts.experiment3_runner import decode_grid32,DetectorS8 as TrainingDetectorS8
from residuos.detector_s8 import DetectorS8


class AppModelTests(unittest.TestCase):
    def test_s8_architecture_and_decoder_match_training(self):
        seed_all(42);a=TrainingDetectorS8();b=DetectorS8();b.load_state_dict(a.state_dict())
        with torch.no_grad():
            x=torch.rand(1,3,256,256);za=a(x)[0];zb=b(x)[0]
        self.assertTrue(torch.equal(za,zb))
        # Use actual logits and a low enough threshold to exercise boxes and NMS.
        self.assertEqual(decode(za,640,480,.005,grid=32),decode_grid32(za,640,480,.005))
        with self.assertRaises(ValueError):decode(za,640,480,.4)

    def test_threshold_rejects_other_checkpoint_and_nan(self):
        with tempfile.TemporaryDirectory() as folder:
            cp=Path(folder)/'best.pt';cp.write_bytes(b'checkpoint-a')
            p=cp.parent/'threshold.json'
            write_json(p,dict(threshold=.4,checkpoint_sha256='wrong'))
            with self.assertRaises(ValueError):threshold_for(cp)
            write_json(p,dict(threshold=.4,checkpoint_sha256=digest(cp)))
            self.assertEqual(threshold_for(cp),.4)
            write_json(p,dict(threshold=float('nan'),checkpoint_sha256=digest(cp)))
            with self.assertRaises(ValueError):threshold_for(cp)

    def test_hidden_gui_identifies_real_checkpoint_and_threshold(self):
        import tkinter as tk
        from residuos.app import Application
        results=[]
        for path,threshold in [('runs/detector_exp03/corrected/best.pt',.4),('runs/detector_exp05/best.pt',.35)]:
            root=tk.Tk();root.withdraw();buf=io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    app=Application(root,data='data/preparado/deteccion',detector=path,device='cpu')
                self.assertIsNotNone(app.model);self.assertEqual(app.threshold,threshold)
                printed=json.loads(buf.getvalue());self.assertEqual(printed['checkpoint_sha256'],digest(path))
                self.assertEqual(app.detector_identity,printed)
                labels=[w.cget('text') for w in root.winfo_children() if w.winfo_class()=='TLabel']
                self.assertTrue(any(printed['checkpoint_sha256'] in s and f'{threshold:.2f}' in s for s in labels))
                results.append(printed)
            finally:root.destroy()
        write_json('runs/verificacion_app_exp03_exp05/gui_startup_checks.json',dict(passed=True,models=results))


if __name__=='__main__':unittest.main()
