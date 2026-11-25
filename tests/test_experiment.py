import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scripts.experiment_detector import Cached
from residuos.detector import encode
from residuos.evaluate import metrics


class ExperimentTests(unittest.TestCase):
    def test_flip_moves_pixels_and_boxes_together(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'pixels.npy'
            pixels=np.zeros((1,256,256,3),dtype=np.uint8);pixels[0,:,0]=255;np.save(path,pixels)
            rows=[dict(width=640,height=480,boxes=[{'class':'papel','xyxy':[0,40,160,240]}])]
            with patch('scripts.experiment_detector.random.random',return_value=0),patch('scripts.experiment_detector.random.uniform',return_value=1):
                image,target=Cached(rows,path,True)[0]
            self.assertEqual(float(image[:,:,-1].mean()),1)
            expected=encode([{'class':'papel','xyxy':[480,40,640,240]}],640,480)
            self.assertTrue(target.equal(expected))

    def test_matching_rejects_wrong_class_and_duplicate(self):
        truth=[{'class':'papel','xyxy':[0,0,100,100]}]
        predictions=[dict(truth[0],score=.9),dict(truth[0],score=.8),{'class':'carton','xyxy':[0,0,100,100],'score':.95}]
        result=metrics([(predictions,truth)])
        self.assertEqual((result['tp'],result['fp'],result['fn']),(1,2,0))
        self.assertEqual(result['classes']['carton']['tp'],0)
        missing=metrics([([],truth)])
        self.assertEqual(missing['classes']['papel']['fn'],1)


if __name__=='__main__':unittest.main()
