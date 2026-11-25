import unittest
import numpy as np
import torch
from scripts.diagnose_detector import overlaps,match,ap50,operating_point
from residuos.localization_loss import aligned_giou,loss_fn_giou
from residuos.detector import encode,Detector,seed_all


class DiagnosticTests(unittest.TestCase):
    def test_ap_perfect_duplicate_wrong_class(self):
        self.assertEqual(ap50([.9,.8],[True,True],2),1.)
        self.assertEqual(ap50([.9],[False],1),0.)
        self.assertAlmostEqual(ap50([.9,.8],[False,True],1),.5)
        xy=np.array([[0,0,10,10]]*3);truth=[{'class':'papel','xyxy':[0,0,10,10]}]
        h,_,reasons,_=match(xy,np.array([1,1,0]),np.array([.9,.8,.7]),truth)
        self.assertEqual(h.tolist(),[True,False,False]);self.assertEqual(reasons,['tp','duplicate','wrong_class'])

    def test_threshold_ties_and_abstention(self):
        x=operating_point([.5,.5],[True,False],1)
        self.assertEqual((x['tp'],x['fp']),(1,1))
        self.assertEqual(operating_point([.5],[False],1)['fp'],0)

    def test_giou_scale_and_gradients(self):
        a=torch.tensor([[.1,.1,.2,.2],[.1,.1,.2,.2]],requires_grad=True)
        b=torch.tensor([[.1,.1,.2,.2],[.3,.3,.4,.4]])
        g=aligned_giou(a,b)
        self.assertAlmostEqual(float(g[0]),1.,places=5);self.assertLess(float(g[1]),0)
        self.assertTrue(torch.allclose(g,aligned_giou(a*2,b*2),atol=1e-6))
        (1-g).mean().backward();self.assertTrue(torch.isfinite(a.grad).all())

    def test_real_architecture_loss_empty_and_positive(self):
        seed_all(20261123);model=Detector();z=model(torch.rand(2,3,256,256))
        target=torch.stack([encode([{'class':'organico','xyxy':[110,110,120,120]}],256,256),encode([],256,256)])
        loss=loss_fn_giou(z,target);self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


if __name__=='__main__':unittest.main()
