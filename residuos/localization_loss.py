"""Experiment 2: replace normalized Smooth L1 by GIoU; all other loss terms fixed."""
import torch
from torch import nn
from .detector import GRID


def aligned_giou(a,b):
    lo=torch.maximum(a[...,:2],b[...,:2]);hi=torch.minimum(a[...,2:],b[...,2:])
    inter=(hi-lo).clamp(min=0).prod(-1)
    area_a=(a[...,2:]-a[...,:2]).clamp(min=0).prod(-1)
    area_b=(b[...,2:]-b[...,:2]).clamp(min=0).prod(-1)
    union=(area_a+area_b-inter).clamp(min=1e-9)
    cover=(torch.maximum(a[...,2:],b[...,2:])-torch.minimum(a[...,:2],b[...,:2])).clamp(min=0).prod(-1).clamp(min=1e-9)
    return inter/union-(cover-union)/cover


def boxes_at_positive(values,indices):
    # Indices: batch, gy, gx, slot. Width/height normalized to entire image.
    gx,gy=indices[:,2].to(values.dtype),indices[:,1].to(values.dtype)
    cx=(values[:,0]+gx)/GRID;cy=(values[:,1]+gy)/GRID
    return torch.stack((cx-values[:,2]/2,cy-values[:,3]/2,cx+values[:,2]/2,cy+values[:,3]/2),-1)


def loss_fn_giou(pred,target):
    positive=target[...,0]==1
    bce=nn.functional.binary_cross_entropy_with_logits(pred[...,0],target[...,0],reduction='none')
    obj=(bce*torch.where(positive,5.,.2)).mean()
    zero=pred.sum()*0
    if positive.any():
        indices=positive.nonzero()
        a=boxes_at_positive(pred[...,1:5][positive].sigmoid(),indices)
        b=boxes_at_positive(target[...,1:5][positive],indices)
        box=(1-aligned_giou(a,b)).mean()
        cls=nn.functional.cross_entropy(pred[...,5:][positive],target[...,5][positive].long())
    else:box=cls=zero
    return obj+8*box+cls
