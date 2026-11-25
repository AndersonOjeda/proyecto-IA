"""Módulo de arquitectura de detección multiobjeto y función de pérdida optimizada para Recall.

Incluye:
- Detector (arquitectura compacta nativa de 248K parámetros, inicializada desde cero).
- DetectorS8 (arquitectura residual escala S=8 con cuadrícula 32x32 para mayor resolución).
- DetectorLoss: Focal Loss para presencia + Localización GIoU/Smooth L1 + Ponderación inversa de clases.
"""
import math
import torch
import torch.nn as nn
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from residuos import CLASSES
from residuos.detector import Detector, SIZE, GRID, SLOTS

# Pesos de clase calculados con base en la frecuencia inversa suave en train
# plastico: 3398, papel: 57, carton: 82, vidrio: 71, metal: 260, organico: 5084
# Clases minoritarias reciben ponderaciones significativamente reforzadas para evitar omisiones (FN)
DEFAULT_CLASS_WEIGHTS = torch.tensor([0.65, 2.80, 2.50, 2.65, 1.60, 0.55], dtype=torch.float32)

def aligned_giou(a, b):
    """Calcula GIoU entre dos conjuntos de cajas xyxy alineadas."""
    lo = torch.maximum(a[..., :2], b[..., :2])
    hi = torch.minimum(a[..., 2:], b[..., 2:])
    inter = (hi - lo).clamp(min=0).prod(-1)
    area_a = (a[..., 2:] - a[..., :2]).clamp(min=0).prod(-1)
    area_b = (b[..., 2:] - b[..., :2]).clamp(min=0).prod(-1)
    union = (area_a + area_b - inter).clamp(min=1e-9)
    cover = (torch.maximum(a[..., 2:], b[..., 2:]) - torch.minimum(a[..., :2], b[..., :2])).clamp(min=0).prod(-1).clamp(min=1e-9)
    return inter / union - (cover - union) / cover

def boxes_at_positive(values, indices, grid_size):
    """Convierte coordenadas normalizadas de celda a xyxy en la imagen normalizada."""
    gx = indices[:, 2].to(values.dtype)
    gy = indices[:, 1].to(values.dtype)
    cx = (values[:, 0] + gx) / grid_size
    cy = (values[:, 1] + gy) / grid_size
    w = values[:, 2]
    h = values[:, 3]
    return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), -1)

class DetectorLoss(nn.Module):
    """Pérdida multiobjeto compuesta optimizada para Recall >= 90%.
    
    1. Focal Loss en presencia (gamma=2.0, alpha=0.75): prioriza ejemplos difíciles y reduce penalización de fondos.
    2. Pérdida híbrida de caja (Smooth L1 + GIoU): premia directamente que el IoU supere 0.5.
    3. Entropía cruzada reponderada: penaliza fuertemente el error en clases minoritarias.
    """
    def __init__(self, class_weights=None, focal_gamma=2.0, focal_alpha=0.75, 
                 obj_scale=2.0, box_weight=10.0, use_giou=True):
        super().__init__()
        self.class_weights = class_weights if class_weights is not None else DEFAULT_CLASS_WEIGHTS
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.obj_scale = obj_scale
        self.box_weight = box_weight
        self.use_giou = use_giou

    def forward(self, pred, target):
        positive = target[..., 0] == 1
        n_pos = max(float(positive.sum()), 1.0)
        
        # 1. Focal Loss para presencia (reducir Falsos Negativos por baja confianza en positivo)
        p = torch.sigmoid(pred[..., 0])
        pt = p * target[..., 0] + (1.0 - p) * (1.0 - target[..., 0])
        focal_weight = (1.0 - pt) ** self.focal_gamma
        alpha_weight = self.focal_alpha * target[..., 0] + (1.0 - self.focal_alpha) * (1.0 - target[..., 0])
        bce = nn.functional.binary_cross_entropy_with_logits(pred[..., 0], target[..., 0], reduction='none')
        fl = alpha_weight * focal_weight * bce
        loss_obj = (fl.sum() / n_pos) * self.obj_scale
        
        zero = pred.sum() * 0
        grid_size = pred.shape[1]  # GRID size (16 o 32)
        
        # 2. Pérdida de localización (caja)
        if positive.any():
            l1_loss = nn.functional.smooth_l1_loss(pred[..., 1:5][positive].sigmoid(), target[..., 1:5][positive])
            if self.use_giou:
                indices = positive.nonzero()
                pred_boxes = boxes_at_positive(pred[..., 1:5][positive].sigmoid(), indices, grid_size)
                true_boxes = boxes_at_positive(target[..., 1:5][positive], indices, grid_size)
                giou_loss = (1.0 - aligned_giou(pred_boxes, true_boxes)).mean()
                loss_box = 0.5 * l1_loss + 0.5 * giou_loss
            else:
                loss_box = l1_loss
        else:
            loss_box = zero
            
        # 3. Pérdida de clasificación con pesos inversos para clases con bajo recall
        if positive.any():
            w = self.class_weights.to(pred.device) if self.class_weights is not None else None
            loss_cls = nn.functional.cross_entropy(pred[..., 5:][positive], target[..., 5][positive].long(), weight=w)
        else:
            loss_cls = zero
            
        return loss_obj + self.box_weight * loss_box + loss_cls

def create_detector(model_type='standard'):
    """Crea detector inicializado estrictamente desde cero (sin pesos externos)."""
    if model_type == 'standard':
        return Detector()
    elif model_type == 's8_residual':
        from scripts.experiment3_runner import DetectorS8
        return DetectorS8()
    else:
        raise ValueError(f"Tipo de modelo desconocido: {model_type}")
