"""Módulo de inferencia para detección multiobjeto en aplicación."""
from pathlib import Path
import sys
import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from residuos import CLASSES
from residuos.detector import load_model, predict, device_for
from residuos.evaluate import threshold_for

class WasteDetector:
    """Detector de residuos multiobjeto con calibración de umbral y modos operativos."""
    def __init__(self, checkpoint='runs/detector/best.pt', device='auto', min_confidence=None, mode='cobertura', min_box_size=None):
        self.device = device_for(device)
        self.checkpoint = Path(checkpoint)
        self.model, self.metadata = load_model(str(self.checkpoint), self.device)
        self.mode = (mode or 'cobertura').lower()
        
        # Modos operativos calibrados sobre fotos reales de validación:
        # - 'cobertura': th=0.35, min_box_size=0 px (prioriza Recall y cobertura, 419 TP)
        # - 'precision': th=0.40, min_box_size=32 px (alta precisión 26.99%, suprime falsos positivos a 836 FP)
        if self.mode == 'precision':
            default_th = 0.40
            default_min_size = 32
        else: # 'cobertura' o estándar
            default_th = None
            default_min_size = 0

        # Umbral calibrado / configurado
        if min_confidence is not None:
            self.threshold = float(min_confidence)
        elif default_th is not None:
            self.threshold = float(default_th)
        else:
            self.threshold = threshold_for(str(self.checkpoint))
            
        # Filtro de tamaño mínimo de caja en píxeles (w >= min_box_size y h >= min_box_size)
        if min_box_size is not None:
            self.min_box_size = int(min_box_size)
        else:
            self.min_box_size = int(default_min_size)
                
    def detect(self, frame):
        """Detecta residuos en una imagen BGR de OpenCV.
        
        Retorna lista de diccionarios: [{'class': str, 'score': float, 'xyxy': [x1, y1, x2, y2]}]
        """
        if frame is None or frame.size == 0:
            return []
        predictions = predict(self.model, frame, self.device, self.threshold)
        if self.min_box_size > 0:
            predictions = [
                b for b in predictions
                if (b['xyxy'][2] - b['xyxy'][0]) >= self.min_box_size
                and (b['xyxy'][3] - b['xyxy'][1]) >= self.min_box_size
            ]
        return predictions
