import unittest
import numpy as np
import torch
from scripts.experiment7_runner import sanear_cajas, validar_sin_fugas, normalizar_ruta
from scripts.experiment3_runner import encode_grid32, GRID, SLOTS
from residuos import CLASSES

class TestExperiment7(unittest.TestCase):
    def test_sanear_cajas_no_mutates_original(self):
        original = [{'class': 'plastico', 'xyxy': [10, 20, 50, 60]}]
        cleaned = sanear_cajas(original, 100, 100)
        self.assertEqual(len(cleaned), 1)
        # Modificar el resultado no debe afectar al original
        cleaned[0]['xyxy'][0] = 999
        self.assertEqual(original[0]['xyxy'], [10, 20, 50, 60])

    def test_sanear_cajas_clipping_and_inverted_coords(self):
        # Cajas con coordenadas fuera de rango e invertidas
        boxes = [
            {'class': 'papel', 'xyxy': [-10, -5, 120, 150]},  # fuera de rango
            {'class': 'metal', 'xyxy': [80, 70, 20, 30]},      # x1>x2, y1>y2
            {'class': 'vidrio', 'xyxy': [40, 40, 40, 60]},     # x1 == x2 (degenerada)
            {'class': 'carton', 'xyxy': [10, 50, 30, 50]},     # y1 == y2 (degenerada)
            {'class': 'plastico', 'xyxy': [200, 200, 250, 250]} # totalmente fuera
        ]
        cleaned = sanear_cajas(boxes, width=100, height=100)
        self.assertEqual(len(cleaned), 2)
        # Primera caja recortada a [0, 0, 100, 100]
        self.assertEqual(cleaned[0]['xyxy'], [0.0, 0.0, 100.0, 100.0])
        self.assertEqual(cleaned[0]['class'], 'papel')
        # Segunda caja invertida reordenada y recortada a [20, 30, 80, 70]
        self.assertEqual(cleaned[1]['xyxy'], [20.0, 30.0, 80.0, 70.0])
        self.assertEqual(cleaned[1]['class'], 'metal')

    def test_horizontal_flip_with_sanear_cajas_and_encode(self):
        w_orig, h_orig = 640, 480
        boxes = [{'class': 'vidrio', 'xyxy': [100.0, 50.0, 200.0, 150.0]}]
        clean = sanear_cajas(boxes, w_orig, h_orig)

        # Aplicar flip horizontal
        flipped = []
        for b in clean:
            x1, y1, x2, y2 = b['xyxy']
            nx1 = w_orig - x2
            nx2 = w_orig - x1
            b_flip = dict(b)
            b_flip['xyxy'] = [min(nx1, nx2), y1, max(nx1, nx2), y2]
            flipped.append(b_flip)
        flipped_clean = sanear_cajas(flipped, w_orig, h_orig)

        self.assertEqual(len(flipped_clean), 1)
        self.assertEqual(flipped_clean[0]['xyxy'], [440.0, 50.0, 540.0, 150.0])

        # Verificar compatibilidad con encode_grid32
        target = encode_grid32(flipped_clean, w_orig, h_orig)
        self.assertEqual(target.shape, (GRID, GRID, SLOTS, 6))
        # Debe haber exactamente 1 celda positiva
        self.assertEqual(int((target[..., 0] == 1).sum().item()), 1)

    def test_validar_sin_fugas_detects_leak_and_raises(self):
        orig_val = [{'id': 1, 'path': 'images/val/sample1.jpg', 'sha256': 'abc123val'}]
        # Caso con fuga por ruta normalizada
        orig_train_leak = [{'id': 10, 'path': 'images\\val\\sample1.jpg', 'sha256': 'def456'}]
        ext_rows = [{'id': 20, 'path': 'images/ext/sample2.jpg'}]
        manual_train = [{'id': 30, 'path': 'images/manual/sample3.jpg'}]
        sep_rows = [{'id': 40, 'path': 'images/sep/sample4.jpg'}]

        with self.assertRaises(RuntimeError) as ctx:
            validar_sin_fugas(orig_val, orig_train_leak, ext_rows, manual_train, sep_rows)
        self.assertIn('ESCUDO ANTI-FUGA ACTIVADO', str(ctx.exception))

        # Caso con fuga por sha256
        orig_train_ok = [{'id': 10, 'path': 'images/train/sample10.jpg', 'sha256': 'different'}]
        manual_train_leak = [{'id': 30, 'path': 'images/manual/other.jpg', 'sha256': 'abc123val'}]

        with self.assertRaises(RuntimeError) as ctx:
            validar_sin_fugas(orig_val, orig_train_ok, ext_rows, manual_train_leak, sep_rows)
        self.assertIn('ESCUDO ANTI-FUGA ACTIVADO', str(ctx.exception))

    def test_validar_sin_fugas_clean(self):
        orig_val = [{'id': 1, 'path': 'images/val/sample1.jpg', 'sha256': 'hash_val'}]
        orig_train = [{'id': 10, 'path': 'images/train/sample10.jpg', 'sha256': 'hash_train'}]
        ext_rows = [{'id': 20, 'path': 'images/ext/sample20.jpg', 'sha256': 'hash_ext'}]
        manual_train = [{'id': 30, 'path': 'images/manual/sample30.jpg', 'sha256': 'hash_manual'}]
        sep_rows = [{'id': 40, 'path': 'images/sep/sample40.jpg', 'sha256': 'hash_sep'}]

        audit = validar_sin_fugas(orig_val, orig_train, ext_rows, manual_train, sep_rows)
        self.assertEqual(audit['status'], 'PASSED_CLEAN')
        self.assertFalse(audit['val_leakage_detected'])
        self.assertEqual(audit['counts']['total_train'], 4)
        self.assertIn('train_overlaps', audit)

if __name__ == '__main__':
    unittest.main()
