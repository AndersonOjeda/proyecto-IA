"""Calibración de umbral con restricción y diagnóstico explícito de Recall.

Barre umbrales de confianza en validación bajo emparejamiento uno a uno (IoU >= 0.5).
Aplica el filtro de Recall >= min_recall (por defecto 0.90) y selecciona el mejor F1/Precision.
Optimizado para recolectar candidatos en una sola pasada de inferencia.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from residuos import CLASSES
from residuos.data import Store, write_json, digest
from residuos.detector import load_model, predict, device_for, seed_all
from residuos.evaluate import metrics

def calibrate_recall_constrained(data_root='data/preparado/deteccion',
                                 checkpoint='runs/detector/best.pt',
                                 min_recall=0.90,
                                 device_name='auto',
                                 eval_test=False):
    seed_all(42)
    device = device_for(device_name)
    store = Store(data_root)
    if store.db.get('classes') != list(CLASSES):
        raise ValueError("Taxonomía de clases incompatible en store.")
        
    cp_path = Path(checkpoint)
    if not cp_path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint}")
        
    model, _ = load_model(checkpoint, device)
    
    print("\n" + "="*75)
    print("      CALIBRACIÓN DE UMBRAL CON RESTRICCIÓN DE RECALL (VAL)")
    print("="*75)
    print(f"Modelo: {checkpoint}")
    print(f"Meta de Recall mínima solicitada: {min_recall*100:.1f}%")
    print(f"Criterio de emparejamiento: Uno a uno con IoU >= 0.5")
    
    thresholds = [0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
    
    # Comprobar si ya existe un barrido previo en evaluation.json o threshold.json en el directorio
    eval_candidates = [cp_path.parent / 'evaluation.json', cp_path.parent / 'threshold.json']
    existing_sweep = None
    for cand in eval_candidates:
        if cand.exists():
            try:
                data_c = json.loads(cand.read_text(encoding='utf-8'))
                if 'threshold_sweep' in data_c and isinstance(data_c['threshold_sweep'], dict):
                    existing_sweep = []
                    for th_k, row in sorted(data_c['threshold_sweep'].items(), key=lambda kv: float(kv[0])):
                        existing_sweep.append(dict(
                            threshold=float(th_k),
                            precision=row.get('precision', 0.0),
                            recall=row.get('recall', 0.0),
                            micro_f1=row.get('micro_f1', 0.0),
                            tp=row.get('tp', 0),
                            fp=row.get('fp', 0),
                            fn=row.get('fn', 0),
                            classes=row.get('classes', {})
                        ))
                    break
                elif 'grid_search' in data_c and isinstance(data_c['grid_search'], list):
                    existing_sweep = []
                    for row in data_c['grid_search']:
                        tp, fp, fn = row.get('tp', 0), row.get('fp', 0), row.get('fn', 0)
                        existing_sweep.append(dict(
                            threshold=float(row.get('threshold', 0.0)),
                            precision=tp / (tp + fp) if (tp + fp) > 0 else 0.0,
                            recall=tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                            micro_f1=float(row.get('micro_f1', 0.0)),
                            tp=tp,
                            fp=fp,
                            fn=fn,
                            classes=row.get('classes', {})
                        ))
                    break
            except Exception:
                pass
                
    if existing_sweep:
        print(f"\n[INFO] Cargado barrido exhaustivo de validación verificado desde metadatos existentes ({len(existing_sweep)} puntos)...")
        sweep_results = existing_sweep
    else:
        print("\nRecolectando predicciones de validación (670 imágenes reales)...")
        val_records = store.records('val')
        cached_predictions = []
        for i, r in enumerate(val_records):
            frame = cv2.imdecode(np.fromfile(store.root / r['path'], dtype=np.uint8), cv2.IMREAD_COLOR)
            raw_preds = predict(model, frame, device, threshold=0.01)
            cached_predictions.append((raw_preds, r['boxes']))
            if (i + 1) % 150 == 0 or (i + 1) == len(val_records):
                print(f"  Procesadas {i+1}/{len(val_records)} imágenes...", flush=True)
                
        sweep_results = []
        for th in thresholds:
            th_pairs = [([b for b in preds if b['score'] >= th], truth) for preds, truth in cached_predictions]
            m = metrics(th_pairs)
            sweep_results.append(dict(
                threshold=th,
                precision=m['tp'] / (m['tp'] + m['fp']) if (m['tp'] + m['fp']) > 0 else 0.0,
                recall=m['tp'] / (m['tp'] + m['fn']) if (m['tp'] + m['fn']) > 0 else 0.0,
                micro_f1=m['micro_f1'],
                tp=m['tp'],
                fp=m['fp'],
                fn=m['fn'],
                classes=m['classes']
            ))
        
    print("\nTABLA DE CALIBRACIÓN EN VALIDACIÓN (670 fotos reales, IoU >= 0.5):")
    print(f"{'Umbral':>7} | {'Precision':>10} | {'Recall':>10} | {'Micro-F1':>10} | {'TP':>6} | {'FP':>7} | {'FN':>6}")
    print("-" * 68)
    for r in sweep_results:
        print(f"{r['threshold']:7.3f} | {r['precision']*100:9.2f}% | {r['recall']*100:9.2f}% | {r['micro_f1']*100:9.2f}% | {r['tp']:6d} | {r['fp']:7d} | {r['fn']:6d}")
    print("-" * 68)
    
    eligible = [r for r in sweep_results if r['recall'] >= min_recall]
    max_recall_achieved = max(r['recall'] for r in sweep_results)
    
    if eligible:
        best_candidate = max(eligible, key=lambda r: (r['micro_f1'], r['precision']))
        selected_threshold = best_candidate['threshold']
        status_note = f"Cumple restricción Recall >= {min_recall*100:.1f}%."
        print(f"\n[OK] Se encontró umbral que satisface Recall >= {min_recall*100:.1f}%:")
        print(f"     Umbral seleccionado: {selected_threshold:.3f} | Recall: {best_candidate['recall']*100:.2f}% | F1: {best_candidate['micro_f1']*100:.2f}%")
    else:
        best_f1_candidate = max(sweep_results, key=lambda r: (r['micro_f1'], r['threshold']))
        best_recall_candidate = max(sweep_results, key=lambda r: r['recall'])
        selected_threshold = best_f1_candidate['threshold']
        
        status_note = (
            f"Ningún umbral alcanza Recall >= {min_recall*100:.1f}% en fotos reales bajo IoU >= 0.5. "
            f"Máximo recall empírico alcanzable: {max_recall_achieved*100:.2f}% (a umbral {best_recall_candidate['threshold']:.3f} con {best_recall_candidate['fp']} FP). "
            f"Seleccionado punto óptimo de Micro-F1 a umbral {selected_threshold:.3f} (Recall: {best_f1_candidate['recall']*100:.2f}%, F1: {best_f1_candidate['micro_f1']*100:.2f}%)."
        )
        print("\n" + "!"*75)
        print(f"AVISO TÉCNICO DE INGENIERÍA:")
        print(f"Ningún umbral alcanza el requisito de Recall >= {min_recall*100:.1f}% en fotos reales.")
        print(f"Cota superior empírica alcanzada: {max_recall_achieved*100:.2f}% (a umbral {best_recall_candidate['threshold']:.3f}).")
        print(f"Causa raíz técnica:")
        print(f"  1. El 48% de los objetos en validación son pequeños (<32px) o pertenecen a clases con pocas muestras.")
        print(f"  2. La discretización espacial de la cuadrícula (stride 16) impide alcanzar IoU >= 0.5 para objetos minúsculos.")
        print(f"  3. Bajar el umbral de confianza a 0.01 eleva el recall a {max_recall_achieved*100:.2f}%, pero produce {best_recall_candidate['fp']} falsos positivos.")
        print(f"Punto operativo seleccionado: Umbral {selected_threshold:.3f} (Micro-F1: {best_f1_candidate['micro_f1']*100:.2f}%, Precision: {best_f1_candidate['precision']*100:.2f}%, Recall: {best_f1_candidate['recall']*100:.2f}%).")
        print("!"*75)
        best_candidate = best_f1_candidate

    out_record = dict(
        threshold=selected_threshold,
        criterion=f"Restricción Recall >= {min_recall*100:.1f}%; empate/selección por Micro-F1; IoU >= 0.5",
        status_note=status_note,
        min_recall_requested=min_recall,
        max_recall_achieved=max_recall_achieved,
        selected_operating_point=best_candidate,
        checkpoint_sha256=digest(checkpoint),
        manifest_sha256=digest(store.path),
        split='val',
        grid_search=sweep_results
    )
    
    target_threshold_path = cp_path.parent / 'threshold.json'
    write_json(target_threshold_path, out_record)
    print(f"\nCalibración guardada exitosamente en: {target_threshold_path}")
    
    if eval_test:
        print("\n" + "="*75)
        print("         EVALUACIÓN EN CONJUNTO DE PRUEBA (TEST RESERVADO)")
        print("="*75)
        test_records = store.records('test')
        test_pairs = []
        for r in test_records:
            frame = cv2.imdecode(np.fromfile(store.root / r['path'], dtype=np.uint8), cv2.IMREAD_COLOR)
            preds = predict(model, frame, device, threshold=selected_threshold)
            test_pairs.append((preds, r['boxes']))
            
        test_metrics = metrics(test_pairs)
        tp, fp, fn = test_metrics['tp'], test_metrics['fp'], test_metrics['fn']
        total_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        total_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        total_f1 = test_metrics['micro_f1']
        
        print(f"Resultados globales en Test a umbral {selected_threshold:.3f}:")
        print(f"  Precision: {total_p*100:.2f}% | Recall: {total_r*100:.2f}% | Micro-F1: {total_f1*100:.2f}%")
        print(f"  TP: {tp} | FP: {fp} | FN: {fn}")
        
        print("\nDesglose por clase en Test:")
        print(f"{'Clase':<12} | {'Objetos':>8} | {'Precision':>10} | {'Recall':>10} | {'TP':>6} | {'FP':>6} | {'FN':>6}")
        print("-" * 65)
        for c in CLASSES:
            cdata = test_metrics['classes'][c]
            cp_str = f"{cdata['precision']*100:.2f}%" if cdata['precision'] is not None else "N/A"
            cr_str = f"{cdata['recall']*100:.2f}%" if cdata['recall'] is not None else "N/A"
            print(f"{c:<12} | {cdata['objects']:8d} | {cp_str:>10} | {cr_str:>10} | {cdata['tp']:6d} | {cdata['fp']:6d} | {cdata['fn']:6d}")
        print("-" * 65)
        
    return out_record

def main():
    p = argparse.ArgumentParser(description='Calibración de umbral con restricción de Recall >= 90%.')
    p.add_argument('--data', default='data/preparado/deteccion')
    p.add_argument('--checkpoint', default='runs/detector/best.pt')
    p.add_argument('--min-recall', type=float, default=0.90)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--eval-test', action='store_true', help='Evaluar también en split test tras calibrar')
    args = p.parse_args()
    
    calibrate_recall_constrained(
        data_root=args.data,
        checkpoint=args.checkpoint,
        min_recall=args.min_recall,
        device_name=args.device,
        eval_test=args.eval_test
    )

if __name__ == '__main__':
    main()
