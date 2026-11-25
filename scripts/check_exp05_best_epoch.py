import json

with open('runs/detector_exp05/history.json', 'r', encoding='utf-8') as f:
    h = json.load(f)

best_epoch = min(h, key=lambda x: x['val_loss'])
print("Exp05 best epoch:", best_epoch['epoch'], "val_loss:", best_epoch['val_loss'], "time:", best_epoch['total_seconds'])
print("Total epochs in Exp05:", len(h), "Total time (mins):", h[-1]['total_seconds'] / 60)
