"""CNN densa desde cero: 16x16 celdas, tres candidatos por celda."""
import random
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from . import CLASSES
from .data import Store, write_json, digest

SIZE, GRID, SLOTS = 256, 16, 3

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(min(4, torch.get_num_threads()))

def device_for(name='auto'):
    if name == 'auto': return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if name == 'cuda' and not torch.cuda.is_available(): raise ValueError('CUDA no disponible. Use cpu o auto.')
    return torch.device(name)

class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        layers, channels = [], 3
        for out in (16, 32, 64, 96):
            layers += [nn.Conv2d(channels, out, 3, stride=2, padding=1), nn.GroupNorm(4, out), nn.SiLU()]
            channels = out
        layers += [nn.Conv2d(96, 96, 3, padding=1), nn.SiLU(), nn.Conv2d(96, 96, 3, padding=1), nn.SiLU()]
        self.features = nn.Sequential(*layers)
        self.head = nn.Conv2d(96, SLOTS*(5+len(CLASSES)), 1)
        # Sólo inicialización local; no hay descarga ni carga de backbones.
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias.view(SLOTS, -1)[:, 0] = -3

    def forward(self, x):
        z = self.head(self.features(x))
        return z.view(len(x), SLOTS, 11, GRID, GRID).permute(0, 3, 4, 1, 2)

def image_tensor(frame):
    rgb = cv2.cvtColor(cv2.resize(frame, (SIZE, SIZE)), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255

def encode(boxes, width, height):
    target = torch.zeros(GRID, GRID, SLOTS, 6)
    cells = {}
    for box in boxes:
        x1,y1,x2,y2 = box['xyxy']
        cx, cy, w, h = (x1+x2)/2/width, (y1+y2)/2/height, (x2-x1)/width, (y2-y1)/height
        gx, gy = min(int(cx*GRID),GRID-1), min(int(cy*GRID),GRID-1)
        cells.setdefault((gy,gx), []).append((cx,cy,w,h,CLASSES.index(box['class'])))
    for (gy,gx), entries in cells.items():
        if len(entries) > SLOTS:
            raise ValueError(f'Más de {SLOTS} centros en la celda ({gx},{gy}). Aumente resolución/slots; no se descartan etiquetas.')
        for slot, (cx,cy,w,h,c) in enumerate(sorted(entries)):
            target[gy,gx,slot] = torch.tensor([1,cx*GRID-gx,cy*GRID-gy,w,h,c])
    return target

class WasteDataset(Dataset):
    def __init__(self, store, split, augment=False):
        self.store, self.rows, self.augment = store, store.records(split), augment
        if not self.rows: raise ValueError(f'No hay imágenes revisadas en {split}. Anote datos reales primero.')
        for r in self.rows: encode(r['boxes'], r['width'], r['height'])

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        frame = cv2.imdecode(np.fromfile(self.store.root/r['path'], dtype=np.uint8), cv2.IMREAD_COLOR)
        frame = cv2.resize(frame, (SIZE, SIZE))
        boxes = [dict(b, xyxy=list(b['xyxy'])) for b in r['boxes']]
        if self.augment and random.random() < .5:
            frame = cv2.flip(frame, 1)
            for b in boxes:
                x1,y1,x2,y2 = b['xyxy']; b['xyxy'] = [r['width']-x2,y1,r['width']-x1,y2]
        if self.augment:
            frame = np.clip(frame.astype(np.float32)*random.uniform(.85,1.15),0,255).astype(np.uint8)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255
        return tensor, encode(boxes, r['width'], r['height'])

def loss_fn(pred, target):
    positive = target[...,0] == 1
    bce = nn.functional.binary_cross_entropy_with_logits(pred[...,0], target[...,0], reduction='none')
    obj = (bce * torch.where(positive, 5., .2)).mean()
    zero = pred.sum()*0
    box = nn.functional.smooth_l1_loss(pred[...,1:5][positive].sigmoid(), target[...,1:5][positive]) if positive.any() else zero
    cls = nn.functional.cross_entropy(pred[...,5:][positive], target[...,5][positive].long()) if positive.any() else zero
    return obj + 8*box + cls

def iou(a, b):
    x1,y1,x2,y2 = max(a[0],b[0]), max(a[1],b[1]), min(a[2],b[2]), min(a[3],b[3])
    inter = max(0,x2-x1)*max(0,y2-y1)
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union > 0 else 0.

def decode(output, width, height, threshold=.25, nms=.45, grid=GRID):
    if output.shape[:2] != (grid, grid):
        raise ValueError('La rejilla del decoder no coincide con la salida del modelo.')
    out = output.detach().cpu()
    probs, classes = out[...,5:].softmax(-1).max(-1)
    scores = out[...,0].sigmoid()*probs
    boxes = []
    for gy,gx,slot in (scores >= threshold).nonzero().tolist():
        dx,dy,w,h = out[gy,gx,slot,1:5].sigmoid().tolist()
        cx,cy = (gx+dx)/grid, (gy+dy)/grid
        coords = [max(0,(cx-w/2)*width),max(0,(cy-h/2)*height),min(width,(cx+w/2)*width),min(height,(cy+h/2)*height)]
        if coords[2] > coords[0] and coords[3] > coords[1]:
            boxes.append({'class':CLASSES[int(classes[gy,gx,slot])], 'xyxy':coords, 'score':float(scores[gy,gx,slot])})
    kept = []
    for b in sorted(boxes, key=lambda b:b['score'], reverse=True):
        if not any(k['class']==b['class'] and iou(k['xyxy'],b['xyxy']) > nms for k in kept): kept.append(b)
        if len(kept) == 100: break
    return kept

def load_model(path, device):
    if not Path(path).exists(): raise ValueError('Faltan pesos propios. Ejecute: python -m residuos train-detector')
    saved = torch.load(path, map_location=device, weights_only=True)
    origin = saved.get('origin')
    s8_origins = {'residuos-exp03-scratch', 'residuos-detector-exp03c', 'residuos-detector-exp05-outdoor-from-scratch'}
    classes = saved.get('classes', saved.get('config', {}).get('classes'))
    # Exp05's historical schema omits taxonomy; its versioned origin fixes this six-class order.
    if classes is None and origin == 'residuos-detector-exp05-outdoor-from-scratch':
        classes = list(CLASSES)
    if origin not in s8_origins | {'residuos-from-scratch-v1'} or classes != list(CLASSES):
        raise ValueError('Checkpoint incompatible; se requieren pesos propios de este proyecto.')
    if origin in s8_origins:
        from .detector_s8 import DetectorS8
        model = DetectorS8().to(device)
    else:
        model = Detector().to(device)
    model.load_state_dict(saved['state'], strict=True); model.eval()
    return model, saved

@torch.no_grad()
def predict(model, frame, device, threshold):
    output = model(image_tensor(frame)[None].to(device))[0]
    return decode(output, frame.shape[1], frame.shape[0], threshold, grid=output.shape[0])

def train(data='data', output='runs/detector', epochs=40, batch=8, lr=.001, seed=42, device='auto'):
    seed_all(seed); device = device_for(device); store = Store(data)
    errors = store.validate()
    if errors: raise ValueError('\n'.join(errors))
    trainset, valset = WasteDataset(store,'train',True), WasteDataset(store,'val')
    if not any(r['boxes'] for r in trainset.rows): raise ValueError('Entrenamiento sin objetos etiquetados.')
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    config = dict(epochs=epochs,batch=batch,lr=lr,seed=seed,device=str(device),size=SIZE,grid=GRID,slots=SLOTS,
                  manifest_sha256=digest(store.path),torch_version=str(torch.__version__),classes=list(CLASSES))
    write_json(out/'config.json', config)
    loaders = {k:DataLoader(v,batch_size=batch,shuffle=(k=='train'),num_workers=0) for k,v in [('train',trainset),('val',valset)]}
    model = Detector().to(device); optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
    best, history, start = float('inf'), [], time.perf_counter()
    for epoch in range(epochs):
        row = {'epoch':epoch+1}
        for split, loader in loaders.items():
            model.train(split=='train'); total = 0.
            for x,y in loader:
                x,y = x.to(device),y.to(device)
                with torch.set_grad_enabled(split=='train'):
                    loss = loss_fn(model(x),y)
                    if not torch.isfinite(loss): raise ValueError('Pérdida no finita.')
                    if split=='train':
                        optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step()
                total += loss.item()*len(x)
            row[split+'_loss'] = total/len(loader.dataset)
        row['seconds_elapsed'] = time.perf_counter()-start
        history.append(row); write_json(out/'history.json',history); print(row, flush=True)
        if row['val_loss'] < best:
            best = row['val_loss']
            torch.save(dict(origin='residuos-from-scratch-v1',classes=list(CLASSES),state=model.state_dict(),config=config,epoch=epoch+1), out/'best.pt')
    return history
