"""Transformer de intenciones aprendido; recomendaciones deterministas revisables."""
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
import torch
from torch import nn
from .data import read_json, write_json, digest
from .detector import seed_all, device_for

INTENTS = ['inventario', 'separar', 'organicos', 'suciedad', 'fuera_alcance']
MAX_LEN = 32
DEFAULT_CORPUS = 'data/intents.json'
RULES = Path(__file__).with_name('recommendations.json')

def tokens(text):
    text = ''.join(c for c in unicodedata.normalize('NFD',text.lower()) if unicodedata.category(c) != 'Mn')
    return re.findall(r'[a-z0-9]+', text)

def vocabulary(rows):
    counts=Counter(t for row in rows for t in tokens(row['text']))
    return {'<pad>':0,'<unk>':1,**{word:i+2 for i,word in enumerate(sorted(counts))}}

def encode_text(text,vocab):
    ids=[vocab.get(t,1) for t in tokens(text)[:MAX_LEN]] or [1]
    return ids+[0]*(MAX_LEN-len(ids))

class IntentTransformer(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,48,padding_idx=0)
        self.position=nn.Embedding(MAX_LEN,48)
        layer=nn.TransformerEncoderLayer(d_model=48,nhead=4,dim_feedforward=96,dropout=.1,batch_first=True)
        self.encoder=nn.TransformerEncoder(layer,num_layers=2,enable_nested_tensor=False)
        # PyTorch clona las capas: reinicializamos cada una independientemente.
        for block in self.encoder.layers:
            for p in block.parameters():
                if p.ndim>1: nn.init.xavier_uniform_(p)
        self.head=nn.Linear(48,len(INTENTS))

    def forward(self,ids):
        mask=ids==0
        x=self.embedding(ids)+self.position(torch.arange(ids.shape[1],device=ids.device))[None]
        x=self.encoder(x,src_key_padding_mask=mask)
        valid=(~mask).unsqueeze(-1)
        pooled=(x*valid).sum(1)/valid.sum(1).clamp_min(1)
        return self.head(pooled)

def validate_corpus(corpus):
    seen={}
    for split in ('train','val','test'):
        if not corpus.get(split): raise ValueError(f'Faltan preguntas en {split}.')
        for r in corpus[split]:
            if r['intent'] not in INTENTS: raise ValueError('Intención desconocida.')
            key=' '.join(tokens(r['text']))
            if key in seen: raise ValueError(f'Pregunta duplicada entre/en conjuntos: {r["text"]}')
            seen[key]=split
    if set(r['intent'] for r in corpus['train']) != set(INTENTS): raise ValueError('Faltan intenciones en entrenamiento.')

def tensor_rows(rows,vocab,device):
    return (torch.tensor([encode_text(r['text'],vocab) for r in rows],device=device),
            torch.tensor([INTENTS.index(r['intent']) for r in rows],device=device))

def train(corpus=DEFAULT_CORPUS,output='runs/assistant',epochs=100,seed=42,device='auto'):
    seed_all(seed); device=device_for(device); data=read_json(corpus); validate_corpus(data)
    vocab=vocabulary(data['train']); model=IntentTransformer(len(vocab)).to(device)
    x,y=tensor_rows(data['train'],vocab,device); vx,vy=tensor_rows(data['val'],vocab,device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002)
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    history=[]; best=float('inf'); start=time.perf_counter()
    for epoch in range(epochs):
        model.train(); order=torch.randperm(len(x),device=device); total=0.
        for batch in order.split(16):
            loss=nn.functional.cross_entropy(model(x[batch]),y[batch])
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step()
            total+=loss.item()*len(batch)
        model.eval()
        with torch.no_grad():
            logits=model(vx); vloss=nn.functional.cross_entropy(logits,vy).item()
            acc=(logits.argmax(-1)==vy).float().mean().item()
        history.append(dict(epoch=epoch+1,train_loss=total/len(x),val_loss=vloss,val_accuracy=acc,seconds_elapsed=time.perf_counter()-start))
        if vloss<best:
            best=vloss
            torch.save(dict(origin='intent-from-scratch-v1',state=model.state_dict(),vocab=vocab,intents=INTENTS,
                            seed=seed,epoch=epoch+1,corpus_sha256=digest(corpus)),out/'best.pt')
        if (epoch+1)%10==0: print(history[-1],flush=True)
    write_json(out/'history.json',history)
    write_json(out/'config.json',dict(epochs=epochs,seed=seed,device=str(device),lr=.002,batch=16,
                                    dimensions=48,heads=4,layers=2,corpus_sha256=digest(corpus)))
    assistant=Assistant(out/'best.pt',device)
    candidates=[]
    for t in (.35,.45,.55,.65,.75,.85,.95):
        correct=sum(assistant.recognize(r['text'],t)[0]==r['intent'] for r in data['val'])
        candidates.append((correct,t))
    threshold=max(candidates)[1]
    write_json(out/'threshold.json',dict(threshold=threshold,checkpoint_sha256=digest(out/'best.pt'),split='val',criterion='accuracy incluyendo fuera_alcance; empate favorece rechazo'))
    return history[-1]

class Assistant:
    def __init__(self,checkpoint='runs/assistant/best.pt',device='cpu'):
        if not Path(checkpoint).exists(): raise ValueError('Faltan pesos propios del transformer. Ejecute: python -m residuos train-assistant')
        self.device=torch.device(device)
        cp=torch.load(checkpoint,map_location=self.device,weights_only=True)
        if cp.get('origin')!='intent-from-scratch-v1' or cp['intents']!=INTENTS: raise ValueError('Checkpoint de intenciones incompatible.')
        self.vocab=cp['vocab']; self.model=IntentTransformer(len(self.vocab)).to(self.device)
        self.model.load_state_dict(cp['state']); self.model.eval()
        self.threshold=.55
        threshold_path=Path(checkpoint).parent/'threshold.json'
        if threshold_path.exists():
            calibration=read_json(threshold_path)
            if calibration['checkpoint_sha256']==digest(checkpoint): self.threshold=calibration['threshold']
        self.rules=read_json(RULES)

    @torch.no_grad()
    def recognize(self,text,threshold=None):
        words=tokens(text)
        if not words or len(words)>MAX_LEN or sum(w not in self.vocab for w in words)/len(words)>.6:
            return 'fuera_alcance',0.
        ids=torch.tensor([encode_text(text,self.vocab)],device=self.device)
        probs=self.model(ids).softmax(-1)[0]; score,index=probs.max(-1)
        intent=INTENTS[int(index)]
        if score.item()<(self.threshold if threshold is None else threshold): intent='fuera_alcance'
        return intent,score.item()

    def answer(self,text,detections):
        intent,score=self.recognize(text)
        return dict(intent=intent,confidence=score,text=respond(intent,text,detections,self.rules))

def respond(intent,text,detections,rules=None):
    rules=rules or read_json(RULES)
    counts=Counter(d['class'] for d in detections)
    if intent=='fuera_alcance': return 'No reconozco con suficiente seguridad esa consulta. Pregunte por los residuos visibles, su separación o limpieza.'
    if not detections: return 'No hay detecciones confiables en este fotograma. No puedo confirmar materiales ni su ausencia física.'
    if intent=='inventario': return 'Detecciones del modelo: '+', '.join(f'{n} {c}' for c,n in counts.items())+'. Verifique visualmente las cajas.'
    if intent=='organicos':
        return (f'Se detectaron {counts["organico"]} restos orgánicos. '+rules['organico']['advice']) if counts['organico'] else 'No se detectaron restos orgánicos con el umbral actual; esto no garantiza que no existan.'
    words=set(tokens(text))
    # Botella/envase describen forma o uso; no demuestran que sea plástico.
    generic={'botella','botellas','envase','envases'}
    mentioned=[c for c,r in rules.items() if words.intersection(set(r['aliases'])-generic)]
    if not mentioned and words.intersection(generic):
        possible=[c for c in counts if c!='organico']
        if len(possible)!=1:
            return 'Botella o envase no identifica por sí solo el material. ¿Es plástico, vidrio, metal u otro material? Confírmelo con el objeto.'
        mentioned=possible
    if len(mentioned)>1: return 'Indique un solo material para precisar la recomendación.'
    if mentioned:
        material=mentioned[0]
        if material not in counts: return f'No hay una detección confiable de {material} en este fotograma. No puedo asociar esa consulta a un objeto visible.'
    elif sum(counts.values())>1:
        return 'Hay varios objetos detectados. ¿A cuál se refiere? Indique el material: '+', '.join(counts)+'.'
    else: material=next(iter(counts))
    return f'{material}: '+rules[material]['advice']+' La cámara no determina limpieza ni contaminación; confirme esas condiciones.'

def evaluate(corpus=DEFAULT_CORPUS,checkpoint='runs/assistant/best.pt',output='runs/assistant/evaluation.json'):
    data=read_json(corpus); validate_corpus(data); assistant=Assistant(checkpoint)
    matrix=[[0]*len(INTENTS) for _ in INTENTS]; rows=[]
    for r in data['test']:
        intent,score=assistant.recognize(r['text']); matrix[INTENTS.index(r['intent'])][INTENTS.index(intent)]+=1
        rows.append(dict(**r,prediction=intent,confidence=score))
    total=len(rows); correct=sum(r['intent']==r['prediction'] for r in rows)
    per_class={}
    for i,c in enumerate(INTENTS):
        support=sum(matrix[i]); predicted=sum(row[i] for row in matrix)
        per_class[c]=dict(support=support,precision=matrix[i][i]/predicted if predicted else None,recall=matrix[i][i]/support if support else None)
    result=dict(accuracy=correct/total,questions=total,intents=INTENTS,confusion_matrix=matrix,classes=per_class,results=rows,
                provenance=data.get('provenance'),checkpoint_sha256=digest(checkpoint),corpus_sha256=digest(corpus),
                warning='Corpus inicial redactado durante implementación. No sustituye preguntas de evaluadores humanos independientes.')
    write_json(output,result); return result
