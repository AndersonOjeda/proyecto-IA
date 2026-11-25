"""Fetch an official wheel in validated byte ranges; verify the index SHA256."""
import concurrent.futures
import hashlib
import html
import re
import threading
import time
from pathlib import Path
from urllib.parse import unquote, urlsplit, urljoin
import requests

base='https://download.pytorch.org/whl/cu130/torch/'
page=requests.get(base,timeout=40);page.raise_for_status()
links=[html.unescape(x) for x in re.findall(r'href="([^"]+)"',page.text)]
link=next(x for x in links if 'torch-2.9.1%2Bcu130-cp312-cp312-win_amd64.whl' in x or 'torch-2.9.1+cu130-cp312-cp312-win_amd64.whl' in x)
url=urljoin(base,link).split('#')[0];expected=link.split('sha256=')[1]
dest=Path('.cache/cuda_wheel');dest.mkdir(parents=True,exist_ok=True)
first=requests.get(url,headers={'Range':'bytes=0-1048575'},timeout=40)
first.raise_for_status();assert first.status_code==206
total=int(first.headers['Content-Range'].split('/')[1]);etag=first.headers.get('ETag');size=1048576
local=threading.local()
def get(start):
    end=min(start+size,total)-1;path=dest/f'{start}.part'
    if path.exists() and path.stat().st_size==end-start+1:return path
    if not hasattr(local,'session'):local.session=requests.Session()
    for attempt in range(15):
        try:
            r=local.session.get(url,headers={'Range':f'bytes={start}-{end}','If-Range':etag},timeout=(10,45))
            assert r.status_code==206 and r.headers['Content-Range']==f'bytes {start}-{end}/{total}'
            assert r.headers.get('ETag')==etag and len(r.content)==end-start+1
            path.write_bytes(r.content);return path
        except Exception:
            if attempt==14:raise
            time.sleep(min(attempt+1,10))
with concurrent.futures.ThreadPoolExecutor(max_workers=40) as pool:
    paths=[]
    for i,path in enumerate(pool.map(get,range(0,total,size))):
        paths.append(path)
        if i%100==0:print(i,total//size,'MiB',flush=True)
wheel=dest/unquote(urlsplit(url).path.split('/')[-1]);h=hashlib.sha256()
with wheel.open('wb') as f:
    for p in paths:
        b=p.read_bytes();h.update(b);f.write(b)
assert h.hexdigest()==expected,'Official SHA256 mismatch'
print('VERIFIED',wheel,h.hexdigest(),flush=True)
