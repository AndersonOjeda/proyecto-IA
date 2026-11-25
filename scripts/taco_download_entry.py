"""Límites de red para una invocación del descargador original, sin editarlo."""
import runpy
import sys
import requests
import concurrent.futures
import re

original,manifest=sys.argv[1:]
get=requests.get
def bounded_get(url,**kwargs):
    kwargs.setdefault('timeout',(10,40))
    size=1024*1024
    first=get(url,headers={'Range':f'bytes=0-{size-1}'},**kwargs)
    if first.status_code!=206:return first
    match=re.fullmatch(r'bytes 0-(\d+)/(\d+)',first.headers.get('Content-Range',''))
    if not match:raise ValueError('Content-Range inicial inesperado')
    end,total=map(int,match.groups());etag=first.headers.get('ETag')
    if len(first.content)!=end+1:raise ValueError('Primer rango truncado')
    def segment(start):
        stop=min(total-1,start+size-1);last=None
        for attempt in range(3):
            try:
                headers={'Range':f'bytes={start}-{stop}'}
                if etag:headers['If-Range']=etag
                r=get(url,headers=headers,**kwargs)
                if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{stop}/{total}':raise ValueError('Rango HTTP inconsistente')
                if etag and r.headers.get('ETag')!=etag:raise ValueError('ETag cambió durante descarga')
                if len(r.content)!=stop-start+1:raise ValueError('Rango truncado')
                return r.content
            except Exception as exc:last=exc
        raise last
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        content=first.content+b''.join(pool.map(segment,range(end+1,total,size)))
    if len(content)!=total:raise ValueError('Longitud final incorrecta')
    response=requests.Response();response.status_code=200;response._content=content
    response.headers=first.headers.copy();response.headers.pop('Content-Range',None);response.headers['Content-Length']=str(total)
    return response
requests.get=bounded_get
sys.argv=[original,'--dataset_path',manifest]
runpy.run_path(original,run_name='__main__')
