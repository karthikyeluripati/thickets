import io, json, struct, urllib.request, zipfile, zlib, hashlib, os
from concurrent.futures import ThreadPoolExecutor
REV='6691f3288bb1ff207d6ead4d841b505de08a6fd8'
need={}
probe=set(json.load(open('results/paper-analysis/r1/frozen_inputs.json'))['probe_uids'])
for s in ('search','validation'):
    for l in open(f'examples/omnispatial-complex-logic/{s}.jsonl',encoding='utf-8'):
        r=json.loads(l)
        if r['uid'] in probe: need[r['image_member']]=r['image_sha256']
def url(n):
    r=urllib.request.urlopen(urllib.request.Request(f'https://huggingface.co/datasets/qizekun/OmniSpatial/resolve/{REV}/{n}',method='HEAD')); return r.url,int(r.headers['Content-Length'])
def rng(u,a,b):
    for _ in range(6):
        try: return urllib.request.urlopen(urllib.request.Request(u,headers={'Range':f'bytes={a}-{b}'}),timeout=120).read()
        except Exception as e: err=e
    raise err
class F(io.RawIOBase):
    def __init__(s,u,n): s.u,s.n,s.p=u,n,0
    def seekable(s): return True
    def readable(s): return True
    def tell(s): return s.p
    def seek(s,o,w=0): s.p={0:o,1:s.p+o,2:s.n+o}[w]; return s.p
    def readinto(s,b):
        if s.p>=s.n: return 0
        d=rng(s.u,s.p,min(s.n,s.p+len(b))-1); b[:len(d)]=d; s.p+=len(d); return len(d)
done=0
for split in ('train','test'):
    u,n=url(f'OmniSpatial-{split}.zip'); z=zipfile.ZipFile(io.BufferedReader(F(u,n),1<<20))
    infos=[i for i in z.infolist() if i.filename in need]
    def get(i):
        h=rng(u,i.header_offset,i.header_offset+29); fl,el=struct.unpack('<HH',h[26:30]); st=i.header_offset+30+fl+el
        raw=rng(u,st,st+i.compress_size-1); data=zlib.decompress(raw,-15) if i.compress_type==8 else raw
        assert hashlib.sha256(data).hexdigest()==need[i.filename], i.filename
        p=f"/workspace/cl-images/{split}/{i.filename.rsplit('/',1)[1]}"; os.makedirs(os.path.dirname(p),exist_ok=True); open(p,'wb').write(data); return 1
    with ThreadPoolExecutor(32) as ex: done+=sum(ex.map(get,infos))
assert done==len(need); print('IMAGES_VERIFIED',done)
