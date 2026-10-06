import io, json, struct, urllib.request, zipfile, zlib, hashlib, os
from concurrent.futures import ThreadPoolExecutor
REV='6691f3288bb1ff207d6ead4d841b505de08a6fd8'
# M1: fetch the frozen held-out candidate images, hash them, and drop any item whose image is byte-identical to a
# SEARCH/RERANK/TEST image (leakage rule in results/paper-analysis/m1/plan_lock.md). Writes holdout.jsonl.
H=json.load(open('results/paper-analysis/m1/holdout_candidates.json',encoding='utf-8'))['items']
used=set()
for s in ('search','validation','test'):
    for l in open(f'examples/omnispatial-perspective-taking/{s}.jsonl',encoding='utf-8'):
        used.add(json.loads(l)['image_sha256'])
need={r['image_member'] for r in H}
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
u,n=url('OmniSpatial-train.zip'); z=zipfile.ZipFile(io.BufferedReader(F(u,n),1<<20))
infos=[i for i in z.infolist() if i.filename in need]
shas={}
def get(i):
    h=rng(u,i.header_offset,i.header_offset+29); fl,el=struct.unpack('<HH',h[26:30]); st=i.header_offset+30+fl+el
    raw=rng(u,st,st+i.compress_size-1); data=zlib.decompress(raw,-15) if i.compress_type==8 else raw
    assert zlib.crc32(data)==i.CRC, i.filename
    p=f"/workspace/m1-images/train/{i.filename.rsplit('/',1)[1]}"; os.makedirs(os.path.dirname(p),exist_ok=True); open(p,'wb').write(data)
    shas[i.filename]=hashlib.sha256(data).hexdigest(); return 1
with ThreadPoolExecutor(32) as ex: got=sum(ex.map(get,infos))
assert got==len(need), (got,len(need))
keep=[{**r,'image_sha256':shas[r['image_member']]} for r in H if shas[r['image_member']] not in used]
os.makedirs('results/paper-analysis/m1/gpu',exist_ok=True)
open('results/paper-analysis/m1/gpu/holdout.jsonl','w',encoding='utf-8').write(chr(10).join(json.dumps(r,sort_keys=True,ensure_ascii=False) for r in keep)+chr(10))
print('M1_IMAGES', got, 'items_kept', len(keep), 'dropped_leak', len(H)-len(keep))
