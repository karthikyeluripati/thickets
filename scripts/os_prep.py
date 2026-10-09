"""OS prep (results/paper-analysis/os-prompt-control/plan_lock.md): fetch the OmniSpatial-train images needed by the M1
hold-out (600 items) and the original SEARCH selection set (200 items) from the pinned HF revision (range requests into
the zip, as scripts/m1_prepare_images.py), and verify every image's sha256 against the committed item records."""
import hashlib
import io
import json
import os
import struct
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor

REV = '6691f3288bb1ff207d6ead4d841b505de08a6fd8'
OUT = '/workspace/os-images/train'
H = [json.loads(l) for l in open('results/paper-analysis/m1/pod/holdout.jsonl', encoding='utf-8') if l.strip()]
S = [json.loads(l) for l in open('examples/omnispatial-perspective-taking/search.jsonl', encoding='utf-8') if l.strip()]
want = {r['image_member']: r['image_sha256'] for r in H + S}


def url(n):
    r = urllib.request.urlopen(urllib.request.Request(f'https://huggingface.co/datasets/qizekun/OmniSpatial/resolve/{REV}/{n}', method='HEAD'))
    return r.url, int(r.headers['Content-Length'])


def rng(u, a, b):
    for _ in range(6):
        try:
            return urllib.request.urlopen(urllib.request.Request(u, headers={'Range': f'bytes={a}-{b}'}), timeout=120).read()
        except Exception as e:  # noqa: BLE001
            err = e
    raise err


class F(io.RawIOBase):
    def __init__(s, u, n): s.u, s.n, s.p = u, n, 0
    def seekable(s): return True
    def readable(s): return True
    def tell(s): return s.p
    def seek(s, o, w=0): s.p = {0: o, 1: s.p + o, 2: s.n + o}[w]; return s.p

    def readinto(s, b):
        if s.p >= s.n: return 0
        d = rng(s.u, s.p, min(s.n, s.p + len(b)) - 1); b[:len(d)] = d; s.p += len(d); return len(d)


def main():
    u, n = url('OmniSpatial-train.zip'); z = zipfile.ZipFile(io.BufferedReader(F(u, n), 1 << 20))
    infos = [i for i in z.infolist() if i.filename in want]
    os.makedirs(OUT, exist_ok=True); bad = []

    def get(i):
        h = rng(u, i.header_offset, i.header_offset + 29); fl, el = struct.unpack('<HH', h[26:30]); st = i.header_offset + 30 + fl + el
        raw = rng(u, st, st + i.compress_size - 1); data = zlib.decompress(raw, -15) if i.compress_type == 8 else raw
        assert zlib.crc32(data) == i.CRC, i.filename
        if hashlib.sha256(data).hexdigest() != want[i.filename]: bad.append(i.filename)
        open(f"{OUT}/{i.filename.rsplit('/', 1)[1]}", 'wb').write(data); return 1
    with ThreadPoolExecutor(32) as ex:
        got = sum(ex.map(get, infos))
    assert got == len(want), (got, len(want)); assert not bad, bad[:5]
    print('OS_IMAGES_OK', got, 'holdout', len(H), 'search', len(S))


if __name__ == '__main__':
    main()
