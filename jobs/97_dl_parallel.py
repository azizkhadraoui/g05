"""Parallel, resumable, multi-connection downloads for the world-model checkpoints (replaces the sequential wget loop of
89_dl_world_models.sh, which ran at ~4 MB/s per file). Each file is split into byte ranges fetched concurrently; a file
already complete is skipped, and an existing *.part from wget is kept as the first range.
Usage: python 97_dl_parallel.py URL=DEST [URL=DEST ...]   (env SEGMENTS, default 8)
"""
import os
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

SEG = int(os.environ.get("SEGMENTS", "8"))


def size_of(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def fetch_range(url, start, end, path):
    """Download bytes [start, end] into path (resuming whatever path already holds)."""
    have = os.path.getsize(path) if os.path.exists(path) else 0
    if start + have > end:
        return
    req = urllib.request.Request(url, headers={"Range": f"bytes={start + have}-{end}"})
    for _ in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(path, "ab") as f:
                while True:
                    b = r.read(1 << 22)
                    if not b:
                        break
                    f.write(b)
            return
        except Exception as e:  # noqa: BLE001 - retry from where the range stopped
            print("retry", path, type(e).__name__, flush=True)
            have = os.path.getsize(path)
            req = urllib.request.Request(url, headers={"Range": f"bytes={start + have}-{end}"})


def download(url, dest):
    total = size_of(url)
    if os.path.exists(dest) and os.path.getsize(dest) == total:
        print("have", dest, flush=True)
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    first = 0
    if os.path.exists(dest + ".part"):  # the sequential wget's partial file becomes the first segment
        first = os.path.getsize(dest + ".part")
        os.replace(dest + ".part", dest + ".seg0")
    rest = total - first
    bounds = [(0, first - 1)] if first else []
    step = -(-rest // SEG)
    bounds += [(first + i * step, min(first + (i + 1) * step, total) - 1) for i in range(SEG) if first + i * step < total]
    with ThreadPoolExecutor(len(bounds)) as ex:
        list(ex.map(lambda ib: fetch_range(url, ib[1][0], ib[1][1], f"{dest}.seg{ib[0]}"), enumerate(bounds)))
    with open(dest + ".tmp", "wb") as out:
        for i in range(len(bounds)):
            with open(f"{dest}.seg{i}", "rb") as f:
                while True:
                    b = f.read(1 << 24)
                    if not b:
                        break
                    out.write(b)
    assert os.path.getsize(dest + ".tmp") == total, (dest, os.path.getsize(dest + ".tmp"), total)
    os.replace(dest + ".tmp", dest)
    for i in range(len(bounds)):
        os.remove(f"{dest}.seg{i}")
    print("done", dest, round(total / 1e9, 1), "GB", flush=True)


if __name__ == "__main__":
    pairs = [a.split("=", 1) for a in sys.argv[1:]]
    threads = [threading.Thread(target=download, args=p) for p in pairs]
    [t.start() for t in threads]
    [t.join() for t in threads]
