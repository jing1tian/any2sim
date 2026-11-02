"""Parallel ranged download of one large file (the HF mirror gives ~2 MB/s per connection).
usage: python environments/fetch.py <url> <dest> [connections]"""
import concurrent.futures as cf
import os
import sys
import urllib.request


def size_of(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"]), r.geturl()


def part(url, dest, a, b, k):
    tmp = f"{dest}.part{k}"
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    if have >= b - a + 1:
        return tmp
    for attempt in range(20):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={a + have}-{b}"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "ab") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            have = os.path.getsize(tmp)
            if have >= b - a + 1:
                return tmp
        except Exception as e:  # retry from where it stopped
            print("retry", k, attempt, e, flush=True)
            have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    raise RuntimeError(f"part {k} failed")


def main():
    url, dest = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    if os.path.exists(dest):
        print("exists", dest)
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    total, final = size_of(url)
    step = -(-total // n)
    ranges = [(i * step, min(total, (i + 1) * step) - 1) for i in range(n)]
    with cf.ThreadPoolExecutor(n) as ex:
        parts = list(ex.map(lambda x: part(final, dest, x[1][0], x[1][1], x[0]), enumerate(ranges)))
    with open(dest + ".tmp", "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                while chunk := f.read(1 << 24):
                    out.write(chunk)
    assert os.path.getsize(dest + ".tmp") == total
    os.replace(dest + ".tmp", dest)
    for p in parts:
        os.remove(p)
    print("done", dest, total)


if __name__ == "__main__":
    main()
