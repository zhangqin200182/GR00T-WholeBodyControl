#!/usr/bin/env python3
"""Fetch real content for LFS pointer files directly via the GitHub LFS batch API.

Usage: cat file_list.txt | python3 lfs_fetch.py <repo_root>
Each line of stdin is a repo-relative path to an LFS pointer file.
Files are downloaded one by one (retryable), replacing the pointer in place.
"""
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

REPO_LFS = "https://github.com/zhangqin200182/GR00T-WholeBodyControl.git/info/lfs"
PROXY = "http://127.0.0.1:7890"
opener = urllib.request.build_opener(
    urllib.request.ProxyHandler({"https": PROXY, "http": PROXY})
)


def read_pointer(path):
    oid = size = None
    for line in pathlib.Path(path).read_text().splitlines():
        if line.startswith("oid sha256:"):
            oid = line.split(":", 1)[1].strip()
        elif line.startswith("size "):
            size = int(line.split()[1])
    return oid, size


def batch(objects):
    body = json.dumps({"operation": "download", "objects": objects}).encode()
    req = urllib.request.Request(
        REPO_LFS + "/objects/batch",
        data=body,
        method="POST",
        headers={
            "Accept": "application/vnd.git-lfs+json",
            "Content-Type": "application/vnd.git-lfs+json",
        },
    )
    try:
        with opener.open(req, timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        print(f"BATCH HTTP {e.code}: {e.read().decode(errors='replace')[:500]}")
        raise


def main(root, files):
    targets = []
    for rel in files:
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            print(f"SKIP missing {rel}")
            continue
        oid, size = read_pointer(p)
        if oid is None:
            print(f"SKIP not-pointer {rel}")
            continue
        targets.append((rel, p, oid, size))
    print(f"{len(targets)} pointer files to fetch")
    if not targets:
        return

    hrefs = {}
    for i in range(0, len(targets), 50):
        chunk = targets[i : i + 50]
        resp = batch([{"oid": o, "size": s} for _, _, o, s in chunk])
        for obj in resp.get("objects", []):
            h = obj.get("actions", {}).get("download", {}).get("href")
            if h:
                hrefs[obj["oid"]] = h
            else:
                print(f"NO-HREF {obj['oid'][:12]} error={obj.get('error')}")
    print(f"got {len(hrefs)} download URLs")

    ok = fail = 0
    for rel, p, oid, size in targets:
        h = hrefs.get(oid)
        if not h:
            fail += 1
            continue
        for attempt in range(5):
            try:
                with opener.open(urllib.request.Request(h), timeout=180) as r:
                    data = r.read()
                if len(data) != size:
                    raise IOError(f"size mismatch {len(data)} != {size}")
                tmp = p + ".tmp"
                with open(tmp, "wb") as f:
                    f.write(data)
                os.replace(tmp, p)
                ok += 1
                print(f"OK  {rel} ({size / 1048576:.2f}MB)")
                break
            except Exception as e:
                print(f"retry {attempt + 1} {rel}: {e}")
                time.sleep(2)
        else:
            fail += 1
            print(f"FAIL {rel}")
    print(f"DONE ok={ok} fail={fail}")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    root = sys.argv[1]
    main(root, [l.strip() for l in sys.stdin if l.strip()])
