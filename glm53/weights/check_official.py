#!/usr/bin/env python3
"""check_official.py DIR [--revision REV] [--jobs 8]: every file of a download of zai-org/GLM-5.3-BF16 against what
Hugging Face lists for that revision of Z.AI's repository: the size and the sha256 of each weight file (Git LFS), the
Git blob hash of each small file (config, tokenizer, chat template, license). Nothing of Fiore-lab's is involved: the
list comes from huggingface.co at the time you run it. Exits 1 on any missing, extra or different file.

Run it before converting (convert.sh does nothing to check its input). Python 3.9+, no other modules; reading 1.5 TB
takes a while (the files are read once, JOBS at a time).
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = "zai-org/GLM-5.3-BF16"
REV = "9d2398f478cab2de883137db3a36ad2c96205e24"   # the revision this recipe's weights were made from
OPTIONAL = {"README.md", ".gitattributes"}         # not read by the conversion: a missing one is noted, not a problem


def listing(rev):
    url = f"https://huggingface.co/api/models/{REPO}/tree/{rev}?recursive=true"
    out = []
    while url:
        with urllib.request.urlopen(url, timeout=60) as r:
            out += json.load(r)
            link = r.headers.get("Link") or ""
        url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part[part.index("<") + 1:part.index(">")]
    return [x for x in out if x.get("type") == "file"]


def digest(path, git_blob):
    h = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        h.update(b"blob %d\0" % os.path.getsize(path))
    with open(path, "rb") as f:
        while True:
            b = f.read(64 << 20)
            if not b:
                break
            h.update(b)
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--revision", default=REV)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    files = listing(a.revision)
    print(f"{REPO} at {a.revision}: {len(files)} files, {sum(x['size'] for x in files) / 1e12:.2f} TB")
    bad = []
    present = []
    for x in files:
        p = os.path.join(a.dir, x["path"])
        if not os.path.isfile(p):
            if x["path"] in OPTIONAL:
                print(f"note: {x['path']} not downloaded (not needed)")
            else:
                bad.append(f"missing {x['path']}")
        elif os.path.getsize(p) != x["size"]:
            bad.append(f"size {x['path']}: {os.path.getsize(p)}, listed {x['size']}")
        else:
            present.append(x)
    listed = {x["path"] for x in files}
    for root, dirs, fs in os.walk(a.dir):
        dirs[:] = [d for d in dirs if d not in (".cache", ".git")]
        for f in fs:
            rel = os.path.relpath(os.path.join(root, f), a.dir)
            if rel not in listed:
                bad.append(f"not in the repository: {rel}")
    t0, done = time.time(), [0]

    def one(x):
        lfs = x.get("lfs")
        want = lfs["oid"] if lfs else x["oid"]
        got = digest(os.path.join(a.dir, x["path"]), git_blob=not lfs)
        done[0] += x["size"]
        return x["path"], got == want

    with ThreadPoolExecutor(a.jobs) as ex:
        for i, (path, ok) in enumerate(ex.map(one, present)):
            if not ok:
                bad.append(f"content {path}: its hash is not the listed one")
            if i % 20 == 19:
                print(f"  {i + 1}/{len(present)} files, {done[0] / 1e9:.0f} GB in {time.time() - t0:.0f} s", flush=True)
    for b in bad:
        print("NG", b)
    print(f"{'OK' if not bad else 'NG'}: {len(present)} files checked, {len(bad)} problems")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
