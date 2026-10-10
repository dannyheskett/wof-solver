"""Renders the synthetic frames the board decoder trains on.

    python make_frames.py --split train --n 200000
    python make_frames.py --split val --n 2000
    python make_frames.py --split test --n 2000
    python make_frames.py --split train --n 20000 --start 100000   # 20k more, appended

Puzzles come from train.csv, dev.csv and control.csv for the train, val and
test splits, so no synthetic test frame shows a puzzle the decoder trained on.
Frame k of a split is rendered from seed (split number, k), so a split is the
same on every run. Each frame is letterboxed to the decoder's input size and
written as data/vision/synth/<split>/<shard>/<k>.jpg, with one JSON line per
frame in data/vision/synth/<split>/labels.jsonl (render.render's label, with
the corners in input pixels).
"""
import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

import render as R
import wof_common as wc

IN_W, IN_H = 640, 360
OUT = wc.DATA / "vision" / "synth"
SPLITS = {"train": (0, "train.csv"), "val": (1, "dev.csv"), "test": (2, "control.csv")}
SHARD = 5000


def letterbox(img):
    """img scaled to fit IN_W x IN_H and centred on black, with the scale and
    offset that map its pixels to the input's."""
    s = min(IN_W / img.width, IN_H / img.height)
    w, h = round(img.width * s), round(img.height * s)
    out = Image.new("RGB", (IN_W, IN_H))
    ox, oy = (IN_W - w) // 2, (IN_H - h) // 2
    out.paste(img.resize((w, h), Image.Resampling.BILINEAR), (ox, oy))
    return out, s, ox, oy


_puzzles = None


def work(args):
    split, k = args
    global _puzzles
    if _puzzles is None:
        _puzzles = R.puzzles(SPLITS[split][1])
    rng = np.random.default_rng([SPLITS[split][0], k])
    img, lab = R.render(rng, _puzzles[int(rng.integers(len(_puzzles)))])
    img, s, ox, oy = letterbox(img)
    lab["corners"] = [[round(x * s + ox, 1), round(y * s + oy, 1)] for x, y in lab["corners"]]
    rel = f"{k // SHARD:03d}/{k:07d}.jpg"
    path = OUT / split / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=90)
    lab["file"] = rel
    return k, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=SPLITS, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--start", type=int, default=0, help="first frame number; above 0 appends to the labels")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    t = time.time()
    labels = [None] * a.n
    with Pool(a.workers) as pool:
        jobs = [(a.split, a.start + k) for k in range(a.n)]
        for i, (k, lab) in enumerate(pool.imap_unordered(work, jobs, chunksize=64)):
            labels[k - a.start] = lab
            if (i + 1) % 10000 == 0:
                print(f"{i + 1} frames, {time.time() - t:.0f} s", flush=True)
    with open(OUT / a.split / "labels.jsonl", "a" if a.start else "w") as f:
        for lab in labels:
            f.write(json.dumps(lab) + "\n")
    print(f"{a.split}: {a.n} frames in {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
