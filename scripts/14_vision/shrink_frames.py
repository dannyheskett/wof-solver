"""Letterboxes every frame in data/vision/raw/<clip>/frames/ to the decoder's
640 x 360 input, into data/vision/raw_lb/<clip>/frames/, so the set can go to
a GPU pod for labelling at a fifth of the size. The decoder letterboxes to
this size anyway, so its reads are the same on either copy.

    python shrink_frames.py [--workers 6]
"""
import argparse
from multiprocessing import Pool

from PIL import Image

import board  # noqa: F401  (puts scripts/ on the path)
import wof_common as wc
from make_frames import letterbox

RAW = wc.DATA / "vision" / "raw"
OUT = wc.DATA / "vision" / "raw_lb"


def one(p):
    dst = OUT / p.relative_to(RAW)
    if dst.exists():
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    letterbox(Image.open(p).convert("RGB"))[0].save(dst, quality=90)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    frames = sorted(p for d in RAW.glob("*/done") for p in (d.parent / "frames").glob("*.jpg"))
    with Pool(a.workers) as pool:
        n = sum(pool.imap_unordered(one, frames, chunksize=256))
    for d in RAW.glob("*/done"):
        (OUT / d.parent.name).mkdir(parents=True, exist_ok=True)
        (OUT / d.parent.name / "done").touch()
    print(f"{len(frames)} frames, {n} written")


if __name__ == "__main__":
    main()
