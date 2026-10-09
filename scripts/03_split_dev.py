#!/usr/bin/env python3
"""Phase 3. Hold DEV_SIZE answers out of the training puzzles for tuning, so
a setting like the SOLVE cutoff is picked on puzzles the transformer never
trained on.

The DEV_SIZE answers with the lowest SHA-1 of "dev|<answer>" go to dev, with
every category they are filed under. The split depends only on the answer
text, so it is the same on every run, and rerunning it (on train.csv plus
dev.csv) gives the same two files.

    data/train.csv -> data/train.csv  (the training sets are built from this)
                      data/dev.csv    (10_tune_solve.py reads this)

control.csv is untouched. smart-solve uses train.csv and dev.csv together as
its word list, since it is never tuned.
"""
import csv
import hashlib

import wof_common as wc

DEV_SIZE = 300


def key(answer):
    return hashlib.sha1(f"dev|{answer}".encode()).hexdigest()


def main():
    pool = wc.read_puzzles("train.csv")
    if (wc.DATA / "dev.csv").exists():
        pool += wc.read_puzzles("dev.csv")
    pool = sorted(set(pool), key=lambda p: (p[1], p[0]))   # dev rows may be in both
    dev_answers = set(sorted({a for a, _ in pool}, key=key)[:DEV_SIZE])
    sides = {"train.csv": [], "dev.csv": []}
    for answer, category in pool:
        sides["dev.csv" if answer in dev_answers else "train.csv"].append((answer, category))
    for name, rows in sides.items():
        with open(wc.DATA / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["answer", "category"])
            w.writerows(rows)
        print(f"{name:10} {len(rows):6} puzzles")


if __name__ == "__main__":
    main()
