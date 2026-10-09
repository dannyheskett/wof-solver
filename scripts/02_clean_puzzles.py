#!/usr/bin/env python3
"""Phase 2. Turn data/puzzles.csv into data/train.csv (what the solvers learn
from) and data/control.csv (held out from everything). Phase 3 then moves
the dev answers out of train.csv.

Rules, applied in this order:
1. Category names are normalised through MERGES.
2. Answers with a character that is not in BOARD_CHARS are dropped.
3. Answers that cannot be laid out on the TV board (rows of 12/14/14/12,
   52 spaces, in use since 1997) are dropped (wof_common.fits_board). The
   source also carries mobile-app puzzles, and the ones longer than the TV
   board go here.
4. Exact repeats of (answer, category) are dropped.
5. Each distinct answer goes to control if the SHA-1 of the answer falls in
   the bottom CONTROL_PCT percent. The split depends only on the answer text,
   so it is the same on every run and an answer filed under two categories
   lands on one side only.

Prints a report of every drop. Columns in both outputs: answer, category.
"""
import csv
import hashlib
from collections import Counter

from wof_common import BOARD_CHARS, DATA, ROOT, fits_board

SRC = DATA / "puzzles.csv"
TRAIN = DATA / "train.csv"
CONTROL = DATA / "control.csv"

CONTROL_PCT = 5
MERGES = {
    "STAR & ROLE": "Star And Role",
    "Character": "Fictional Character",
    "Fictional Place": "Place",
}


def in_control(answer):
    h = int(hashlib.sha1(answer.encode()).hexdigest(), 16)
    return h % 100 < CONTROL_PCT


def main():
    rows = list(csv.DictReader(open(SRC, newline="")))
    dropped = Counter()
    bad_chars = Counter()
    seen = set()
    clean, control = [], []

    for r in rows:
        answer = r["answer"]
        category = MERGES.get(r["category"], r["category"])
        extra = set(answer) - BOARD_CHARS
        if extra:
            dropped["characters"] += 1
            bad_chars.update(extra)
            continue
        if not fits_board(answer):
            dropped["too long for board"] += 1
            continue
        if (answer, category) in seen:
            dropped["exact repeat"] += 1
            continue
        seen.add((answer, category))
        (control if in_control(answer) else clean).append((answer, category))

    for path, data in ((TRAIN, clean), (CONTROL, control)):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["answer", "category"])
            w.writerows(sorted(data, key=lambda x: (x[1], x[0])))

    print(f"read     {len(rows):6}")
    for reason, n in dropped.items():
        print(f"dropped  {n:6}  {reason}")
    print(f"         bad characters: {''.join(sorted(bad_chars))}")
    print(f"train    {len(clean):6}  -> {TRAIN.relative_to(ROOT)}")
    print(f"control  {len(control):6}  -> {CONTROL.relative_to(ROOT)}"
          f"  ({len({a for a, _ in control})} distinct answers)")
    overlap = {a for a, _ in clean} & {a for a, _ in control}
    assert not overlap, f"answers on both sides: {sorted(overlap)[:5]}"


if __name__ == "__main__":
    main()
