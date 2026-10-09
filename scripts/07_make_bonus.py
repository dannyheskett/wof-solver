#!/usr/bin/env python3
"""Phase 7. Build the BONUS (one shot) dataset.

Bonus round rules: R, S, T, L, N and E are revealed for free, the player
names 3 more consonants and 1 more vowel together, those are revealed, and
the player gets one SOLVE. Y counts as a consonant.

Per puzzle:
    one PICK line    the RSTLNE board, target the best set of 3 consonants
                     and 1 vowel (see wof_common.BoardWords.best_set)
    SOLVE lines      the board after each of up to PICK_SETS different sets
                     is revealed: the best set, the frequency set (the most
                     common letters left), and random sets. Target is always
                     SOLVE, since the round allows exactly one.

Repeated lines within a puzzle are dropped. Random sets are seeded from SEED
and the puzzle, so every run gives the same output.

Line format, tab-separated, no header:
    category  board  called  target  solution
PICK targets read "PICK BCD A": the consonants A-Z, then the vowel.

    data/train.csv -> data/bonus_train.tsv
    data/control.csv -> data/bonus_control.tsv
"""
import random
import sys
import time
from multiprocessing import Pool

import wof_common as wc

FREE = set("RSTLNE")
PICK_SETS = 10
SEED = "wof-bonus-v1"
JOBS = [
    ("train.csv", "bonus_train.tsv", ["train.csv"]),
    ("control.csv", "bonus_control.tsv", ["train.csv", "control.csv"]),
]
V = ORDER = None
CONS = [c for c in wc.CONSONANTS if c not in FREE]
VOWS = [c for c in "AEIOU" if c not in FREE]


def pick_text(letters):
    cons = "".join(sorted(c for c in letters if c not in wc.VOWELS))
    vow = "".join(c for c in letters if c in wc.VOWELS)
    return f"PICK {cons} {vow}"


def puzzle_lines(item):
    solution, category = item
    rng = random.Random(wc.seed_for(SEED, category, solution))
    bw = wc.BoardWords(V, solution, category)
    for c in sorted(FREE):
        bw.call(c)
    free = "".join(sorted(FREE))
    smart, _ = bw.best_set([c for c in ORDER if c in CONS],
                           [c for c in ORDER if c in VOWS], 3, 1, ORDER)
    lines = [f"{category}\t{wc.board(solution, FREE)}\t{free}\t{pick_text(smart)}\t{solution}\n"]
    sets = [frozenset(smart),
            frozenset([c for c in ORDER if c in CONS][:3] + [c for c in ORDER if c in VOWS][:1])]
    tries = 0
    while len(set(sets)) < PICK_SETS and tries < 100:
        sets.append(frozenset(rng.sample(CONS, 3) + rng.sample(VOWS, 1)))
        tries += 1
    seen = set()
    for s in sets:
        called = FREE | s
        line = (f"{category}\t{wc.board(solution, called)}\t{''.join(sorted(called))}"
                f"\tSOLVE {solution}\t{solution}\n")
        if line not in seen:
            seen.add(line)
            lines.append(line)
    return lines, len(sets) - len(seen)


def run(src, dst, vocab_files, limit=None):
    global V, ORDER
    V = wc.Vocab(vocab_files, ["train.csv"])
    ORDER = wc.letter_order(wc.read_puzzles("train.csv"))
    puzzles = wc.read_puzzles(src)[:limit]
    t0 = time.time()
    written = repeats = 0
    with Pool(11) as pool, open(wc.DATA / dst, "w") as f:
        for n, (lines, rep) in enumerate(pool.imap(puzzle_lines, puzzles, chunksize=32), 1):
            f.writelines(lines)
            written += len(lines)
            repeats += rep
            if n % 5000 == 0:
                print(f"  {n:,}/{len(puzzles):,} puzzles, {time.time() - t0:,.0f}s", flush=True)
    size = (wc.DATA / dst).stat().st_size / 1e6
    print(f"{src:12} {len(puzzles):6} puzzles -> {dst}: {written:,} lines "
          f"({len(puzzles):,} PICK), {repeats:,} repeats dropped, {size:,.0f} MB, "
          f"{time.time() - t0:,.0f}s")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    for src, dst, vocab_files in JOBS:
        run(src, dst, vocab_files, limit)
