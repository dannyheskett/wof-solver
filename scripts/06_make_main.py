#!/usr/bin/env python3
"""Phase 6. Build the MAIN (play the game) dataset.

Each puzzle is walked RANDOM_WALKS times in a random letter order, once in
order of letter frequency, and once following the smart pick. Every board a
walk passes through becomes one line, labelled with the best move there,
whatever the walk itself called next:

    SOLVE <solution>   every board word has one candidate left
    CALL <consonant>   otherwise, the smart pick when it is a consonant
    BUY <vowel>        otherwise, the smart pick when it is a vowel

Every move follows the money rule (wof_common.can_buy): until a called
consonant shows on the board, the walks and the labels call consonants only.

A walk ends at its first SOLVE line. Misses write lines too. Repeated lines
within a puzzle are dropped by hash. Each puzzle's random orders are seeded
from SEED and the puzzle, so every run gives the same output.

Line format, tab-separated, no header:
    category  board  called  target  solution
called is the letters called so far, sorted A-Z, or '-' before any call.

    data/train.csv -> data/main_train.tsv
    data/control.csv -> data/main_control.tsv
"""
import hashlib
import random
import sys
import time
from multiprocessing import Pool

import wof_common as wc

RANDOM_WALKS = 3
SEED = "wof-main-v1"
JOBS = [
    ("train.csv", "main_train.tsv", ["train.csv"]),
    ("control.csv", "main_control.tsv", ["train.csv", "control.csv"]),
]
V = ORDER = None


def walk(bw, solution, category, policy, rng):
    called = set()
    order = rng.sample(wc.LETTERS, 26) if policy == "random" else ORDER
    while True:
        b = wc.board(solution, called)
        c = "".join(sorted(called)) or "-"
        if bw.solved():
            yield f"{category}\t{b}\t{c}\tSOLVE {solution}\t{solution}\n"
            return
        vowels = wc.can_buy(b, called)
        best = bw.smart_pick(called, ORDER, vowels)
        yield f"{category}\t{b}\t{c}\t{wc.move(best)}\t{solution}\n"
        nxt = best if policy == "smart" else next(
            x for x in order if x not in called and (vowels or x not in wc.VOWELS))
        called.add(nxt)
        bw.call(nxt)


def puzzle_lines(item):
    solution, category = item
    rng = random.Random(wc.seed_for(SEED, category, solution))
    start = wc.BoardWords(V, solution, category)
    seen, out, repeats = set(), [], 0
    for policy in ["random"] * RANDOM_WALKS + ["frequency", "smart"]:
        for line in walk(start.copy(), solution, category, policy, rng):
            key = hashlib.sha1(line.encode()).digest()[:12]
            if key in seen:
                repeats += 1
                continue
            seen.add(key)
            out.append(line)
    return out, repeats


def run(src, dst, vocab_files, limit=None):
    global V, ORDER
    V = wc.Vocab(vocab_files, ["train.csv"])
    ORDER = wc.letter_order(wc.read_puzzles("train.csv"))
    puzzles = wc.read_puzzles(src)[:limit]
    t0 = time.time()
    written = repeats = solves = 0
    with Pool(11) as pool, open(wc.DATA / dst, "w") as f:
        for n, (lines, rep) in enumerate(pool.imap(puzzle_lines, puzzles, chunksize=32), 1):
            f.writelines(lines)
            written += len(lines)
            repeats += rep
            solves += sum("\tSOLVE " in ln for ln in lines)
            if n % 5000 == 0:
                print(f"  {n:,}/{len(puzzles):,} puzzles, {time.time() - t0:,.0f}s", flush=True)
    size = (wc.DATA / dst).stat().st_size / 1e6
    print(f"{src:12} {len(puzzles):6} puzzles -> {dst}: {written:,} lines "
          f"({solves:,} SOLVE), {repeats:,} repeats dropped, {size:,.0f} MB, "
          f"{time.time() - t0:,.0f}s")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    for src, dst, vocab_files in JOBS:
        run(src, dst, vocab_files, limit)
