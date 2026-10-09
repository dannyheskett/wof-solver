#!/usr/bin/env python3
"""Phase 10. Pick transformer-solve's SOLVE confidence cutoff, on dev puzzles.

Plays MAIN on SAMPLE puzzles drawn from data/dev.csv (03_split_dev.py), which
no training stream contains, and never control.csv. It plays them with
every early SOLVE turned down, so each game runs until the board is full.
Each turned-down SOLVE is logged with its answer, the transformer's probability
for it, and whether it was right.

A game with cutoff p plays exactly like this one up to the first SOLVE
with probability at least p, because a turned-down SOLVE always leads to
the same letter move. So that first SOLVE decides the game: right is a
solve after that many letters, wrong is a loss. With none, the game ends
with the board full. One run therefore scores every cutoff.

    python3 scripts/10_tune_solve.py runs/ft2b/ckpt.pt [SAMPLE]

Writes data/eval/tune.tsv, one row per logged SOLVE, and prints
solve rate and letters for a range of cutoffs.
"""
import csv
import random
import statistics
import sys
import time
from multiprocessing import Pool

import wof_common as wc
from solvers import TransformerSolver, check_move, one_thread

SAMPLE = 300   # all of dev.csv
SEED = "wof-tune-v1"
CUTOFFS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
S = None


def play(job):
    n, solution, category = job
    S.new_puzzle(None)
    called, bad = [], []
    while True:
        board = wc.board(solution, set(called))
        mv = S.main(category, board, "".join(sorted(called)))
        if mv.startswith("SOLVE "):
            final = (mv[6:] == solution, len(called))
            break
        letter = check_move(mv, board, called)
        if letter is None:   # invalid move, as 11_evaluate.py scores it: a loss
            final = (False, len(called))
            bad.append(mv)
            break
        called.append(letter)
    tried = [(p, ans == solution, k, ans) for ans, p, k in S.attempts]
    return n, solution, category, tried, final, bad


def score(games, cut):
    solved, letters = 0, []
    for _, _, _, tried, final, _ in games:
        hit = next(((ok, k) for p, ok, k, _ in tried if p >= cut), final)
        if hit[0]:
            solved += 1
            letters.append(hit[1])
    return solved / len(games), letters


def main():
    global S
    ckpt = sys.argv[1]
    sample = int(sys.argv[2]) if len(sys.argv) > 2 else SAMPLE
    puzzles = wc.read_puzzles("dev.csv")
    random.Random(SEED).shuffle(puzzles)
    puzzles = puzzles[:sample]
    S = TransformerSolver(ckpt, min_p=2.0)   # above any probability: turn every early SOLVE down
    games = []
    t0 = time.time()
    with Pool(11, initializer=one_thread) as pool:
        for g in pool.imap_unordered(play, [(n, s, c) for n, (s, c) in enumerate(puzzles)]):
            games.append(g)
            if len(games) % 25 == 0:
                print(f"  {len(games)}/{len(puzzles)} puzzles, {time.time() - t0:,.0f}s", flush=True)
    games.sort(key=lambda g: g[0])   # games finish in any order, write them in puzzle order
    with open(wc.DATA / "eval" / "tune.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["category", "solution", "letters_called", "p", "right", "answer"])
        for _, sol, cat, tried, _, _ in games:
            for p, ok, k, ans in tried:
                w.writerow([cat, sol, k, f"{p:.4f}", int(ok), ans])
    n_tried = sum(len(g[3]) for g in games)
    bad = [mv for g in games for mv in g[5]]
    print(f"{len(games)} dev puzzles, {n_tried} early SOLVEs logged, "
          f"{len(bad)} invalid moves {sorted(set(bad))[:10]}")
    print("cutoff  solved  letters_median  letters_mean")
    for cut in CUTOFFS:
        rate, letters = score(games, cut)
        print(f"{cut:6.2f}  {rate:6.1%}  {statistics.median(letters):14}  {statistics.mean(letters):12.2f}")


if __name__ == "__main__":
    main()
