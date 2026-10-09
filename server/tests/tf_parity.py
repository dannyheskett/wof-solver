#!/usr/bin/env python3
"""Check transformer-solve in C (server/src/transformer.c) against the model
solver in scripts/solvers.py: play control puzzles in PyTorch and ask the C
one for a move at every state on the way. Every move must be the same.

    make -C server build/tf_dump
    python3 server/tests/tf_parity.py runs/ft2b/ckpt.pt runs/ft2b/wof.bin [N]
"""
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import wof_common as wc  # noqa: E402
from solvers import FREE, TransformerSolver, check_move, one_thread  # noqa: E402

CKPT = sys.argv[1]
S = None


def init():
    global S
    one_thread()
    S = TransformerSolver(CKPT, 0.99)


def job(item):
    """(mode, category, board, called, move) for every move of one MAIN and one BONUS game."""
    answer, cat = item
    out, called = [], []
    S.new_puzzle(None)
    while len(called) < 40:
        board, c = wc.board(answer, set(called)), "".join(sorted(called))
        mv = S.main(cat, board, c)
        out.append(("main", cat, board, c, mv))
        letter = None if mv.startswith("SOLVE ") else check_move(mv, board, called)
        if letter is None:
            break
        called.append(letter)
    board, c = wc.board(answer, FREE), "".join(sorted(FREE))
    pick = S.bonus_pick(cat, board, c)
    out.append(("bonus_pick", cat, board, c, pick))
    parts = pick.split()
    if len(parts) == 3:
        called = FREE | set(parts[1] + parts[2])
        board, c = wc.board(answer, called), "".join(sorted(called))
        out.append(("bonus_solve", cat, board, c, S.bonus_solve(cat, board, c)))
    return out


def main():
    weights = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    puzzles = wc.read_puzzles("control.csv")[:n]
    with Pool(8, initializer=init) as pool:
        states = [s for game in pool.imap(job, puzzles) for s in game]
    r = subprocess.run([str(ROOT / "server/build/tf_dump"), weights],
                       input="".join(f"{m}\t{c}\t{b}\t{k}\n" for m, c, b, k, _ in states),
                       capture_output=True, text=True, check=True)
    got = [line.split("\t")[0] for line in r.stdout.split("\n")]
    games = {}
    for (mode, cat, board, called, want), g in zip(states, got):
        games.setdefault(mode, [0, 0])
        games[mode][0] += 1
        if want != g:
            games[mode][1] += 1
            print(f"differs: {mode} {cat!r} {board!r} called {called!r}\n  torch {want!r}\n  C     {g!r}")
    for mode, (total, bad) in games.items():
        print(f"{mode}: {total - bad} of {total} moves the same")
    print("C:", r.stderr.strip())


if __name__ == "__main__":
    main()
