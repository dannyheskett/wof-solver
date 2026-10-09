#!/usr/bin/env python3
"""Check smart-solve in C (server/src/smart.c) against scripts/solvers.py:
play control puzzles with the Python solver and ask the C one for a move
at every state on the way. Every move must be the same.

    make -C server build/smart_dump
    python3 server/tests/smart_parity.py [N]      N puzzles per round, default 300
"""
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import wof_common as wc  # noqa: E402
from solvers import FREE, SmartSolver  # noqa: E402


def states(solver, puzzles):
    """(mode, category, board, called, python move) for every move of every game."""
    out = []
    for answer, cat in puzzles:
        called = set()
        for _ in range(30):                       # MAIN
            board = wc.board(answer, called)
            mv = solver.main(cat, board, "".join(sorted(called)))
            out.append(("main", cat, board, "".join(sorted(called)), mv))
            if mv.startswith("SOLVE"):
                break
            called.add(mv.split()[1])
        called = set(FREE)                        # BONUS
        board = wc.board(answer, called)
        mv = solver.bonus_pick(cat, board, "".join(sorted(called)))
        out.append(("bonus_pick", cat, board, "".join(sorted(called)), mv))
        called |= set(mv.split()[1] + mv.split()[2])
        board = wc.board(answer, called)
        mv = solver.bonus_solve(cat, board, "".join(sorted(called)))
        out.append(("bonus_solve", cat, board, "".join(sorted(called)), mv))
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    puzzles = random.Random(11).sample(wc.read_puzzles("control.csv"), n)
    solver = SmartSolver()
    st = states(solver, puzzles)
    r = subprocess.run([str(ROOT / "server/build/smart_dump"), str(wc.WORDLIST), ",".join(str(wc.DATA / n) for n in wc.SMART_FILES)],
                       input="".join(f"{m}\t{c}\t{b}\t{k}\n" for m, c, b, k, _ in st),
                       capture_output=True, text=True, check=True)
    got = r.stdout.split("\n")
    bad = [(s, g) for s, g in zip(st, got) if s[4] != g]
    for s, g in bad[:10]:
        print(f"differs: {s[0]} {s[1]!r} {s[2]!r} called {s[3]!r}\n  python {s[4]!r}\n  C      {g!r}")
    print(f"{n} puzzles, {len(st)} moves: {len(st) - len(bad)} identical, {len(bad)} differ")
    print("C:", r.stderr.strip().replace("\n", "; "))


if __name__ == "__main__":
    main()
