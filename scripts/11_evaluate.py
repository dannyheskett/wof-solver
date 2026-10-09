#!/usr/bin/env python3
"""Phase 11. Play a solver against the control set and score it.

The harness holds the answer. The solver only gets the category, the board
and the called letters (see solvers.py), and every move is checked against
the rules (solvers.check_move).

MAIN, one game per puzzle
    The solver calls consonants and buys vowels one at a time until it says
    SOLVE. The game ends at the first SOLVE, right or wrong. A letter
    called twice, a consonant bought, a vowel called, or a vowel bought
    before a called consonant shows on the board (money rule, see
    wof_common.can_buy) is "invalid" and also ends the game. Scored per puzzle: solved or not, letters called before
    the SOLVE, how many were vowels bought, and misses (letters not in the
    puzzle; on the show a miss loses the turn).

BONUS, one round per puzzle
    The board starts with R, S, T, L, N, E shown. The solver picks 3 more
    consonants and 1 more vowel, those are shown, and it gets one SOLVE.

A full run (no limit) writes per-puzzle results to data/eval/<solver>.tsv
(smart-solve.tsv or transformer-solve.tsv) and appends one summary line per
mode to data/eval/summary.tsv. A limited run is a test: its rows go to
/tmp/wof-eval-<solver>.tsv and no summary is written.

    python3 scripts/11_evaluate.py smart-solve                  # both modes, all control
    python3 scripts/11_evaluate.py smart-solve main 200         # MAIN, first 200 only
    python3 scripts/11_evaluate.py transformer-solve:runs/ft2b/ckpt.pt         # cutoff 0.99
    python3 scripts/11_evaluate.py transformer-solve:runs/ft2b/ckpt.pt,p=0.9   # another cutoff
"""
import resource
import statistics
import sys
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import wof_common as wc
from solvers import FREE, check_move, make, one_thread

SEED = "wof-eval-v1"
WORKERS = 11
PROGRESS = 250   # print a progress line every this many puzzles
OUT = wc.DATA / "eval"
S = None


def play_main(n, solution, category):
    S.new_puzzle(wc.seed_for(SEED, str(n)))
    called = []
    while True:
        board = wc.board(solution, set(called))
        mv = S.main(category, board, "".join(sorted(called)))
        if mv.startswith("SOLVE "):
            return "solved" if mv[6:] == solution else "wrong", called
        letter = check_move(mv, board, called)
        if letter is None:
            return "invalid", called
        called.append(letter)


def play_bonus(n, solution, category):
    S.new_puzzle(wc.seed_for(SEED, str(n)))
    free = "".join(sorted(FREE))
    mv = S.bonus_pick(category, wc.board(solution, FREE), free)
    parts = mv.split()
    ok = (len(parts) == 3 and parts[0] == "PICK" and len(parts[1]) == 3 and len(parts[2]) == 1
          and all(c in wc.CONSONANTS and c not in FREE for c in parts[1])
          and len(set(parts[1])) == 3 and parts[2] in wc.VOWELS and parts[2] not in FREE)
    if not ok:
        return "invalid", mv
    called = FREE | set(parts[1] + parts[2])
    sv = S.bonus_solve(category, wc.board(solution, called), "".join(sorted(called)))
    return ("solved" if sv == "SOLVE " + solution else "wrong"), mv


def work(job):
    mode, n, solution, category = job
    t0 = time.process_time()
    if mode == "main":
        result, called = play_main(n, solution, category)
        detail = {"letters": len(called), "buys": sum(c in wc.VOWELS for c in called),
                  "misses": sum(c not in solution for c in called), "order": "".join(called) or "-"}
    else:
        result, pick = play_bonus(n, solution, category)
        detail = {"pick": pick}
    detail.update(getattr(S, "notes", {}))
    cpu = time.process_time() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return mode, category, solution, result, detail, cpu, rss


def summarize(name, mode, rows, wall):
    n = len(rows)
    count = lambda r: sum(row[3] == r for row in rows)
    solved = [row for row in rows if row[3] == "solved"]
    cpu = sum(row[5] for row in rows)
    rss = max(row[6] for row in rows)
    out = {"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "solver": name, "mode": mode,
           "puzzles": n, "solved": f"{count('solved') / n:.1%}",
           "wrong": count("wrong"), "invalid": count("invalid")}
    out.update(dict.fromkeys(["letters_median", "letters_mean", "buys_mean", "misses_mean"], "-"))
    if mode == "main":
        letters = [row[4]["letters"] for row in solved]
        out.update({
            "letters_median": statistics.median(letters) if letters else "-",
            "letters_mean": f"{statistics.mean(letters):.2f}" if letters else "-",
            "buys_mean": f"{statistics.mean(r[4]['buys'] for r in solved):.2f}" if solved else "-",
            "misses_mean": f"{statistics.mean(r[4]['misses'] for r in solved):.2f}" if solved else "-",
        })
    out.update({"cpu_ms_per_puzzle": f"{1000 * cpu / n:.1f}", "wall_s": f"{wall:.0f}",
                "worker_peak_mb": f"{rss:.0f}"})
    return out


def oov_report(puzzles):
    words = [w for sol, _ in puzzles for w in wc.word_keys(sol)]
    missing = [w for w in words if w not in S.v.index.get(wc.shape(w), {})]
    hit = sum(any(w not in S.v.index.get(wc.shape(w), {}) for w in wc.word_keys(sol))
              for sol, _ in puzzles)
    print(f"control words not in the solver's word list: {len(missing):,} of {len(words):,} "
          f"({len(missing) / len(words):.1%}), in {hit:,} of {len(puzzles):,} puzzles "
          f"({hit / len(puzzles):.1%})")


def main():
    global S
    name = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "both"
    if mode not in ("main", "bonus", "both"):
        sys.exit(f"mode must be main, bonus or both, not {mode!r}")
    modes = ["main", "bonus"] if mode == "both" else [mode]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    t0 = time.time()
    S = make(name)
    load = time.time() - t0
    base_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    puzzles = wc.read_puzzles("control.csv")[:limit]
    print(f"{name}: loaded in {load:.1f}s, {base_mb:.0f} MB")
    if hasattr(S, "v"):
        oov_report(puzzles)
    OUT.mkdir(exist_ok=True)
    summary_path = OUT / "summary.tsv"
    new = not summary_path.exists()
    label = name.split(":")[0]   # smart-solve or transformer-solve
    # a limited run is a test: its rows go to /tmp so they never replace a full run's
    rows_path = Path(f"/tmp/wof-eval-{label}.tsv") if limit else OUT / f"{label}.tsv"
    with open(rows_path, "w") as f, open(summary_path, "a") as sf:
        f.write("mode\tcategory\tsolution\tresult\tdetail\n")
        for mode in modes:
            jobs = [(mode, n, sol, cat) for n, (sol, cat) in enumerate(puzzles)]
            t1 = time.time()
            rows = []
            with Pool(WORKERS, initializer=one_thread) as pool:
                for row in pool.imap(work, jobs, chunksize=4):
                    rows.append(row)
                    if len(rows) % PROGRESS == 0:
                        print(f"    {mode}: {len(rows):,}/{len(jobs):,} puzzles, "
                              f"{time.time() - t1:,.0f}s", flush=True)
            for mode_, cat, sol, result, detail, _, _ in rows:
                d = " ".join(f"{k}={v}" for k, v in detail.items())
                f.write(f"{mode_}\t{cat}\t{sol}\t{result}\t{d}\n")
            s = summarize(label, mode, rows, time.time() - t1)
            s["load_s"], s["load_mb"] = f"{load:.1f}", f"{base_mb:.0f}"
            print("  " + "  ".join(f"{k}={v}" for k, v in s.items() if k != "date"))
            if limit:
                continue
            if new:
                sf.write("\t".join(s) + "\n")
                new = False
            sf.write("\t".join(str(v) for v in s.values()) + "\n")


if __name__ == "__main__":
    main()
