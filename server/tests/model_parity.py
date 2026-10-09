#!/usr/bin/env python3
"""Check the C forward pass (server/src/model.c) against PyTorch: the same
next-token log probabilities and the same greedy completions, on MAIN and
BONUS prompts built from control puzzles part way through a game.

    make -C server build/model_dump
    python3 server/tests/model_parity.py runs/ft2b/ckpt.pt runs/ft2b/wof.bin [N]
"""
import random
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import wof_common as wc  # noqa: E402
from model import GPT, Config  # noqa: E402


def prompts(n, rng):
    puzzles = wc.read_puzzles("control.csv")
    out = []
    for answer, category in rng.sample(puzzles, n):
        if rng.random() < 0.5:
            called = "".join(sorted(rng.sample(wc.LETTERS, rng.randint(0, 8))))
            mode = "MAIN"
        else:
            called = "".join(sorted(set("RSTLNE") | set(rng.sample(wc.LETTERS, rng.choice([0, 4])))))
            mode = "BONUS"
        board = "".join("_" if c in wc.LETTERS and c not in called else c for c in answer)
        out.append(f"{mode}|{category.upper()}|{board}|{called or '-'}|")
    return out


def main():
    ckpt, weights = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    ps = prompts(n, random.Random(7))
    r = subprocess.run([str(ROOT / "server/build/model_dump"), weights], input="\n".join(ps) + "\n",
                       capture_output=True, text=True, check=True)
    lines = r.stdout.split("\n")
    torch.set_num_threads(1)
    ck = torch.load(ckpt, map_location="cpu")
    a = ck["alphabet"]
    model = GPT(Config(**ck["config"]))
    model.load_state_dict(ck["model"])
    model.eval()
    stop = a.index("\n")
    worst, same = 0.0, 0
    for i, p in enumerate(ps):
        ids = [a.index(c) for c in p]
        with torch.no_grad():
            lp = torch.log_softmax(model(torch.tensor([ids]))[0, -1], 0)
        c_lp = torch.tensor([float(x) for x in lines[2 * i].split()])
        worst = max(worst, float((lp - c_lp).abs().max()))
        want = "".join(a[t] for t in model.complete([ids], stop, 80)[0])
        if want == lines[2 * i + 1]:
            same += 1
        else:
            print(f"differs: {p!r}\n  torch {want!r}\n  C     {lines[2 * i + 1]!r}")
    print(f"{n} prompts: max |log p difference| {worst:.2e}, completions identical {same}/{n}")
    print("C timing:", r.stderr.strip())


if __name__ == "__main__":
    main()
