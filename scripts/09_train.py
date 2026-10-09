#!/usr/bin/env python3
"""Phase 9. Train the 59M-parameter transformer on the files from
08_make_train.py.

    pretrain   corpus only, loss on every character
    finetune   rows drawn from the streams by --mix; task rows pack whole
               lines one after another, with loss on targets only (see
               08_make_train.py), corpus rows as in pretrain

Both phases use AdamW with linear warmup and cosine decay to lr/10, and
bf16 autocast on CUDA. Every --eval_every steps it reports validation
loss per stream and saves runs/<name>/ckpt.pt, which holds the weights,
the model config, the alphabet and these arguments. log.tsv in the same
folder records step, losses, tokens per second and elapsed time, for the
cost log.

CPU smoke test (a few minutes):
    python3 scripts/09_train.py --name smoke --phase pretrain --n_layer 2 --n_head 2 \\
        --n_embd 64 --batch 16 --steps 200 --eval_every 100
The GPU run, with the exact settings used, is scripts/09_pod/run.sh.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from model import GPT, Config

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "train"
RUNS = ROOT / "runs"
ALPHABET = (TRAIN / "alphabet.txt").read_text()
NEWLINE = ALPHABET.index("\n")
DEFAULT_MIX = "main=0.35,bonus=0.30,list_bonus=0.20,list=0.05,corpus=0.10"


def args_():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--phase", choices=["pretrain", "finetune"], required=True)
    p.add_argument("--init", help="checkpoint to start from")
    p.add_argument("--mix", default=DEFAULT_MIX)
    p.add_argument("--n_layer", type=int, default=12)
    p.add_argument("--n_head", type=int, default=10)
    p.add_argument("--n_embd", type=int, default=640)
    p.add_argument("--block", type=int, default=256)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


class Lines:
    """One task stream: token file plus (start, prompt length, length)."""

    def __init__(self, name):
        self.tok = np.memmap(TRAIN / f"{name}.bin", dtype=np.uint8, mode="r")
        self.idx = np.load(TRAIN / f"{name}.idx.npy")

    def row(self, rng, n):
        """n + 1 tokens of whole random lines, and the loss mask for each
        predicted position (True where the next token is target)."""
        x, m = [], []
        while len(x) < n + 1:
            start, plen, length = self.idx[rng.integers(len(self.idx))]
            x.extend(self.tok[start:start + length])
            m.extend([False] * plen + [True] * (length - plen))
        return np.array(x[:n + 1], dtype=np.int64), np.array(m[1:n + 1])


class Text:
    def __init__(self, name):
        self.tok = np.memmap(TRAIN / f"{name}.bin", dtype=np.uint8, mode="r")

    def row(self, rng, n):
        s = rng.integers(len(self.tok) - n - 1)
        return self.tok[s:s + n + 1].astype(np.int64), np.ones(n, dtype=bool)


def batch(sources, weights, rng, B, T, device):
    X = np.empty((B, T + 1), dtype=np.int64)
    Y = np.empty((B, T), dtype=np.int64)
    picks = rng.choice(len(sources), size=B, p=weights)
    for r, k in enumerate(picks):
        x, m = sources[k].row(rng, T)
        X[r] = x
        Y[r] = np.where(m, x[1:], -1)
    return (torch.from_numpy(X[:, :-1]).to(device, non_blocking=True),
            torch.from_numpy(Y).to(device, non_blocking=True))


def main():
    a = args_()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = RUNS / a.name
    out.mkdir(parents=True, exist_ok=True)

    if a.init:
        ck = torch.load(a.init, map_location="cpu")
        cfg = Config(**ck["config"])
        model = GPT(cfg)
        model.load_state_dict(ck["model"])
    else:
        cfg = Config(vocab=len(ALPHABET), block=a.block, n_layer=a.n_layer,
                     n_head=a.n_head, n_embd=a.n_embd)
        model = GPT(cfg)
    model.to(device)
    print(f"{a.name}: {model.params() / 1e6:.2f}M parameters, device {device}"
          + (f" ({torch.cuda.get_device_name()})" if device == "cuda" else ""), flush=True)

    if a.phase == "pretrain":
        train = [(Text("corpus"), 1.0)]
    else:
        mix = dict(kv.split("=") for kv in a.mix.split(","))
        train = [(Text("corpus") if k == "corpus" else Lines(k), float(w)) for k, w in mix.items()]
    sources = [s for s, _ in train]
    weights = np.array([w for _, w in train])
    weights /= weights.sum()
    vals = {"corpus_val": Text("corpus_val")}
    if a.phase == "finetune":
        vals.update(main_val=Lines("main_val"), bonus_val=Lines("bonus_val"))

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), fused=device == "cuda")
    step_fn = torch.compile(model) if a.compile else model
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.autocast("cpu", enabled=False)

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        frac = (s - a.warmup) / max(1, a.steps - a.warmup)
        return a.lr / 10 + 0.5 * (a.lr - a.lr / 10) * (1 + math.cos(math.pi * frac))

    @torch.no_grad()
    def evaluate():
        model.eval()
        erng = np.random.default_rng(0)
        res = {}
        for name, src in vals.items():
            losses = []
            for _ in range(a.eval_batches):
                x, y = batch([src], np.array([1.0]), erng, a.batch, cfg.block, device)
                with amp:
                    losses.append(float(step_fn(x, y)))
            res[name] = sum(losses) / len(losses)
        model.train()
        return res

    log = open(out / "log.tsv", "a")
    if log.tell() == 0:
        log.write("step\ttrain_loss\t" + "\t".join(vals) + "\ttokens_per_s\telapsed_s\n")
    t0 = time.time()
    tick, tokens, recent = time.time(), 0, []
    for s in range(a.steps + 1):
        if s % a.eval_every == 0 or s == a.steps:
            ev = evaluate()
            el = time.time() - t0
            tps = tokens / max(1e-9, time.time() - tick)
            tl = sum(recent) / len(recent) if recent else float("nan")
            print(f"step {s:6d}  train {tl:.4f}  " + "  ".join(f"{k} {v:.4f}" for k, v in ev.items())
                  + f"  {tps:,.0f} tok/s  {el:,.0f}s", flush=True)
            log.write(f"{s}\t{tl:.4f}\t" + "\t".join(f"{v:.4f}" for v in ev.values())
                      + f"\t{tps:.0f}\t{el:.0f}\n")
            log.flush()
            torch.save({"model": model.state_dict(), "config": vars(cfg), "alphabet": ALPHABET,
                        "args": vars(a), "step": s}, out / "ckpt.pt")
            tick, tokens, recent = time.time(), 0, []
            if s == a.steps:
                break
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        x, y = batch(sources, weights, rng, a.batch, cfg.block, device)
        with amp:
            loss = step_fn(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tokens += x.numel()
        recent.append(float(loss.detach()))
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))


if __name__ == "__main__":
    main()
