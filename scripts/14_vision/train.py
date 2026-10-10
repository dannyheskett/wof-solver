"""Trains the board decoder on frames from make_frames.py.

    python train.py --name dec1 --steps 30000 --batch 128 --workers 20

Reads data/vision/synth/{train,val}/. Every --eval_every steps it scores the
val frames and writes runs/vision/<name>/ckpt.pt and a line to log.tsv.

With --real, real frames labelled by label_real.py join in: --real_frac of
each batch is drawn from the "train" split of that labels file, and the
"test" split is scored at every eval as real:<source>. Real frames train the
squares only; their category text is the decoder's own read, so it is not a
target.

Loss: cross-entropy on the 52 squares and the 24 category characters (both
only on frames with a board), plus binary cross-entropy on whether a board is
there. Scores:
    square    share of squares right, frames with a board
    board     share of frames with all 52 squares right
    cat       share of frames with the category text exactly right
    present   share of frames called right on board / no board
each also broken down by source and era.
"""
import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import board as B
import decoder_net as D
import wof_common as wc

SYNTH = wc.DATA / "vision" / "synth"
RUNS = wc.ROOT / "runs" / "vision"


def grid_targets(grid):
    return [B.CLASS_IDX[grid[r][c]] for r, c in B.SQUARES]


class Frames(Dataset):
    def __init__(self, split, augment):
        self.dir = SYNTH / split
        self.labels = [json.loads(l) for l in open(self.dir / "labels.jsonl")]
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        lab = self.labels[i]
        img = np.asarray(Image.open(self.dir / lab["file"]).convert("RGB"), dtype=np.float32) / 255
        x = torch.from_numpy(img).permute(2, 0, 1)
        if self.augment:
            g = torch.Generator().manual_seed(int(torch.randint(0, 2 ** 31, ())))
            r = torch.rand(5, generator=g)
            x = x * (0.75 + 0.5 * r[0]) + (r[1] - 0.5) * 0.2              # contrast, brightness
            gray = x.mean(0, keepdim=True)
            x = gray + (x - gray) * (0.6 + 0.8 * r[2])                     # saturation
            x = x * (0.92 + 0.16 * torch.rand(3, 1, 1, generator=g))       # colour cast
            x = x.clamp(0, 1)
        sq = torch.tensor(grid_targets(lab["grid"]) if lab["present"] else [-100] * D.N_SQ)
        cat = torch.tensor(D.encode_category(lab["category"]) if lab["present"] else [-100] * D.CAT_LEN)
        return x, sq, cat, torch.tensor(float(lab["present"])), i


RAW = wc.DATA / "vision" / "raw"


class RealFrames(Dataset):
    """Frames from data/vision/raw/ with label_real.py's labels, letterboxed
    to the decoder's input. Same item shape as Frames. The category target is
    the puzzle's majority read snapped to the nearest show category
    (board.snap_category), and masked when nothing is near enough."""

    def __init__(self, labels_path, split, augment, root=RAW):
        self.labels = [l for l in map(json.loads, open(labels_path)) if l["split"] == split]
        self.augment = augment
        self.root = Path(root)
        choices = B.show_categories()
        snapped = {}
        for l in self.labels:
            r = l.get("category_read", "")
            if r not in snapped:
                snapped[r] = B.snap_category(r, choices)
            l["category"] = snapped[r]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        from make_frames import letterbox
        lab = self.labels[i]
        img, *_ = letterbox(Image.open(self.root / lab["clip"] / "frames" / lab["frame"]).convert("RGB"))
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255).permute(2, 0, 1)
        if self.augment:
            r = torch.rand(3)
            x = (x * (0.8 + 0.4 * r[0]) + (r[1] - 0.5) * 0.15).clamp(0, 1)
        sq = torch.tensor(grid_targets(lab["grid"]))
        cat = torch.tensor(D.encode_category(lab["category"])) if lab["category"] else torch.full((D.CAT_LEN,), -100)
        return x, sq, cat, torch.tensor(1.0), i


def losses(out, sq, cat, present):
    sq_l, cat_l, pr_l = out
    l_sq = F.cross_entropy(sq_l.flatten(0, 1), sq.flatten(), ignore_index=-100, label_smoothing=0.02)
    l_cat = F.cross_entropy(cat_l.flatten(0, 1), cat.flatten(), ignore_index=-100)
    l_pr = F.binary_cross_entropy_with_logits(pr_l, present)
    if torch.isnan(l_sq):   # a batch with no board
        l_sq = sq_l.sum() * 0
    if torch.isnan(l_cat):  # no category targets (no board, or only real frames)
        l_cat = cat_l.sum() * 0
    return l_sq + 0.5 * l_cat + 0.2 * l_pr, (l_sq.item(), l_cat.item(), l_pr.item())


@torch.no_grad()
def evaluate(model, loader, labels, device):
    model.eval()
    tally = defaultdict(lambda: [0, 0])
    for x, sq, cat, present, idx in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            sq_l, cat_l, pr_l = model(x)
        sq_p = sq_l.argmax(-1).cpu()
        cat_p = cat_l.argmax(-1).cpu()
        pr_p = (pr_l.float().cpu() > 0)
        for k in range(len(idx)):
            lab = labels[int(idx[k])]
            keys = ["all", lab["source"], lab["era"]]
            for key in keys:
                tally[("present", key)][0] += int(pr_p[k] == bool(present[k]))
                tally[("present", key)][1] += 1
            if not lab["present"]:
                continue
            right = (sq_p[k] == sq[k])
            for key in keys:
                tally[("square", key)][0] += int(right.sum())
                tally[("square", key)][1] += D.N_SQ
                tally[("board", key)][0] += int(right.all())
                tally[("board", key)][1] += 1
                tally[("cat", key)][0] += int(D.decode_category(cat_p[k].tolist()) == lab["category"].strip())
                tally[("cat", key)][1] += 1
    model.train()
    return {k: v[0] / max(v[1], 1) for k, v in tally.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--init", help="a checkpoint to start from")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--val_limit", type=int, default=0, help="score only the first n val frames")
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--real", help="label_real.py labels to mix in, e.g. data/vision/real/labels.jsonl")
    ap.add_argument("--real_frac", type=float, default=0.3)
    ap.add_argument("--real_root", default=str(RAW), help="where the real frames are (raw_lb on a pod)")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    out_dir = RUNS / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    train, val = Frames(a.train_split, True), Frames("val", False)
    if a.val_limit:
        val.labels = val.labels[:a.val_limit]
    real_test = None
    if a.real:
        real_train = RealFrames(a.real, "train", True, a.real_root)
        real_test = RealFrames(a.real, "test", False, a.real_root)
        mixed = torch.utils.data.ConcatDataset([train, real_train])
        w = torch.cat([torch.full((len(train),), (1 - a.real_frac) / len(train)),
                       torch.full((len(real_train),), a.real_frac / max(1, len(real_train)))])
        sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=len(train), replacement=True)
        tl = DataLoader(mixed, a.batch, sampler=sampler, num_workers=a.workers, drop_last=True,
                        pin_memory=True, persistent_workers=a.workers > 0)
        print(f"{len(real_train)} real train frames, {len(real_test)} real test frames", flush=True)
    else:
        tl = DataLoader(train, a.batch, shuffle=True, num_workers=a.workers, drop_last=True,
                        pin_memory=True, persistent_workers=a.workers > 0)
    vl = DataLoader(val, a.batch, num_workers=a.workers, pin_memory=True)
    rl = DataLoader(real_test, a.batch, num_workers=a.workers) if real_test and len(real_test) else None
    model = D.BoardDecoder().to(device)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=device)["model"])
    fwd = torch.compile(model) if a.compile else model
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    rest = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05},
                             {"params": rest, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.98))

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        p = (s - a.warmup) / max(1, a.steps - a.warmup)
        return a.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * p)))

    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters, "
          f"{len(train)} train frames, {len(val)} val frames, {device}", flush=True)
    log = open(out_dir / "log.tsv", "a")
    step, t0, seen, run = 0, time.time(), 0, np.zeros(3)
    best = -1.0
    while step < a.steps:
        for x, sq, cat, present, _ in tl:
            if step >= a.steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            x, sq, cat, present = (t.to(device, non_blocking=True) for t in (x, sq, cat, present))
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = fwd(x)
            loss, parts = losses([o.float() for o in out], sq, cat, present)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run = 0.98 * run + 0.02 * np.array(parts) if step else np.array(parts)
            step += 1
            seen += len(x)
            if step % a.log_every == 0:
                el = time.time() - t0
                print(f"step {step} loss sq {run[0]:.4f} cat {run[1]:.4f} present {run[2]:.4f} "
                      f"lr {lr_at(step):.2e} {seen / el:.0f} frames/s", flush=True)
            if step % a.eval_every == 0 or step == a.steps:
                sc = evaluate(model, vl, val.labels, device)
                keys = ["all", "broadcast", "offair", "phone", "new", "old"]
                line = "  ".join(f"{m} " + "/".join(f"{sc.get((m, k), 0):.3f}" for k in keys)
                                 for m in ("square", "board", "cat", "present"))
                print(f"eval {step} ({'/'.join(keys)})  {line}", flush=True)
                if rl:
                    rs = evaluate(model, rl, [dict(l, era="real", present=True)
                                              for l in real_test.labels], device)
                    keys_r = ["all", "broadcast", "offair", "phone"]
                    print(f"real {step} ({'/'.join(keys_r)})  " + "  ".join(
                        f"{m} " + "/".join(f"{rs.get((m, k), 0):.3f}" for k in keys_r)
                        for m in ("square", "board", "cat")), flush=True)
                    sc.update({("real_" + m, k): v for (m, k), v in rs.items()})
                log.write(f"{step}\t{run[0]:.4f}\t{run[1]:.4f}\t{run[2]:.4f}\t"
                          + json.dumps({f"{m}:{k}": round(v, 4) for (m, k), v in sc.items()}) + "\n")
                log.flush()
                torch.save({"model": model.state_dict(), "step": step, "args": vars(a), "scores":
                            {f"{m}:{k}": v for (m, k), v in sc.items()}}, out_dir / "ckpt.pt")
                if sc[("board", "all")] > best:
                    best = sc[("board", "all")]
                    torch.save({"model": model.state_dict(), "step": step}, out_dir / "best.pt")


if __name__ == "__main__":
    main()
