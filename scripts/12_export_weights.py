#!/usr/bin/env python3
"""Phase 12. Write a checkpoint from 09_train.py as the flat weights file the C server
loads (server/src/model.c).

Layout, all little-endian:
    magic   "WOFM" and a uint32 version (2)
    uint32  vocab, block, n_layer, n_head, n_embd
    char    the alphabet, vocab bytes; a token's id is its index
    float16 tensors (IEEE half), in order:
        tok        [vocab][n_embd]   (also the output layer, tied)
        pos        [block][n_embd]
        per layer: ln1.w, ln1.b [n_embd]
                   qkv  [3 n_embd][n_embd]
                   proj [n_embd][n_embd]
                   ln2.w, ln2.b [n_embd]
                   fc   [4 n_embd][n_embd]
                   out  [n_embd][4 n_embd]
        ln.w, ln.b [n_embd]
A linear layer's weight is [out][in], as in PyTorch. Half precision keeps the
file (and the server's memory) at about 118 MB; the server converts each
value to float32 as it computes.

    python3 scripts/12_export_weights.py runs/ft2b/ckpt.pt runs/ft2b/wof.bin
"""
import struct
import sys

import numpy as np
import torch


def main():
    src, dst = sys.argv[1], sys.argv[2]
    ck = torch.load(src, map_location="cpu")
    c, sd, alphabet = ck["config"], ck["model"], ck["alphabet"]
    assert len(alphabet) == c["vocab"] and all(ord(ch) < 128 for ch in alphabet)
    names = ["tok.weight", "pos.weight"]
    for k in range(c["n_layer"]):
        b = f"blocks.{k}."
        names += [b + "ln1.weight", b + "ln1.bias", b + "attn.qkv.weight", b + "attn.proj.weight",
                  b + "ln2.weight", b + "ln2.bias", b + "mlp.0.weight", b + "mlp.2.weight"]
    names += ["ln.weight", "ln.bias"]
    with open(dst, "wb") as f:
        f.write(b"WOFM" + struct.pack("<6I", 2, c["vocab"], c["block"], c["n_layer"], c["n_head"],
                                      c["n_embd"]))
        f.write(alphabet.encode("ascii"))
        for n in names:
            f.write(sd[n].float().numpy().astype("<f2").tobytes())
    print(f"{dst}: {len(names)} tensors, {sum(sd[n].numel() for n in names):,} values as float16")


if __name__ == "__main__":
    main()
