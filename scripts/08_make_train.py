#!/usr/bin/env python3
"""Phase 8. Turn the TSV datasets and the corpus into model-ready token files.

The model reads characters. ALPHABET below is every character it sees, and
a character's token id is its index there. Each training line is one
prompt and one target, ending in a newline:

    MAIN|THING|A _A__L_ __ ___ ______|AL|BUY E
    BONUS|THING|_ __N_LE _N T_E __N___|ELNRST|PICK DHM A
    BONUS|THING|A _ANDLE _N THE __ND__|ADEHLMNRST|SOLVE A CANDLE IN THE WINDOW
    LIST|ON THE MAP|SAN ANTONIO TEXAS

The prompt is everything up to the last '|'. Training puts loss on the
target and the newline only. The category is uppercased.

Streams, each a token file plus an index of (start, prompt length, length):

    Puzzles held out in data/dev.csv (03_split_dev.py) are in none of them.

    main         MAIN lines from data/main_train.tsv
    bonus        BONUS lines from data/bonus_train.tsv
    list         LIST lines: every train.csv puzzle and every source
                 list phrase under its category (the model's category
                 knowledge), except phrases that are a dev answer
    list_bonus   BONUS SOLVE lines made from the source list phrases: RSTLNE
                 plus a random 3 consonants and 1 vowel shown, seeded per
                 phrase. Phrases already fully shown are skipped.
    main_val, bonus_val   the same from the control files, for validation
                          loss only, never for training

The corpus (data/corpus/train.txt and val.txt) becomes corpus.bin and
corpus_val.bin: the text as token ids, no index, loss on everything.

Output: data/train/
"""
import random
import time

import numpy as np

import wof_common as wc

ALPHABET = "\n !&',-./0123456789?ABCDEFGHIJKLMNOPQRSTUVWXYZ_|"
OUT = wc.DATA / "train"
SEED = "wof-train-v1"
FREE = set("RSTLNE")
CONS = [c for c in wc.CONSONANTS if c not in FREE]
VOWS = [c for c in "AEIOU" if c not in FREE]
TABLE = np.full(256, 255, dtype=np.uint8)
for i, ch in enumerate(ALPHABET):
    TABLE[ord(ch)] = i


def encode(text):
    ids = TABLE[np.frombuffer(text.encode("ascii"), dtype=np.uint8)]
    if (ids == 255).any():
        bad = sorted({ch for ch in text if ch not in ALPHABET})
        raise ValueError(f"characters outside the alphabet: {bad}")
    return ids


class Stream:
    def __init__(self, name):
        self.name = name
        self.f = open(OUT / f"{name}.bin", "wb")
        self.index = []
        self.pos = 0
        self.buf = []

    def add(self, prompt, target):
        line = prompt + target + "\n"
        self.index.append((self.pos, len(prompt), len(line)))
        self.pos += len(line)
        self.buf.append(line)
        if len(self.buf) >= 100_000:
            self.flush()

    def flush(self):
        if self.buf:
            self.f.write(encode("".join(self.buf)).tobytes())
            self.buf = []

    def close(self):
        self.flush()
        self.f.close()
        np.save(OUT / f"{self.name}.idx.npy", np.array(self.index, dtype=np.int64))
        print(f"  {self.name}: {len(self.index):,} lines, {self.pos / 1e6:,.1f}M tokens", flush=True)


def task_lines(src, mode, stream):
    s = Stream(stream)
    for line in open(wc.DATA / src):
        cat, board, called, target, _ = line.rstrip("\n").split("\t")
        s.add(f"{mode}|{cat.upper()}|{board}|{called}|", target)
    s.close()


def list_lines():
    dev = {sol for sol, _ in wc.read_puzzles("dev.csv")}
    pairs = [(cat, sol) for sol, cat in wc.read_puzzles("train.csv")]
    for path in sorted((wc.DATA / "sources").glob("*.tsv")):
        for line in open(path):
            cat, phrase = line.rstrip("\n").split("\t")
            pairs.append((cat, phrase))
    lst, bon = Stream("list"), Stream("list_bonus")
    seen = set()
    for cat, phrase in pairs:
        cat = cat.upper()
        if (cat, phrase) in seen or phrase in dev:
            continue
        seen.add((cat, phrase))
        lst.add(f"LIST|{cat}|", phrase)
        rng = random.Random(wc.seed_for(SEED, cat, phrase))
        called = FREE | set(rng.sample(CONS, 3) + rng.sample(VOWS, 1))
        board = wc.board(phrase, called)
        if "_" in board:
            bon.add(f"BONUS|{cat}|{board}|{''.join(sorted(called))}|", f"SOLVE {phrase}")
    lst.close()
    bon.close()


def corpus(src, dst):
    n = 0
    with open(wc.DATA / "corpus" / src, encoding="ascii") as f, open(OUT / dst, "wb") as out:
        while chunk := f.read(64 << 20):
            out.write(encode(chunk).tobytes())
            n += len(chunk)
    print(f"  {dst}: {n / 1e6:,.1f}M tokens", flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "alphabet.txt").write_text(ALPHABET)
    t0 = time.time()
    task_lines("main_train.tsv", "MAIN", "main")
    task_lines("bonus_train.tsv", "BONUS", "bonus")
    task_lines("main_control.tsv", "MAIN", "main_val")
    task_lines("bonus_control.tsv", "BONUS", "bonus_val")
    list_lines()
    corpus("train.txt", "corpus.bin")
    corpus("val.txt", "corpus_val.bin")
    print(f"done in {time.time() - t0:,.0f}s")


if __name__ == "__main__":
    main()
