"""Shared cleaning and output for the category list builders.

Every builder produces (category, phrase) pairs and hands them to write().
write() normalises each phrase to board characters, keeps it only if it fits
the TV board, drops anything that is a control solution, removes repeats,
and writes data/sources/<name>.tsv as "category<TAB>phrase" lines.
"""
import csv
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from wof_common import BOARD_CHARS, fits_board  # noqa: E402

DATA = ROOT / "data"
SOURCES = DATA / "sources"
RAW = SOURCES / "raw"

TRANSLATE = str.maketrans({
    "‘": "'", "’": "'", "ʼ": "'", "`": "'",
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": " ",
    "ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D", "ð": "d", "Ð": "D",
    "ı": "i", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ß": "ss", "þ": "th", "Þ": "TH",
    "_": " ",
})
PARENS = re.compile(r"\s*[\(\[][^)\]]*[\)\]]")
SPACES = re.compile(r"\s+")


def normalize(text):
    """Board form of a phrase, or None if it holds characters a board cannot
    show (digits, symbols, other scripts). Parenthesised notes are dropped:
    "Titanic (1997 film)" becomes TITANIC."""
    text = PARENS.sub("", text.translate(TRANSLATE))
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).upper()
    text = SPACES.sub(" ", text).strip(" ,.")
    if not text or set(text) - BOARD_CHARS or not any(ch.isalpha() for ch in text):
        return None
    return text


def control_solutions():
    with open(DATA / "control.csv", newline="") as f:
        return {r["answer"] for r in csv.DictReader(f)}


def write(name, pairs):
    control = control_solutions()
    seen = set()
    counts = Counter()
    dropped = Counter()
    for category, phrase in pairs:
        p = normalize(phrase)
        if p is None:
            dropped["characters"] += 1
        elif not fits_board(p):
            dropped["too long for board"] += 1
        elif p in control:
            dropped["control solution"] += 1
        elif (category, p) in seen:
            dropped["repeat"] += 1
        else:
            seen.add((category, p))
            counts[category] += 1
    path = SOURCES / f"{name}.tsv"
    with open(path, "w") as f:
        for category, p in sorted(seen):
            f.write(f"{category}\t{p}\n")
    print(f"{name}: {len(seen):,} lines -> {path.relative_to(ROOT)}")
    for c, n in counts.most_common():
        print(f"    {n:8,}  {c}")
    if dropped:
        print("    dropped: " + ", ".join(f"{k} {v:,}" for k, v in dropped.items()))
