#!/usr/bin/env python3
"""Phase 5. Build the pretraining corpus: about TARGET characters of plain English,
half English Wikipedia and half FineWeb-Edu, in board characters only.

Run with the project venv (it needs pyarrow):
    .venv/bin/python scripts/05_make_corpus.py

Sources, downloaded to data/corpus/raw/ (see data/corpus/SOURCE.md):
    wiki-*.parquet      wikimedia/wikipedia 20231101.en, column "text"
    fineweb-*.parquet   HuggingFaceFW/fineweb-edu sample/10BT, column "text"

Cleaning, per paragraph (one line of the source text):
    accents are stripped (E for É), curly quotes become ', en dashes and
    other hyphen marks become -, em dashes become a space
    letters that do not break down into a base letter are mapped by hand
    (L for Ł, SS for ß), the text is uppercased and every character
    outside A-Z, space and - & ' . ! ? , becomes a space, then runs of
    spaces collapse, and the stray punctuation left where numbers and
    brackets were removed is tidied ("IN , THE" becomes "IN, THE")
    paragraphs with fewer than MIN_LETTERS letters are dropped
    Wikipedia articles are cut at their References / See also / External
    links / Notes / Further reading section

Output, one paragraph per line, documents separated by a blank line:
    data/corpus/train.txt   documents from both sources, interleaved
    data/corpus/val.txt     about 1% of documents, chosen by hash of the text
Each source stops once it has written TARGET/2 characters.
"""
import hashlib
import re
import sys
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"
RAW = CORPUS / "raw"
TARGET = 1_000_000_000
MIN_LETTERS = 40
VAL_PCT = 1
WIKI_STOP = re.compile(r"^(References|See also|External links|Notes|Further reading"
                       r"|Bibliography|Sources|Citations|Footnotes)\s*$")
TRANSLATE = str.maketrans({
    "‘": "'", "’": "'", "ʼ": "'", "`": "'",
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": " ",
    "ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D", "ð": "d", "Ð": "D",
    "ı": "i", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ß": "ss", "þ": "th", "Þ": "TH",
})
NOT_BOARD = re.compile(r"[^A-Z \-&'.!?,]+")
SPACES = re.compile(r" {2,}")
STRAY = re.compile(r"\s+([,.!?])")
REPEAT = re.compile(r"([,.!?])(?:\s*[,.])+")


def clean_line(line):
    line = unicodedata.normalize("NFKD", line.translate(TRANSLATE))
    line = "".join(ch for ch in line if not unicodedata.combining(ch)).upper()
    line = SPACES.sub(" ", NOT_BOARD.sub(" ", line))
    line = REPEAT.sub(r"\1", STRAY.sub(r"\1", line)).strip(" ,.")
    letters = sum(ch.isalpha() for ch in line)
    return line if letters >= MIN_LETTERS else None


def documents(pattern, wiki):
    for path in sorted(RAW.glob(pattern)):
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            for text in pf.read_row_group(rg, columns=["text"]).column("text").to_pylist():
                lines = []
                for raw in text.split("\n"):
                    if wiki and WIKI_STOP.match(raw.strip()):
                        break
                    c = clean_line(raw)
                    if c:
                        lines.append(c)
                if lines:
                    yield "\n".join(lines) + "\n\n"


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else TARGET
    sources = {"wiki": documents("wiki-*.parquet", True),
               "web": documents("fineweb-*.parquet", False)}
    done = {k: 0 for k in sources}
    docs = {k: 0 for k in sources}
    val_chars = 0
    with open(CORPUS / "train.txt", "w") as train, open(CORPUS / "val.txt", "w") as val:
        while sources:
            name = min(sources, key=lambda k: done[k])
            try:
                doc = next(sources[name])
            except StopIteration:
                print(f"  {name}: source ran out at {done[name]:,} characters")
                del sources[name]
                continue
            h = int(hashlib.sha1(doc.encode()).hexdigest(), 16) % 100
            if h < VAL_PCT:
                val.write(doc)
                val_chars += len(doc)
            else:
                train.write(doc)
                done[name] += len(doc)
                docs[name] += 1
            if done[name] >= target // 2:
                del sources[name]
            if sum(docs.values()) % 100_000 == 0:
                print(f"  {done}", flush=True)
    for k in done:
        print(f"{k:5} {docs[k]:9,} documents {done[k]:14,} characters")
    print(f"train {sum(done.values()):,} characters, val {val_chars:,} characters")


if __name__ == "__main__":
    main()
