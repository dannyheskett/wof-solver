"""Drops doubtful real labels before training: label_real.py's <name>.jsonl
in, <name>_clean.jsonl out.

A puzzle whose answer was spelled from the frames (not looked up in
puzzles.csv) is dropped when the decoder disagreed with its labels on more
than MAX_REVIEW squares a frame on average, when it has a word of one
letter other than A or I, a word of two or three letters used fewer than
MIN_SHORT times in puzzles.csv, or a word in neither the word list nor
puzzles.csv, or when it starts with punctuation. A hand check of 40 spelled
answers found about a quarter wrong (B PAWN COFFEE D T Y, MITHRA MATURE
CUBIST); these rules catch most, and what passes is mostly one wrong letter
(DON'T WAIT FUR ME).

The test split keeps only answers looked up in puzzles.csv, so real-frame
scores are measured against answers known to be right.

    python clean_labels.py [data/vision/real/labels.jsonl]
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import label_real as LR
import wof_common as wc

MAX_REVIEW = 2.5
MIN_SHORT = 5


def doubtful(answer, rows, counts, words):
    if sum(r["review"] for r in rows) / len(rows) > MAX_REVIEW:
        return "disagreement"
    if not answer or not answer[0].isalpha():
        return "empty or punctuation first"
    for w in wc.word_keys(answer):
        if len(w) == 1 and w not in ("A", "I"):
            return "one letter"
        if len(w) <= 3 and w not in ("A", "I") and counts[w] < MIN_SHORT:
            return "rare short word"
        if w not in words and not counts[w]:
            return "unknown word"
    return None


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else wc.DATA / "vision" / "real" / "labels.jsonl")
    LR.vocab()
    counts = LR._counts
    words = {w.strip().upper() for w in open(wc.WORDLIST)}
    rows = [json.loads(l) for l in open(src)]
    seg = defaultdict(list)
    for r in rows:
        seg[(r["clip"], r["answer"])].append(r)
    keep, why = [], Counter()
    for (_, answer), rs in seg.items():
        if rs[0]["how"] == "spelled":
            if rs[0]["split"] == "test":
                why["spelled, test split"] += len(rs)
                continue
            reason = doubtful(answer, rs, counts, words)
            if reason:
                why[reason] += len(rs)
                continue
        keep.extend(rs)
    out = src.with_name(src.stem + "_clean.jsonl")
    out.write_text("".join(json.dumps(r) + "\n" for r in keep))
    print(f"{out}: kept {len(keep)} of {len(rows)} frames; dropped {dict(why)}")
    print("kept by split and source", Counter((r["split"], r["source"]) for r in keep))


if __name__ == "__main__":
    main()
