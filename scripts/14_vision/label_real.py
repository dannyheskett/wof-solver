"""Labels the real frames in data/vision/raw/ from the decoder's own reads,
corrected by each puzzle's answer.

    python label_real.py --ckpt runs/vision/dec1/best.pt [--clips id ...] [--every 1]
        [--raw data/vision/raw_lb]   # the frames to read (shrink_frames.py's copy on a pod)

For each clip, in frame order:
1. The decoder reads every frame (every --every'th one).
2. Frames with a board are cut into puzzles: a new puzzle starts when more
   than FLICKER squares in play change. Letters are not used here, since an
   early decoder misreads them (A as R, O as D). Runs of SHORT frames or
   fewer (a puzzle's reveal animation, a cut-away) join the next puzzle if
   they fit it, and are dropped if not.
3. A puzzle's squares in play are pooled over its frames, and the answer is
   looked up among the 68k puzzles in data/puzzles.csv whose show-style
   layout puts squares in play in exactly those places. Each candidate
   scores the squares where the decoder's confident letter read agrees with
   it. The best is taken when at least MIN_AGREE of the confident reads
   agree and it leads the next answer by MARGIN squares; otherwise the
   puzzle is spelled instead (spell_answer): each word on the board is the
   word list entry of its shape that best fits the decoder's letter
   probabilities over every frame where the word shows. Most puzzles from
   Season 40 on are not in puzzles.csv, so this is how they get answers.
4. Each frame is relabelled from the answer. A square the answer leaves
   unused is '#', even when the host stands in front of it. A square in
   play shows its letter from the first frame where the decoder reads that
   letter there with confidence, and every frame after: the host reveals a
   called letter's squares one at a time, so a letter can show in one
   square and not yet in another. Before that it is '_', or '*' where the
   decoder reads it lit. Punctuation always shows. The decoder's '~'
   (covered) reads on squares in play are kept.

Writes data/vision/real/labels.jsonl, one line per labelled frame, with the
clip, frame, source (from clips.tsv), answer, category, grid, the number of
squares where the decoder disagreed with the label (review), and split:
"test" for the 5% of answers 02_clean_puzzles.py holds out as control
(hash of the answer), "train" otherwise. So a puzzle shown in two clips
lands in the same split both times.
"""
import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

import board as B
import wof_common as wc

RAW = wc.DATA / "vision" / "raw"
OUT = wc.DATA / "vision" / "real"
SURE = 0.9          # a read this confident counts toward the answer and reveals a letter
PRESENT = 0.5
FLICKER = 2        # squares in play that may differ within one puzzle (misreads)
SHORT = 2          # frames; a shorter run is a reveal animation or a cut-away
MIN_AGREE = 0.7     # share of confident letter reads the answer must agree with
MARGIN = 3          # squares the best answer must lead the next one by


def in_control(answer):
    return int(hashlib.sha1(answer.encode()).hexdigest(), 16) % 100 < 5


def play_mask(grid):
    """Per square: True in play, False unused, None covered or unsure."""
    out = []
    for r, c in B.SQUARES:
        ch = grid[r][c]
        out.append(None if ch == B.COVERED else ch != B.OFF)
    return out


def layout_index():
    """In-play mask -> [(answer, category, layout grid)] over every puzzle."""
    idx = defaultdict(list)
    seen = set()
    with open(wc.DATA / "puzzles.csv", newline="") as f:
        for r in csv.DictReader(f):
            a = r["answer"]
            if a in seen or set(a) - wc.BOARD_CHARS:
                continue
            seen.add(a)
            for lay in B.layouts(a, all_starts=True):
                cols = {(row, start + k) for row, (start, text) in lay.items()
                        for k, ch in enumerate(text) if ch != " "}
                key = tuple((r_, c_) in cols for r_, c_ in B.SQUARES)
                idx[key].append((a, r["category"], lay))
    return idx


def same_puzzle(mask_a, mask_b):
    return sum(a is not None and b is not None and a != b for a, b in zip(mask_a, mask_b)) <= FLICKER


def pooled_mask(seg):
    votes = [Counter() for _ in B.SQUARES]
    for _, rd in seg:
        for j, v in enumerate(play_mask(rd["grid"])):
            if v is not None:
                votes[j][v] += 1
    return [v.most_common(1)[0][0] if v else None for v in votes]


def segments(reads):
    """Runs of board frames that show the same puzzle."""
    segs, cur, cur_mask = [], [], None
    for k, rd in reads:
        if rd["present"] < PRESENT:
            continue
        m = play_mask(rd["grid"])
        if cur and not same_puzzle(cur_mask, m):
            segs.append(cur)
            cur, cur_mask = [], None
        cur.append((k, rd))
        cur_mask = m if cur_mask is None else [a if a is not None else b for a, b in zip(cur_mask, m)]
    if cur:
        segs.append(cur)
    out = []
    carry = []                           # short runs waiting for the next puzzle
    for seg in segs:
        if len(seg) <= SHORT:
            carry.extend(seg)
            continue
        if carry:
            m = pooled_mask(seg)
            seg = [fr for fr in carry if same_puzzle(play_mask(fr[1]["grid"]), m)] + seg
            carry = []
        if out and same_puzzle(pooled_mask(out[-1]), pooled_mask(seg)):
            out[-1] = out[-1] + seg          # one puzzle split by a cut-away
        else:
            out.append(seg)
    return out


def find_answer(seg, idx):
    """The one puzzle that fits the segment, or None."""
    votes = [Counter() for _ in B.SQUARES]
    letters = [Counter() for _ in B.SQUARES]   # confident letter reads per square
    for _, rd in seg:
        for j, (r, c) in enumerate(B.SQUARES):
            ch = rd["grid"][r][c]
            if ch != B.COVERED:
                votes[j][ch != B.OFF] += 1
            if ch in wc.IDX and rd["conf"][j] > SURE:
                letters[j][ch] += 1
    if any(not v for v in votes):
        return None
    key = tuple(v.most_common(1)[0][0] for v in votes)
    n_read = sum(1 for l in letters if l)
    if not n_read:
        return None
    best = {}
    for a, cat, lay in idx.get(key, []):
        g = B.place(lay, set(wc.LETTERS))     # everything shown
        score = sum(1 for j, (r, c) in enumerate(B.SQUARES)
                    if letters[j] and letters[j].most_common(1)[0][0] == g[r][c])
        if a not in best or score > best[a][0]:
            best[a] = (score, cat, g)
    ranked = sorted(best.items(), key=lambda kv: -kv[1][0])
    if not ranked:
        return None
    a, (score, cat, g) = ranked[0]
    runner = ranked[1][1][0] if len(ranked) > 1 else 0
    if score < MIN_AGREE * n_read or score - runner < MARGIN:
        return None
    return a, cat, g


_vocab = None


_counts = Counter()
PRIOR = 1.0         # weight on log(1 + times the word appears in puzzles.csv)


def vocab():
    """Words by shape: smart-solve's word list plus every puzzle word. Fills
    _counts with how often each word appears in the puzzles."""
    global _vocab
    if _vocab is None:
        words = set()
        for line in open(wc.WORDLIST):
            w = line.strip().upper()
            if w and set(w) <= set(wc.LETTERS + "'"):
                words.add(w)
        with open(wc.DATA / "puzzles.csv", newline="") as f:
            for r in csv.DictReader(f):
                ws = wc.word_keys(r["answer"])
                words.update(ws)
                _counts.update(ws)
        _vocab = defaultdict(list)
        for w in words:
            _vocab[wc.shape(w)].append(w)
    return _vocab


LETTER_IDX = [B.CLASS_IDX[c] for c in wc.LETTERS]
WORD_MARGIN = 3.0   # log-probability a spelled word must lead the next by
OOV_CONF = 0.97     # mean letter probability to accept a word not in the list


def spell_answer(seg):
    """The answer spelled from the frames, as a grid, or None. Needs every
    letter square to show a letter in at least one frame."""
    import numpy as np
    votes = [Counter() for _ in B.SQUARES]
    logp = np.zeros((len(B.SQUARES), 26))
    seen = np.zeros(len(B.SQUARES), dtype=int)
    punct = [Counter() for _ in B.SQUARES]
    for _, rd in seg:
        P = rd["probs"]
        for j, (r, c) in enumerate(B.SQUARES):
            ch = rd["grid"][r][c]
            if ch == B.COVERED:
                continue
            votes[j][ch != B.OFF] += 1
            if ch in B.PUNCT:
                punct[j][ch] += 1
            pl = P[j][LETTER_IDX]
            if pl.sum() > 0.5:                     # the square shows a letter here
                logp[j] += np.log(pl / pl.sum() + 1e-6)
                seen[j] += 1
    if any(not v for v in votes):
        return None
    play = [v.most_common(1)[0][0] for v in votes]
    g = B.blank_grid()
    for r in range(B.ROWS):
        run = []
        for c in list(range(B.COLS)) + [None]:
            j = B.SQUARES.index((r, c)) if c is not None and B.is_square(r, c) else None
            if j is not None and play[j]:
                run.append(j)
                continue
            if run:
                chars = {}
                letters = [k for k in run if not punct[k] or seen[k] > sum(punct[k].values())]
                for k in run:
                    if k not in letters:
                        chars[k] = punct[k].most_common(1)[0][0]
                if any(seen[k] == 0 for k in letters):
                    return None                    # a letter never showed
                key = "".join("'" if chars.get(k) == "'" else "_" for k in run if k in letters or chars.get(k) == "'")
                cands = vocab().get(key, [])
                best, second = None, -1e9
                for w in cands:
                    sc = sum(logp[k][wc.IDX[ch]] for k, ch in zip([k for k in run if k in letters], w.replace("'", "")))
                    sc += PRIOR * math.log1p(_counts[w])
                    if best is None or sc > best[0]:
                        best, second = (sc, w), (best[0] if best else -1e9)
                    elif sc > second:
                        second = sc
                argmax = "".join(wc.LETTERS[int(np.argmax(logp[k]))] for k in letters)
                mean_conf = float(np.mean([np.exp(logp[k].max() / seen[k]) for k in letters]))
                if best and best[0] - second >= WORD_MARGIN:
                    word = best[1].replace("'", "")
                elif mean_conf >= OOV_CONF:
                    word = argmax
                else:
                    return None
                it = iter(word)
                for k in run:
                    rr, cc = B.SQUARES[k]
                    g[rr][cc] = chars[k] if k in chars else next(it)
                run = []
    return g


def relabel(seg, answer_grid):
    """Labels for each frame of a segment from the answer's grid."""
    shown = set()                       # squares (by index) revealed so far
    out = []
    for k, rd in seg:
        for j, (r, c) in enumerate(B.SQUARES):
            ch = rd["grid"][r][c]
            if ch in wc.IDX and rd["conf"][j] > SURE and ch == answer_grid[r][c]:
                shown.add(j)
        g = B.blank_grid()
        diff = 0
        for j, (r, c) in enumerate(B.SQUARES):
            want, read = answer_grid[r][c], rd["grid"][r][c]
            if want == B.OFF:
                lab = B.OFF
            elif read == B.COVERED:
                lab = B.COVERED
            elif want not in wc.IDX:
                lab = want                                  # punctuation always shows
            elif read == B.LIT and j not in shown:
                lab = B.LIT
            else:
                lab = want if j in shown else B.HIDDEN
            g[r][c] = lab
            diff += lab != read
        out.append((k, B.grid_text(g), diff))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--clips", nargs="*")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--raw", default=str(RAW), help="where the clips' frames are")
    ap.add_argument("--out", default=str(OUT / "labels.jsonl"))
    a = ap.parse_args()
    from infer import Decoder
    dec = Decoder(a.ckpt, threads=a.threads)
    meta = {r["id"]: r for r in csv.DictReader(open(wc.DATA / "vision" / "clips.tsv"), delimiter="\t")}
    raw = Path(a.raw)
    clips = a.clips or sorted(p.parent.name for p in raw.glob("*/done"))
    t = time.time()
    idx = layout_index()
    print(f"layout index: {sum(len(v) for v in idx.values())} placements, {time.time() - t:.0f} s", flush=True)
    known = B.known_categories()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    with open(a.out, "a") as f:
        for clip in clips:
            frames = sorted((raw / clip / "frames").glob("*.jpg"))[::a.every]
            reads = list(enumerate(dec.read(frames, probs=True)))
            for seg in segments(reads):
                stats["segments"] += 1
                found = find_answer(seg, idx)
                if found:
                    answer, cat, g = found
                    stats["looked up"] += 1
                else:
                    g = spell_answer(seg) if len(seg) >= 3 else None
                    if g is None:
                        stats["no answer"] += 1
                        continue
                    answer, cat = B.to_board(g)[0], ""
                    stats["spelled"] += 1
                stats["puzzles"] += 1
                shown_cat = Counter(rd["category"] for _, rd in seg).most_common(1)[0][0]
                for k, grid, diff in relabel(seg, g):
                    stats["frames"] += 1
                    stats["review"] += diff > 0
                    f.write(json.dumps(dict(
                        clip=clip, frame=frames[k].name, source=meta.get(clip, {}).get("source", ""),
                        answer=answer, how="looked up" if found else "spelled",
                        category=cat, category_read=shown_cat,
                        category_mapped=B.category_from_strip(shown_cat, known),
                        grid=grid, review=diff, split="test" if in_control(answer) else "train")) + "\n")
            print(f"{clip}: {dict(stats)} {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
