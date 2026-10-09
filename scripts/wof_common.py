"""Shared pieces for the MAIN and BONUS dataset builders.

Board words and the word list
    A board is split into words on spaces and hyphens, and each word is kept
    as its letters and apostrophes (other punctuation always shows and says
    nothing about the letters). The vocabulary is the SCOWL word list in
    data/wordlist/ plus every word in the puzzle files passed to Vocab, so
    every solution word is in it. Words are grouped by shape (length and
    apostrophe positions), and each group holds a matrix of letter position
    bitmasks: P[i, c] has bit k set when word i has letter c at position k.

Candidates
    A board word's candidates are the vocabulary words of its shape whose
    letters match every called letter exactly: in the shown squares and
    nowhere else. Called letters that missed must not appear at all.

Weights
    Each candidate weighs 1 plus the number of times that word appears in
    the training puzzles of the board's category. This is how the category
    steers the picks. Weights come from the training puzzles only
    (train.csv for the training sets), so the control and dev sets never
    shape a label. smart-solve itself weighs words by train.csv and dev.csv
    together (SMART_FILES).

Pinned and smart picks
    A board word is pinned when it has one candidate. The board is solved
    when every word is pinned. The smart pick is the letter that leaves the
    lowest expected log2(candidates left), summed over the words, with each
    outcome weighted as above. Ties go to the more common letter.

Money rule
    On the show a vowel costs $250, and money only comes from consonants
    that hit. The games here keep no money, so the rule is read from the
    board: a vowel can be bought once a called consonant shows on it (or
    once every consonant is called). Before that the smart pick is a
    consonant.

Bonus set pick
    The best set is the 3 consonants and 1 vowel that give the highest
    chance that filling each board word with its heaviest candidate is
    right, with candidates weighted as above and words taken as
    independent. Ties go to the more common letters.
"""
import csv
import hashlib
import itertools
import math
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LETTERS = string.ascii_uppercase
IDX = {c: i for i, c in enumerate(LETTERS)}
VOWELS = set("AEIOU")
CONSONANTS = [c for c in LETTERS if c not in VOWELS]
WORDLIST = DATA / "wordlist" / "american-english.txt"
CACHE_MIN = 2000      # cache letter scores for words with at least this many candidates
CACHE_MAX = 300_000   # entries before the cache is cleared
# What smart-solve learns from: its word list and category weights
SMART_FILES = ["train.csv", "dev.csv"]

# The TV board: rows of 12/14/14/12 squares, in use since 1997
BOARD_ROWS = (12, 14, 14, 12)
BOARD_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ -&'.!?,")


def pieces(answer):
    """Split into (text, joined) pieces. A row can break between words or
    right after a hyphen. joined=True means the piece continues the previous
    one with no space, e.g. BLACK-AND-WHITE -> BLACK- AND- WHITE."""
    out = []
    for word in answer.split():
        parts = word.split("-")
        for k, part in enumerate(parts):
            text = part + ("-" if k < len(parts) - 1 else "")
            if text:
                out.append((text, k > 0))
    return out


def fits_board(answer):
    """Greedy wrap. Filling each row as far as it goes is optimal when the
    rows are fixed and in order, so a False here means no layout fits."""
    ps = pieces(answer)
    i = 0
    for width in BOARD_ROWS:
        used = 0
        while i < len(ps):
            text, joined = ps[i]
            need = len(text) + (1 if used and not joined else 0)
            if used + need > width:
                break
            used += need
            i += 1
    return i == len(ps)


def read_puzzles(name):
    with open(DATA / name, newline="") as f:
        return [(r["answer"], r["category"]) for r in csv.DictReader(f)]


def board(solution, called):
    return "".join("_" if ch in IDX and ch not in called else ch for ch in solution)


def can_buy(board, called):
    """The money rule: True once a called consonant shows on the board, or
    once every consonant is called."""
    cons = [c for c in called if c not in VOWELS]
    return len(cons) == len(CONSONANTS) or any(c in board for c in cons)


def word_keys(text):
    """Board words of a solution or board, as (key, start) pairs. The key is
    the word's letters, apostrophes and '_' squares."""
    out = []
    for m in re.finditer(r"[^ \-]+", text):
        key = re.sub(r"[^A-Z'_]", "", m.group())
        if any(ch != "'" for ch in key):
            out.append(key)
    return out


def shape(key):
    return "".join("'" if ch == "'" else "_" for ch in key)


def posmasks(word):
    row = [0] * 26
    for k, ch in enumerate(word):
        if ch in IDX:
            row[IDX[ch]] |= 1 << k
    return row


def seed_for(*parts):
    return hashlib.sha1("|".join(parts).encode()).digest()


def letter_order(puzzles):
    """All 26 letters, most common in the puzzles first."""
    freq = Counter(ch for sol, _ in puzzles for ch in sol if ch in IDX)
    return sorted(LETTERS, key=lambda c: (-freq[c], c))


def read_all(names):
    """read_puzzles over several files, in order."""
    return [p for name in names for p in read_puzzles(name)]


class Vocab:
    def __init__(self, vocab_files, weight_files=SMART_FILES):
        words = set()
        for line in open(WORDLIST):
            w = line.strip().upper()
            if re.fullmatch(r"[A-Z']+", w) and any(ch != "'" for ch in w):
                words.add(w)
        for name in vocab_files:
            for sol, _ in read_puzzles(name):
                words.update(word_keys(sol))
        groups = defaultdict(list)
        for w in words:
            groups[shape(w)].append(w)
        self.words, self.index, self.P = {}, {}, {}
        for sh, ws in groups.items():
            ws.sort()
            self.words[sh] = ws
            self.index[sh] = {w: i for i, w in enumerate(ws)}
            self.P[sh] = np.array([posmasks(w) for w in ws], dtype=np.int32)
        self.counts = defaultdict(Counter)
        for sol, cat in read_all(weight_files):
            self.counts[cat].update(word_keys(sol))
        self._weights = {}
        self.cache = {}

    def weights(self, category, sh):
        key = (category, sh)
        if key not in self._weights:
            counts = self.counts[category]
            self._weights[key] = np.array(
                [1.0 + counts.get(w, 0) for w in self.words[sh]])
        return self._weights[key]


class BoardWords:
    """Candidate sets for every word of one puzzle, narrowed call by call."""

    def __init__(self, vocab, solution, category):
        self.v, self.category = vocab, category
        self.items = []
        for key in word_keys(solution):
            sh = shape(key)
            row = vocab.index[sh][key]
            self.items.append({
                "key": key, "shape": sh, "sol": vocab.P[sh][row],
                "idx": np.arange(len(vocab.words[sh])),
            })

    def copy(self):
        new = object.__new__(BoardWords)
        new.v, new.category = self.v, self.category
        new.items = [dict(it) for it in self.items]
        return new

    def call(self, letter):
        c = IDX[letter]
        for it in self.items:
            P = self.v.P[it["shape"]]
            it["idx"] = it["idx"][P[it["idx"], c] == it["sol"][c]]
            assert len(it["idx"]) >= 1

    def solved(self):
        return all(len(it["idx"]) == 1 for it in self.items)

    def pattern(self, it, called):
        return "".join(ch if ch == "'" or ch in called else "_" for ch in it["key"])

    def word_scores(self, it, called, letters):
        """Expected log2(candidates left) for this word after each letter."""
        idx = it["idx"]
        if len(idx) == 1:
            return dict.fromkeys(letters, 0.0)
        ckey = None
        if len(idx) >= CACHE_MIN:
            ckey = (self.category, it["shape"], self.pattern(it, called),
                    "".join(sorted(called)), "".join(letters))
            hit = self.v.cache.get(ckey)
            if hit is not None:
                return hit
        P = self.v.P[it["shape"]][idx]
        w = self.v.weights(self.category, it["shape"])[idx]
        total = w.sum()
        out = {}
        for letter in letters:
            _, inv, n = np.unique(P[:, IDX[letter]], return_inverse=True,
                                  return_counts=True)
            W = np.bincount(inv, weights=w)
            out[letter] = float((W / total * np.log2(n)).sum())
        if ckey is not None:
            if len(self.v.cache) > CACHE_MAX:
                self.v.cache.clear()
            self.v.cache[ckey] = out
        return out

    def smart_pick(self, called, order, vowels=True):
        """vowels=False leaves the vowels out, for when none can be bought."""
        letters = [c for c in order if c not in called and (vowels or c not in VOWELS)]
        totals = dict.fromkeys(letters, 0.0)
        for it in self.items:
            if len(it["idx"]) > 1:
                for c, s in self.word_scores(it, called, letters).items():
                    totals[c] += s
        rank = {c: i for i, c in enumerate(order)}
        return min(letters, key=lambda c: (round(totals[c], 9), rank[c]))

    def set_pick(self, allowed_consonants, allowed_vowels, n_cons, n_vow, order):
        """Greedy choice of n_cons consonants and n_vow vowels named together,
        with no feedback between them: each step adds the letter that most
        lowers expected log2(candidates left) over the joint outcome of the
        letters chosen so far."""
        rank = {c: i for i, c in enumerate(order)}
        live = [it for it in self.items if len(it["idx"]) > 1]
        labels = [np.zeros(len(it["idx"]), dtype=np.int64) for it in live]
        Ps = [self.v.P[it["shape"]][it["idx"]] for it in live]
        ws = [self.v.weights(self.category, it["shape"])[it["idx"]] for it in live]
        chosen = []
        need = {"c": n_cons, "v": n_vow}
        while need["c"] or need["v"]:
            pool = ([c for c in allowed_consonants if c not in chosen] if need["c"] else []) + \
                   ([c for c in allowed_vowels if c not in chosen] if need["v"] else [])
            best = None
            for letter in pool:
                score = 0.0
                for lab, P, w in zip(labels, Ps, ws):
                    joint = lab * 16384 + P[:, IDX[letter]]
                    _, inv, n = np.unique(joint, return_inverse=True, return_counts=True)
                    W = np.bincount(inv, weights=w)
                    score += float((W / w.sum() * np.log2(n)).sum())
                key = (round(score, 9), rank[letter])
                if best is None or key < best[0]:
                    best = (key, letter)
            letter = best[1]
            chosen.append(letter)
            need["v" if letter in VOWELS else "c"] -= 1
            for k, P in enumerate(Ps):
                joint = labels[k] * 16384 + P[:, IDX[letter]]
                labels[k] = np.unique(joint, return_inverse=True)[1].astype(np.int64)
        return chosen

    def best_set(self, consonants, vowels, n_cons, n_vow, order):
        """The n_cons consonants and n_vow vowels, named together, that give
        the highest chance that the heaviest candidate of every word is
        right once they show. Returns (letters, chance)."""
        rank = {c: i for i, c in enumerate(order)}
        letters = list(consonants) + list(vowels)
        live = []
        for it in self.items:
            if len(it["idx"]) < 2:
                continue
            w = self.v.weights(self.category, it["shape"])[it["idx"]]
            # heaviest first, stable, so the first of a group is what fill() would pick
            o = np.argsort(-w, kind="stable")
            P = self.v.P[it["shape"]][it["idx"][o]]
            codes = {}
            for c in letters:
                inv = np.unique(P[:, IDX[c]], return_inverse=True)[1].astype(np.int64)
                codes[c] = (inv, int(inv.max()) + 1)
            live.append((codes, w[o], float(w.sum())))
        best = None
        for cs in itertools.combinations(consonants, n_cons):
            for vs in itertools.combinations(vowels, n_vow):
                chosen = cs + vs
                tie = sorted(rank[c] for c in chosen)
                p = 1.0
                for codes, w, total in live:
                    key = np.zeros(len(w), dtype=np.int64)
                    for c in chosen:
                        inv, n = codes[c]
                        key = key * n + inv
                    first = np.unique(key, return_index=True)[1]
                    p *= w[first].sum() / total
                    if best is not None and p < best[0][0] - 1e-12:
                        break
                k = (round(p, 12), [-r for r in tie])
                if best is None or k > best[0]:
                    best = (k, chosen)
        return list(best[1]), best[0][0]


def move(letter):
    return f"{'BUY' if letter in VOWELS else 'CALL'} {letter}"
