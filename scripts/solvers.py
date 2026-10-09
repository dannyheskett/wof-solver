"""The two solvers, as 11_evaluate.py plays them. Each one sees only what a
contestant sees: the category, the board, and the letters called so far.

Interface, one object per worker process:

    new_puzzle(seed)                     before each puzzle
    main(category, board, called)        -> "CALL X", "BUY X" or "SOLVE <text>"
    bonus_pick(category, board, called)  -> "PICK BCD A" (3 consonants, 1 vowel)
    bonus_solve(category, board, called) -> "SOLVE <text>"

board shows hidden letters as '_'. called is the letters called so far,
sorted A-Z, as in the training lines.

smart-solve (SmartSolver)
    Candidates for each board word come from the board alone: the word list
    words of its shape that match every called letter exactly (shown where
    it shows, absent everywhere else). The word list is SCOWL plus the words
    of train.csv and dev.csv, weighted by how often each word appears in
    those puzzles of the category, so no control puzzle is in it. A control
    word missing from the list has no candidates, or only wrong ones, and
    can only be filled by calling its letters.

    MAIN: SOLVE when every hidden word has one candidate, else call the
    smart pick (wof_common.BoardWords.smart_pick), keeping to the money rule
    (no vowel until a called consonant shows). BONUS: pick the best set
    (BoardWords.best_set), then fill each hidden word with its heaviest
    candidate.

transformer-solve (TransformerSolver), the 59M-parameter transformer
    Each move is the greedy completion of the training-format prompt, e.g.
        MAIN|THING|A _A__L_ __ ___ ______|AL|  ->  BUY E
    Masks hold it to the rules and the board: BUY is masked out until a
    vowel can be bought (money rule), after CALL or BUY a letter already
    called is masked out, and after SOLVE each character must fit its
    square (shown letters, spaces and punctuation copied, an uncalled letter
    in each blank, ending where the board ends), so every word fills its
    slot exactly. "masked" and "fitted" in the results count the characters
    each mask changed. Anything else it writes goes to the harness as is,
    so malformed moves are invalid.

    A MAIN SOLVE is only accepted when the transformer gives its answer at
    least probability min_p (0.99, tuned by 10_tune_solve.py on dev.csv) or
    no blank is left. Otherwise it makes its best letter move: the first
    character is held to C, or B when a vowel can be bought. "rejected"
    counts those.
"""
import math
import re
import sys

import numpy as np

import wof_common as wc

FREE = set("RSTLNE")


def check_move(mv, board, called):
    """The letter of a valid CALL or BUY, or None: a letter called twice, a
    consonant bought, a vowel called, or a vowel bought before a called
    consonant shows (the money rule) is invalid."""
    verb, _, letter = mv.partition(" ")
    if len(letter) != 1 or letter not in wc.IDX or letter in called:
        return None
    if (verb == "BUY") != (letter in wc.VOWELS) or verb not in ("BUY", "CALL"):
        return None
    if verb == "BUY" and not wc.can_buy(board, called):
        return None
    return letter


def one_thread():
    """Each worker is one process on one core: no extra torch threads."""
    if "torch" in sys.modules:
        sys.modules["torch"].set_num_threads(1)


class SmartSolver:
    def __init__(self, vocab=None):
        """vocab: a prebuilt wc.Vocab(wc.SMART_FILES), to skip building it."""
        self.v = vocab or wc.Vocab(wc.SMART_FILES)
        self.order = wc.letter_order(wc.read_all(wc.SMART_FILES))

    def new_puzzle(self, seed):
        pass

    def view(self, category, board, called):
        """The board's hidden words with their candidates, as a BoardWords
        for the pickers plus (blank positions, item) per hidden word."""
        bw = object.__new__(wc.BoardWords)
        bw.v, bw.category, bw.items = self.v, category, []
        cols = [wc.IDX[c] for c in called]
        spans = []
        for m in re.finditer(r"[^ \-]+", board):
            blanks = [m.start() + k for k, ch in enumerate(m.group()) if ch == "_"]
            if not blanks:
                continue
            key = re.sub(r"[^A-Z'_]", "", m.group())
            sh = wc.shape(key)
            P = self.v.P.get(sh)
            if P is None:
                idx = np.arange(0)
            elif cols:
                want = np.array(wc.posmasks(key), dtype=np.int32)[cols]
                idx = np.nonzero((P[:, cols] == want).all(axis=1))[0]
            else:
                idx = np.arange(len(P))
            it = {"key": key, "shape": sh, "idx": idx}
            if len(idx):
                bw.items.append(it)
            spans.append((blanks, it))
        return bw, spans

    def fill(self, board, category, spans):
        out = list(board)
        for blanks, it in spans:
            idx = it["idx"]
            if not len(idx):
                continue
            w = self.v.weights(category, it["shape"])[idx]
            word = self.v.words[it["shape"]][idx[int(np.argmax(w))]]
            letters = [word[k] for k, ch in enumerate(it["key"]) if ch == "_"]
            for pos, ch in zip(blanks, letters):
                out[pos] = ch
        return "".join(out)

    def main(self, category, board, called):
        bw, spans = self.view(category, board, called)
        if all(len(it["idx"]) == 1 for _, it in spans):
            return "SOLVE " + self.fill(board, category, spans)
        vowels = wc.can_buy(board, called)
        return wc.move(bw.smart_pick(set(called), self.order, vowels))

    def bonus_pick(self, category, board, called):
        cons = [c for c in self.order if c in wc.CONSONANTS and c not in called]
        vows = [c for c in self.order if c in wc.VOWELS and c not in called]
        bw, _ = self.view(category, board, called)
        picked, _ = bw.best_set(cons, vows, 3, 1, self.order)
        c = "".join(sorted(x for x in picked if x not in wc.VOWELS))
        v = "".join(x for x in picked if x in wc.VOWELS)
        return f"PICK {c} {v}"

    def bonus_solve(self, category, board, called):
        _, spans = self.view(category, board, called)
        return "SOLVE " + self.fill(board, category, spans)


class TransformerSolver:
    def __init__(self, path, min_p=0.99):
        self.min_p = min_p
        import torch
        from model import GPT, Config
        torch.set_num_threads(1)
        ck = torch.load(path, map_location="cpu")
        self.alphabet = ck["alphabet"]
        self.model = GPT(Config(**ck["config"]))
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.stop = self.alphabet.index("\n")

    def new_puzzle(self, seed):
        self.notes = {"masked": 0, "fitted": 0, "rejected": 0}
        self.attempts = []

    def ask(self, mode, category, board, called, ban=None):
        prompt = f"{mode}|{category.upper()}|{board}|{called or '-'}|"
        ids = [self.alphabet.index(ch) for ch in prompt]
        new = self.model.complete([ids], self.stop, 80, ban)[0]
        out = "".join(self.alphabet[i] for i in new)
        key = "fitted" if out.startswith("SOLVE ") else "masked"
        self.notes[key] += self.model.banned[0]
        # confidence of a SOLVE: probability of the answer after "SOLVE "
        self.last_p = math.exp(sum(self.model.logp[0][len("SOLVE "):]))
        return out

    def rules(self, board, called):
        """The ban function for one move. BUY is banned as the first word
        until a vowel can be bought. After CALL or BUY a called letter
        is banned. After SOLVE the answer must fit the board square by
        square: shown letters, spaces and punctuation are copied, a blank
        takes only an uncalled letter, and the answer ends with the board."""
        a = self.alphabet
        everything = set(range(len(a)))
        repeat = [a.index(c) for c in called]
        verbs = [[a.index(ch) for ch in v] for v in ("CALL ", "BUY ")]
        solve = [a.index(ch) for ch in "SOLVE "]
        blank = {a.index(c) for c in wc.LETTERS if c not in called}
        allowed = [blank if ch == "_" else {a.index(ch)} for ch in board] + [{self.stop}]
        no_buy = [] if wc.can_buy(board, called) else [a.index("B")]

        def ban(_, new):
            if not new:
                return no_buy
            if new in verbs:
                return repeat
            if len(new) >= len(solve) and new[:len(solve)] == solve:
                k = len(new) - len(solve)
                return list(everything - allowed[k]) if k < len(allowed) else []
            return ()
        return ban

    def main(self, category, board, called):
        ban = self.rules(board, called)
        out = self.ask("MAIN", category, board, called, ban)
        if out.startswith("SOLVE ") and "_" in board:
            self.attempts.append((out[6:], self.last_p, len(called)))
            if self.last_p < self.min_p:
                # not sure enough: take the best letter move instead
                self.notes["rejected"] += 1
                # the move must start CALL (or BUY, when a vowel can be bought)
                a = self.alphabet
                ok = {a.index("C")} | ({a.index("B")} if wc.can_buy(board, called) else set())
                not_move = [k for k in range(len(a)) if k not in ok]
                out = self.ask("MAIN", category, board, called,
                               lambda i, new: not_move if not new else ban(i, new))
        return out

    def bonus_pick(self, category, board, called):
        return self.ask("BONUS", category, board, called)

    def bonus_solve(self, category, board, called):
        return self.ask("BONUS", category, board, called, self.rules(board, called))


def make(name):
    """smart-solve, or transformer-solve:<checkpoint>[,p=<min_p>]."""
    if name == "smart-solve":
        return SmartSolver()
    if name.startswith("transformer-solve:"):
        path, *opts = name[len("transformer-solve:"):].split(",")
        opts = dict(o.split("=") for o in opts)
        return TransformerSolver(path, float(opts.get("p", 0.99)))
    sys.exit(f"unknown solver {name!r}: smart-solve or transformer-solve:<checkpoint>")
