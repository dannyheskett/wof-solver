"""The puzzle board as the vision phase sees it: 52 squares on a 14x4 grid.

Grid
    Rows of 12/14/14/12 squares (wof_common.BOARD_ROWS). The two 12-square
    rows sit one column in from each side, so every square has a (row, col)
    on a 14x4 grid and the four corner positions are not squares. The board's
    corners, as the corner model finds them, are the outer corners of that
    14x4 grid. A grid column is PITCH_W wide and a row PITCH_H tall; on the
    show a square is about 1.17 times as tall as it is wide.

Square classes
    '#' unused (green), '_' in play and hidden (white), '*' lit (blue: the
    letter was just called and not yet touched), a letter or punctuation
    shown, and '~' covered (more than half hidden behind the host or out of
    frame). A label grid is four strings of 14 characters, with ' ' at the
    four corner positions.

Layout
    The show starts every line of a puzzle in the same column and centres the
    block. layouts() lists the ways a puzzle can sit on the board that way,
    and to_board() turns a grid of squares back into the solvers' board text.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import wof_common as wc  # noqa: E402

ROWS, COLS = 4, 14
PITCH_W, PITCH_H = 32, 38            # canonical pixels per grid column and row
CANON_W, CANON_H = COLS * PITCH_W, ROWS * PITCH_H

PUNCT = "-&'.!?,"
CLASSES = "#_*" + wc.LETTERS + PUNCT + "~"
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}
OFF, HIDDEN, LIT, COVERED = "#", "_", "*", "~"


def is_square(r, c):
    return 0 <= c < COLS and (r in (1, 2) or 1 <= c <= COLS - 2)


SQUARES = [(r, c) for r in range(ROWS) for c in range(COLS) if is_square(r, c)]
assert len(SQUARES) == 52


def blank_grid():
    return [[OFF if is_square(r, c) else " " for c in range(COLS)] for r in range(ROWS)]


def grid_text(grid):
    return ["".join(row) for row in grid]


def _splits(ps, k):
    """Every way to cut the pieces into k non-empty lines, in order."""
    if k == 1:
        yield [ps]
        return
    for i in range(1, len(ps) - k + 2):
        for rest in _splits(ps[i:], k - 1):
            yield [ps[:i]] + rest


def _line_text(line):
    out = ""
    for text, joined in line:
        out += text if (not out or joined) else " " + text
    return out


# Rows a block of k lines may use, middle rows first
ROW_PLANS = {1: [(1,), (2,)], 2: [(1, 2), (0, 1), (2, 3)],
             3: [(0, 1, 2), (1, 2, 3)], 4: [(0, 1, 2, 3)]}


LINE_COST = 3   # an extra line is worth this many squares off the longest line


def layout_score(layout):
    """Lower is more show-like. The show trades lines against line length:
    MARGARITA / MOJITO / MIMOSA / MUDSLIDE, PUT DOWN / THE PHONE and
    TAKING A / BREAK TO / SOCIALIZE all score lowest of their placements."""
    widest = max(len(t) for _, t in layout.values())
    middle = sum(r in (1, 2) for r in layout)
    return widest + LINE_COST * (len(layout) - 1) - 0.5 * middle


def layouts(answer, all_starts=False):
    """Every show-style placement of the answer as {row: (start col, text)},
    most show-like first (layout_score). The block is centred; all_starts
    also gives every other start column that fits, for matching real boards
    whose centring differs by a column."""
    ps = wc.pieces(answer)
    out = []
    for k in range(1, min(4, len(ps)) + 1):
        for split in _splits(ps, k):
            lines = [_line_text(l) for l in split]
            widest = max(len(t) for t in lines)
            for rows in ROW_PLANS[k]:
                for start in sorted(range(COLS), key=lambda s: abs(2 * s + widest - COLS)):
                    if all(is_square(r, start) and is_square(r, start + len(t) - 1)
                           for r, t in zip(rows, lines)):
                        out.append(dict(zip(rows, ((start, t) for t in lines))))
                        if not all_starts:
                            break
    out.sort(key=layout_score)
    return out


def place(layout, called):
    """The label grid of a placed answer with the called letters shown."""
    g = blank_grid()
    for r, (start, text) in layout.items():
        for k, ch in enumerate(text):
            if ch == " ":
                continue
            g[r][start + k] = HIDDEN if (ch in wc.IDX and ch not in called) else ch
    return g


def to_board(grid):
    """The solvers' board text from a grid of square classes, plus a list of
    the squares that had to be guessed. Lit and covered squares in play count
    as hidden. A covered square is taken as in play when squares in play lie
    on both sides of it in its row, or when it runs left from its row's
    first square in play no further than the column where the other rows
    start (the show starts every line in the same column)."""
    lines, guessed = [], []
    firsts = [min((c for c in range(COLS) if is_square(r, c) and grid[r][c] not in (OFF, COVERED)), default=None)
              for r in range(ROWS)]
    start = min((f for f in firsts if f is not None), default=COLS)
    for r in range(ROWS):
        row = [grid[r][c] for c in range(COLS) if is_square(r, c)]
        cols = [c for c in range(COLS) if is_square(r, c)]
        play = [ch not in (OFF, COVERED) for ch in row]
        lead = set()                     # covered squares that start the line
        if firsts[r] is not None:
            i = cols.index(firsts[r]) - 1
            while i >= 0 and row[i] == COVERED and cols[i] >= start:
                lead.add(i)
                i -= 1
        cells = []
        for i, ch in enumerate(row):
            if ch == COVERED:
                inside = (any(play[:i]) and any(play[i + 1:])) or i in lead
                if inside:
                    guessed.append((r, cols[i]))
                cells.append(HIDDEN if inside else OFF)
            elif ch == LIT:
                guessed.append((r, cols[i]))
                cells.append(HIDDEN)
            else:
                cells.append(ch)
        text = re.sub(r"#+", " ", "".join(cells)).strip()
        if text:
            lines.append(text)
    board = ""
    for line in lines:
        board += line if (not board or board.endswith("-")) else " " + line
    return board, guessed


def shown_letters(board):
    return "".join(sorted(set(ch for ch in board if ch in wc.IDX)))


# --- categories -----------------------------------------------------------

def known_categories():
    names = set()
    for name in ("train.csv", "dev.csv"):
        names.update(cat for _, cat in wc.read_puzzles(name))
    return sorted(n for n in names if n and n[0].isalpha() and n == n.strip())


def category_from_strip(text, known):
    """Map the category as written on screen (FOOD & DRINK, LIVING THINGS,
    WHAT ARE YOU DOING?) to the dataset's name for it (Food And Drink, Living
    Thing, What Are You Doing): an exact match, the same without a final S,
    or the closest name, it or its singular at most 25% different. A name
    the dataset lacks comes back title-cased as written, and the solvers
    then weigh its words equally. server/src/main.c map_category is the same."""
    t = re.sub(r"[^A-Z0-9& ']", "", text.upper()).replace("&", " AND ")
    t = " ".join(t.split()).title().replace("'S", "'s")
    if not t:
        return ""
    by_lower = {k.lower(): k for k in known}
    for cand in (t, t[:-1] if t.endswith("s") else None):
        if cand and cand.lower() in by_lower:
            return by_lower[cand.lower()]
    lo = t.lower()
    forms = [lo, lo[:-1]] if lo.endswith("s") and len(lo) > 1 else [lo]
    best, best_d = None, None
    for name in by_lower:
        d = min(_edits(f, name) for f in forms)
        if best_d is None or d < best_d:
            best, best_d = name, d
    if best is not None and best_d * 100 <= 25 * max(len(lo), len(best)):
        return by_lower[best]
    return t


# Categories the show uses that the puzzle data does not have (seen on screen)
SHOW_EXTRA = ["TRAVEL TIPS", "OPPOSITES ATTRACT", "CHARACTER", "CHARACTERS", "FICTIONAL PLACE",
              "TV TITLE", "MOVIE QUOTE", "BEST SELLER", "COMPOUND PHRASE", "STAR & ROLE"]


def show_categories():
    """Every category as the show can write it: each dataset name with &
    for AND, singular and plural, the ? on the question ones, and
    SHOW_EXTRA."""
    out = set(SHOW_EXTRA)
    for name in known_categories():
        t = name.upper().replace(" AND ", " & ")
        out.update({t, t + "S"})
        if t.startswith(("WHAT ", "WHERE ")):
            out.add(t + "?")
    return sorted(out)


def snap_category(read, choices, max_ratio=0.35):
    """The show category closest to a (noisy) read, or "" when none is
    within max_ratio edits per character."""
    read = read.strip().upper()
    if not read:
        return ""
    best, best_d = "", 1e9
    for c in choices:
        d = _edits(read, c)
        if d < best_d:
            best, best_d = c, d
    return best if best_d <= max_ratio * max(len(read), len(best)) else ""


def _edits(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# The category as the show writes it, for rendering
def show_category(name, rng):
    t = name.upper().replace(" AND ", " & ")
    if t in ("WHAT ARE YOU DOING", "WHERE ARE WE GOING", "WHAT ARE YOU WEARING"):
        t += "?"
    if t in ("THING", "PLACE", "LIVING THING", "EVENT", "FICTIONAL CHARACTER",
             "LANDMARK", "OCCUPATION", "PROPER NAME", "TITLE") and rng.random() < 0.4:
        t += "S"
    return t
