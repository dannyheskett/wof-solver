#!/usr/bin/env python3
"""Phase 13. Write data/demo.json, the one data file behind the demo endpoints of the
move API (/categories, /puzzle, /presets).

    categories  every category in the training puzzles, in order of how
                many there are, with "auto": true when the control set has
                a puzzle in it (Auto mode only draws from those)
    puzzles     the control set: puzzles neither solver was built from
    presets     hand-picked control puzzles where transformer-solve beats
                smart-solve, the first one featured. They are wins, not
                typical games: over the control set smart-solve wins most.

    python3 scripts/13_build_demo.py
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
import wof_common as wc  # noqa: E402

PRESETS = [
    ("bonus", "Phrase", "YOU CRACK ME UP",
     "Same four letters picked, two-thirds of the board blank. smart-solve "
     "says YOU CRAZY ME UP."),
    ("main", "Thing", "CELEBRITY HEADSHOT",
     "smart-solve's word list has no HEADSHOT. It solves CELEBRITY HELMSMAN."),
    ("main", "Thing", "ROLLERBALL PENS",
     "smart-solve's word list has no ROLLERBALL. It solves VOLLEYBALL PENS."),
    ("bonus", "Phrase", "LIGHTNING IN A BOTTLE",
     "Same letters picked. smart-solve says LIGHTNING IN A BATTLE."),
    ("bonus", "Phrase", "FREE SHIPPING AND HANDLING",
     "Same letters picked. smart-solve says FREE SHOPPING AND HANDLING."),
    ("bonus", "Thing", "TRAVEL MUG",
     "Same letters picked. smart-solve says TRAVEL FOG."),
    ("bonus", "Food And Drink", "GOOD CUP OF COFFEE",
     "Same letters picked. smart-solve says FOOD CUP OF COFFEE."),
]


def main():
    train = Counter(cat for _, cat in wc.read_all(wc.SMART_FILES))
    control = wc.read_puzzles("control.csv")
    in_control = {cat for _, cat in control}
    known = {(sol, cat) for sol, cat in control}
    for mode, cat, sol, _ in PRESETS:
        assert (sol, cat) in known, f"preset not in the control set: {cat} / {sol}"
    out = {
        "categories": [{"name": c, "auto": c in in_control} for c, _ in train.most_common()],
        "puzzles": [[cat, sol] for sol, cat in control],
        "presets": [{"mode": m, "category": c, "answer": a, "note": n} for m, c, a, n in PRESETS],
    }
    path = wc.DATA / "demo.json"
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"{path}: {len(out['categories'])} categories, {len(out['puzzles'])} puzzles, "
          f"{len(out['presets'])} presets, {path.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
