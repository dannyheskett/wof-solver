#!/usr/bin/env python3
"""Phase 4. Wordplay category lists, built from the other lists. Run it last.

Sources: data/sources/{wordnet,wiktionary,wikidata}.tsv, data/train.csv, data/dev.csv,
         data/sources/raw/cmudict.dict (CMU Pronouncing Dictionary, BSD-style)
Output:  data/sources/wordplay.tsv

    Same Letter       phrases of 2+ words that all start with one letter
                      (CREEPY CRAWLY CREATURES)
    Rhyme Time        phrases of 2+ words where two different words of 3+
                      letters rhyme: same sounds from the last stressed vowel
                      on, per CMU, and neither word ends with the other
                      (HIKING AND BIKING)
    Before And After  phrase A whose last word starts phrase B, joined on
                      that word (RUBBER BAND + BAND OF BROTHERS ->
                      RUBBER BAND OF BROTHERS). Sampled, JOIN_LIMIT at most.
    Same Name         two phrases ending in the same word, joined as
                      "A1 & A2 WORD" (BOWLING BALL + DEBUTANTE BALL ->
                      BOWLING & DEBUTANTE BALL). Sampled, JOIN_LIMIT at most.

Joins only use phrases of 2-4 words, and the sampling is seeded, so every
run gives the same output.
"""
import csv
import random
import re
from collections import defaultdict

from src_common import DATA, RAW, SOURCES, write

JOIN_LIMIT = 20_000
SEED = "wof-wordplay-v1"
LISTS = ["wordnet", "wiktionary", "wikidata"]
# lines that are already two things joined (SONG BY ARTIST), not used as input
JOINED = {"Song Artist", "Title/Author", "Star And Role", "Husband And Wife"}
# a Same Name head can't end on one of these (THE CITY OF & ...)
FILLER = {"A", "AN", "THE", "OF", "IN", "ON", "AT", "TO", "FOR", "BY", "WITH", "AND", "&", "OR", "FROM",
          "MY", "YOUR", "HIS", "HER", "OUR", "THEIR", "ITS", "IS", "ARE", "ME"}


def load_phrases():
    phrases = set()
    for name in LISTS:
        path = SOURCES / f"{name}.tsv"
        if not path.exists():
            print(f"  missing {path.name}, skipped")
            continue
        for line in open(path):
            cat, phrase = line.rstrip("\n").split("\t")
            if cat not in JOINED:
                phrases.add(phrase)
    for name in ("train.csv", "dev.csv"):
        with open(DATA / name, newline="") as f:
            phrases.update(r["answer"] for r in csv.DictReader(f))
    return phrases


def load_rhymes():
    rhyme = {}
    for line in open(RAW / "cmudict.dict", encoding="latin-1"):
        word, *phones = line.split("#")[0].split()
        word = re.sub(r"\(\d+\)$", "", word).upper()
        stressed = [i for i, p in enumerate(phones) if p[-1] in "12"]
        if stressed and word not in rhyme:
            rhyme[word] = " ".join(p.rstrip("012") for p in phones[stressed[-1]:])
    return rhyme


def words_of(p):
    return [w for w in re.split(r"[ \-]", p) if w and w not in ("&", "AND", "N", "'N")]


def main():
    phrases = load_phrases()
    rhyme = load_rhymes()
    print(f"{len(phrases):,} phrases, {len(rhyme):,} pronunciations")
    pairs = []

    for p in phrases:
        ws = words_of(p)
        if len(ws) >= 2 and len({w[0] for w in ws}) == 1 and all(w[0].isalpha() for w in ws):
            pairs.append(("Same Letter", p))
        content = [re.sub(r"[^A-Z]", "", re.sub(r"'S?$", "", w)) for w in ws]
        content = [w for w in content if len(w) >= 3 and w in rhyme]
        done = False
        for i, a in enumerate(content):
            for b in content[i + 1:]:
                if (a != b and rhyme[a] == rhyme[b]
                        and not a.endswith(b) and not b.endswith(a)):
                    pairs.append(("Rhyme Time", p))
                    done = True
                    break
            if done:
                break

    short = sorted(p for p in phrases if 2 <= len(p.split()) <= 4 and "&" not in p)
    by_first = defaultdict(list)
    by_last = defaultdict(list)
    for p in short:
        ws = p.split()
        by_first[ws[0]].append(p)
        by_last[ws[-1]].append(p)
    rng = random.Random(SEED)

    joins = []
    for p in short:
        last = p.split()[-1]
        for q in by_first.get(last, []):
            if q != p:
                joins.append(p + q[len(last):])
    rng.shuffle(joins)
    pairs += [("Before And After", j) for j in joins[:JOIN_LIMIT]]

    same = []
    for word, group in by_last.items():
        if len(group) < 2:
            continue
        heads = [g[: -len(word)].strip() for g in group]
        heads = [h for h in heads if h.split()[-1] not in FILLER]
        if len(heads) < 2:
            continue
        for _ in range(min(len(group), 6)):
            a, b = rng.sample(heads, 2)
            if a and b and a != b:
                same.append(f"{a} & {b} {word}")
    rng.shuffle(same)
    pairs += [("Same Name", s) for s in same[:JOIN_LIMIT]]
    print(f"  candidates: Before And After {len(joins):,}, Same Name {len(same):,}")
    write("wordplay", pairs)


if __name__ == "__main__":
    main()
