#!/usr/bin/env python3
"""Phase 4. Everyday-noun category lists from Open English WordNet 2025 (CC BY 4.0).

Source: data/sources/raw/english-wordnet-2025.xml.gz
Output: data/sources/wordnet.tsv

Every noun synset has a lexfile (noun.food, noun.artifact, ...), and
hypernym links that form a tree. Categories come from whole lexfiles, or
from everything below a root synset ("furniture", "kitchen utensil"):

    Food And Drink      lexfile noun.food
    Living Thing        lexfiles noun.animal, noun.plant
    Thing               lexfiles noun.artifact, noun.object
    Person              lexfile noun.person
    People              plurals of the Person lemmas (simple English rules)
    Occupation          below worker, professional, skilled worker, ...
    Around The House    below furniture, home appliance, furnishing, ...
    In The Kitchen      below kitchen utensil, cookware, tableware, ...
    Fun And Games       below toy, game, sport, outdoor game, ...
    Event               below social event, ceremony, celebration, contest, ...

Lemmas that start with a capital letter (proper names, Latin genus names)
are skipped, as are lemmas naming taxonomic groups ("genus ..."), and
lemmas with any word missing from the SCOWL size-50 list in data/wordlist/
(this drops rare words like CUSPIDATION). A root can carry a word from its
definition, to pick one sense of an ambiguous lemma.
"""
import gzip
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

from src_common import DATA, RAW, write

SRC = RAW / "english-wordnet-2025.xml.gz"

LEXFILES = {
    "noun.food": ["Food And Drink"],
    "noun.animal": ["Living Thing"],
    "noun.plant": ["Living Thing"],
    "noun.artifact": ["Thing"],
    "noun.object": ["Thing"],
    "noun.person": ["Person"],
}
# (category, lexfile, lemma of the root synset[, word in its definition])
ROOTS = [
    ("Occupation", "noun.person", "professional"),
    ("Occupation", "noun.person", "skilled worker"),
    ("Occupation", "noun.person", "employee"),
    ("Occupation", "noun.person", "athlete"),
    ("Occupation", "noun.person", "entertainer"),
    ("Around The House", "noun.artifact", "furniture"),
    ("Around The House", "noun.artifact", "home appliance"),
    ("Around The House", "noun.artifact", "furnishing"),
    ("Around The House", "noun.artifact", "bedclothes"),
    ("Around The House", "noun.artifact", "room"),
    ("Around The House", "noun.artifact", "lamp"),
    ("Around The House", "noun.artifact", "decoration"),
    ("In The Kitchen", "noun.artifact", "kitchen utensil"),
    ("In The Kitchen", "noun.artifact", "cookware"),
    ("In The Kitchen", "noun.artifact", "kitchen appliance"),
    ("In The Kitchen", "noun.artifact", "tableware"),
    ("In The Kitchen", "noun.artifact", "cutlery", "eating"),
    ("In The Kitchen", "noun.artifact", "crockery"),
    ("Fun And Games", "noun.artifact", "toy"),
    ("Fun And Games", "noun.act", "game"),
    ("Fun And Games", "noun.act", "sport"),
    ("Fun And Games", "noun.act", "recreation"),
    ("Fun And Games", "noun.artifact", "sports equipment"),
    ("Event", "noun.act", "ceremony"),
    ("Event", "noun.act", "celebration"),
    ("Event", "noun.act", "contest"),
    ("Event", "noun.event", "social event"),
    ("Event", "noun.time", "holiday"),
]
SKIP = re.compile(r"^(genus|family|order|class|phylum|division|subfamily|suborder|subclass|tribe)\b")


def load():
    lemma = {}
    synsets = {}
    for _, el in ET.iterparse(gzip.open(SRC), events=("end",)):
        if el.tag == "LexicalEntry":
            lemma[el.get("id")] = el.find("Lemma").get("writtenForm")
            el.clear()
        elif el.tag == "Synset":
            hyper = [r.get("target") for r in el.findall("SynsetRelation")
                     if r.get("relType") in ("hypernym", "instance_hypernym")]
            d = el.find("Definition")
            synsets[el.get("id")] = (el.get("lexfile"), el.get("members").split(), hyper,
                                     d.text if d is not None else "")
            el.clear()
    return lemma, synsets


def plural(word):
    """People category: plural of a person noun, last word only."""
    head, _, last = word.rpartition(" ")
    if last.endswith("man") and not last.endswith(("human", "shaman", "german")):
        p = last[:-3] + "men"
    elif last.endswith("child"):
        p = last + "ren"
    elif last.endswith("person"):
        p = last[:-6] + "people"
    elif re.search(r"[^aeiou]y$", last):
        p = last[:-1] + "ies"
    elif re.search(r"(s|x|z|ch|sh)$", last):
        p = last + "es"
    else:
        p = last + "s"
    return f"{head} {p}".strip()


def main():
    lemma, synsets = load()
    print(f"loaded {len(lemma):,} entries, {len(synsets):,} synsets")
    hypo = defaultdict(list)
    common = {w.strip().lower() for w in open(DATA / "wordlist" / "american-english.txt")}
    for sid, (_, _, hyper, _) in synsets.items():
        for h in hyper:
            hypo[h].append(sid)
    words = lambda sid: [lemma[m] for m in synsets[sid][1]]
    keep = lambda w: (w[:1].islower() and not SKIP.match(w)
                      and all(part in common for part in re.split(r"[ \-]", w) if part))

    pairs = []
    for sid, (lexfile, _, _, _) in synsets.items():
        for cat in LEXFILES.get(lexfile, []):
            for w in words(sid):
                if keep(w):
                    pairs.append((cat, w))
                    if cat == "Person":
                        pairs.append(("People", plural(w)))

    for cat, lexfile, root_word, *gloss in ROOTS:
        roots = [sid for sid, (lf, _, _, d) in synsets.items()
                 if lf == lexfile and root_word in words(sid)
                 and (not gloss or gloss[0] in d)]
        if not roots:
            print(f"  no root synset for {root_word!r} in {lexfile}")
            continue
        stack, seen = list(roots), set()
        while stack:
            sid = stack.pop()
            if sid in seen:
                continue
            seen.add(sid)
            stack.extend(hypo[sid])
        n = 0
        for sid in seen:
            for w in words(sid):
                if keep(w):
                    pairs.append((cat, w))
                    n += 1
        print(f"  {cat:17} {root_word:18} {len(roots)} root(s), {len(seen):5,} synsets, {n:6,} lemmas")
    write("wordnet", pairs)


if __name__ == "__main__":
    main()
