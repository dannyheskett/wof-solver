#!/usr/bin/env python3
"""Phase 4. Phrase and What Are You Doing lists from English Wiktionary (CC BY-SA),
using the kaikki.org extract.

Source: data/sources/raw/kaikki-english.jsonl
Output: data/sources/wiktionary.tsv

    Phrase              multiword entries whose part of speech is phrase,
                        proverb, prepositional phrase or interjection, plus
                        multiword nouns, adjectives and adverbs with a sense
                        tagged idiomatic
    What Are You Doing  multiword verbs ("take a walk"), with the first word
                        turned into its present participle ("TAKING A WALK"),
                        read from that verb's own Wiktionary forms when it
                        has them, else by spelling rules

An entry is skipped when every sense is obsolete, archaic or a form of
another entry, when any sense is tagged vulgar, offensive, derogatory or a
slur, or when any word is missing from the SCOWL size-50 list (this drops
Latin and other foreign phrases). "one's" becomes MY and "oneself" becomes
MYSELF, the way the show writes them. Verb phrases with "someone" or
"something" are skipped.
"""
import json
import re

from src_common import DATA, RAW, write

SRC = RAW / "kaikki-english.jsonl"
PHRASE_POS = {"phrase", "proverb", "prep_phrase", "intj"}
IDIOM_POS = {"noun", "adj", "adv"}
DEAD = {"obsolete", "archaic", "form-of", "alt-of", "misspelling"}
BAD = {"vulgar", "offensive", "derogatory", "slur", "ethnic"}
PLACEHOLDER = re.compile(r"\b(someone|somebody|something|sth|sb|so|one)\b")


def participle(verb):
    """Spelling rules, used only when Wiktionary has no form for the verb."""
    if verb.endswith("ie"):
        return verb[:-2] + "ying"
    if verb.endswith("e") and not verb.endswith(("ee", "ye", "oe")):
        return verb[:-1] + "ing"
    if re.search(r"[^aeiou][aeiou][bdgklmnprt]$", verb) and len(verb) <= 4:
        return verb + verb[-1] + "ing"
    return verb + "ing"


def main():
    common = {w.strip().lower() for w in open(DATA / "wordlist" / "american-english.txt")}
    is_common = lambda text: all(p in common for p in re.split(r"[ \-]", text.lower()) if p)
    participles = {}
    phrases = []
    verbs = []
    with open(SRC) as f:
        for line in f:
            d = json.loads(line)
            word, pos = d.get("word", ""), d.get("pos")
            senses = d.get("senses", [])
            tags = [set(s.get("tags", [])) for s in senses]
            if pos == "verb" and " " not in word:
                for form in d.get("forms", []):
                    ft = set(form.get("tags", []))
                    if {"present", "participle"} <= ft and form.get("form", "").endswith("ing"):
                        participles.setdefault(word, form["form"])
            if " " not in word or not senses:
                continue
            if all(t & DEAD for t in tags) or any(t & BAD for t in tags):
                continue
            if pos in PHRASE_POS or (pos in IDIOM_POS and any("idiomatic" in t for t in tags)):
                if is_common(word):
                    phrases.append(word)
            elif pos == "verb":
                verbs.append(word)

    pairs = [("Phrase", re.sub(r"\boneself\b", "myself", re.sub(r"\bone's\b", "my", p)))
             for p in phrases]
    doing = 0
    for v in verbs:
        v = re.sub(r"\bone's\b", "my", v)
        v = re.sub(r"\boneself\b", "myself", v)
        if PLACEHOLDER.search(v) or not is_common(v):
            continue
        head, _, rest = v.partition(" ")
        pairs.append(("What Are You Doing", f"{participles.get(head) or participle(head)} {rest}"))
        doing += 1
    print(f"participles {len(participles):,}, phrases {len(phrases):,}, verb phrases {doing:,}")
    write("wiktionary", pairs)


if __name__ == "__main__":
    main()
