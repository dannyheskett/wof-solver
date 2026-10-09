#!/usr/bin/env python3
"""Phase 4. Named-entity category lists from the Wikidata files fetched by
fetch_wikidata.py.

Source: data/sources/raw/wikidata/*.json
Output: data/sources/wikidata.tsv

Formats follow the show:
    Song Artist        SONG BY ARTIST
    Title/Author       TITLE BY AUTHOR
    Star And Role      ACTOR AS CHARACTER, only for films in films.json and
                       actors in any of the people lists
    Husband And Wife   FIRST & FIRST LAST when both share the last words, else
                       A & B, only when both are in the people lists
    On The Map         US cities also as CITY STATE
"""
import json

from src_common import RAW, write

WD = RAW / "wikidata"
SIMPLE = {
    "films": ["Movie Title"],
    "tv_series": ["Tv Show Title"],
    "novels": ["Title"],
    "fictional_characters": ["Fictional Character"],
    "bands": ["Proper Name"],
    "sports_teams": ["Proper Name"],
    "brands": ["Proper Name"],
    "companies": ["Proper Name"],
    "countries": ["On The Map"],
    "us_states": ["On The Map"],
    "cities": ["On The Map"],
    "rivers": ["On The Map"],
    "mountains": ["On The Map"],
    "lakes": ["On The Map"],
    "islands": ["On The Map"],
    "seas_oceans_deserts": ["On The Map"],
    "landmarks": ["Landmark"],
    "dishes": ["Food And Drink"],
    "drinks": ["Food And Drink"],
    "games": ["Fun And Games"],
    "video_games": ["Fun And Games"],
    "holidays_events": ["Event"],
    "slogans": ["Slogan"],
}
PEOPLE = {
    "actors": ["Proper Name", "Show Biz"],
    "singers": ["Proper Name", "Show Biz"],
    "musicians": ["Proper Name", "Show Biz"],
    "film_directors": ["Proper Name", "Show Biz"],
    "tv_presenters": ["Proper Name", "Show Biz"],
    "comedians": ["Proper Name", "Show Biz"],
    "writers": ["Proper Name"],
    "politicians": ["Proper Name"],
    "football_players": ["Proper Name"],
    "basketball_players": ["Proper Name"],
    "baseball_players": ["Proper Name"],
    "american_football_players": ["Proper Name"],
}


def load(name):
    path = WD / f"{name}.json"
    if not path.exists():
        print(f"  missing {path.name}, skipped")
        return []
    return json.loads(path.read_text())


def couple(a, b):
    """Shared trailing words go once, at the end: BETTINA VON ARNIM and
    LUDWIG ACHIM VON ARNIM -> BETTINA & LUDWIG ACHIM VON ARNIM."""
    aw, bw = a.split(), b.split()
    n = 0
    while n < min(len(aw), len(bw)) - 1 and aw[-1 - n] == bw[-1 - n]:
        n += 1
    if n:
        return f"{' '.join(aw[:-n])} & {' '.join(bw[:-n])} {' '.join(aw[-n:])}"
    return f"{a} & {b}"


def main():
    pairs = []
    for name, cats in SIMPLE.items():
        for r in load(name):
            pairs += [(c, r["label"]) for c in cats]
    people = set()
    for name, cats in PEOPLE.items():
        for r in load(name):
            people.add(r["item"])
            pairs += [(c, r["label"]) for c in cats]
    for r in load("books_authors"):
        pairs.append(("Title/Author", f"{r['label']} by {r['author']}"))
    for r in load("songs"):
        pairs.append(("Song Title", r["label"]))
    for r in load("songs_artists"):
        pairs.append(("Song Artist", f"{r['label']} by {r['artist']}"))
    for r in load("us_city_list"):
        pairs.append(("On The Map", r["label"]))
    for r in load("us_cities"):
        pairs.append(("On The Map", f"{r['label']} {r['state']}"))
    films = {r["item"] for r in load("films")}
    for r in load("star_and_role"):
        if r["item"] in films and r["a"] in people:
            pairs.append(("Star And Role", f"{r['actor']} as {r['char']}"))
    for r in load("spouses"):
        if r["q"] in people:
            pairs.append(("Husband And Wife", couple(r["label"], r["b"])))
    write("wikidata", pairs)


if __name__ == "__main__":
    main()
