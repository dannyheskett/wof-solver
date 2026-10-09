#!/usr/bin/env python3
"""Phase 4. Fetch named-entity lists from Wikidata (CC0).

Queries go to QLever's public Wikidata endpoint (Uni Freiburg), which serves
the same data as query.wikidata.org. The official service was limited to one
request per minute during an outage when this was written.

QLever stops any query at 30 seconds and answers 429 when it does. Joins
over a whole class (books with their authors) run longer than that, so a
join is fetched in two steps: the base list with a simple query, then the
linked values for the kept base items, BATCH items at a time in a VALUES
block. Fame counts are fetched with each item and the cut is applied here,
in Python, before saving: QLever drops results when a sitelinks comparison
stands alone in a FILTER, and runs past 30 seconds when it is combined with
others.

Each list is saved as JSON in data/sources/raw/wikidata/<name>.json and
skipped on later runs if that file exists. "links" is the number of
Wikipedia language editions with an article on the item, used as a fame
filter. build_wikidata.py turns the raw files into puzzle lines.

    python3 scripts/04_category_lists/fetch_wikidata.py            # all lists
    python3 scripts/04_category_lists/fetch_wikidata.py films      # just these
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "sources" / "raw" / "wikidata"
ENDPOINT = "https://qlever.dev/api/wikidata"
UA = "wof-solver-dataset/0.1 (personal research project)"
SPACING = 20   # seconds between full-class queries
BATCH = 500    # items per VALUES block in a join, about 1.5s each
PREFIXES = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""


def links_at_least(n):
    return lambda row: int(row["links"]) >= n


def simple(classes, min_links, prop="wdt:P31"):
    """Items whose `prop` is one of `classes`, with an English label."""
    values = " ".join(f"wd:{c}" for c in classes)
    return f"""SELECT DISTINCT ?item ?label ?links WHERE {{
  VALUES ?class {{ {values} }}
  ?item {prop} ?class; wikibase:sitelinks ?links; rdfs:label ?label.
  FILTER(LANG(?label) = "en")
}}""", links_at_least(min_links)


def people(occupation, min_links):
    return f"""SELECT DISTINCT ?item ?label ?links WHERE {{
  ?item wdt:P106 wd:{occupation}; wdt:P31 wd:Q5; wikibase:sitelinks ?links; rdfs:label ?label.
  FILTER(LANG(?label) = "en")
}}""", links_at_least(min_links)


class Linked:
    """A join: `pattern` (on ?item) run for the items in the base files that
    have at least `min_links`. Each result row also gets its base row's
    label and links."""

    def __init__(self, bases, select, pattern, min_links=0):
        self.bases = [bases] if isinstance(bases, str) else bases
        self.select, self.pattern, self.min_links = select, pattern, min_links


def label_of(base, prop, var, min_links=0):
    """The English label of the item at the end of `prop` (author, performer)."""
    return Linked(base, f"?item ?{var}", f"""?item {prop} ?o. ?o rdfs:label ?{var}.
  FILTER(LANG(?{var}) = "en")""", min_links)


PEOPLE = ["actors", "singers", "musicians", "film_directors", "writers", "politicians",
          "tv_presenters", "comedians", "football_players", "basketball_players",
          "baseball_players", "american_football_players"]

QUERIES = {
    # titles
    "films": simple(["Q11424"], 10),
    "tv_series": simple(["Q5398426"], 8),
    "novels": simple(["Q8261", "Q7725634"], 10),
    "songs": simple(["Q7366", "Q134556"], 8),
    # people and groups
    "actors": people("Q33999", 30),
    "singers": people("Q177220", 30),
    "musicians": people("Q639669", 35),
    "film_directors": people("Q2526255", 30),
    "writers": people("Q36180", 50),
    "politicians": people("Q82955", 60),
    "tv_presenters": people("Q947873", 20),
    "comedians": people("Q245068", 20),
    "football_players": people("Q937857", 50),
    "basketball_players": people("Q3665646", 25),
    "baseball_players": people("Q10871364", 12),
    "american_football_players": people("Q19204627", 10),
    "bands": simple(["Q215380", "Q5741069"], 15),
    "sports_teams": simple(["Q17156793", "Q13393265", "Q13027888", "Q4498974"], 10),
    "brands": simple(["Q431289", "Q167270"], 15),
    "companies": simple(["Q4830453", "Q891723"], 40),
    "fictional_characters": simple(
        ["Q95074", "Q15632617", "Q15773347", "Q15773317", "Q3658341", "Q1114461",
         "Q15711870", "Q4271324"], 10),
    # places
    "countries": simple(["Q6256", "Q3624078"], 0),
    "us_states": simple(["Q35657"], 0),
    "us_city_list": simple(["Q1093829"], 25),
    "cities": simple(["Q1549591", "Q5119"], 40),
    "rivers": simple(["Q4022"], 25),
    "mountains": simple(["Q8502"], 25),
    "lakes": simple(["Q23397"], 25),
    "islands": simple(["Q23442"], 40),
    "seas_oceans_deserts": simple(["Q165", "Q9430", "Q8514", "Q39594", "Q34763"], 15),
    "landmarks": simple(
        ["Q570116", "Q4989906", "Q12280", "Q11303", "Q46169", "Q33506", "Q16560",
         "Q23413", "Q2977", "Q12518", "Q483110", "Q1107656", "Q34038"], 20),
    # food, games, events
    "dishes": simple(["Q746549", "Q2095", "Q182940", "Q10943"], 8),
    "drinks": simple(["Q40050", "Q134768", "Q282", "Q154"], 8),
    "games": simple(["Q131436", "Q142714", "Q11422", "Q349"], 8),
    "video_games": simple(["Q7889"], 40),
    "holidays_events": simple(["Q1445650", "Q1197685", "Q132241", "Q18608583"], 12),
    "slogans": ("""SELECT DISTINCT ?item ?label ?links WHERE {
  ?item wdt:P1451 ?label; wikibase:sitelinks ?links.
  FILTER(LANG(?label) = "en")
}""", links_at_least(5)),
    # joins on the lists above, so they run last
    "books_authors": label_of("novels", "wdt:P50", "author", min_links=15),
    "songs_artists": label_of("songs", "wdt:P175", "artist"),
    "us_cities": Linked("us_city_list", "?item ?state", """?item wdt:P131+ ?o.
  ?o wdt:P31 wd:Q35657; rdfs:label ?state.
  FILTER(LANG(?state) = "en")"""),
    "star_and_role": Linked("films", "?item ?a ?actor ?char", """?item p:P161 ?st.
  ?st ps:P161 ?a; pq:P453 ?c. ?a rdfs:label ?actor. ?c rdfs:label ?char.
  FILTER(LANG(?actor) = "en" && LANG(?char) = "en")"""),
    "spouses": Linked(PEOPLE, "?item ?q ?b", """?item wdt:P26 ?q. ?q rdfs:label ?b.
  FILTER(LANG(?b) = "en")"""),
}


def run(query):
    """POST one query, retrying with backoff. Returns rows as plain dicts."""
    data = urllib.parse.urlencode({"query": PREFIXES + query}).encode()
    req = urllib.request.Request(ENDPOINT, data=data, headers={
        "User-Agent": UA, "Accept": "application/sparql-results+json"})
    for attempt in range(6):
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                # labels can hold raw control characters, which strict JSON rejects
                rows = json.loads(r.read(), strict=False)["results"]["bindings"]
            return [{k: v["value"] for k, v in row.items()} for row in rows]
        except Exception as e:
            print(f"    attempt {attempt + 1} failed after {time.time() - t0:.0f}s: {e}",
                  flush=True)
            time.sleep(90 * (attempt + 1))
    raise RuntimeError("gave up")


def fetch_linked(spec):
    base = {}
    for name in spec.bases:
        for r in json.loads((RAW / f"{name}.json").read_text()):
            if int(r["links"]) >= spec.min_links:
                base[r["item"]] = r
    items = list(base)
    rows = []
    for i in range(0, len(items), BATCH):
        values = " ".join(f"<{q}>" for q in items[i:i + BATCH])
        rows += run(f"""SELECT DISTINCT {spec.select} WHERE {{
  VALUES ?item {{ {values} }}
  {spec.pattern}
}}""")
    for r in rows:
        r["label"], r["links"] = base[r["item"]]["label"], base[r["item"]]["links"]
    return rows, len(items)


def fetch(name, spec):
    path = RAW / f"{name}.json"
    if path.exists():
        print(f"  {name}: cached")
        return
    t0 = time.time()
    try:
        if isinstance(spec, Linked):
            rows, n = fetch_linked(spec)
            note = f"{len(rows):,} rows for {n:,} base items"
        else:
            query, keep = spec
            rows = run(query)
            total = len(rows)
            rows = [r for r in rows if keep(r)]
            note = f"{len(rows):,} of {total:,} rows kept"
            time.sleep(SPACING)
    except RuntimeError:
        print(f"  {name}: GAVE UP", flush=True)
        return
    path.write_text(json.dumps(rows))
    print(f"  {name}: {note}, {time.time() - t0:.0f}s", flush=True)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:] or list(QUERIES)
    for name in names:
        fetch(name, QUERIES[name])


if __name__ == "__main__":
    main()
