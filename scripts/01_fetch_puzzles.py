#!/usr/bin/env python3
"""Phase 1. Pull every puzzle (answer + category) from the wofanswers.com API.

The API returns at most 1000 rows per call and its offset paging overlaps
(rows repeat across pages, others are skipped), so this never pages. It
splits the data with filters until every slice fits in one call:
category, then word_count, then letter_count. At each split the child
totals must add up to the parent total, so nothing can go missing silently.

Responses are cached in data/.cache/fetch/ (not in git), so a rerun resumes
without refetching. Output: data/puzzles.csv with columns answer, category,
one row per distinct (answer, category): the API lists 24 of them twice.
"""
import csv
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

SITE = "https://wofanswers.com"
API = SITE + "/api/collections/wof-answers/items"
LIMIT = 1000
DELAY = 1.0
SPLITS = ["word_count", "letter_count"]
RANGES = {"word_count": range(1, 21), "letter_count": range(1, 81)}
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / ".cache" / "fetch"
calls = 0
# Optional earlier (incomplete) pull, used only to guess which split values exist.
HINT_FILE = Path("/tmp/puzzles_paged.csv")
HINT = list(csv.DictReader(open(HINT_FILE))) if HINT_FILE.exists() else []


def fetch(url):
    global calls
    req = urllib.request.Request(url, headers={"User-Agent": "wof-solver dataset fetch"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                body = r.read()
            calls += 1
            time.sleep(DELAY)
            return body
        except Exception as e:
            if attempt == 2:
                raise
            print(f"  {e}, retrying")
            time.sleep(5 * (attempt + 1))


def query(filters):
    """One call for this filter set, cached on disk."""
    params = {f"filter[{k}]": v for k, v in sorted(filters.items())}
    params.update(limit=LIMIT, offset=0, fields="answer,category")
    url = API + "?" + urllib.parse.urlencode(params)
    path = RAW / (hashlib.sha1(url.encode()).hexdigest()[:16] + ".json")
    if not path.exists():
        path.write_text(json.dumps({"url": url, "response": json.loads(fetch(url))}))
    return json.loads(path.read_text())["response"]


def categories():
    """Category filter names from the solver script on the site's pages."""
    page = fetch(SITE + "/phrase").decode()
    cats = json.loads(re.search(r"WOF_CATS=(\[\[.*?\]\]);", page, re.S).group(1))
    return [c[2] for c in cats]


def word_count(answer):
    return len(answer.split())


def letter_count(answer):
    return sum(ch.isalpha() for ch in answer)


def collect(filters, total, depth, rows, seen):
    """Collect every row matching filters, splitting until slices fit in one call."""
    resp = query(filters)
    if resp["meta"]["total"] != total:
        raise SystemExit(f"{filters}: total {resp['meta']['total']} != expected {total}")
    if total <= LIMIT:
        if len(resp["data"]) != total:
            raise SystemExit(f"{filters}: got {len(resp['data'])} rows, total says {total}")
        rows += resp["data"]
        return
    if depth == len(SPLITS):
        raise SystemExit(f"{filters}: {total} rows and no splits left")
    field = SPLITS[depth]
    # Try the values seen in this slice (plus any earlier pull) first; fall
    # back to the full range.
    calc = {"word_count": word_count, "letter_count": letter_count}
    known = resp["data"] + [r for r in HINT if all(
        r["category"] == v if k == "category" else calc[k](r["answer"]) == v
        for k, v in filters.items())]
    guess = sorted({calc[field](r["answer"]) for r in known})
    for values in (guess, RANGES[field]):
        children = []
        for v in values:
            child = dict(filters, **{field: v})
            n = query(child)["meta"]["total"]
            if n:
                children.append((child, n))
        if sum(n for _, n in children) == total:
            break
        print(f"  {filters}: {field} values seen cover {sum(n for _, n in children)}/{total}, trying full range")
    else:
        raise SystemExit(f"{filters}: {field} children sum to {sum(n for _, n in children)}, not {total}")
    for child, n in children:
        collect(child, n, depth + 1, rows, seen)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    grand = query({})["meta"]["total"]
    cats = categories()
    print(f"total={grand} categories={len(cats)}")

    rows = []
    for cat in cats:
        n = query({"category": cat})["meta"]["total"]
        before = len(rows)
        collect({"category": cat}, n, 0, rows, set())
        print(f"{cat}: {len(rows) - before} rows (calls so far: {calls})")

    if len(rows) != grand:
        raise SystemExit(f"collected {len(rows)} rows, API total is {grand}")

    unique = sorted({(r["answer"], r["category"]) for r in rows}, key=lambda p: (p[1], p[0]))
    with open(ROOT / "data" / "puzzles.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["answer", "category"])
        w.writerows(unique)
    print(f"rows={len(rows)} expected={grand} written (distinct)={len(unique)} calls={calls}")


if __name__ == "__main__":
    main()
