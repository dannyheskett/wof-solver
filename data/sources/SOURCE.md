# Category list sources

The four TSVs here are built by the phase 4 scripts in
`scripts/04_category_lists/` and are committed. Their raw inputs go in `raw/`
(not committed, about 3.2 GB):

| File | Source | License | SHA-256 |
|---|---|---|---|
| `raw/wikidata/*.json` (42 files) | `fetch_wikidata.py`, QLever Wikidata endpoint, fetched 2026-10-06. Published as `wikidata-raw.tar.gz` in the `v1` release | CC0 | see the release |
| `raw/kaikki-english.jsonl` | https://kaikki.org/dictionary/English/kaikki.org-dictionary-English.jsonl, downloaded 2026-10-06. kaikki.org rebuilds it, so a new download differs | CC BY-SA | `9978ce34256e4143c3498387564d293a9a2971ef376c1e038d369a2021c02195` |
| `raw/english-wordnet-2025.xml.gz` | https://github.com/globalwordnet/english-wordnet/releases/download/2025-edition/english-wordnet-2025.xml.gz | CC BY 4.0 | `9ca6d1dcb75f822fdd66617f7d9da48142ace38dd544d6ad5e2feca1674ad3fe` |
| `raw/cmudict.dict` | https://raw.githubusercontent.com/cmusphinx/cmudict/0f8072f814306c5ee4fbf992ed853601b12c01f9/cmudict.dict | BSD-style | `81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22` |

```bash
cd data/sources/raw
curl -L https://github.com/dannyheskett/wof-solver/releases/download/v1/wikidata-raw.tar.gz | tar -xz
curl -L -o kaikki-english.jsonl https://kaikki.org/dictionary/English/kaikki.org-dictionary-English.jsonl
curl -L -o english-wordnet-2025.xml.gz https://github.com/globalwordnet/english-wordnet/releases/download/2025-edition/english-wordnet-2025.xml.gz
curl -L -o cmudict.dict https://raw.githubusercontent.com/cmusphinx/cmudict/0f8072f814306c5ee4fbf992ed853601b12c01f9/cmudict.dict
```

With today's kaikki.org file, `build_wiktionary.py` will not reproduce the
committed `wiktionary.tsv` exactly. The committed file is the one every later
phase used.
