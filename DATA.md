# Data

Everything in `data/`, what it is, where it comes from, and which phase builds it. Phases are the numbered scripts in `scripts/` (see "Pipeline" in `NOTES.md`). "Git" says whether the file is committed. Files marked "local" are rebuilt by their script and are not committed.

## How the puzzle files relate

```
wofanswers.com ──1───> puzzles.csv (68,197)
                          │
                          2   drop puzzles off the TV board, merge categories
                          │
          ┌───────────────┴──────────────────┐
     control.csv (3,349)                 train.csv
     5% of answers by hash,                  │
     held out from everything                3   300 answers by hash
                                             │
                                  ┌──────────┴──────────┐
                             train.csv (64,389)     dev.csv (300)
```

- **`control.csv`** is held out from everything. Neither solver learns from it, and every result in `eval/` is measured on it.
- **`dev.csv`** is held out from the transformer's training. It is used only to tune transformer-solve's SOLVE cutoff (phase 10).
- **smart-solve** builds its word list and weights from `train.csv` and `dev.csv` together. It has no setting to tune, so it never uses dev for tuning.
- **The splits use only the answer text** (SHA-1), so an answer filed under two categories always lands on one side. Phases 2 and 3 give the same files on every run.

## Puzzles (git)

All four files are CSV with a header row and two columns: `answer`, `category`. Answers are upper case, using only `A-Z`, space and `- & ' . ! ? ,`.

| File | Rows | Size | Source | Phase |
|---|---|---|---|---|
| `puzzles.csv` | 68,197 | 2.1 MB | every puzzle on wofanswers.com (TV show and mobile app), one row per distinct answer and category, sorted by category then answer | 1 `01_fetch_puzzles.py` |
| `train.csv` | 64,389 | 2.0 MB | the puzzles both solvers learn from | 2, then 3 |
| `dev.csv` | 300 | 12 KB | held out from model training, for tuning | 3 `03_split_dev.py` |
| `control.csv` | 3,349 | 108 KB | held out from everything, for scoring | 2 `02_clean_puzzles.py` |

Phase 2 drops 133 puzzles with characters the board can't show and 26 that don't fit the TV board (rows of 12, 14, 14 and 12 squares). It also merges three category names (`STAR & ROLE`, `Character`, `Fictional Place`).

## Word lists and category lists (git)

| Path | Rows | Format | Source | Phase |
|---|---|---|---|---|
| `wordlist/american-english.txt` | 104,334 | one word per line | Debian `wamerican` (SCOWL size 50), see `wordlist/SOURCE.md` and `wordlist/COPYRIGHT` | downloaded |
| `sources/wikidata.tsv` | 116,403 | TSV `category`, `phrase`, no header | named entities from Wikidata (CC0) | 4 `fetch_wikidata.py`, `build_wikidata.py` |
| `sources/wiktionary.tsv` | 21,487 | as above | phrases and "-ing" activities from English Wiktionary (CC BY-SA) | 4 `build_wiktionary.py` |
| `sources/wordnet.tsv` | 38,256 | as above | everyday nouns from Open English WordNet 2025 (CC BY 4.0) | 4 `build_wordnet.py` |
| `sources/wordplay.tsv` | 47,032 | as above | Same Letter, Rhyme Time, Before And After and Same Name, built from the lists above, the puzzles and the CMU Pronouncing Dictionary | 4 `build_wordplay.py` |

**How they are used:**
- smart-solve's candidate words come from the word list plus every word in `train.csv` and `dev.csv`.
- The category lists become the LIST training streams in phase 8.
- Every list phrase is fitted to the TV board.
- No phrase that is a control answer is kept.

## Demo data (git)

`demo.json` (127 KB) holds everything behind the demo's endpoints. Built by phase 13, `13_build_demo.py`.

| Key | What |
|---|---|
| `categories` | `[{"name", "auto"}]`, all 49 training categories, most common first. `auto` is true when control has a puzzle in it. |
| `puzzles` | `[[category, answer]]`, the 3,349 control puzzles, served by `GET /puzzle` |
| `presets` | `[{"mode", "category", "answer", "note"}]`, the 7 Demo examples |

## Results (git)

All files are TSV with a header row, in `eval/`.

| File | Rows | What | Phase |
|---|---|---|---|
| `smart-solve.tsv` | 6,698 | One row per control puzzle and mode. Columns `mode`, `category`, `solution`, `result` (`solved`, `wrong` or `invalid`), `detail` (letters, buys, misses, call order, or the BONUS pick). | 11 `11_evaluate.py` |
| `transformer-solve.tsv` | 6,698 | As above, plus in `detail`: `masked` and `fitted` (characters the rule masks changed) and `rejected` (SOLVEs under the cutoff that were turned down). | 11 |
| `summary.tsv` | 4 | One row per solver and mode: solve rate, wrong, invalid, letters, buys, misses, CPU time and memory. | 11 |
| `tune.tsv` | 2,819 | Every early SOLVE transformer-solve wrote on the 300 dev puzzles. Columns `category`, `solution`, `letters_called`, `p` (its confidence), `right`, `answer`. Used to pick the 0.99 cutoff. | 10 `10_tune_solve.py` |

## Board decoder (phase 14)

| Path | Git | What | Phase |
|---|---|---|---|
| `vision/clips.tsv` | git | The 705 YouTube clips the real frames come from: `id`, `source` (`broadcast`, `offair` or `phone`), `channel`, `title`. 480 were fetched; the rest are broadcast clips not needed yet, two that YouTube refused and one age-restricted. | 14 |
| `vision/raw/<id>/` | local | Each clip's `video.mp4` and one frame a second in `frames/`, 93,605 frames, 22 GB. The show's broadcast: never committed. | 14 `fetch_clips.sh` |
| `vision/raw_lb/` | local | The same frames letterboxed to the decoder's 640 x 360 input, 5.7 GB, for a GPU pod | 14 `shrink_frames.py` |
| `vision/real/labels2.jsonl`, `labels2_clean.jsonl` | local | Real frame labels: `clip`, `frame`, `source`, `answer`, `how` (`looked up` or `spelled`), `category`, `category_read`, `grid` (4 rows of 14 square classes), `review` (squares where the decoder disagreed) and `split` (`test` for control answers). The clean file drops doubtful spelled answers and keeps only looked-up answers in test. | 14 `label_real.py`, `clean_labels.py` |
| `vision/synth/<split>/` | local | Synthetic frames, `<shard>/<k>.jpg` and `labels.jsonl`, rebuilt the same from seeds | 14 `make_frames.py` |
| `vision/fonts/` | local | The six fonts frames are drawn with, copied from the system by `pod/pack.sh` so a pod renders the same frames | 14 |

## Local only (not in git)

| Path | Size | What | Phase |
|---|---|---|---|
| `corpus/train.txt`, `corpus/val.txt` | 1.0 GB | Pretraining text: Wikipedia and FineWeb-Edu, cleaned to board characters. See `corpus/SOURCE.md` (in git) for how to fetch the raw files. | 5 `05_make_corpus.py` |
| `corpus/raw/` | 3.1 GB | The downloaded Parquet files | downloaded |
| `sources/raw/` | 3.2 GB | Wikidata, Wiktionary, WordNet and CMU dumps behind the category lists | 4 |
| `main_train.tsv`, `main_control.tsv` | 5,131,187 and 267,216 lines | MAIN game states: `category`, `board`, `called`, `move`, `answer` | 6 `06_make_main.py` |
| `bonus_train.tsv`, `bonus_control.tsv` | 708,279 and 36,839 lines | BONUS picks and solves, same columns | 7 `07_make_bonus.py` |
| `train/` | 1.5 GB | Token files for training: one byte per character (`alphabet.txt`, 48 characters), `<stream>.bin` with `<stream>.idx.npy` line offsets for `main`, `bonus`, `list`, `list_bonus`, `corpus` and their `_val` versions | 8 `08_make_train.py` |
| `.cache/fetch/` | | Cached API responses, so a rerun of 1 resumes | 1 |

Model checkpoints are in `runs/` (local). `runs/pre` is the pretrained transformer. `runs/ft2b` is the finetuned model behind both solvers' results, with `wof.bin`, its float16 export for the server (phase 12). `runs/vision/dec2` is the board decoder behind `/decode`, with `decoder.bin`, its export for the server.
