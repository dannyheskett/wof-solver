# wof-solver notes

Tracking issue: dannyheskett/danheskett#147

## Goal

Can a small transformer learn to play TV-rules Wheel of Fortune? Two solvers play the same held-out puzzles, with what each cost to build and to run:

1. **transformer-solve:** a 59M-parameter character-level transformer. It is pretrained on general English, then finetuned on game states, and it plays each move as a text completion.
2. **smart-solve:** deterministic. It finds word-list candidates for each board word, calls the letter that narrows them most, and solves when every word has one candidate.

The demo (https://wof-solver.fly.dev, for danheskett.com) plays them head to head.

Modes, for both solvers:

- **MAIN** (play the game): from any board, CALL a consonant, BUY a vowel or SOLVE. Scored on solve rate, letters called before a correct SOLVE, and wrong SOLVEs.
- **BONUS** (one shot): R S T L N E are shown, then PICK 3 consonants and 1 vowel, then one SOLVE. Scored on solve rate.

Fair comparison rules:

- Neither solver sees a control answer. Both know only the training puzzles, the word lists and category lists, and (transformer-solve) the corpus.
- Both get the same control puzzles, the same board states and the same move rules.

## Pipeline

Each phase has its own script in `scripts/`. `DATA.md` documents every file they read and write.

| Phase | Script | What it does |
|---|---|---|
| 1 | `01_fetch_puzzles.py` | Pulls all 68,197 distinct puzzles from wofanswers.com |
| 2 | `02_clean_puzzles.py` | Drops puzzles off the TV board, merges categories, holds 5% of answers out as control |
| 3 | `03_split_dev.py` | Holds 300 answers out of training, for tuning |
| 4 | `04_category_lists/` | Category phrase lists from Wikidata, Wiktionary, WordNet, and wordplay joins |
| 5 | `05_make_corpus.py` | 1.0B characters of pretraining text (Wikipedia and FineWeb-Edu) |
| 6 | `06_make_main.py` | MAIN game states labelled with smart-solve's move |
| 7 | `07_make_bonus.py` | BONUS picks and solves |
| 8 | `08_make_train.py` | Token files for training |
| 9 | `09_train.py`, `09_pod/` | Pretrain, then finetune, on a rented H100 |
| 10 | `10_tune_solve.py` | Picks transformer-solve's SOLVE cutoff on the dev puzzles |
| 11 | `11_evaluate.py` | Plays both solvers on all 3,349 control puzzles |
| 12 | `12_export_weights.py` | Exports the transformer as float16 weights for the C server |
| 13 | `13_build_demo.py` | Builds the demo's puzzle and category data |
| 14 | `14_vision/` | The board decoder: reads a screencap or photo of the TV board for `POST /decode` |

Shared code: `wof_common.py` (board rules, word list, smart picks), `solvers.py` (both solvers and the move check), `model.py` (the transformer).

## The rules

- A letter called twice, a consonant bought, or a vowel called is invalid, and it ends that game.
- A vowel can be bought only once a called consonant shows on the board, the money rule (`wof_common.can_buy`). On the show, money only comes from consonants that hit.
- The first SOLVE ends the game, right or wrong.
- No wheel, money or turns are played. Each solver plays its own board, and a head-to-head match compares the two games.

## smart-solve

- **Candidates:** each board word's candidates are the words of its shape that match every called letter exactly. They come from the SCOWL list plus every word of `train.csv` and `dev.csv`.
- **Weights:** each candidate weighs 1 plus how often it appears in that category's training puzzles.
- **MAIN:** smart-solve calls the letter that leaves the lowest expected log2(candidates left), keeping to the money rule. It solves when every hidden word has one candidate.
- **BONUS:** it picks the 3 consonants and 1 vowel most likely to leave a right fill (`best_set`), then fills each word with its heaviest candidate.
- **Training labels:** the same picks label the MAIN and BONUS training lines in phases 6 and 7.

## The transformer

It reads characters, 48 of them (`data/train/alphabet.txt`). Every task line is a prompt and a target:

    MAIN|THING|A _A__L_ __ ___ ______|AL|BUY E
    BONUS|THING|_ __N_LE _N T_E __N___|ELNRST|PICK DHM A
    BONUS|THING|A _ANDLE _N THE __ND__|ADEHLMNRST|SOLVE A CANDLE IN THE WINDOW
    LIST|ON THE MAP|SAN ANTONIO TEXAS

Loss is on the target only.

| Stream | Lines | Tokens | Used for |
|---|---|---|---|
| corpus | | 1,000M | pretraining, and 10% of finetune rows |
| main | 5,131,187 | 283M | finetune, 35% |
| bonus | 708,279 | 52M | finetune, 30% |
| list_bonus | 280,147 | 20M | finetune, 20%: BONUS SOLVE lines made from the category lists |
| list | 281,171 | 10M | finetune, 5%: category and phrase |
| main_val, bonus_val | 304,055 | 17M | validation loss only (control puzzles) |

- **Architecture:** `model.py` is a nanoGPT-style transformer with 12 layers, 10 heads, 640 wide and block 256, which is 59.0M parameters.
- **Training:** `09_pod/run.sh` pretrains for 30,500 steps (1.0B tokens, `runs/pre`), then finetunes for 15,000 steps (0.49B tokens, `runs/ft2b`), at batch 128 × 256.
- **Playing:** as transformer-solve, each move is the greedy completion of the prompt. Masks hold it to the rules and the board: no BUY before a vowel can be bought, an uncalled letter after CALL or BUY, and a SOLVE that fits the board square by square.
- **The SOLVE cutoff:** a MAIN SOLVE is accepted only when the transformer gives that answer at least 99% probability, or no blank is left. Otherwise it calls its best letter instead. The cutoff was tuned on the 300 dev puzzles, which it never trained on: 0.99 solved 98.3% at 12.08 letters, and with no cutoff it solved 64.0%.

## Results on the control set

`11_evaluate.py` plays every control puzzle (3,349) in both modes. Per-puzzle results are in `data/eval/<solver>.tsv` and summaries in `data/eval/summary.tsv`.

| Solver | MAIN solved | Wrong / invalid | Letters (median / mean) | Misses (mean) | BONUS solved | CPU per puzzle (MAIN / BONUS) |
|---|---|---|---|---|---|---|
| smart-solve | 98.3% | 57 / 0 | 9 / 9.04 | 1.96 | 79.8% | 65 / 246 ms |
| transformer-solve | 98.7% | 42 / 0 | 12 / 12.33 | 3.77 | 77.2% | 16.7 / 3.2 s |

- **The two solve about as often, but smart-solve needs fewer letters.** On MAIN, scoring each puzzle by fewer misses (an unsolved puzzle loses): smart-solve wins 2,245 (67%), transformer-solve 439 (13%), and 665 tie.
- **Missing words:** 1.5% of control words (163 of 10,649) are not in smart-solve's word list, in 159 puzzles. All 57 of smart-solve's wrong MAIN solves are in those puzzles: a list word fits the board, so it solves early with the wrong word (FILM JOUR FESTIVAL, THE COUPONS for THE GORGONS). Some are typos in the source data (HADRIN, LASANGA, CHEDDER). Only 10 of transformer-solve's 42 wrong solves are in those puzzles.
- **BONUS:** on the 1,348 puzzles where both pick the same letters, smart-solve solves 1,135 and transformer-solve 1,096. They miss different puzzles: smart-solve solves 414 that transformer-solve doesn't, and transformer-solve 326 the other way. Taking either one's answer would solve 89.5%.
- **Openings:** transformer-solve opens the same way in most games: CALL S, then BUY E and BUY A once a consonant shows. GOOEY COOKIE DOUGH costs it five misses (S R L N T) before the first hit.
- **Timing caveat:** the CPU times are from runs that shared the machine with other jobs.
- **Human comparison:** contestants solve roughly 40% of bonus rounds (unconfirmed). That isn't comparable as is, since they get 10 seconds and no word list.

## The board decoder (phase 14)

`POST /decode` takes a screencap or a phone photo of the puzzle board and returns what `/move` takes: the category, the board and the letters showing. The solvers are unchanged.

**The model** (`14_vision/decoder_net.py`): a 1.96M-parameter CNN (ReLU convs, batch norm, features at 1/8, 1/16 and 1/32 scale merged to 45 x 80 tokens) and a 3-layer transformer decoder with 77 learned queries: 52 squares, 24 category characters, and whether a board is there at all. Input is the image letterboxed to 640 x 360. Each square is one of 37 classes: `#` unused, `_` hidden, `*` lit, a letter or punctuation shown, `~` covered.

**Synthetic frames** (`render.py`, `make_frames.py`): puzzles from `train.csv` laid out the show's way (every line starts in the same column; `board.layout_score` picks among placements), drawn as the 2022-on LED board or the 1997-2022 board, with a host figure, lit squares, the category strip, the floor reflection, insets, captions and station logos, then passed through one of three sources: a clean capture, an off-air recording (low resolution, 4:3 crops, compression), or a phone filming the TV (bezel, angle, moire, glare). 10% of frames have no board (the wheel, people, title cards). Val and test frames use `dev.csv` and `control.csv`.

**Real frames** (`clips.tsv`, `fetch_clips.sh`): 480 YouTube clips at one frame a second, 93,605 frames: 134 broadcast clips from the show's channel (Season 40 on), 262 off-air recordings and 84 phone recordings of a TV. The frames stay local; only the clip list is committed. `label_real.py` labels them with the decoder's own reads corrected by each puzzle's answer: frames are cut into puzzles, and the answer is looked up in `puzzles.csv` by its layout and the letters read, or, for puzzles newer than `puzzles.csv` (most of Season 40 on), spelled word by word from the word list and the decoder's letter probabilities. `clean_labels.py` drops doubtful spelled answers (a hand check found about a quarter wrong before the filter). Test frames are the 5% of answers 02 holds out as control, and only answers looked up in `puzzles.csv`, so they are known to be right.

**Training** (`train.py`, `pod/`): `dec1` trained 30,000 steps at batch 128 on 100,000 synthetic frames; `dec2` finetuned it 8,000 steps on 120,000 synthetic frames with 30% of each batch real (24,677 labelled frames). Both on a rented A40.

**Results.** On 2,000 synthetic test-style frames dec2 gets 99.4% of squares and 87.0% of whole boards right (broadcast 92.1%, off-air 91.6%, phone 75.9%). On 752 real held-out frames, the board text the solvers would get is exactly right for:

| Model | All | Broadcast | Off-air | Phone (20 frames) | Squares |
|---|---|---|---|---|---|
| dec1, synthetic only | 58.4% | 49.3% | 62.6% | 55% | 97.2% |
| dec2, with real frames | 74.5% | 75.8% | 74.3% | 65% | 98.3% |

- **Most remaining errors are reveal timing**: a square read as its letter while the label still has it hidden, on frames caught mid-reveal at one frame a second. Some of these are the labels, which come from the decoder's own reads.
- **The real labels are partly the model's own reads** (which squares are lit or covered, when a letter first shows), so the real scores lean favourable. The answers themselves are checked against `puzzles.csv`.
- **Phone photos are barely tested**: 20 real frames. Real phone photos of a TV are the next test set.
- **Board or not**: dec2 still calls some title cards a board. The server also requires at most 12 squares under 80% confidence (real boards: 99% have 7 or fewer), which rejects them.

**Snap and solve.** `/decode?solve=1` also returns transformer-solve's single best SOLVE for the board, however unsure (`tf_guess`: its MAIN prompt with only a SOLVE allowed), with its confidence. On the 752 real test frames the guess is the exact answer for 19.5% of the 750 not fully revealed, and for none of the 239 with under a quarter of the letters showing, but the confidence tells right from wrong: at 0.9 or more it is right 90% of the time (92 frames), at 0.3 or more 65% (223 frames). When smart-solve's move is a SOLVE it is the guess instead, confidence 0.97: its SOLVEs were right 63 times in 65 on the same frames, and taking them with the transformer's guesses of 0.9 or more gives 97 frames, 91% right. So a bot can call a guess of 0.9 or more the answer, 0.3 to 0.9 a best guess, and anything lower too early. `/decode` reads JPEG and PNG (stb_image) and WebP (libwebp, from Debian). HEIC is not read on the server: phone cameras hand a web page JPEG, and a .heic file from a photo library is converted to JPEG in the browser.

**The server** (`server/src/decoder.c`): the same forward pass in C, float32, weights exported by `export_decoder.py` (`runs/vision/dec2/decoder.bin`, 9.7 MB). Images are decoded by `stb_image.h` (JPEG and PNG) and letterboxed exactly as Pillow's BILINEAR resize does. `server/tests/decoder_parity.py` checks it against PyTorch: identical letterboxed bytes, square probabilities within 6e-6, the same class for every square and the same category text. A read takes 0.4-1.5 s on this laptop under load.

## Hosting

Both solvers are served together at https://wof-solver.fly.dev by one C program, `server/` (`wof-server`). It also serves the WASM client (`client/`). It runs on Fly.io in the personal org: app `wof-solver`, region iad, shared-cpu-1x, 256 MB.

    POST /move  {"mode": "main" | "bonus_pick" | "bonus_solve",
                 "category": ..., "board": ..., "called": ...}
    -> {"smart": {"move": ..., "ms": ...},
        "transformer": {"move": ..., "confidence": ..., "letters": [...],
                        "turned_down": ..., "ms": ...}}

The caller holds the answer. The service sees only the category, board and called letters, as in `11_evaluate.py`. `POST /decode` (phase 14) reads those three from an image instead. The full API is at the top of `server/src/main.c`.

- **`server/src/model.c`:** runs the transformer on one thread, with the same layers as `scripts/model.py` and exact (erf) GELU. Weights are float16 (`12_export_weights.py`, 118 MB), mapped from the file rather than copied into memory. All arithmetic is float32.
- **`server/src/transformer.c`:** decodes each move greedily under the same rules as the masks above, and turns down a MAIN SOLVE under 0.99. A SOLVE's confidence is renormalised over the characters each square allows.
- **`server/src/smart.c`:** smart-solve, ported from `solvers.py` and `wof_common.py`. It sums scores in numpy's pairwise order and rounds as Python does, so ties break the same way.
- **Libraries:** HTTP parsing is picohttpparser (vendored, MIT), and JSON is cJSON.
- **Parity tests** (`server/tests/`):
  - The forward pass with float16 weights matches PyTorch on 200 game prompts. The largest log-probability difference is 0.0096, and every greedy completion is the same.
  - Whole games against PyTorch: 1,329 of 1,331 MAIN moves, and all 100 BONUS picks and 100 BONUS solves, are the same. The two that differ are close calls that float16 tips the other way (CALL S for CALL R, BUY O for BUY A).
  - smart-solve makes the same move as Python at every state of all 3,349 control puzzles, MAIN and BONUS (40,223 moves).
- **Memory:** the process's own memory is about 30 MB. The weights sit in the page cache, which the kernel can drop and reread. Copying the weights into the process (160 MB) got it killed for lack of memory on the 256 MB machine. The board decoder's 10 MB of weights are mapped the same way, and a `/decode` read's 40 MB of scratch comes from mmap and is trimmed afterwards: the first deploy of `/decode` held them as process memory (47 MB in all), the transformer's weights no longer fit in the page cache, and every move took 9-14 s until it was rolled back. With the fix the process holds 28 MB and all 118 MB of weights stay cached.
- **Speed on Fly:** a transformer move takes about 240 ms. The first move after a fresh start takes about 5.6 s while the weights are read from disk.
- **Suspend:** idle machines suspend after about 6 minutes. Waking one and making a move takes about 1.1 s.
- **Earlier version:** before the C server, the model ran in llama.cpp's `llama-server` behind a Python service. Its RAM prompt cache (8 GiB by default) grew under steady traffic until the weights were paged out on the 1 GB machine, and moves took 50-60 s.

## Cost log

Machine for everything below unless noted: WSL2 on an Intel Core i5-1335U, 12 logical CPUs, 4.9 GB RAM, no GPU.

| Date | Step | Wall time | Compute | Output size |
|---|---|---|---|---|
| 2026-10-05 | 1 Fetch puzzles (658 API calls, 1 s apart) | ~15 min | 1 core, network bound | 2.1 MB CSV |
| 2026-10-06 | 2, 3 Clean and split | < 10 s | 1 core | 2.2 MB |
| 2026-10-06 | 4 Wikidata lists (QLever, 30 s query cap) | ~75 min, 43 of it the cast join | network | 17 MB JSON |
| 2026-10-06 | 4 Category lists build (all four) | ~40 s | 1 core | 223k lines, 6.3 MB |
| 2026-10-06 | 5 Corpus download | ~1 min | network | 3.1 GB parquet |
| 2026-10-06 | 5 Corpus build | ~10 min | 1 core | 1.01 GB text |
| 2026-10-06 | 6, 7, 8 MAIN, BONUS and token files | 4,298 s (MAIN 2,614, BONUS 1,647, tokens 28) | 11 cores | 1.6 GB |
| 2026-10-06 | 9 Runpod pod, H100 SXM 80GB, Secure Cloud | ~60 min up, $3.49/hr, ~$3.50 | 1 GPU, 28 vCPU | |
| 2026-10-06 | 9 Upload bundle to pod (rsync) | 168 s, 8.8 MB/s | network | 1.55 GB |
| 2026-10-06 | 9 Pretrain, 30,500 steps, 1.0B tokens | 1,220 s | H100, ~820k tok/s | 237 MB checkpoint |
| 2026-10-07 | 9 Runpod pod, H100 SXM 80GB, Secure Cloud | ~28 min up, $3.49/hr, ~$1.65 | 1 GPU, 26 vCPU | |
| 2026-10-07 | 9 Finetune, 15,000 steps, 0.49B tokens | 722 s | H100, ~682k tok/s | 237 MB checkpoint |
| 2026-10-07 | 10 Cutoff tuning on 300 dev puzzles | 1,538 s | 11 cores, shared | 0.1 MB |
| 2026-10-06 | 11 Evaluate smart-solve, both modes | 104 s | 11 cores | 0.6 MB |
| 2026-10-07 | 11 Evaluate transformer-solve, both modes | 6,235 s | 11 cores, shared with other jobs | 0.7 MB |
| 2026-10-07 | Fly.io deploys (remote builder) | several minutes each | Fly builder | |
| 2026-10-08 | Fly.io machine, shared-cpu-1x 256 MB, iad | $2.19/mo always on; suspended: $0.15/GB/mo of rootfs | 1 shared CPU | |

| 2026-10-10 | 14 YouTube clips, 480 of 705 listed (`fetch_clips.sh`) | several hours, network bound | yt-dlp, ffmpeg | 22 GB video and frames (local) |
| 2026-10-10 | 14 Runpod pod, A40 48GB, Secure Cloud | 6.46 h up, $0.59/hr, ~$3.81 | 1 GPU, 9 vCPU (7.65 usable) | |
| 2026-10-10 | 14 Render 100,000 synthetic frames | 4,041 s | 6 workers on the pod | ~5 GB |
| 2026-10-10 | 14 Train dec1, 30,000 steps | ~2.7 h | A40, ~395 frames/s (data loading bound) | 7.9 MB checkpoint |
| 2026-10-10 | 14 Label 93,605 real frames, twice | ~15 min each | A40 | 9 MB labels |
| 2026-10-10 | 14 Finetune dec2, 8,000 steps | ~45 min | A40 | 7.9 MB checkpoint |

Total GPU spend: about $8.96.
