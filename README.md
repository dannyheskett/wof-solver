# WOF Solver

A 59M-parameter transformer trained from scratch to play Wheel of Fortune,
head to head against smart-solve, a word-list solver, on 3,349 puzzles
neither has seen.

- Play it: https://wof-solver.fly.dev/
- The write-up, phase by phase: https://danheskett.com/projects/wof-solver/

| Solver | MAIN solved | Wrong | Letters (median) | BONUS solved |
|---|---|---|---|---|
| smart-solve | 98.3% | 57 | 9 | 79.8% |
| transformer-solve | 98.7% | 42 | 12 | 77.2% |

## Setup

```bash
git clone https://github.com/dannyheskett/wof-solver.git
cd wof-solver
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
```

The trained checkpoints, the float16 server weights and the raw Wikidata
query results are in the [v1 release](https://github.com/dannyheskett/wof-solver/releases/tag/v1).

## Layout

| Path | What it is |
|---|---|
| `scripts/` | the pipeline, one script per phase (`01_fetch_puzzles.py` to `13_build_demo.py`), the model and both solvers |
| `data/` | puzzles, category lists and results. [`DATA.md`](DATA.md) documents every file, [`data/SHA256SUMS`](data/SHA256SUMS) has their checksums |
| `server/` | the C server that serves both solvers |
| `client/` | the raylib game client, compiled to WebAssembly |
| [`NOTES.md`](NOTES.md) | the rules, the results and the cost log |

## License

The code is MIT, see [`LICENSE`](LICENSE). The data sources keep their own
licenses, listed in the write-up.
