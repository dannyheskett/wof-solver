# Pretraining corpus sources

Built by `scripts/05_make_corpus.py` (phase 5) into `train.txt` and `val.txt` (not
committed, about 1 GB). The raw files in `raw/` are not committed either.
Fetch them again with these URLs, pinned to the dataset revisions used:

```bash
cd data/corpus/raw
W=https://huggingface.co/datasets/wikimedia/wikipedia/resolve/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/20231101.en
for n in 00000 00010 00020 00030; do
  curl -sSL -o wiki-$n.parquet "$W/train-$n-of-00041.parquet"
done
curl -sSL -o fineweb-000_00000.parquet "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/sample/10BT/000_00000.parquet"
```

- **wikimedia/wikipedia**, config `20231101.en`, files 0, 10, 20 and 30 of
  41, revision `b04c8d1c`. Wikipedia text is licensed CC BY-SA 4.0.
- **HuggingFaceFW/fineweb-edu**, `sample/10BT`, file `000_00000`, revision
  `87f09149`. Licensed ODC-By 1.0.

SHA-256 of each file (also in `data/SHA256SUMS`):

```
382e7f6f09e488b24793a7f7cfc659879d5a22da2cf2efec6491665f0c019677  wiki-00000.parquet
cd8b4e9ee2506ba5f47d552c56e2a7569402a6814d9fa171c0e995e56859e51b  wiki-00010.parquet
79ee06627f4ba2fec056fcd23c077a29238bbcbacc3187e8d1a0b47e663e9c7a  wiki-00020.parquet
1de329963a4aaed0c40d820a564487dc98ef113c8ca6b88b11d495d6a4b98774  wiki-00030.parquet
b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871  fineweb-000_00000.parquet
```
