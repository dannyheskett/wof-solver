#!/usr/bin/env bash
# Round 2 on the GPU pod, after run.sh trained dec1: label the real frames,
# drop doubtful labels, render 20,000 more synthetic frames (with title
# cards), and finetune dec1 into dec2 with 30% real frames a batch.
#
# Needs on the pod: the bundle from pack.sh unpacked here, data/puzzles.csv
# and data/wordlist/ beside it, the round-1 frames in data/vision/synth/,
# runs/vision/dec1/best.pt, and shrink_frames.py's real frames in REAL_ROOT.
#
#   REAL_ROOT=/root/real/raw_lb bash scripts/14_vision/pod/round2.sh
#
# The labels and dec2 used for the results in NOTES.md were made this way,
# except that the frames were labelled twice: once with dec1 (labels.jsonl,
# which trained dec2) and once with dec2 after label_real.py tracked reveals
# square by square (labels2.jsonl, which the real scores are measured on).
set -euo pipefail
cd "$(dirname "$0")/../../.."
REAL_ROOT=${REAL_ROOT:?the letterboxed real frames}
REAL=data/vision/real
mkdir -p $REAL
cd scripts/14_vision
python3 label_real.py --ckpt ../../runs/vision/dec1/best.pt --raw "$REAL_ROOT" \
    --out ../../$REAL/labels.jsonl --threads 6 > ../../$REAL/label.log 2>&1
python3 clean_labels.py ../../$REAL/labels.jsonl
python3 make_frames.py --split train --n 20000 --start 100000 --workers 6
python3 train.py --name dec2 --init ../../runs/vision/dec1/best.pt --steps 8000 --lr 3e-4 --warmup 200 \
    --batch 128 --workers 6 --eval_every 1000 --real ../../$REAL/labels_clean.jsonl \
    --real_root "$REAL_ROOT" --real_frac 0.3 2>&1 | tee ../../dec2.log
