#!/usr/bin/env bash
# Phase 9. Bundle what a GPU pod needs to train: the training scripts and the
# token files from 08_make_train.py, plus runs/pre/ckpt.pt when it exists
# (run.sh finetune starts from it). Writes /tmp/wof-train.tar (about
# 1.6-1.8 GB; token files barely compress, so it is not gzipped).
set -euo pipefail
cd "$(dirname "$0")/../.."
extra=()
[ -f runs/pre/ckpt.pt ] && extra=(runs/pre/ckpt.pt)
tar -cf /tmp/wof-train.tar scripts/model.py scripts/09_train.py scripts/09_pod/run.sh data/train "${extra[@]}"
ls -la /tmp/wof-train.tar
