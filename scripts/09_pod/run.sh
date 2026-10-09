#!/usr/bin/env bash
# Phase 9, on the GPU pod, from the bundle 09_pod/pack.sh made. Pretrains the
# 59M-parameter transformer on the corpus (runs/pre), then finetunes it on
# the task mix (runs/ft2b, the model both solvers' results come from).
#
#   bash scripts/09_pod/run.sh             # pretrain, then finetune
#   bash scripts/09_pod/run.sh finetune    # finetune only, from runs/pre/ckpt.pt
#   bash scripts/09_pod/run.sh timing      # 300 steps, to measure tokens/s
#
# Batch 128 x 256 = 32,768 tokens a step. Pretrain 30,500 steps is 1.0B
# tokens, finetune 15,000 steps is 0.49B. Logs and checkpoints go to runs/.
set -euo pipefail
cd "$(dirname "$0")/../.."
nvidia-smi --query-gpu=name,memory.total --format=csv | tee runs-gpu.txt
python3 -c "import torch; print('torch', torch.__version__)"
if [ "${1:-}" = "timing" ]; then
    python3 scripts/09_train.py --name timing --phase pretrain --steps 300 \
        --eval_every 100 --warmup 50 --compile
    exit
fi
if [ "${1:-}" != "finetune" ]; then
    python3 scripts/09_train.py --name pre --phase pretrain --steps 30500 \
        --eval_every 2000 --compile 2>&1 | tee pre.log
fi
python3 scripts/09_train.py --name ft2b --phase finetune --init runs/pre/ckpt.pt \
    --steps 15000 --eval_every 1000 --lr 3e-4 --warmup 200 --compile 2>&1 | tee ft2b.log
