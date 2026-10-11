#!/usr/bin/env bash
# On the GPU pod, from the bundle pod/pack.sh made, unpacked in /workspace.
# Renders the synthetic frames on every CPU but two, then trains the board
# decoder. Frames go on the container disk (data/vision/synth, recreated from
# seeds at will); the log and checkpoints go to runs/vision/<name>/ under
# /workspace, which is the network volume.
#
#   bash scripts/14_vision/pod/run.sh             # render, then train
#   bash scripts/14_vision/pod/run.sh train       # train only (frames exist)
set -euo pipefail
cd "$(dirname "$0")/../../.."
NAME=${NAME:-dec1}
N_TRAIN=${N_TRAIN:-300000}
# nproc shows the whole host on a pod; the container's CPU limit is in its cgroup
CPUS=$(nproc)
if [ -r /sys/fs/cgroup/cpu.max ] && read -r q p < /sys/fs/cgroup/cpu.max && [ "$q" != max ]; then
  CPUS=$(( q / p ))
fi
W=$(( CPUS > 3 ? CPUS - 1 : 2 ))
nvidia-smi --query-gpu=name,memory.total --format=csv | tee runs-gpu.txt
echo "cpus $CPUS, workers $W" | tee -a runs-gpu.txt
python3 -m pip install -q --break-system-packages pillow==12.3.0 numpy==1.26.4
python3 -c "import torch; print('torch', torch.__version__, torch.cuda.is_available())"
SYNTH=/root/synth
mkdir -p $SYNTH data/vision && ln -sfn $SYNTH data/vision/synth
if [ "${1:-}" != "train" ]; then
  for s in val test; do python3 scripts/14_vision/make_frames.py --split $s --n 2000 --workers $W; done
  python3 scripts/14_vision/make_frames.py --split train --n $N_TRAIN --workers $W 2>&1 | tee render.log
fi
cd scripts/14_vision
python3 train.py --name $NAME --steps ${STEPS:-30000} --batch 128 --workers $W \
    --eval_every 2000 2>&1 | tee ../../$NAME.log
