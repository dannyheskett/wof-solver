#!/usr/bin/env bash
# Bundles what a GPU pod needs to render the frames and train the board
# decoder: the vision scripts, wof_common.py, the puzzle files the renderer
# draws from, and the fonts (copied into data/vision/fonts/ first, so local
# renders use the same ones). Writes /tmp/wof-vision.tar.gz.
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p data/vision/fonts
for f in /usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf \
         /usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf \
         /usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf \
         /usr/share/fonts/truetype/freefont/FreeSansBold.ttf \
         /usr/share/fonts/opentype/urw-base35/NimbusSansNarrow-Bold.otf \
         /usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf; do
  cp -n "$f" data/vision/fonts/
done
tar -czf /tmp/wof-vision.tar.gz scripts/wof_common.py scripts/14_vision/*.py scripts/14_vision/pod \
    data/train.csv data/dev.csv data/control.csv data/vision/fonts
ls -la /tmp/wof-vision.tar.gz
