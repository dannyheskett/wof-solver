#!/usr/bin/env bash
# Downloads each clip in data/vision/clips.tsv at 720p and pulls one frame a
# second into data/vision/raw/<id>/frames/. Everything under raw/ is the
# show's broadcast and stays local: it is gitignored and never committed.
# Needs ffmpeg, uv and node (yt-dlp uses node to read YouTube's player).
# SOURCES limits the run to those sources (column 2 of clips.tsv), e.g.
#   SOURCES="offair phone" scripts/14_vision/fetch_clips.sh
# A clip is skipped once raw/<id>/done exists. A failed download is retried
# three times, then reported, and the rest still run. Run it again to retry.
set -uo pipefail
cd "$(dirname "$0")/../.."
fail=0
SOURCES=${SOURCES:-broadcast offair phone}
for id in $(tail -n +2 data/vision/clips.tsv | awk -F'\t' -v s=" $SOURCES " 'index(s, " " $2 " ") {print $1}'); do
  dir=data/vision/raw/$id
  [ -f "$dir/done" ] && continue
  mkdir -p "$dir/frames"
  if [ ! -s "$dir/video.mp4" ]; then
    ok=0
    for try in 1 2 3; do
      uvx --from "yt-dlp[default]" yt-dlp --js-runtimes node -q --no-progress --no-warnings \
        --sleep-requests 1 -f "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720]" \
        -o "$dir/video.mp4" "https://www.youtube.com/watch?v=$id" </dev/null && { ok=1; break; }
      sleep $((try * 20))
    done
    [ $ok = 1 ] || { echo "$id: download failed"; fail=1; continue; }
  fi
  rm -f "$dir"/frames/*.jpg
  ffmpeg -nostdin -loglevel error -i "$dir/video.mp4" -vf fps=1 -q:v 3 "$dir/frames/%04d.jpg" \
    || { echo "$id: frame extraction failed"; fail=1; continue; }
  touch "$dir/done"
  echo "$id $(ls "$dir/frames" | wc -l) frames"
  sleep 3
done
exit $fail
