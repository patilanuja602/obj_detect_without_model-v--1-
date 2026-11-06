#!/usr/bin/env bash
# Baseline vs integrated app on fixed segments of the three test videos (same frames for both systems).  Usage: bash scripts/run_comparisons.sh [VIDEO_DIR] [OUT_DIR] [extra compare_baseline args]
VD=${1:-/mnt/user-data/uploads}; OUT=${2:-results/runs}; shift 2 2>/dev/null
cd "$(dirname "$0")/.."
mkdir -p "$OUT" "$OUT/review"
seg() { python3 scripts/compare_baseline.py --video "$VD/$1" --start "$2" --frames "$3" --out "$OUT/cmp_${1%.mp4}_$2.json" --review "$OUT/review" --review-every 25 "${@:4}" > "$OUT/log_${1%.mp4}_$2.txt" 2>&1; }
seg 99.mp4 0 150 "$@"
seg 99.mp4 500 150 "$@"
seg 99.mp4 1500 150 "$@"
seg 99.mp4 2500 150 "$@"
seg 99.mp4 3000 200 "$@"
seg 99.mp4 3560 150 "$@"
seg 99.mp4 4100 150 "$@"
seg 99.mp4 5640 150 "$@"
seg 12.mp4 0 150 "$@"
seg 12.mp4 2000 150 "$@"
seg nD_20.mp4 0 150 "$@"
seg nD_20.mp4 300 150 "$@"
echo ALLDONE > "$OUT/DONE"
