#!/bin/bash
# Runs BEFORE (baseline reconstruction) and AFTER (default) on the three test segments.  Usage: eval/run_sequences_all.sh UHD.mp4 HD.mp4 NMOUNTAIN.mp4
cd "$(dirname "$0")/.."
U=$1; H=$2; N=$3
for tag in before after; do
  P=default.json; [ $tag = before ] && P=baseline_reconstruction.json
  python3 eval/run_sequence.py --video "$U" --preset $P --start 24 --frames 640 --out results/seq_${tag}_uhd.pkl
  python3 eval/run_sequence.py --video "$H" --preset $P --start 20 --frames 400 --out results/seq_${tag}_hd.pkl
  python3 eval/run_sequence.py --video "$N" --preset $P --start 0  --frames 420 --out results/seq_${tag}_nd20.pkl
done
echo ALLDONE
