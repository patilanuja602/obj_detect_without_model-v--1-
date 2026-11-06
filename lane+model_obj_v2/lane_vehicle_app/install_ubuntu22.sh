#!/bin/bash
# One-time setup on Ubuntu 22.04.   Run:  bash install_ubuntu22.sh
set -e
cd "$(dirname "$0")"
sudo apt update
sudo apt install -y python3-venv python3-pip
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
if ! pip install -r requirements.txt; then
  echo "newest openvino not available for this Python; falling back to openvino 2024.6.0"
  pip install "openvino==2024.6.0" "opencv-python>=4.8" "numpy>=1.24" "psutil>=5.9"
fi
python scripts/selftest.py
echo
echo "Installed.  Run:   bash run.sh --video /path/to/video.mp4"
