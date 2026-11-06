#!/usr/bin/env bash
# One-shot setup for Ubuntu 22.04 (ThinkPad T480).  Creates .venv next to this script.  Needs internet for pip only.
set -euo pipefail
cd "$(dirname "$0")"
sudo apt update
sudo apt install -y python3-venv python3-pip ffmpeg
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt || pip install "openvino==2024.6.0" "opencv-python>=4.8" "numpy>=1.24" "psutil>=5.9"
python3 scripts/check_install.py
echo
echo "Installed.  Optional, for stable benchmark numbers (resets on reboot):"
echo "  sudo apt install -y linux-tools-common linux-tools-\$(uname -r) && sudo cpupower frequency-set -g performance"
echo "Run:  ./run.sh /path/to/video.mp4 --show"
