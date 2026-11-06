#!/bin/bash
# Launch the application inside the virtual environment.   bash run.sh --video /path/to/video.mp4  [--output file] [--autotune] ...
cd "$(dirname "$0")"
source .venv/bin/activate
python run_app.py "$@"
