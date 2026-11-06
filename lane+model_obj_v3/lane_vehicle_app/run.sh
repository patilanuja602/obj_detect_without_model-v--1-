#!/usr/bin/env bash
# Launch:  ./run.sh VIDEO_OR_CAMERA_INDEX [options]      e.g.  ./run.sh drive.mp4 --show      ./run.sh drive.mp4 --save out.mp4
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate
exec python3 lane_vehicle_app.py "$@"
