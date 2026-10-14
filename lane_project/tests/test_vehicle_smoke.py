"""Smoke test: a dark box ('car') sitting on a synthetic road in the lane to the right of the ego lane
must be found, and an empty road must give no boxes.   Run: python tests/test_vehicle_smoke.py"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lane_detector import Config, LaneDetector  # noqa: E402
from test_synthetic import make_frame, H  # noqa: E402
from vehicle_detector import VehicleDetector  # noqa: E402


def add_car(img, x, y, w, h):
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 160), -1)            # body
    cv2.rectangle(img, (x + w // 6, y + h // 8), (x + 5 * w // 6, y + h // 2), (200, 200, 210), -1)  # window
    cv2.rectangle(img, (x, y + h - 6), (x + w, y + h), (10, 10, 10), -1)    # shadow band
    for k in range(6):                                                         # lights / plate texture
        cv2.rectangle(img, (x + 5 + k * w // 6, y + h // 2 + 4), (x + 12 + k * w // 6, y + h // 2 + 12), (30 + 30 * (k % 2), 220, 240), -1)


def run(with_car):
    lane, veh = LaneDetector(Config(proc_width=640)), VehicleDetector()
    out = []
    for k in range(30):
        f = make_frame(0.45 * H, seed=k)
        if with_car:
            add_car(f, 370, 190, 90, 60)
        out = veh.process(lane.process(f))
    return out


def test_empty_road_gives_no_boxes():
    assert run(False) == []


def test_car_is_found():
    boxes = run(True)
    assert any(abs((v.box[0] + v.box[2]) / 2 - 415) < 50 for v in boxes), boxes


if __name__ == "__main__":
    test_empty_road_gives_no_boxes()
    print("empty ok")
    try:
        test_car_is_found()
        print("car ok")
    except AssertionError as e:
        print("car FAIL", e)
