#!/usr/bin/env python3
"""Smoke / unit tests for vehicle_detector.py (synthetic data only; no video needed).   python3 -m unittest -v test_vehicle_detector"""
import unittest
import numpy as np
import cv2
from vehicle_detector import VehicleDetector, VehicleDetection, VCFG, _iou, _ramp

W, H = 640, 360


def corridor(ext=246.0, yref=353.0, cx=320.0, n=30, wm=(2.5, -589.0), src=("OBSERVED", "OBSERVED"), state="LANE"):
    ys = np.linspace(yref, ext, n)                       # V6 order: bottom -> top
    wd = wm[0] * ys + wm[1]
    return {"state": state, "extent_y": ext, "corridor": {"ys": ys, "xl": cx - wd / 2, "xr": cx + wd / 2},
            "L": {"source": src[0]}, "R": {"source": src[1]}}


def road_frame():
    img = np.full((H, W, 3), (235, 200, 150), np.uint8)            # sky (BGR)
    yh = 231
    pts = np.array([[0, H], [W, H], [W, yh], [0, yh]], np.int32)
    cv2.fillPoly(img, [pts], (90, 90, 90))                          # asphalt
    return img


class TestHelpers(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertAlmostEqual(_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        self.assertAlmostEqual(_iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3, places=5)

    def test_ramp(self):
        self.assertEqual(_ramp(0.0, 0.2, 0.4, 0.8, 1.0), 0.0)
        self.assertEqual(_ramp(0.5, 0.2, 0.4, 0.8, 1.0), 1.0)
        self.assertAlmostEqual(_ramp(0.3, 0.2, 0.4, 0.8, 1.0), 0.5)


class TestPolicyAndRobustness(unittest.TestCase):
    def test_suppress_policy_gives_no_detections_when_not_lane(self):
        vd = VehicleDetector(W, H, cfg=dict(unreliable_policy="suppress"))
        vf = vd.process(road_frame(), {"lane": {"state": "NONE", "corridor": None}})
        self.assertEqual(vf.road_ref, "SUPPRESSED")
        self.assertEqual(vf.detections, [])

    def test_fallback_policy_is_flagged(self):
        vd = VehicleDetector(W, H, cfg=dict(unreliable_policy="fallback"))
        vf = vd.process(road_frame(), {"lane": {"state": "ONE_SIDED", "corridor": None}})
        self.assertEqual(vf.road_ref, "ROI_FALLBACK")
        self.assertTrue(vf.notes)

    def test_malformed_corridors_do_not_crash(self):
        vd = VehicleDetector(W, H)
        img = road_frame()
        bad = [None, {}, {"lane": None}, {"lane": {"state": "LANE", "corridor": None}},
               {"lane": {"state": "LANE", "corridor": {"ys": np.array([1.0]), "xl": np.array([1.0]), "xr": np.array([2.0])}}},
               {"lane": {"state": "LANE", "corridor": {"ys": np.full(30, np.nan), "xl": np.zeros(30), "xr": np.ones(30)}}},
               {"lane": {"state": "LANE", "corridor": {"ys": np.linspace(353, 246, 30), "xl": np.full(30, 400.0), "xr": np.full(30, 300.0)}}}]
        for b in bad:
            vf = vd.process(img, b)                      # must fall back (not LANE-quality) instead of raising
            self.assertIn(vf.road_ref, ("ROI_FALLBACK", "SUPPRESSED"))

    def test_blank_frame_has_no_detections(self):
        vd = VehicleDetector(W, H)
        for _ in range(5):
            vf = vd.process(road_frame(), corridor() and {"lane": corridor()})
        self.assertEqual([d for d in vf.detections if d.observed], [])


class TestAssociation(unittest.TestCase):
    def _det(self, cx, cy):
        return VehicleDetection(track_id=1, state="ACCEPTED", box=(cx - 10, cy - 20, cx + 10, cy), raw_box=(0, 0, 0, 0),
                                center=(cx, cy - 10), contact=(cx, cy), contact_method="bbox_bottom", score=0.9, parts={})

    def _geom(self, **kw):
        vd = VehicleDetector(W, H)
        g = vd._geometry(corridor(**kw))
        return vd, g

    def test_inside_left_right(self):
        vd, g = self._geom()
        d = self._det(320, 320); vd._associate(d, g); self.assertEqual(d.lane, "IN_CORRIDOR")
        d = self._det(100, 320); vd._associate(d, g); self.assertEqual(d.lane, "LEFT_OF_CORRIDOR")
        d = self._det(560, 320); vd._associate(d, g); self.assertEqual(d.lane, "RIGHT_OF_CORRIDOR")

    def test_no_extrapolation_beyond_corridor_extent(self):
        vd, g = self._geom()
        d = self._det(320, 230); vd._associate(d, g)
        self.assertEqual((d.lane, d.lane_reason), ("UNKNOWN", "BEYOND_CORRIDOR_EXTENT"))

    def test_inferred_boundary_is_not_trusted(self):
        vd, g = self._geom(src=("INFERRED", "OBSERVED"))
        d = self._det(100, 320); vd._associate(d, g)
        self.assertEqual((d.lane, d.lane_reason), ("UNKNOWN", "INFERRED_BOUNDARY"))
        d = self._det(320, 320); vd._associate(d, g)
        self.assertEqual(d.lane, "UNKNOWN")                       # cannot claim IN_CORRIDOR with one guessed side
        d = self._det(560, 320); vd._associate(d, g)
        self.assertEqual(d.lane, "RIGHT_OF_CORRIDOR")             # the observed side is still trustworthy

    def test_near_boundary_is_ambiguous(self):
        vd, g = self._geom()
        ys, xl, xr = g["corr"]
        L = float(np.interp(320, ys, xl))
        d = self._det(L + 2, 320); vd._associate(d, g)
        self.assertEqual(d.lane, "AMBIGUOUS")


class TestTracker(unittest.TestCase):
    def _cand(self, box, score=0.9):
        return VehicleDetection(track_id=-1, state="CANDIDATE", box=box, raw_box=box, center=((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
                                contact=((box[0] + box[2]) / 2, box[3]), contact_method="bbox_bottom", score=score, parts={})

    def test_confirm_coast_expire(self):
        vd = VehicleDetector(W, H)
        box = (300.0, 280.0, 340.0, 310.0)
        for _ in range(3):
            vd._track([self._cand(box)], [])
        self.assertTrue(vd._tracks[0]["confirmed"])
        tid = vd._tracks[0]["id"]
        for k in range(VCFG["max_coast"]):
            vd._track([], [])
            self.assertEqual(len(vd._tracks), 1, "track should coast for %d frames" % VCFG["max_coast"])
        vd._track([], [])
        self.assertEqual(vd._tracks, [], "track must expire after max_coast misses (no perpetual false detections)")
        vd._track([self._cand(box)], [])
        self.assertNotEqual(vd._tracks[0]["id"], tid)

    def test_weak_candidate_cannot_create_track(self):
        vd = VehicleDetector(W, H)
        vd._track([], [self._cand((300.0, 280.0, 340.0, 310.0), 0.45)])
        self.assertEqual(vd._tracks, [])

    def test_predicted_is_marked_not_observed(self):
        vd = VehicleDetector(W, H)
        box = (300.0, 280.0, 340.0, 310.0)
        for _ in range(3):
            vd._track([self._cand(box)], [])
            from vehicle_detector import VehicleFrame
            vf = vd._emit(VehicleFrame(), {"mode": "ROI_FALLBACK"}, None, False)
        vd._track([], [])
        vf = vd._emit(VehicleFrame(), {"mode": "ROI_FALLBACK"}, None, False)
        self.assertEqual(len(vf.detections), 1)
        self.assertEqual(vf.detections[0].state, "PREDICTED")
        self.assertFalse(vf.detections[0].observed)

    def test_nms_removes_duplicates(self):
        vd = VehicleDetector(W, H)
        a, b = self._cand((300.0, 280.0, 340.0, 310.0), 0.9), self._cand((302.0, 281.0, 341.0, 311.0), 0.7)
        c = self._cand((500.0, 280.0, 540.0, 310.0), 0.8)
        out = vd._nms([a, b, c])
        self.assertEqual(len(out), 2)
        self.assertIn(a, out)


class TestSyntheticVehicle(unittest.TestCase):
    """Mechanics check on a synthetic scene (NOT an accuracy test).  The detector must first see an empty road: its road-colour model
    is initialised from the first frames (documented cold-start limitation: a vehicle filling the corridor core at start-up is
    learned as 'road').  The body is textured because the tuned edge-density rule (edge_dens_min) rejects flat-colour regions."""

    def test_box_on_road_is_detected_in_corridor(self):
        lane = {"lane": corridor()}
        vd = VehicleDetector(W, H, cfg=dict(edge_dens_min=0.2))       # synthetic texture is weaker than real footage (default 0.5)
        for _ in range(5):                                           # warm-up on empty road
            vd.process(road_frame(), lane)
        x0, y0, x1, y1 = 285, 285, 355, 345                         # 70 x 60 px "vehicle" in the host lane
        rng = np.random.RandomState(0)
        img = road_frame()
        body = np.clip(np.array([40, 40, 160]) + rng.randint(-45, 46, (y1 - y0, x1 - x0, 3)), 0, 255).astype(np.uint8)
        img[y0:y1, x0:x1] = body
        cv2.rectangle(img, (x0 + 8, y0 + 8), (x1 - 8, y0 + 22), (20, 20, 20), -1)        # window
        cv2.rectangle(img, (x0 - 2, y1 - 8), (x1 + 2, y1 + 1), (15, 15, 15), -1)         # dark underside
        found = None
        for _ in range(6):
            vf = vd.process(img, lane)
            obs = [d for d in vf.detections if d.observed]
            if obs:
                found = max(obs, key=lambda d: d.score)
        self.assertIsNotNone(found, "synthetic vehicle not detected")
        self.assertGreater(_iou(found.box, (x0, y0, x1, y1)), 0.4)
        self.assertEqual(found.lane, "IN_CORRIDOR")
        self.assertAlmostEqual(found.contact[0], (x0 + x1) / 2, delta=12)
        self.assertEqual(found.state, "TRACKED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
