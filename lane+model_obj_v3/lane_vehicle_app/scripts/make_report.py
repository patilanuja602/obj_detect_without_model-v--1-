#!/usr/bin/env python3
"""Aggregates results/runs/cmp_*.json (written by compare_baseline.py) into a markdown table: python3 scripts/make_report.py results/runs"""
import json, sys
from pathlib import Path
d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/runs")
rows = []
for f in sorted(d.glob("cmp_*.json"), key=lambda p: (p.stem.split("_")[1], int(p.stem.split("_")[2]))):
    r = json.load(open(f)); m = r["meta"]
    b, v = r["baseline"], r["v7"]
    rows.append((Path(m["video"]).name, m["start"], m["frames"], b, v, r["lane_diff"], r["v7"].get("zones", {}), r["v7"]["tracker_stats"]))
def pc(x): return "%.0f%%" % (100 * x) if x is not None else "-"
def fx(x, n=2): return ("%." + str(n) + "f") % x if x is not None else "-"
print("| video @start (frames) | LANE % base→v7 | ONE_SIDED % | NONE % | jitter px base→v7 | width jumps | state differs | boxes/frame | dup pairs | flicker gaps | class chg | lane overlay on vehicles base→v7 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
tot = dict(bl=0, vl=0, n=0)
for vid, s, n, b, v, ld, z, ts in rows:
    bl, vl = b["lane"], v["lane"]
    print("| %s @%d (%d) | %s → %s | %s → %s | %s → %s | %s → %s | %d → %d | %s | %s → %s | %d → %d | %d → %d | %d → %d | %s → %s |" % (
        vid, s, n, pc(bl["LANE"]), pc(vl["LANE"]), pc(bl["ONE_SIDED"]), pc(vl["ONE_SIDED"]), pc(bl["NONE"]), pc(vl["NONE"]),
        fx(bl["jitter_px_mean"]), fx(vl["jitter_px_mean"]), bl["width_jumps"], vl["width_jumps"], pc(ld["state_differs"]),
        fx(b["vehicle"]["boxes_per_frame"]), fx(v["vehicle"]["boxes_per_frame"]), b["vehicle"]["duplicate_pairs"], v["vehicle"]["duplicate_pairs"],
        b["vehicle"]["flicker_gaps"], v["vehicle"]["flicker_gaps"], b["vehicle"]["class_changes"], v["vehicle"]["class_changes"],
        fx(b["vehicle"]["overlay_px_in_vehicles"], 3), fx(v["vehicle"]["overlay_px_in_vehicles"], 3)))
    tot["bl"] += bl["LANE"] * n; tot["vl"] += vl["LANE"] * n; tot["n"] += n
print("\nweighted LANE share over all segments: baseline %s, v7 %s (%d frames)" % (pc(tot["bl"] / tot["n"]), pc(tot["vl"] / tot["n"]), tot["n"]))
print("\nzones of displayed vehicles (v7), per segment:")
for vid, s, n, b, v, ld, z, ts in rows:
    print("- %s @%d: %s | tracker %s" % (vid, s, z, ts))
