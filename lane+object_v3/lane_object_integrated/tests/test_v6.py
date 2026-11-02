#!/usr/bin/env python3
"""Behavioural tests for lane_path_v6 (imports ONLY lane_path_v6).  Run:  python3 tests/test_v6.py
Each test drives the real _track/_classify/draw code with synthetic boundary measurements (the dict format _fit_side returns) and checks
(a) the internal decision (result['lane']) and (b) the DISPLAYED pixels (what draw() paints on a blank frame).
Global rule checked on EVERY rendered frame of every test: no amber / blue / red lane colours are ever drawn - the overlay is green only."""
import os, re, sys
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import lane_path_v6 as m
CFG = dict(m.CFG); CFG.update(m.PROFILES["mountain"])
BG = 128; VIOL = []
HUD = []
_put = cv2.putText
def _spy(img, text, *a, **k):
    HUD.append(text); return _put(img, text, *a, **k)
cv2.putText = _spy

def mk(cfg=None, cfg6=None):
    c = dict(CFG); c.update(cfg or {})
    d = m.PathDetectorV6(640, 360, c, cfg6=dict(dict(overlay_top_y=0), **(cfg6 or {}))); d._yref = d.y1 - 1; return d   # T01-T24: overlay clip OFF (they test the full corridor)
def coef(d, xb, vx, vy, curv=0.0): return (curv, (vx - xb) / (vy - d._yref) * d.H, float(xb))
def meas(c, ytop, ridge=1.0, edge=1.0, mode="A"):
    return dict(coef=tuple(c), y_top=float(ytop), ridge_frac=ridge, edge_frac=edge, mode=mode, windows=None, seed_len=300.0, seed_ridge=10.0)
def step(d, Lm, Rm):
    out = {s: d._track(s, mm, d._yref) for s, mm in (("L", Lm), ("R", Rm))}
    d.result = d._classify(out, d._yref); return d.result
def good(d, vp=(320, 215), xl=130, xr=480, top=None, **kw):
    top = d._yref - 0.9 * d.H if top is None else top
    return meas(coef(d, xl, vp[0], vp[1]), top, **kw), meas(coef(d, xr, vp[0], vp[1]), top, **kw)
def warm(d, n=6, **kw):
    for _ in range(n): r = step(d, *good(d, **kw))
    return r
def blend(c): return 0.7 * BG + 0.3 * np.array(c)
def render(d):
    HUD.clear(); img = np.full((720, 1280, 3), BG, np.uint8)
    out = d.draw(img, None, None, None, False); body = out[34:].astype(int)
    near = lambda col, tol: np.abs(body - np.array(col)).sum(2) < tol
    G = near(blend(m.GREEN), 14); G &= (G.sum(1, keepdims=True) >= 20)                      # fill pixels (>=20 per row)
    line = near(d.c6["line_bgr"], 90) & ~G                                                   # boundary-line pixels (green, any style)
    bad = near(m.AMBER, 60) | near(blend(m.AMBER), 40) | near(m.LEFT_C, 90) | near(m.RIGHT_C, 90)
    if bad.any(): VIOL.append("non-green lane colour drawn (%d px)" % int(bad.sum()))
    top = lambda mask: None if not mask.any() else (np.nonzero(mask.any(1))[0][0] + 34) / 2.0
    half = body.shape[1] // 2
    return dict(img=out, fill_top=top(G), fill_n=int(G.sum()), line_top=top(line), line_n=int(line.sum()),
                left_n=int(line[:, :half].sum()), right_n=int(line[:, half:].sum()), hud=HUD[-1] if HUD else "")
def ok(c, msg):
    if not c: raise AssertionError(msg)

RES = []
def test(name):
    def deco(f):
        VIOL.clear()
        try: f(); r = True; why = ""
        except AssertionError as e: r = False; why = str(e)
        except Exception as e: r = False; why = "ERROR %r" % e
        if r and VIOL: r = False; why = VIOL[0]
        RES.append(r); print("[%s] %s%s" % ("PASS" if r else "FAIL", name, "" if r else "\n        -> " + why)); return f
    return deco
LANE = "LANE"

@test("T01 both boundaries valid: green corridor + both lines")
def _():
    d = mk(); r = warm(d); ln = r["lane"]; rd = render(d)
    ok(ln["state"] == LANE and ln["L"]["source"] == "OBSERVED" and ln["R"]["source"] == "OBSERVED", ln["state"])
    ok(rd["fill_n"] > 0 and rd["left_n"] > 0 and rd["right_n"] > 0, "fill/lines missing")

@test("T02 left detected, right missing, ROI edge plausible -> inferred right, green corridor (Case A)")
def _():
    d = mk(cfg=dict(bot_x=(0.0, 0.76), top_x=(0.22, 0.55)))            # ROI whose right edge is a plausible lane edge
    for _ in range(50): step(d, good(d)[0], good(d)[1])                 # builds lane-width history
    for _ in range(40): r = step(d, good(d)[0], None)
    ln = r["lane"]; rd = render(d)
    ok(ln["state"] == LANE and ln["R"]["source"] == "INFERRED" and ln["R"]["type"] == m.EST, "%s R=%s %s" % (ln["state"], ln["R"] and ln["R"]["source"], ln["note"]))
    ok(rd["fill_n"] > 0 and rd["left_n"] > 0 and rd["right_n"] > 0, "corridor/lines not drawn")

@test("T03 right detected, left missing, ROI edge plausible -> inferred left (Case B)")
def _():
    d = mk(cfg=dict(bot_x=(0.20, 1.0), top_x=(0.22, 0.74)))
    for _ in range(50): step(d, *good(d))
    for _ in range(40): r = step(d, None, good(d)[1])
    ln = r["lane"]; rd = render(d)
    ok(ln["state"] == LANE and ln["L"]["source"] == "INFERRED", "%s %s" % (ln["state"], ln["note"]))
    ok(rd["fill_n"] > 0 and rd["left_n"] > 0, "not drawn")

@test("T04 one boundary tracked (hidden for 3 frames): tracked position preferred over ROI edge (Case C)")
def _():
    d = mk(); warm(d); xs = d.state["R"]["coef"][2]
    for _ in range(3): r = step(d, good(d)[0], None)
    ln = r["lane"]; ok(ln["state"] == LANE and ln["R"]["source"] == "TRACKED", "%s %s" % (ln["state"], ln["R"]))
    ok(abs(ln["R"]["coef"][2] - xs) < 1.0, "tracked R moved to %.1f (was %.1f)" % (ln["R"]["coef"][2], xs))
    ok(render(d)["right_n"] > 0, "tracked boundary not drawn")

@test("T05 both boundaries temporarily missing: lane kept <= max_hold2 frames, then removed (nothing stale drawn)")
def _():
    d = mk(); warm(d); k = d.c2["max_hold2"]; st = [step(d, None, None)["lane"]["state"] for _ in range(k + 3)]
    ok(all(x == LANE for x in st[:k]) and all(x == "NONE" for x in st[k:]), "states %s" % st)
    rd = render(d); ok(rd["fill_n"] == 0 and rd["line_n"] == 0, "stale lane still drawn")

@test("T06 one-frame dropout: lane stays (dashed), recovers to OBSERVED")
def _():
    d = mk(); warm(d); r = step(d, None, None); ok(r["lane"]["state"] == LANE and r["lane"]["L"]["source"] == "TRACKED", "dropout")
    ok(render(d)["fill_n"] > 0, "fill gone during dropout")
    r = step(d, *good(d)); ok(r["lane"]["state"] == LANE and r["lane"]["L"]["source"] == "OBSERVED", "no recovery")

@test("T07 several consecutive misses then reacquisition: smooth, solid lines again (T17)")
def _():
    d = mk(); warm(d); [step(d, None, None) for _ in range(4)]
    for _ in range(3): r = step(d, *good(d))
    ln = r["lane"]; ok(ln["state"] == LANE and ln["L"]["source"] == "OBSERVED" and abs(ln["L"]["coef"][2] - 130) < 2, str(ln["L"]))

@test("T08 small far-field crossing: valid ~80+% of the corridor is KEPT, invalid end clipped, everything agrees")
def _():
    d = mk(); Ln = lambda: meas(coef(d, 130, 322, 218), d.y0); Rn = lambda: meas(coef(d, 480, 318, 218), d.y0)    # curves cross just below the ROI top
    for _ in range(6): r = step(d, Ln(), Rn())
    ln = r["lane"]; ok(ln["state"] == LANE, "whole lane removed: %s" % ln["note"])
    frac = (d._yref - ln["ext_valid"]) / d.H; ok(0.6 < frac < 1.0, "valid fraction %.2f (expected most of the corridor)" % frac)
    rd = render(d); e = ln["extent_y"]
    ok(abs(rd["fill_top"] - e) < 3 and rd["line_top"] >= e - 3, "fill %.1f line %.1f extent %.1f" % (rd["fill_top"], rd["line_top"], e))
    c = ln["corridor"]; ok((c["xr"] - c["xl"]).min() >= d.c4["corr_min_w"] - 1e-6, "drawn rows narrower than min width")

@test("T09 crossing closer to the camera: only the valid section (~40%) is shown; if nothing valid remains -> removed")
def _():
    d = mk(); Ln = lambda: meas(coef(d, 130, 330, 262), d.y0); Rn = lambda: meas(coef(d, 480, 310, 262), d.y0)
    for _ in range(6): r = step(d, Ln(), Rn())
    ln = r["lane"]; frac = (d._yref - (ln["ext_valid"] or d._yref)) / d.H
    ok(ln["state"] == LANE and 0.2 < frac < 0.7, "%s frac %.2f note %s" % (ln["state"], frac, ln["note"]))
    rd = render(d); ok(rd["fill_top"] is not None and rd["line_top"] >= ln["extent_y"] - 3, "extent mismatch")
    d2 = mk(); Ln = lambda: meas(coef(d2, 130, 330, 340), d2.y0); Rn = lambda: meas(coef(d2, 480, 310, 340), d2.y0)   # crosses almost at the bottom
    for _ in range(6): r = step(d2, Ln(), Rn())
    ok(r["lane"]["state"] != LANE and render(d2)["fill_n"] == 0, "implausible sliver was displayed")

def shifted(d, dx):
    L, R = good(d); c = list(L["coef"]); c[2] += dx; L["coef"] = tuple(c); return L, R
@test("T10 one boundary jumps because of a false measurement: lane kept, jump NOT blended, recovers")
def _():
    d = mk(); warm(d); before = d.state["L"]["coef"][2]; r = step(d, *shifted(d, 60))
    ok(r["lane"]["state"] == LANE, "lane removed by one outlier"); ok(abs(d.state["L"]["coef"][2] - before) < 1.0, "outlier leaked into track")
    ok(r["lane"]["L"]["source"] == "TRACKED" and r["lane"]["L"]["outlier"], "outlier not flagged")
    r = step(d, *good(d)); ok(r["lane"]["L"]["source"] == "OBSERVED" and not r["lane"]["L"]["outlier"], "no recovery")

@test("T11 repeated consistent conflicting measurements = genuine change: lane kept meanwhile, then replaced")
def _():
    d = mk(); warm(d); sts = []
    for _ in range(3): r = step(d, *shifted(d, 45)); sts.append(r["lane"]["state"])
    ok(all(s == LANE for s in sts), str(sts)); ok(abs(d.state["L"]["coef"][2] - 175) < 2, "track did not move (%.1f)" % d.state["L"]["coef"][2])

@test("T12 persistent mutually inconsistent conflicts: held only a bounded time (no stale lane forever)")
def _():
    d = mk(); warm(d); sts = []
    for dx in (60, -60, 90, -90, 40, -120, 70, -45, 110, -80, 55, -100): r = step(d, *shifted(d, dx)); sts.append(r["lane"]["state"])      # never two consistent in a row
    ok(sts[0] == LANE and sts[-1] != LANE, str(sts)); ok(d.stats["L:conflict_expire"] >= 1, "held boundary never expired")

@test("T13 genuine curved-road transition: no dropout, tracking lag < 3 px")
def _():
    sts = []; d = mk()
    for k in range(70):
        cv = 0.6 * k; Ln = meas(coef(d, 130, 320, 215, curv=-cv), d._yref - 0.9 * d.H); Rn = meas(coef(d, 480, 320, 215, curv=-cv), d._yref - 0.9 * d.H)
        r = step(d, Ln, Rn); sts.append(r["lane"]["state"])
        if k > 8:
            tt = np.linspace(-0.8, 0, 5); err = np.abs(np.polyval(d.state["L"]["coef"], tt) - np.polyval(Ln["coef"], tt)).max(); ok(err < 3.0, "lag %.1f px at frame %d" % (err, k))
    ok(all(s == LANE for s in sts[8:]), "dropouts: %d" % sum(s != LANE for s in sts[8:]))

@test("T14 dashed marking disappearing for 2 of every 4 frames: lane never removed, gaps drawn dashed")
def _():
    d = mk(); warm(d); sts = []
    for k in range(40):
        gap = (k % 4) in (2, 3); r = step(d, None if gap else good(d)[0], None if gap else good(d)[1]); sts.append(r["lane"]["state"])
        if gap: ok(r["lane"]["L"]["source"] == "TRACKED", "gap frame must be TRACKED")
    ok(all(s == LANE for s in sts), str(sts))

@test("T15 concrete wall that legitimately borders the road: 'edge' type accepted, green lane (no orange)")
def _():
    d = mk()
    for _ in range(40): r = step(d, good(d, ridge=0.1, edge=1.0)[0], good(d)[1])                  # left = wall base (edge evidence, no paint ridge), right = paint
    ln = r["lane"]; ok(ln["state"] == LANE and ln["L"]["type"] == m.ROAD_EDGE, "%s L=%s" % (ln["state"], ln["L"] and ln["L"]["type"]))
    ok(r["mode"] == "ROAD_CORRIDOR" and render(d)["fill_n"] > 0, "classification label should be ROAD_CORRIDOR but the display must still show the lane")

@test("T16 roadside divider appearing for 3 frames at a far offset: corridor not pulled, lane kept")
def _():
    d = mk()
    for _ in range(60): step(d, *good(d))                                  # lane-width history (needed to recognise a much wider corridor)
    before = d.state["R"]["coef"][2]
    for _ in range(3):
        L, R = good(d); c = list(R["coef"]); c[2] = 700; R["coef"] = tuple(c); r = step(d, L, R); ok(r["lane"]["state"] == LANE, "lane removed")
    ok(abs(d.state["R"]["coef"][2] - before) < 1.0, "divider pulled the boundary (%.1f -> %.1f)" % (before, d.state["R"]["coef"][2]))

@test("T17 divider as the ONLY 'boundary' (weak evidence, never supported): not drawn as a lane side")
def _():
    d = mk()
    for _ in range(20): r = step(d, good(d, ridge=0.05, edge=0.05)[0], good(d, ridge=0.05, edge=0.05)[1])
    ok(r["lane"]["state"] != LANE and render(d)["fill_n"] == 0, "unsupported boundaries produced a lane: %s" % r["lane"]["state"])

@test("T18 vehicle edge (near-vertical) detected as left boundary: never becomes a lane side; established boundary survives it")
def _():
    d = mk()
    for _ in range(20): r = step(d, meas((0.0, 4.0, 300.0), d._yref - 0.5 * d.H), good(d)[1])       # near-vertical = vehicle side / pole
    ok(r["lane"]["state"] != LANE and render(d)["fill_n"] == 0, "vehicle edge accepted as lane boundary")
    d = mk(); warm(d)
    for _ in range(2): r = step(d, meas((0.0, 4.0, 300.0), d._yref - 0.5 * d.H), good(d)[1]); ok(r["lane"]["state"] == LANE, "established lane removed by vehicle edge")
    ok(abs(d.state["L"]["coef"][2] - 130) < 1.0, "vehicle edge pulled the boundary")

@test("T19 unsuitable ROI fallback (default ROI -> corridor 1.4x lane width): no invented corridor, thin one-sided line, limitation exposed")
def _():
    d = mk()
    for _ in range(50): step(d, *good(d))
    for _ in range(25): r = step(d, good(d)[0], None)
    ln = r["lane"]; rd = render(d)
    ok(ln["state"] == "ONE_SIDED" and not ln["fb"]["accepted"] and "ROI fallback rejected" in ln["note"], "%s | %s" % (ln["state"], ln["note"]))
    ok(rd["fill_n"] == 0 and rd["left_n"] > 0 and rd["right_n"] == 0 and "ONE BOUNDARY" in rd["hud"], "drawing/HUD wrong: %s" % rd["hud"])

@test("T19b plausible-width ROI edge that would make the boundary JUMP from where it was displayed is rejected (no abrupt edge change)")
def _():
    d = mk(cfg=dict(bot_x=(0.094, 1.0)))                                   # ROI left edge ~60 px: width 420 (1.2x lane) -> plausible by width alone
    for _ in range(50): step(d, *good(d))
    for _ in range(12): r = step(d, None, good(d)[1])
    ln = r["lane"]; ok(ln["state"] == "ONE_SIDED" and "last displayed" in ln["note"], "%s | %s" % (ln["state"], ln["note"]))

@test("T20 fill, boundary lines and HUD share the same far end (open / converging / crossing / flare / inferred)")
def _():
    cases = [dict(vp=(320, 215)), dict(vp=(320, 232)), dict(vp=(320, 232), top=None)]
    for vp in ((320, 215), (320, 232), (320, 226)):
        d = mk(); r = warm(d, 6, vp=vp, top=d.y0 + 2); ln = r["lane"]
        if ln["state"] != LANE: continue
        rd = render(d); e = ln["extent_y"]
        ok(abs(rd["fill_top"] - e) < 3 and rd["line_top"] >= e - 3, "vp %s: fill %.1f line %.1f ext %.1f" % (vp, rd["fill_top"], rd["line_top"], e))
        mm = re.search(r"to (\d+)% of ROI", rd["hud"]); ok(mm and abs(int(mm.group(1)) - 100 * (d._yref - e) / d.H) <= 1, "HUD %r" % rd["hud"])
    d = mk(); Ln = lambda: meas(coef(d, 130, 320, 215), d.y0); Rn = lambda: meas(coef(d, 480, 320, 215, curv=300.0), d.y0)
    for _ in range(6): r = step(d, Ln(), Rn())
    ln = r["lane"]; ok(ln["state"] == LANE and ln["kind"] == "diverged", "flare not cut: %s %s" % (ln["state"], ln.get("kind")))
    rd = render(d); ok(abs(rd["fill_top"] - ln["extent_y"]) < 3 and rd["line_top"] >= ln["extent_y"] - 3, "flare: endpoints differ")

@test("T21 genuine road widening: gradual +17% accepted without dropout; abrupt jump accepted only after longer agreement")
def _():
    d = mk(); sts = []
    for k in range(90): sts.append(step(d, meas(coef(d, 130, 320, 215), d._yref - 0.9 * d.H), meas(coef(d, 480 + min(k, 50) * 1.2, 320, 215), d._yref - 0.9 * d.H))["lane"]["state"])
    ok(all(s == LANE for s in sts[d.c6["acq_n"]:]), "dropouts on gradual widening")
    d = mk()
    for _ in range(60): step(d, *good(d))
    n = 0
    while n < 30:
        R = good(d)[1]; c = list(R["coef"]); c[2] = 680; R["coef"] = tuple(c); r = step(d, good(d)[0], R); n += 1
        ok(r["lane"]["state"] == LANE, "lane removed during the widening transition")
        if abs(d.state["R"]["coef"][2] - 680) < 2: break
    ok(d.c2["reacquire_n"] * 2 <= n <= 30, "wide jump accepted after %d frames (expected >= %d)" % (n, d.c2["reacquire_n"] * 2))

@test("T21b abrupt NARROWING from a false boundary (road symbol inside the lane): needs 2x longer agreement, lane never removed")
def _():
    d = mk()
    for _ in range(60): step(d, *good(d))
    n = 0
    while n < 30:
        L = good(d)[0]; c = list(L["coef"]); c[2] = 330; L["coef"] = tuple(c); r = step(d, L, good(d)[1]); n += 1       # left boundary jumps to mid-lane: width 480-330 = 150 (0.43x)
        ok(r["lane"]["state"] == LANE, "lane removed during the narrowing transition")
        if abs(d.state["L"]["coef"][2] - 330) < 2: break
    ok(d.c2["reacquire_n"] * 2 <= n <= 30, "false narrow boundary accepted after %d frames (expected >= %d)" % (n, d.c2["reacquire_n"] * 2))

@test("T22 hysteresis: alternating evidence never flickers; sustained loss of evidence DOES remove the lane (bounded)")
def _():
    d = mk(); warm(d, 6); sts = []
    for k in range(20):
        L, R = good(d); L["ridge_frac"] = 0.2 if k % 2 else 1.0; sts.append(step(d, L, R)["lane"]["state"])
    ok(all(s == LANE for s in sts), "alternating evidence flickered")
    sts = []
    for k in range(40):
        L, R = good(d); L["ridge_frac"] = 0.05; L["edge_frac"] = 0.05; sts.append(step(d, L, R)["lane"]["state"])
    lim = d.c6["type_dn_n"] + d.c6["unsup_grace"] + 2
    ok(sts[0] == LANE and sts[-1] != LANE and sts.index("ONE_SIDED") <= lim, "removal not bounded: first non-lane at %s (limit %d)" % ([i for i, s in enumerate(sts) if s != LANE][:1], lim))

@test("T23 no lane evidence at all: nothing drawn")
def _():
    d = mk()
    for _ in range(6): r = step(d, None, None)
    rd = render(d); ok(r["lane"]["state"] == "NONE" and rd["fill_n"] == 0 and rd["line_n"] == 0, "drew something without evidence")

@test("T24 display is green in EVERY lane state (observed / tracked / inferred / one-sided / corridor-clipped)")
def _():
    d = mk(cfg=dict(bot_x=(0.0, 0.76), top_x=(0.22, 0.55)))
    seen = set()
    for _ in range(50): step(d, *good(d)); render(d)
    for k in range(30):
        r = step(d, good(d)[0], None if k < 20 else good(d)[1]); render(d); seen.add(r["lane"]["R"]["source"] if r["lane"]["R"] else "-")
    for _ in range(3): r = step(d, None, None); render(d); seen.add(r["lane"]["L"]["source"])
    d2 = mk(); Ln = lambda: meas(coef(d2, 130, 322, 218), d2.y0); Rn = lambda: meas(coef(d2, 480, 318, 218), d2.y0)
    for _ in range(6): step(d2, Ln(), Rn()); render(d2)
    ok({"OBSERVED", "INFERRED", "TRACKED"} <= seen, "states exercised: %s" % seen)       # colour violations are collected by render() and fail the test

# ------------------------------------------------------------------ display-only overlay clip (T25-T31)
import copy
def snap(lane):            # deep copy of everything a downstream consumer reads
    return copy.deepcopy({k: lane[k] for k in ("state", "extent_y", "ext_valid", "pts_L", "pts_R", "corridor", "kind", "note") if k in lane})
def same(a, b):
    if isinstance(a, dict): return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, np.ndarray): return b is not None and a.shape == b.shape and np.array_equal(a, b)
    return a == b
def fill_rows(img, top=34):
    body = img[top:].astype(int); G = np.abs(body - np.array(blend(m.GREEN))).sum(2) < 14; G &= (G.sum(1, keepdims=True) >= 20)
    r = np.nonzero(G.any(1))[0]; return (r[0] + top, r[-1] + top) if len(r) else (None, None)

@test("T25 green fill and lines stop at overlay_top_y; everything below that row is pixel-identical to the unclipped overlay")
def _():
    YC = 260
    a, b = mk(), mk(cfg6=dict(overlay_top_y=YC)); warm(a, 8); warm(b, 8)
    ok(a.result["lane"]["extent_y"] < YC - 10, "test setup: corridor must reach well above the clip row")
    ra, rb = render(a), render(b)
    ok(abs(rb["fill_top"] - YC) <= 1.0, "fill top %.1f != clip row %d" % (rb["fill_top"], YC))
    ok(rb["line_top"] >= YC - 1.0, "boundary line continues above the clip row: %.1f" % rb["line_top"])
    ok(ra["fill_top"] < YC - 10 and ra["fill_n"] > rb["fill_n"], "unclipped reference should extend higher")
    lo = int(YC * 2) + 14      # = one corridor sample step (~9 display px) + line thickness: only the last partial segment next to the cut may differ
    ok(np.array_equal(ra["img"][lo:], rb["img"][lo:]), "pixels below the clip row differ from the original overlay")
    band = np.abs(ra["img"][int(YC * 2) + 1:lo].astype(int) - rb["img"][int(YC * 2) + 1:lo].astype(int))      # sub-pixel anti-aliasing of the cut segment's rounded endpoint
    gm = lambda im: (np.abs(im[int(YC * 2) + 1:lo].astype(int) - np.array(blend(m.GREEN))).sum(2) < 14)
    ok(np.abs(gm(ra["img"]).sum(1) - gm(rb["img"]).sum(1)).max() <= 4, "fill width just below the cut differs from the original")   # fill extent identical (+-line-edge AA pixels)
    ok((band.sum(2) > 0).sum() <= 40 and band.max() <= 80, "too much change just below the cut: %d px, max %d" % ((band.sum(2) > 0).sum(), band.max()))
    ok((rb["img"][34:int(YC * 2) - 6] == BG).all(), "something is drawn above the clip row")      # 6 display px = cap of the 3-px boundary line (+AA); the fill itself ends exactly on the row

@test("T26 internal geometry and lane state are identical with and without clipping (all of result['lane'], extent, corridor, pts_L/R); draw() does not mutate them")
def _():
    a, b = mk(), mk(cfg6=dict(overlay_top_y=270))
    seq = [(good(a), "both")] * 8 + [(good(a), "L")] * 25 + [(good(a), "none")] * 14 + [(good(a), "both")] * 6
    for (Lm, Rm), mode in seq:
        Lm2 = {k: (copy.deepcopy(v)) for k, v in Lm.items()}; Rm2 = {k: copy.deepcopy(v) for k, v in Rm.items()}
        ra = step(a, Lm if mode != "none" else None, Rm if mode == "both" else None)
        rb = step(b, Lm2 if mode != "none" else None, Rm2 if mode == "both" else None)
        before = snap(rb["lane"]); partial_b = copy.deepcopy(rb["partial"]); render(b); render(a)
        ok(same(before, snap(rb["lane"])) and same(partial_b, rb["partial"]), "draw() mutated result['lane']/partial")
        ok(same(snap(ra["lane"]), snap(rb["lane"])), "lane geometry/state differs with clipping (%s)" % mode)
        ok(same(ra["corridor"], rb["corridor"]) and ra["extent_y"] == rb["extent_y"] and ra["mode"] == rb["mode"], "corridor/extent/mode differ")

@test("T27 LANE / ONE_SIDED / NONE: ONE_SIDED line is clipped at the same row, NONE draws nothing, states unchanged")
def _():
    YC = 260; d = mk(cfg6=dict(overlay_top_y=YC)); seen = set()
    for _ in range(8): step(d, *good(d))
    seen.add(render(d)["hud"][:4])
    for _ in range(40): r = step(d, good(d)[0], None); rd = render(d)
    ln = r["lane"]; seen.add(ln["state"])
    ok(ln["state"] in ("LANE", "ONE_SIDED"), "state %s" % ln["state"])
    if ln["state"] == "ONE_SIDED": ok(rd["fill_n"] == 0 and rd["line_top"] >= YC - 1.0, "one-sided line above clip row: %s" % rd["line_top"])
    else: ok(rd["fill_top"] >= YC - 1.0 and rd["line_top"] >= YC - 1.0, "inferred lane above clip row")
    d2 = mk(cfg6=dict(overlay_top_y=YC))
    for _ in range(6): r2 = step(d2, None, None)
    rd2 = render(d2); ok(r2["lane"]["state"] == "NONE" and rd2["fill_n"] == 0 and rd2["line_n"] == 0, "NONE drew something")

@test("T28 clip row above the whole corridor (or 0 / frac 0) = original overlay bit-for-bit; clip row below the whole corridor = nothing drawn but lane state still LANE")
def _():
    a, b, c, e = mk(), mk(cfg6=dict(overlay_top_y=-50)), mk(cfg6=dict(overlay_top_y=None, overlay_top_frac=0.0)), mk(cfg6=dict(overlay_top_y=500))
    for d in (a, b, c, e): warm(d, 8)
    ra, rb, rc, re_ = render(a), render(b), render(c), render(e)
    ok(np.array_equal(ra["img"], rb["img"]) and np.array_equal(ra["img"], rc["img"]), "clip row above corridor changed the overlay")
    ok(e.result["lane"]["state"] == LANE and re_["fill_n"] == 0 and re_["line_n"] == 0, "clip below corridor: state/drawing wrong")

@test("T29 coordinate systems: clip row is in PROCESSING px and scaled with the corridor (display 1x, 2x, 3x) ; default fraction follows the processing height (360 and 480)")
def _():
    YC = 255; d = mk(cfg6=dict(overlay_top_y=YC)); warm(d, 8)
    for sc in (1, 2, 3):
        img = np.full((360 * sc, 640 * sc, 3), BG, np.uint8); out = d.draw(img, None, None, None, False)
        top, bot = fill_rows(out); ok(abs(top - YC * sc) <= 1.5, "display x%d: fill top %s expected %d" % (sc, top, YC * sc))
    c = dict(CFG); d2 = m.PathDetectorV6(640, 480, c, cfg6=None); d2._yref = d2.y1 - 1; warm(d2, 8)       # default overlay_top_frac on a 640x480 processing frame
    out = d2.draw(np.full((480, 640, 3), BG, np.uint8), None, None, None, False); top, _ = fill_rows(out)
    exp = d2.c6["overlay_top_frac"] * 480; ok(abs(top - exp) <= 1.5, "640x480: fill top %s expected %.1f" % (top, exp))

@test("T30 _clip_top: interpolates the boundary exactly at the cut row, keeps prefix points, never touches its input, no triangular artefact")
def _():
    pts = np.stack([np.linspace(100, 300, 11), np.linspace(350, 150, 11)], 1); orig = pts.copy()      # straight line x = 100 + (350 - y) -> 1:1
    q = m._clip_top(pts, 233.0); ok(np.array_equal(pts, orig), "input modified")
    ok(np.allclose(q[-1], [100 + (350 - 233.0), 233.0]), "wrong interpolation: %s" % q[-1])
    ok(np.array_equal(q[:-1], pts[:len(q) - 1]) and (q[:, 1] >= 233.0 - 1e-9).all(), "prefix changed / point above cut kept")
    ok(m._clip_top(pts, 100.0) is pts and m._clip_top(pts, None) is pts, "no-op case must return the input unchanged")
    ok(m._clip_top(pts, 360.0) is None, "everything above the cut -> None")
    L = np.stack([np.linspace(130, 300, 30), np.linspace(354, 223, 30)], 1); R = np.stack([np.linspace(480, 330, 30), np.linspace(354, 223, 30)], 1)
    l, r = m._clip_top(L, 270.0), m._clip_top(R, 270.0); ok(len(l) == len(r) and l[-1][1] == r[-1][1] == 270.0, "two sides do not end on the same row")
    ok(r[-1][0] > l[-1][0], "clipped polygon is inverted at the cut row")

print("\n%d/%d tests passed" % (sum(RES), len(RES))); sys.exit(0 if all(RES) else 1)
