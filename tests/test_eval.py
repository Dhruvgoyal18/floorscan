"""Seam: evaluation harness (GT + result -> records, gates, calibration). Hand-written fixtures."""
import numpy as np

from eval import gates
from eval.calibrate import conformal_k, fit
from eval.match import match


def M(v, h=0.01):
    return {"value": v, "ci90": [v - h, v + h]}


def _result(walls, openings, adjacency=(), tier="lidar", rid="R1"):
    ws = [{"id": f"{rid}W{i + 1}", "start_m": [0, 0], "end_m": [1, 0], "length_m": M(L), "height_m": M(2.5),
           "net_area_m2": M(1.0)} for i, L in enumerate(walls)]
    ops = [{"id": f"{rid}O{i + 1}", "type": t, "wall_id": f"{rid}W{w + 1}", "offset_m": off, "width_m": M(wd),
            "height_m": M(2.03)} for i, (t, w, off, wd) in enumerate(openings)]
    room = {"id": rid, "name": rid, "polygon_m": [[0, 0], [1, 0], [1, 1]], "walls": ws, "ceiling_height_m": M(2.51),
            "floor_area_m2": M(walls[0] * walls[1]), "perimeter_m": M(sum(walls)), "openings": ops}
    return {"tier": tier, "rooms": [room], "damage": [],
            "stitched": {"adjacency": list(adjacency), "net_floor_area_m2": M(walls[0] * walls[1]), "overlap_m2": 0.0}}


GT = {"rooms": [{"id": "bed", "walls": [4.0, 3.0, 4.0, 3.0], "ceiling_height": [2.50], "floor_area": 12.0,
                 "openings": [{"id": "door", "type": "door", "wall": 0, "offset": 0.5, "width": 0.80, "height": 2.03},
                              {"id": "win", "type": "window", "wall": 2, "offset": 1.0, "width": 1.20, "height": 1.2}],
                 "damage": []}], "adjacency": []}


def test_rotated_wall_order_is_aligned_by_lengths_and_openings():
    # predicted walls start one wall later (3,4,3,4) and the door sits on predicted wall 4
    res = _result([3.01, 4.02, 3.0, 4.01], [("door", 3, 0.51, 0.81), ("window", 1, 1.0, 1.19)])
    m = match(res, {"items": []}, GT, ["bed"], "c", "p")
    errs = {r["gt_wall"]: r["err"] for r in m.records if r["kind"] == "wall_length"}
    assert np.allclose([errs[0], errs[1], errs[2], errs[3]], [0.01, 0.01, 0.02, 0.0], atol=1e-9)
    assert m.openings == {"gt": 2, "matched": 2, "missed": [], "phantom": []}


def test_missed_and_phantom_openings_both_fail_the_gate():
    res = _result([4.0, 3.0, 4.0, 3.0], [("door", 0, 0.5, 0.80), ("window", 1, 1.0, 1.0)])  # window on wrong wall
    m = match(res, {"items": []}, GT, ["bed"], "c", "p")
    assert m.openings["missed"] == ["bed:win"] and len(m.openings["phantom"]) == 1
    g = gates.opening_gate([m], "lidar")
    assert g["within_2cm"] == 1 and g["gt_openings"] + g["phantom"] == 3 and g["pass"] is False


def test_conformal_factor_is_the_finite_sample_quantile():
    z = np.arange(1, 11, dtype=float)  # n=10, q = ceil(11*0.9)/10 = 1.0 -> max
    assert np.isclose(conformal_k(z), 10 / 1.645)
    assert conformal_k(np.full(20, 0.1)) == 1.0, "never shrink below the analytic se"


def test_loro_never_uses_the_scored_room():
    recs = []
    for room, err in (("a", 0.05), ("b", 0.001), ("c", 0.001), ("d", 0.001), ("e", 0.001), ("f", 0.001), ("g", 0.001)):
        recs.append({"tier": "lidar", "kind": "wall_length", "property": "p", "room": room, "err": err, "se": 0.001,
                     "pred": 4.0, "gt": 4.0 - err})
    _, ann = fit(recs)
    a = next(r for r in ann if r["room"] == "a")
    assert not a["covered_loro"], "room a's large error must not calibrate its own interval"


def test_repeatability_gate_uses_1cm_or_half_percent():
    class Mm:
        pass
    ms = []
    for cap, L in (("t1", 4.000), ("t2", 4.015)):
        m = Mm()
        m.tier, m.capture, m.property, m.walls = "lidar", cap, "p", {("bed", 0): L}
        ms.append(m)
    g = gates.repeatability(ms, "lidar")
    assert np.isclose(g["rows"][0]["tol_cm"], 0.5 * 4.0075) and g["pass"] is True
