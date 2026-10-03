"""Gates from the brief, computed from match records. Thresholds live here and nowhere else.

Round-1 gates are not reproduced in the brief we have; where a Round-1 number is needed (LiDAR wall
length, floor area) the threshold below is our assumption and is labelled as such in the report.
"""
from __future__ import annotations

import itertools

import numpy as np

WALL_TOL = {"lidar": 0.01, "video": 0.03, "photo": 0.08}          # relative; lidar = assumption (Round 1)
AREA_TOL = {"lidar": 0.02, "video": 0.06, "photo": 0.16}          # relative; assumption (2x wall tolerance)
OPENING_ABS = 0.02
OPENING_RATE = 0.85
CEIL_ABS = 0.015
CEIL_SPREAD = 0.01
REPEAT_ABS, REPEAT_REL = 0.01, 0.005
PHOTO_FOOTPRINT = 0.08
WALL_PASS_SHARE = 0.9


def opening_gate(matches, tier):
    ms = [m for m in matches if m.tier == tier]
    recs = [r for m in ms for r in m.records if r["kind"] == "opening_width"]
    n_gt = sum(m.openings["gt"] for m in ms)
    n_ph = sum(len(m.openings["phantom"]) for m in ms)
    ok = sum(abs(r["err"]) <= OPENING_ABS for r in recs)
    denom = n_gt + n_ph
    rate = ok / denom if denom else float("nan")
    errs = np.abs([r["err"] for r in recs]) if recs else np.zeros(0)
    return {"tier": tier, "within_2cm": ok, "gt_openings": n_gt, "missed": n_gt - len(recs),
            "phantom": n_ph, "rate": rate, "pass": bool(rate >= OPENING_RATE) if denom else None,
            "median_abs_err_cm": float(np.median(errs) * 100) if len(errs) else None,
            "p90_abs_err_cm": float(np.percentile(errs, 90) * 100) if len(errs) else None,
            "missed_list": [x for m in ms for x in m.openings["missed"]],
            "phantom_list": [x for m in ms for x in m.openings["phantom"]]}


def ceiling_gate(matches, tier):
    recs = [r for m in matches if m.tier == tier for r in m.records if r["kind"] == "ceiling_height"]
    by_room = {}
    for r in recs:
        by_room.setdefault((r["property"], r["room"]), []).append(r["err"])
    rows = []
    for (p, room), errs in sorted(by_room.items()):
        e = np.array(errs)
        spread = float(e.max() - e.min()) if len(e) > 1 else None
        rows.append({"room": f"{p}/{room}", "n": len(e), "max_abs_err_cm": float(np.abs(e).max() * 100),
                     "bias_cm": float(e.mean() * 100), "spread_cm": spread * 100 if spread is not None else None})
    errs_ok = all(r["max_abs_err_cm"] <= CEIL_ABS * 100 for r in rows)
    spread_ok = all(r["spread_cm"] is None or r["spread_cm"] <= CEIL_SPREAD * 100 for r in rows)
    multi = [r for r in rows if r["spread_cm"] is not None]
    if not rows:
        diagnosis = "no data"
    elif errs_ok and spread_ok:
        diagnosis = "accurate and repeatable"
    elif spread_ok and multi:
        diagnosis = "repeatable-but-biased"
    elif not spread_ok:
        diagnosis = "unrepeatable" + (" and biased" if not errs_ok else "")
    else:
        diagnosis = "biased (no repeat captures to judge repeatability)"
    return {"tier": tier, "rooms": rows, "pass": bool(errs_ok and spread_ok) if rows else None, "diagnosis": diagnosis}


def wall_gate(matches, tier):
    recs = [r for m in matches if m.tier == tier for r in m.records if r["kind"] == "wall_length"]
    rel = np.array([abs(r["err"]) / r["gt"] for r in recs]) if recs else np.zeros(0)
    share = float(np.mean(rel <= WALL_TOL[tier])) if len(rel) else float("nan")
    return {"tier": tier, "n": len(rel), "tolerance_rel": WALL_TOL[tier], "share_within": share,
            "median_rel_err_pct": float(np.median(rel) * 100) if len(rel) else None,
            "max_rel_err_pct": float(rel.max() * 100) if len(rel) else None,
            "pass": bool(share >= WALL_PASS_SHARE) if len(rel) else None}


def area_gate(matches, tier):
    recs = [r for m in matches if m.tier == tier for r in m.records if r["kind"] == "floor_area"]
    rel = np.array([abs(r["err"]) / r["gt"] for r in recs]) if recs else np.zeros(0)
    return {"tier": tier, "n": len(rel), "tolerance_rel": AREA_TOL[tier],
            "share_within": float(np.mean(rel <= AREA_TOL[tier])) if len(rel) else None,
            "median_rel_err_pct": float(np.median(rel) * 100) if len(rel) else None}


def repeatability(matches, tier):
    """Same room, same tier, different takes: per GT wall, |L_a - L_b| <= max(1 cm, 0.5 %)."""
    groups = {}
    for m in matches:
        if m.tier != tier:
            continue
        for (room, wi), L in m.walls.items():
            groups.setdefault((m.property, room), {}).setdefault(m.capture, {})[wi] = L
    rows = []
    for (p, room), caps in groups.items():
        if len(caps) < 2:
            continue
        for a, b in itertools.combinations(sorted(caps), 2):
            for wi in sorted(set(caps[a]) & set(caps[b])):
                La, Lb = caps[a][wi], caps[b][wi]
                tol = max(REPEAT_ABS, REPEAT_REL * (La + Lb) / 2)
                rows.append({"room": f"{p}/{room}", "wall": wi, "a": a, "b": b, "La": La, "Lb": Lb,
                             "diff_cm": abs(La - Lb) * 100, "tol_cm": tol * 100, "ok": abs(La - Lb) <= tol})
    return {"tier": tier, "rows": rows, "share_ok": float(np.mean([r["ok"] for r in rows])) if rows else None,
            "pass": bool(all(r["ok"] for r in rows)) if rows else None}


def photo_stitch_gate(matches):
    rows = []
    for m in matches:
        if m.tier != "photo" or m.footprint.get("n_rooms_gt", 0) < 3:
            continue
        lo, hi = m.footprint["ci90"]
        rows.append({"capture": m.capture, "adjacency_correct": m.adjacency["correct"],
                     "missing": m.adjacency["missing"], "extra": m.adjacency["extra"],
                     "overlap_m2": m.footprint["overlap_m2"], "footprint_rel_err_pct": m.footprint["rel_err"] * 100,
                     "gt_in_ci90": lo <= m.footprint["gt"] <= hi,
                     "rooms": f"{m.footprint['n_rooms_pred']}/{m.footprint['n_rooms_gt']}"})
    ok = [r["adjacency_correct"] and r["overlap_m2"] <= 0.05 and abs(r["footprint_rel_err_pct"]) <= PHOTO_FOOTPRINT * 100
          and r["gt_in_ci90"] for r in rows]
    return {"rows": rows, "pass": bool(all(ok)) if rows else None}


def coverage(annotated, tier):
    out = {}
    for r in annotated:
        if r["tier"] != tier:
            continue
        out.setdefault(r["kind"], []).append(r)
    rows = []
    for kind, rs in sorted(out.items()):
        cov = float(np.mean([r["covered_loro"] for r in rs]))
        width = float(np.median([2 * r["half_width_loro"] for r in rs]))
        rows.append({"kind": kind, "n": len(rs), "coverage": cov, "median_width": width,
                     "median_abs_err": float(np.median([abs(r["err"]) for r in rs])),
                     "confident_garbage": cov < 0.7})
    return rows
