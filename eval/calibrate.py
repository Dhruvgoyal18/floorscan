"""Conformal calibration of interval widths, leave-one-room-out.

For each (tier, kind): z = |error| / se over the benchmark. The factor applied to a record from room r
is the split-conformal 90 % quantile of z over records from *other* rooms (property, room), divided by
1.645, never below 1. Coverage is then evaluated with exactly the half-width formula the pipeline uses.
The factor fitted on all rooms is written to scan/calibrate/calibration.json for use on new captures.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scan.calibrate.intervals import PRIOR, Z90, Calibration
from scan.types import Estimate

MIN_RECORDS = 6


def conformal_k(z: np.ndarray, target: float = 0.9) -> float:
    z = np.sort(z[np.isfinite(z)])
    if len(z) == 0:
        return np.nan
    q = min(1.0, np.ceil((len(z) + 1) * target) / len(z))
    return max(1.0, float(np.quantile(z, q, method="higher")) / Z90)


def fit(records: list[dict], target: float = 0.9, status: str = "fitted") -> tuple[dict, list[dict]]:
    """Returns (calibration table, records annotated with LORO half-width and coverage)."""
    table = json.loads(json.dumps(PRIOR))
    table["status"] = status
    table["k_kind"] = {}
    usable = [r for r in records if np.isfinite(r["se"]) and r["se"] > 0]
    by = {}
    for r in usable:
        by.setdefault((r["tier"], r["kind"]), []).append(r)
    by_tier = {}
    for r in usable:
        by_tier.setdefault(r["tier"], []).append(r)
    for tier, rs in by_tier.items():
        table["k"][tier] = round(conformal_k(np.array([abs(r["err"]) / r["se"] for r in rs]), target), 3)
    for (tier, kind), rs in by.items():
        if len(rs) >= MIN_RECORDS:
            table["k_kind"].setdefault(tier, {})[kind] = round(
                conformal_k(np.array([abs(r["err"]) / r["se"] for r in rs]), target), 3)
    # leave-one-room-out evaluation
    out = []
    for r in usable:
        room = (r["property"], r["room"])
        same = by[(r["tier"], r["kind"])]
        others = [x for x in same if (x["property"], x["room"]) != room]
        if len(others) < MIN_RECORDS:
            others = [x for x in by_tier[r["tier"]] if (x["property"], x["room"]) != room]
        k = conformal_k(np.array([abs(x["err"]) / x["se"] for x in others]), target) if others else np.nan
        tab = json.loads(json.dumps(table))
        tab["k_kind"] = {r["tier"]: {r["kind"]: k}} if np.isfinite(k) else {}
        cal = Calibration(tab)
        h = cal.half_width(Estimate(r["pred"], r["se"]), r["tier"], r["kind"])
        out.append({**r, "k_loro": k, "half_width_loro": h, "covered_loro": abs(r["err"]) <= h})
    return table, out


def save(table: dict, path: Path):
    path.write_text(json.dumps(table, indent=1))
