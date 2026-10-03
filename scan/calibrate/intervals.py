"""The one place interval widths are set.

ci90 half-width = max( 1.645 * k[tier][kind] * se ,  floor_rel[tier][kind] * |value| ,  floor_abs[kind] )

* ``se``: analytic standard error from the geometry core (plane-fit residuals, effective sample count,
  correlated depth-bias term) combined with the frame-bootstrap spread when bootstrapping is on.
* ``k``: conformal scale factor fitted on the benchmark, leave-one-room-out (eval/calibrate.py), so a
  room is never scored with a factor fitted on itself. Until fitted, conservative priors are used and
  the output says so.
* floors: thin tiers can never claim tighter than their documented accuracy (no confident garbage).
"""
from __future__ import annotations

import json
from pathlib import Path

from ..types import Estimate

Z90 = 1.645
DEFAULT_PATH = Path(__file__).with_name("calibration.json")

PRIOR = {
    "status": "prior (not yet fitted on a benchmark)",
    "target": 0.9,
    "k": {"lidar": 2.0, "video": 2.5, "photo": 3.0},
    "k_kind": {},
    "floor_rel": {"lidar": 0.002, "video": 0.01, "photo": 0.03},
    "floor_abs": {"wall_length": 0.003, "ceiling_height": 0.003, "opening_width": 0.004, "opening_height": 0.005,
                  "floor_area": 0.02, "footprint": 0.05, "damage_area": 0.002, "perimeter": 0.01, "quantity": 0.0},
}


class Calibration:
    def __init__(self, table: dict | None = None):
        self.t = table or PRIOR

    @classmethod
    def load(cls, path: Path | None = None) -> "Calibration":
        p = Path(path) if path else DEFAULT_PATH
        if p.exists():
            return cls(json.loads(p.read_text()))
        return cls(PRIOR)

    @property
    def status(self) -> str:
        return self.t.get("status", "unknown")

    def factor(self, tier: str, kind: str) -> float:
        kk = self.t.get("k_kind", {}).get(tier, {}).get(kind)
        return float(kk if kk is not None else self.t["k"].get(tier, 3.0))

    def half_width(self, est: Estimate, tier: str, kind: str) -> float:
        k = self.factor(tier, kind)
        rel = self.t["floor_rel"].get(tier, 0.03)
        if kind in ("floor_area", "footprint", "damage_area"):
            rel *= 2  # areas: two lengths' relative error
        return max(Z90 * k * est.se, rel * abs(est.value), self.t["floor_abs"].get(kind, 0.0))

    def measure(self, est: Estimate, tier: str, kind: str, digits: int = 4) -> dict:
        h = self.half_width(est, tier, kind)
        out = {"value": round(est.value, digits), "ci90": [round(est.value - h, digits), round(est.value + h, digits)]}
        if not est.observed:
            out["observed"] = False
        return out
