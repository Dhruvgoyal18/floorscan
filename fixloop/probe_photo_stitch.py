"""Feedback loop for the photo-tier stitch gate (fix-loop diagnosis). ~1 min, deterministic.

Runs the photo pipeline on the two multi-room photo captures and prints, per room: area error vs GT,
doorways detected vs GT, and the solver's doorway pairs vs GT adjacency.

  python fixloop/probe_photo_stitch.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.match import match  # noqa: E402
from scan.pipeline import run  # noqa: E402

CAPS = ["home_multiroom_photo_t1", "holdout2026_multiroom_photo_t1"]


def main():
    man = yaml.safe_load((ROOT / "data" / "benchmark_synthetic.yaml").read_text())
    for c in [c for c in man["captures"] if c["id"] in CAPS]:
        out = ROOT / "out" / "probe" / c["id"]
        res = run(ROOT / c["path"], out, tier="photo", capture_id=c["id"], render=True)
        est = json.loads((out / "estimates.json").read_text())
        gt = yaml.safe_load((ROOT / "data" / "gt" / "synthetic" / f"{c['property']}.yaml").read_text())
        m = match(res, est, gt, c["rooms"], c["id"], c["property"])
        g = {r["id"]: r for r in gt["rooms"]}
        print(f"\n== {c['id']}  footprint {m.footprint['rel_err'] * 100:+.1f} %  adjacency correct: {m.adjacency['correct']}")
        print("   missing:", m.adjacency["missing"], " extra:", m.adjacency["extra"])
        for r in res["rooms"]:
            gid = m.rooms.get(r["id"])
            ga = g[gid]["floor_area"] if gid else float("nan")
            n_gt_doors = sum(o["type"] == "door" for o in g[gid]["openings"]) if gid else -1
            dets = [(o["type"][0], round(o["width_m"]["value"], 2)) for o in r["openings"]]
            print(f"   {r['id']} {r['name']:>8} -> {gid}: area {r['floor_area_m2']['value']:.2f} vs {ga:.2f} "
                  f"({(r['floor_area_m2']['value'] / ga - 1) * 100:+.1f} %), walls {len(r['walls'])}, GT doors {n_gt_doors}, "
                  f"detected {dets}")
        for e in res["stitched"]["adjacency"]:
            if e["kind"] == "door":
                print("   pair", e["rooms"], e["via"], "->", [m.rooms.get(x) for x in e["rooms"]])


if __name__ == "__main__":
    main()
