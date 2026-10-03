"""Part 3 head-to-head: our LiDAR-tier output vs a consumer scanning app on the same rooms.

Inputs (committed with the raw data):
  data/exports/<app>/<room>.yaml   the app's dimensions transcribed from its export (export file alongside):
      app: Polycam   version: "x.y.z"   export: polycam_bedroom.pdf
      room: bedroom  capture: <our capture id for the same room>
      walls: [..]  (CCW, same start wall as GT)   ceiling_height: ..   floor_area: ..
      openings: [{id: door, width: ..}, ...]      (ids as in our GT file)
  data/gt/<manifest>/<property>.yaml               tape/laser ground truth

Tie rule (fixed before measuring): |our error| and |their error| within 0.5 cm of each other
(1 % for areas). Win/tie share must be >= 70 % of shared dimensions.

  python -m eval.head_to_head --manifest data/benchmark_real.yaml --runs out/bench --out reports/real/head_to_head.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .match import match

TIE_ABS = 0.005
TIE_REL_AREA = 0.01


def compare(ours: dict, theirs: dict, gt_room: dict) -> list[dict]:
    rows = []

    def add(dim, gtv, ov, tv, area=False):
        if ov is None or tv is None:
            return
        eo, et = abs(ov - gtv), abs(tv - gtv)
        tie = abs(eo - et) <= (TIE_REL_AREA * gtv if area else TIE_ABS)
        rows.append({"dimension": dim, "gt": gtv, "ours": ov, "ours_err": ov - gtv, "theirs": tv, "theirs_err": tv - gtv,
                     "winner": "tie" if tie else ("ours" if eo < et else "theirs")})

    for i, g in enumerate(gt_room["walls"]):
        add(f"wall {i + 1}", g, ours["walls"].get(i), (theirs.get("walls") or [None] * 99)[i])
    add("ceiling height", gt_room["ceiling_height"][0] if isinstance(gt_room["ceiling_height"], list) else gt_room["ceiling_height"],
        ours.get("ceiling"), theirs.get("ceiling_height"))
    add("floor area", gt_room["floor_area"], ours.get("area"), theirs.get("floor_area"), area=True)
    for o in theirs.get("openings", []):
        g = next((x for x in gt_room.get("openings", []) if x["id"] == o["id"]), None)
        if g:
            add(f"opening {o['id']} width", g["width"], ours["openings"].get(o["id"]), o["width"])
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--runs", type=Path, default=Path("out/bench"))
    ap.add_argument("--exports", type=Path, default=Path("data/exports"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    man = yaml.safe_load(a.manifest.read_text())
    caps = {c["id"]: c for c in man["captures"]}
    all_rows, apps = [], set()
    for f in sorted(a.exports.glob("*/*.yaml")):
        th = yaml.safe_load(f.read_text())
        c = caps[th["capture"]]
        gt = yaml.safe_load((a.manifest.parent / "gt" / man["name"] / f"{c['property']}.yaml").read_text())
        res = json.loads((a.runs / c["id"] / "result.json").read_text())
        est = json.loads((a.runs / c["id"] / "estimates.json").read_text())
        m = match(res, est, gt, c["rooms"], c["id"], c["property"])
        g = next(r for r in gt["rooms"] if r["id"] == th["room"])
        ours = {"walls": {}, "openings": {}}
        for r in m.records:
            if r["room"] != th["room"]:
                continue
            if r["kind"] == "wall_length":
                ours["walls"][r["gt_wall"]] = r["pred"]
            elif r["kind"] == "ceiling_height":
                ours["ceiling"] = r["pred"]
            elif r["kind"] == "floor_area":
                ours["area"] = r["pred"]
            elif r["kind"] == "opening_width":
                ours["openings"][r["opening"]] = r["pred"]
        for row in compare(ours, th, g):
            all_rows.append({"room": th["room"], **row})
        apps.add(f"{th['app']} {th.get('version', '?')}")
    n = len(all_rows)
    good = sum(r["winner"] in ("ours", "tie") for r in all_rows)
    L = [f"# Head-to-head vs {', '.join(sorted(apps)) or 'n/a'}", "",
         f"Tie = errors within {TIE_ABS * 100:.1f} cm ({TIE_REL_AREA * 100:.0f} % for area). Beat-or-tie: **{good}/{n}"
         f" = {good / n:.0%}** (gate >= 70 %: {'PASS' if n and good / n >= 0.7 else 'FAIL'})." if n else "No exports found.", "",
         "| Room | Dimension | GT | Ours | Our err (cm) | Theirs | Their err (cm) | Winner |", "|---|---|---|---|---|---|---|---|"]
    for r in all_rows:
        L.append(f"| {r['room']} | {r['dimension']} | {r['gt']:.3f} | {r['ours']:.3f} | {r['ours_err'] * 100:+.1f} | "
                 f"{r['theirs']:.3f} | {r['theirs_err'] * 100:+.1f} | {r['winner']} |")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L) + "\n")
    print(a.out.read_text())


if __name__ == "__main__":
    main()
