"""make benchmark: run every capture in a manifest, score against ground truth, write the report.

  python -m eval.run --manifest data/benchmark_synthetic.yaml --out reports/synthetic

Every number in the report comes from this script's output; nothing is typed by hand.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from scan.pipeline import pipeline_version, run

from . import gates
from .calibrate import fit, save
from .match import match
from .report import write_report

ROOT = Path(__file__).resolve().parents[1]


def run_capture(c: dict, out: Path, reuse: bool, drift: bool = True) -> tuple[dict, dict]:
    out.mkdir(parents=True, exist_ok=True)
    rj, ej = out / "result.json", out / "estimates.json"
    ver = pipeline_version()
    stamp = out / "stamp.json"
    if reuse and rj.exists() and ej.exists() and stamp.exists() and json.loads(stamp.read_text()).get("version") == ver:
        return json.loads(rj.read_text()), json.loads(ej.read_text())
    t = time.time()
    res = run(Path(c["path"]), out, tier=c["tier"], capture_id=c["id"], device=c.get("device"), drift=drift,
              backend=c.get("backend", "auto"))
    stamp.write_text(json.dumps({"version": ver, "seconds": time.time() - t}))
    return res, json.loads(ej.read_text())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "benchmark_synthetic.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "synthetic")
    ap.add_argument("--runs", type=Path, default=ROOT / "out" / "bench")
    ap.add_argument("--reuse", action="store_true", help="reuse runs made by the same pipeline version")
    ap.add_argument("--only", default=None)
    ap.add_argument("--write-calibration", action="store_true",
                    help="write the fitted factors to scan/calibrate/calibration.json (real benchmark only)")
    a = ap.parse_args(argv)
    man = yaml.safe_load(a.manifest.read_text())
    gt_dir = a.manifest.parent / "gt" / man["name"]
    caps = man["captures"]
    if a.only:
        caps = [c for c in caps if c["id"] in a.only.split(",")]
    matches, results, timings, ablation = [], {}, [], []
    for c in caps:
        print(f"[bench] {c['id']}", flush=True)
        res, est = run_capture(c, a.runs / c["id"], a.reuse)
        gt = yaml.safe_load((gt_dir / f"{c['property']}.yaml").read_text())
        m = match(res, est, gt, c["rooms"], c["id"], c["property"])
        m.holdout = c.get("holdout", False)
        m.set = c["set"]
        matches.append(m)
        results[c["id"]] = res
        timings.append({"capture": c["id"], "tier": c["tier"], **res["timing_s"]})
        if c["set"] == "multiroom" and c["tier"] in ("lidar", "video"):
            off, est_off = run_capture(c, a.runs / f"{c['id']}__nodrift", a.reuse, drift=False)
            m_off = match(off, est_off, gt, c["rooms"], c["id"] + "__nodrift", c["property"])
            ablation.append(_ablation_row(c, res, off, m, m_off))

    tiers = [t for t in ("lidar", "video", "photo") if any(m.tier == t for m in matches)]
    records = [r for m in matches for r in m.records]
    status = f"fitted LORO on {man['name']} benchmark ({len(caps)} captures)"
    table, annotated = fit(records, status=status)
    a.out.mkdir(parents=True, exist_ok=True)
    save(table, a.out / "calibration_fitted.json")
    if a.write_calibration:
        save(table, ROOT / "scan" / "calibrate" / "calibration.json")

    summary = {
        "manifest": str(a.manifest.relative_to(ROOT) if a.manifest.is_absolute() else a.manifest),
        "pipeline_version": pipeline_version(),
        "gates": {
            "opening_widths": {t: gates.opening_gate(matches, t) for t in tiers},
            "ceiling_height": {t: gates.ceiling_gate(matches, t) for t in tiers},
            "repeatability": {t: gates.repeatability(matches, t) for t in tiers},
            "wall_lengths": {t: gates.wall_gate(matches, t) for t in tiers},
            "floor_area": {t: gates.area_gate(matches, t) for t in tiers},
            "photo_stitch": gates.photo_stitch_gate(matches),
            "drift_ablation": ablation,
        },
        "holdout": {t: {"walls": gates.wall_gate([m for m in matches if m.holdout], t),
                        "openings": gates.opening_gate([m for m in matches if m.holdout], t)} for t in tiers},
        "calibration": {t: gates.coverage(annotated, t) for t in tiers},
        "calibration_table": table,
        "damage": [{"capture": m.capture, **{k: v for k, v in m.damage.items()}} for m in matches if m.damage.get("gt")],
        "adjacency": [{"capture": m.capture, **m.adjacency} for m in matches if len(m.adjacency["gt"]) > 0],
        "footprint": [{"capture": m.capture, "tier": m.tier, **m.footprint} for m in matches],
        "timing": timings,
        "records": records,
    }
    (a.out / "benchmark.json").write_text(json.dumps(summary, indent=1, default=_np))
    write_report(summary, a.out / "benchmark.md", results)
    print(f"[bench] wrote {a.out / 'benchmark.md'}")


def _ablation_row(c, on, off, m_on, m_off):
    def thick_spread(res):
        rooms = {r["id"]: r for r in res["rooms"]}
        gaps = []
        for e in res["stitched"]["adjacency"]:
            if e["kind"] != "shared_wall":
                continue
            wa, wb = e["via"]
            ra, rb = rooms[e["rooms"][0]], rooms[e["rooms"][1]]
            A = next(w for w in ra["walls"] if w["id"] == wa)
            B = next(w for w in rb["walls"] if w["id"] == wb)
            d = np.array(A["end_m"]) - np.array(A["start_m"])
            n = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9)
            gaps.append(abs((np.array(B["start_m"]) - np.array(A["start_m"])) @ n))
        return float(np.std(gaps) * 100) if len(gaps) > 1 else None

    def fp(m):
        return m.footprint["rel_err"] * 100

    return {"capture": c["id"], "tier": c["tier"],
            "footprint_err_pct_on": fp(m_on), "footprint_err_pct_off": fp(m_off),
            "overlap_m2_on": on["stitched"]["overlap_m2"], "overlap_m2_off": off["stitched"]["overlap_m2"],
            "rooms_on": len(on["rooms"]), "rooms_off": len(off["rooms"]),
            "adjacency_correct_on": m_on.adjacency["correct"], "adjacency_correct_off": m_off.adjacency["correct"],
            "wall_thickness_spread_cm_on": thick_spread(on), "wall_thickness_spread_cm_off": thick_spread(off),
            "loop_closure_error_cm": (on["stitched"]["drift"].get("loop_closure_error_m") or 0) * 100,
            "loop_closure_error_after_cm": (on["stitched"]["drift"].get("loop_closure_error_after_m") or 0) * 100,
            "median_wall_err_cm_on": float(np.median([abs(r["err"]) for r in m_on.records if r["kind"] == "wall_length"]) * 100),
            "median_wall_err_cm_off": float(np.median([abs(r["err"]) for r in m_off.records if r["kind"] == "wall_length"]) * 100)
            if any(r["kind"] == "wall_length" for r in m_off.records) else None,
            "plan_on": f"out/bench/{c['id']}/plan.png", "plan_off": f"out/bench/{c['id']}__nodrift/plan.png"}


def _np(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(type(o))


if __name__ == "__main__":
    main()
