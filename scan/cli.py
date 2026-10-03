"""Command line: ``scan run <capture>`` (one command per capture)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="scan", description="Phone capture -> dimensioned, stitched floor plan.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="process one capture (Stray Scanner folder, video, or folder of room photo folders)")
    r.add_argument("capture", type=Path)
    r.add_argument("-o", "--out", type=Path, default=None, help="output folder (default: out/<capture name>)")
    r.add_argument("--tier", choices=["lidar", "video", "photo"], default=None, help="override auto-detection")
    r.add_argument("--backend", default="auto", help="video/photo depth backend: auto|mapanything|depthanything|oracle")
    r.add_argument("--no-drift-correction", action="store_true", help="ablation only: use odometry poses as-is")
    r.add_argument("--bootstrap", type=int, default=None, help="frame-bootstrap resamples (default per tier)")
    r.add_argument("--calibration", type=Path, default=None, help="calibration table (default: shipped one)")
    r.add_argument("--id", default=None, help="capture id (default: folder name)")
    r.add_argument("--device", default=None, help="device name for the report (e.g. 'iPhone 15 Pro')")
    r.add_argument("--no-render", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        from .pipeline import run
        out = a.out or Path("out") / (a.id or a.capture.stem)
        res = run(a.capture, out, tier=a.tier, backend=a.backend, drift=not a.no_drift_correction,
                  bootstrap=a.bootstrap, calibration=a.calibration, capture_id=a.id, device=a.device,
                  render=not a.no_render)
        st = res["stitched"]
        print(json.dumps({"capture_id": res["capture_id"], "tier": res["tier"], "rooms": len(res["rooms"]),
                          "net_floor_area_m2": st["net_floor_area_m2"], "adjacency": len(st["adjacency"]),
                          "damage": len(res["damage"]), "warnings": res["quality"]["warnings"],
                          "seconds": res["timing_s"]["total"], "out": str(out)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
