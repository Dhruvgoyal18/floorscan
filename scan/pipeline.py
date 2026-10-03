"""One capture in -> result.json (schema-validated) + plan.svg/png out."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator
from shapely.geometry import Polygon
from shapely.ops import unary_union

from . import __version__
from .calibrate.intervals import Calibration
from .damage.detect import detect_damage
from .damage.rules import concealed_flags, scope_items
from .geometry.core import CoreConfig, OpeningEst, RoomEst, _adjacency, extract_plan
from .io.media import IMG_EXT, find_video, photo_groups
from .io.stray import is_stray
from .render.plan import render_plan
from .stitch.drift import DriftConfig
from .types import Estimate

SCHEMA = json.loads((Path(__file__).parent / "schema" / "output.schema.json").read_text())
BOOTSTRAP = {"lidar": 4, "video": 4, "photo": 8}


def detect_tier(capture: Path) -> str:
    capture = Path(capture)
    if capture.is_dir() and is_stray(capture):
        return "lidar"
    if find_video(capture) is not None:
        return "video"
    if capture.is_dir() and photo_groups(capture):
        return "photo"
    raise ValueError(f"{capture}: not a Stray Scanner folder, a video, or a folder of photo folders")


def pipeline_version() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).parent, timeout=5).stdout.strip()
    except Exception:
        sha = ""
    return f"{__version__}+{sha}" if sha else __version__


def _offset(room: RoomEst, o: OpeningEst) -> float:
    w = room.walls[o.edge]
    b = 1 - w.axis
    sgn = np.sign(w.end[b] - w.start[b]) or 1.0
    u0 = (o.t0.value - w.start[b]) * sgn
    u1 = (o.t1.value - w.start[b]) * sgn
    return float(min(u0, u1))


def _rename(room: RoomEst, rid: str):
    room.id = rid
    for j, w in enumerate(room.walls):
        w.id = f"{rid}W{j + 1}"
    for j, o in enumerate(room.openings):
        o.id = f"{rid}O{j + 1}"


def run(capture: Path, out_dir: Path, tier: str | None = None, backend: str = "auto", drift: bool = True,
        bootstrap: int | None = None, calibration: Path | None = None, capture_id: str | None = None,
        device: str | None = None, render: bool = True) -> dict:
    t_start = time.time()
    timing: dict[str, float] = {}
    capture = Path(capture)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tier = tier or detect_tier(capture)
    calib = Calibration.load(calibration)
    B = BOOTSTRAP[tier] if bootstrap is None else bootstrap
    warnings: list[str] = []
    groups = []   # (obs, plan, room list, name hint)
    drift_info = {"method": "n/a (per-room photo folders share no poses; rooms placed by doorway matching)",
                  "enabled": False}
    used_backend = "none (LiDAR depth + ARKit poses)"
    dev = device or ""

    t = time.time()
    if tier == "lidar":
        from .tiers.lidar import lidar_observations
        obs, rep = lidar_observations(capture, DriftConfig(enabled=drift))
        drift_info = rep.as_dict()
        timing["load_and_drift"] = time.time() - t
        t = time.time()
        plan = extract_plan(obs, CoreConfig(bootstrap=B))
        groups.append((obs, plan, plan.rooms, None))
        dev = dev or "iPhone/iPad Pro (LiDAR)"
    else:
        from .backends.base import get_backend
        if backend == "auto":
            from .backends.oracle import is_synthetic
            if is_synthetic(capture):
                backend = "oracle"
        be = get_backend(backend)
        used_backend = be.name
        if tier == "video":
            from .tiers.rgb import video_observations
            obs, rep, info = video_observations(capture, be, out_dir / "work", DriftConfig(
                enabled=drift, accept_rmse=0.05, sigma_icp_t=0.02, frag_seconds=4.0))
            drift_info = rep.as_dict()
            dev = dev or info.get("device") or "iPhone (video)"
            timing["load_and_reconstruct"] = time.time() - t
            t = time.time()
            plan = extract_plan(obs, CoreConfig(bootstrap=B))
            groups.append((obs, plan, plan.rooms, None))
        else:
            from .tiers.rgb import photo_observations
            obs_list, info = photo_observations(capture, be)
            dev = dev or info.get("device") or "iPhone (photos)"
            timing["load_and_reconstruct"] = time.time() - t
            t = time.time()
            for o in obs_list:
                try:
                    plan = extract_plan(o, CoreConfig(bootstrap=B, single_room=True))
                except RuntimeError as e:
                    warnings.append(f"room folder '{o.group}' produced no room: {e}")
                    continue
                main = max(plan.rooms, key=lambda r: (r.n_cams, r.area.value))
                groups.append((o, plan, [main], o.group))
    timing["geometry"] = time.time() - t

    # ids, damage, rules (in each group's own plan frame, before stitching)
    t = time.time()
    rooms: list[RoomEst] = []
    names: dict[str, str] = {}
    damage_regions, flags, scope = [], [], []
    for obs, plan, rlist, hint in groups:
        warnings.extend(w for w in plan.warnings if w not in warnings)
        for r in rlist:
            rid = f"R{len(rooms) + 1}"
            _rename(r, rid)
            names[rid] = _room_name(hint, len(rooms))
            for o in r.openings:
                o.offset = _offset(r, o)
            rooms.append(r)
        plan.rooms = rlist
        regs = detect_damage(obs, plan)
        for d in regs:
            d.id = f"D{len(damage_regions) + 1}"
            damage_regions.append(d)
        flags += concealed_flags(regs)
        scope += scope_items(regs, concealed_flags(regs))
    for i, s in enumerate(scope):
        s["id"] = f"S{i + 1}"
    timing["damage"] = time.time() - t

    # stitch
    t = time.time()
    if tier == "photo" and len(rooms) > 1:
        from .stitch.layout import overlap_area, seen_through_doors, solve_layout, transform_room
        from .geometry.core import prepare_cloud
        seen = []
        for obs, plan, rlist, _ in groups:
            c = prepare_cloud(obs, CoreConfig(voxel=0.05))
            P = c["P"] @ plan.R_plan.T
            N = c["N"] @ plan.R_plan.T
            wall = (np.abs(N[:, 2]) < 0.3) & (P[:, 2] - plan.z0 > 0.3)
            seen.append(seen_through_doors(rlist[0], P[wall, :2]))
        pl, lw = solve_layout(rooms, seen=seen)
        warnings += lw
        for i, r in enumerate(rooms):
            transform_room(r, pl.k[i], pl.t[i])
        method = "doorway matching across room folders (beam search + loop-closing least squares)"
    else:
        method = "single capture session" + (" (drift-corrected)" if drift and tier != "photo" else "")
    adjacency, thickness = _adjacency(rooms, None)
    overlap = _overlap(rooms)
    timing["stitch"] = time.time() - t

    result = _assemble(capture_id or capture.name, tier, dev, used_backend, rooms, names, adjacency, overlap,
                       method, drift_info, damage_regions, flags, scope, calib, warnings, thickness)
    timing["total"] = time.time() - t_start
    result["timing_s"] = {k: round(v, 2) for k, v in timing.items()}
    errs = sorted(Draft202012Validator(SCHEMA).iter_errors(result), key=lambda e: list(e.path))
    if errs:
        raise ValueError("output does not match the published schema: " + "; ".join(
            f"{list(e.path)}: {e.message}" for e in errs[:5]))
    if render:
        t = time.time()
        render_plan(result, out_dir / "plan.svg", out_dir / "plan.png")
        result["artifacts"] = {"plan_svg": "plan.svg", "plan_png": "plan.png", "json": "result.json"}
        result["timing_s"]["render"] = round(time.time() - t, 2)
    (out_dir / "result.json").write_text(json.dumps(result, indent=1))
    return result


def _room_name(hint: str | None, k: int) -> str:
    if hint:
        parts = hint.split("_", 1)
        return parts[1].replace("_", " ") if len(parts) == 2 and parts[0].isdigit() else hint
    return f"Room {k + 1}"


def _overlap(rooms) -> float:
    tot = 0.0
    polys = [Polygon(r.polygon) for r in rooms]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            tot += polys[i].intersection(polys[j]).area
    return float(tot)


def _assemble(cid, tier, device, backend, rooms, names, adjacency, overlap, method, drift_info, regions, flags,
              scope, calib: Calibration, warnings, thickness) -> dict:
    M = calib.measure
    out_rooms = []
    for r in rooms:
        walls = []
        for w in r.walls:
            open_area = sum(o.width.value * o.height.value for o in r.openings if r.walls[o.edge] is w)
            area = Estimate(max(w.length.value * r.height.value - open_area, 0.0),
                            float(np.hypot(w.length.se * r.height.value, r.height.se * w.length.value)),
                            w.length.observed)
            walls.append({"id": w.id, "start_m": [round(float(x), 4) for x in w.start],
                          "end_m": [round(float(x), 4) for x in w.end],
                          "length_m": M(w.length, tier, "wall_length"), "height_m": M(r.height, tier, "ceiling_height"),
                          "net_area_m2": M(area, tier, "floor_area", 3)})
        ops = []
        for o in r.openings:
            d = {"id": o.id, "type": o.kind, "wall_id": r.walls[o.edge].id, "offset_m": round(getattr(o, "offset", 0.0), 4),
                 "width_m": M(o.width, tier, "opening_width"), "height_m": M(o.height, tier, "opening_height"),
                 "connects_to": o.connects_to, "confidence": round(float(o.confidence), 3)}
            if o.sill is not None:
                d["sill_m"] = M(o.sill, tier, "opening_height")
            ops.append(d)
        out_rooms.append({"id": r.id, "name": names.get(r.id, r.id),
                          "polygon_m": [[round(float(x), 4), round(float(y), 4)] for x, y in r.polygon],
                          "walls": walls, "ceiling_height_m": M(r.height, tier, "ceiling_height"),
                          "floor_area_m2": M(r.area, tier, "floor_area", 3),
                          "perimeter_m": M(r.perimeter, tier, "perimeter"), "openings": ops})
    areas = [r.area for r in rooms]
    if tier == "photo":   # independent reconstructions: errors add in quadrature
        net_se = float(np.sqrt(sum(a.se ** 2 for a in areas)))
    else:                 # one session, shared scale: errors are correlated
        net_se = float(sum(a.se for a in areas))
    net = Estimate(float(sum(a.value for a in areas)), net_se, all(a.observed for a in areas))
    T = thickness.value if thickness is not None else 0.12
    gross_poly = unary_union([Polygon(r.polygon).buffer(T / 2 + 0.01, join_style=2) for r in rooms])
    gross_poly = gross_poly.buffer(0.3, join_style=2).buffer(-0.3, join_style=2)
    gross = Estimate(float(gross_poly.area), net_se * (gross_poly.area / max(net.value, 1e-6)))
    dmg = []
    for d in regions:
        dmg.append({"id": d.id, "class": d.cls, "surface_id": d.surface.id, "area_m2": M(d.area, tier, "damage_area", 4),
                    "extent_m": {"width": M(d.width, tier, "damage_area", 3), "height": M(d.height, tier, "damage_area", 3)},
                    "centroid_height_m": round(d.centroid_height, 3),
                    "bbox_on_surface_m": [round(float(x), 3) for x in d.bbox], "score": round(d.score, 3),
                    "n_views": d.n_views})
    sc = [{"id": s["id"], "surface_id": s["surface_id"], "action": s["action"],
           "quantity": M(s["quantity"], tier, "quantity", 3), "unit": s["unit"], "reason": s["reason"]} for s in scope]
    return {
        "schema_version": "1.0", "capture_id": cid, "tier": tier, "device": device, "pipeline_version": pipeline_version(),
        "backend": backend, "units": "metric", "rooms": out_rooms,
        "stitched": {"adjacency": adjacency, "net_floor_area_m2": M(net, tier, "footprint", 3),
                     "gross_footprint_m2": M(gross, tier, "footprint", 3), "overlap_m2": round(overlap, 4),
                     "method": method, "drift": drift_info},
        "damage": dmg, "concealed_flags": flags, "scope": sc,
        "quality": {"warnings": warnings, "calibration": calib.status,
                    "wall_thickness_m": round(T, 3)},
        "timing_s": {},
    }


__all__ = ["run", "detect_tier", "IMG_EXT"]
