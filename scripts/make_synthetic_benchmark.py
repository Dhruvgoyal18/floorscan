"""Generate the synthetic benchmark (deterministic from seeds) + ground truth + manifest.

The synthetic set mirrors the composition the brief requires of the real one:
  * multi-room capture (>= 3 rooms + connector) at all three tiers   -> home / multiroom
  * furnished room with staged damage of two classes                 -> home / bedroom
  * one room captured repeatedly at the same tier (repeatability)    -> bedroom takes t1..t3
  * a property never used for development (walk-in stand-in)         -> holdout2026 / multiroom
Real captures replace this with data/benchmark_real.yaml; nothing else changes.

usage: python scripts/make_synthetic_benchmark.py [--root data] [--quick]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scan.synth.capture import write_photos, write_stray, write_video  # noqa: E402
from scan.synth.presets import home, random_property  # noqa: E402
from scan.synth.scene import build_scene, ground_truth  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data"))
    ap.add_argument("--quick", action="store_true", help="single take per tier (CI smoke)")
    ap.add_argument("--only", default=None, help="comma list of capture ids to (re)generate")
    a = ap.parse_args()
    raw = a.root / "raw" / "synthetic"
    gtd = a.root / "gt" / "synthetic"
    gtd.mkdir(parents=True, exist_ok=True)
    props = {"home": home(), "holdout2026": random_property(2026)}
    for name, spec in props.items():
        (gtd / f"{name}.yaml").write_text(yaml.safe_dump(ground_truth(spec), sort_keys=False))

    plan = []  # (id, property, set, tier, take, rooms, seed, holdout)
    h = props["home"]
    plan.append(("home_multiroom_lidar_t1", "home", "multiroom", "lidar", 1, h.walk, 7, False))
    plan.append(("home_multiroom_video_t1", "home", "multiroom", "video", 1, h.walk, 8, False))
    plan.append(("home_multiroom_photo_t1", "home", "multiroom", "photo", 1, ["hall", "living", "kitchen", "bedroom", "bath"], 9, False))
    takes = 1 if a.quick else 3
    for t in range(1, takes + 1):
        plan.append((f"home_bedroom_lidar_t{t}", "home", "bedroom", "lidar", t, ["bedroom"], 20 + t, False))
        plan.append((f"home_bedroom_photo_t{t}", "home", "bedroom", "photo", t, ["bedroom"], 40 + t, False))
        if t <= 2:
            plan.append((f"home_bedroom_video_t{t}", "home", "bedroom", "video", t, ["bedroom"], 30 + t, False))
    ho = props["holdout2026"]
    ho_rooms = [r.name for r in ho.rooms]
    plan.append(("holdout2026_multiroom_lidar_t1", "holdout2026", "multiroom", "lidar", 1, ho.walk, 101, True))
    plan.append(("holdout2026_multiroom_video_t1", "holdout2026", "multiroom", "video", 1, ho.walk, 102, True))
    plan.append(("holdout2026_multiroom_photo_t1", "holdout2026", "multiroom", "photo", 1, ho_rooms, 103, True))

    only = set(a.only.split(",")) if a.only else None
    manifest_path = a.root / "benchmark_synthetic.yaml"
    scenes = {k: build_scene(v) for k, v in props.items()}
    manifest = {"name": "synthetic", "note": "rendered by scan/synth; regenerate with make synthetic", "captures": []}
    for cid, prop, cset, tier, take, rooms, seed, holdout in plan:
        path = raw / prop / cset / tier / f"t{take}"
        manifest["captures"].append({"id": cid, "property": prop, "set": cset, "tier": tier, "take": take,
                                     "path": str(path), "rooms": sorted(set(rooms)), "holdout": holdout,
                                     "device": "synthetic iPhone 15 Pro" if tier == "lidar" else "synthetic iPhone 15"})
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    for cid, prop, cset, tier, take, rooms, seed, holdout in plan:
        spec = props[prop]
        path = raw / prop / cset / tier / f"t{take}"
        if only and cid not in only:
            continue
        if path.exists() and not only:
            continue
        t0 = time.time()
        if tier == "lidar":
            write_stray(path, spec, rooms, seed=seed, scene=scenes[prop])
        elif tier == "video":
            write_video(path, spec, rooms, seed=seed, scene=scenes[prop])
        else:
            write_photos(path, spec, rooms, seed=seed, scene=scenes[prop])
        print(f"{cid}: {time.time() - t0:.0f}s", flush=True)
    print("manifest:", manifest_path)


if __name__ == "__main__":
    main()
