"""LiDAR tier: Stray Scanner capture -> drift-corrected Observations. No learned weights needed."""
from __future__ import annotations

from pathlib import Path

from ..io.common import LIDAR_DEPTH, build_observations
from ..io.stray import load_stray_frames
from ..stitch.drift import DriftConfig, DriftReport, correct_drift
from ..types import Observations


def lidar_observations(capture: Path, drift: DriftConfig | None = None, target_fps: float = 5.0,
                       poses_override=None) -> tuple[Observations, DriftReport]:
    frames, meta = load_stray_frames(Path(capture), target_fps=target_fps, poses_override=poses_override)
    warnings = []
    if len(frames) < 20:
        warnings.append(f"only {len(frames)} LiDAR frames; capture may be too short")
    frames, rep = correct_drift(frames, drift or DriftConfig(), conf_min=LIDAR_DEPTH.conf_min, max_depth=4.0)
    obs = build_observations(frames, "lidar", "all", LIDAR_DEPTH, warnings=warnings, meta=meta)
    return obs, rep
