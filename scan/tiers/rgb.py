"""Video and photo tiers: images -> backend depth/poses -> gravity-aligned Observations."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..backends.base import ReconstructionBackend
from ..geometry.core import CoreConfig
from ..geometry.cloud import backproject, estimate_normals, voxel_reduce
from ..geometry.frame import align_up, gravity_from_normals
from ..io.common import DepthModel, build_observations
from ..io.media import find_video, photo_groups, photo_view, video_keyframes
from ..stitch.drift import DriftConfig, DriftReport, correct_drift
from ..types import Frame, Observations

VIDEO_DEPTH = DepthModel(abs=0.005, rel=0.015, conf_min=0.3, max_depth=6.0)
PHOTO_DEPTH = DepthModel(abs=0.008, rel=0.02, conf_min=0.3, max_depth=7.0)


def gravity_align(frames: list[Frame]) -> tuple[list[Frame], np.ndarray]:
    """Rotate a backend reconstruction so that up = +z (no IMU in these tiers)."""
    pts, cams_down = [], []
    for f in frames:
        if f.depth is None:
            continue
        good = (f.depth > 0.2) & (f.depth < 6) & ((f.conf if f.conf is not None else 1) >= 0.3)
        p, _ = backproject(f.depth, f.K, f.T_wc, good, stride=4)
        pts.append(p)
        cams_down.append(f.T_wc[:3, 1])   # OpenCV y axis = image down
    P = voxel_reduce(np.concatenate(pts), 0.05)[0]
    N, pl = estimate_normals(P, k=12)
    up = gravity_from_normals(N, pl, np.mean(cams_down, 0))
    R = np.eye(4)
    R[:3, :3] = align_up(up)
    return [Frame(f.index, R @ f.T_wc, f.K, f.depth, f.conf, f.load_image, f.timestamp, f.name) for f in frames], R


def video_observations(capture: Path, backend: ReconstructionBackend, workdir: Path,
                       drift: DriftConfig | None = None) -> tuple[Observations, DriftReport, dict]:
    path = find_video(capture)
    if path is None:
        raise FileNotFoundError(f"no video in {capture}")
    views, info = video_keyframes(path, workdir / "keyframes")
    frames = backend.reconstruct(views, "video")
    warnings = []
    if len(frames) < 0.8 * len(views):
        warnings.append(f"backend registered only {len(frames)}/{len(views)} keyframes")
    if info.get("blurry_share", 0) > 0.3:
        warnings.append(f"{info['blurry_share']:.0%} of keyframes are blurry (low light / fast motion); intervals widened")
    frames, _ = gravity_align(frames)
    dcfg = drift or DriftConfig(accept_rmse=0.05, sigma_icp_t=0.02, frag_seconds=4.0)
    frames, rep = correct_drift(frames, dcfg, conf_min=VIDEO_DEPTH.conf_min, max_depth=5.0)
    obs = build_observations(frames, "video", "all", VIDEO_DEPTH, voxel=0.015, warnings=warnings, meta=info)
    return obs, rep, info


def photo_observations(capture: Path, backend: ReconstructionBackend) -> tuple[list[Observations], dict]:
    groups = photo_groups(capture)
    if not groups:
        raise FileNotFoundError(f"no photos in {capture}")
    out, device = [], ""
    for name, paths in groups:
        views = []
        for p in paths:
            v, model = photo_view(p, name)
            views.append(v)
            device = device or model
        warnings = []
        if len(views) < 2:
            warnings.append(f"room '{name}' has {len(views)} photo(s); protocol asks for 4-8")
        frames = backend.reconstruct(views, "photo")
        if len(frames) < len(views):
            warnings.append(f"room '{name}': {len(views) - len(frames)} photo(s) could not be registered")
        if not frames:
            continue
        frames, _ = gravity_align(frames)
        out.append(build_observations(frames, "photo", name, PHOTO_DEPTH, voxel=0.015, warnings=warnings,
                                      meta={"n_photos": len(views)}))
    return out, {"device": device, "groups": [g for g, _ in groups]}
