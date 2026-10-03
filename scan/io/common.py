"""Frames -> Observations. The one place depth becomes points, for every tier."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.cloud import backproject, pixel_rays, voxel_reduce
from ..types import Frame, Observations


@dataclass
class DepthModel:
    """1-sigma range noise = abs + rel * z (doubled for low-confidence pixels)."""

    abs: float
    rel: float
    conf_min: float      # drop pixels below this confidence (0..1)
    max_depth: float
    far_dist: float = 6.0


LIDAR_DEPTH = DepthModel(abs=0.002, rel=0.003, conf_min=0.5, max_depth=4.5)


def build_observations(frames: list[Frame], tier: str, group: str, model: DepthModel, voxel: float = 0.01,
                       far_stride: int = 2, point_stride: int = 1, warnings: list[str] | None = None,
                       meta: dict | None = None) -> Observations:
    pts, sig, fidx, far, far_f = [], [], [], [], []
    for k, fr in enumerate(frames):
        if fr.depth is None:
            continue
        depth = fr.depth
        conf = fr.conf if fr.conf is not None else np.ones_like(depth)
        good = (conf >= model.conf_min) & (depth > 0.15) & (depth < model.max_depth)
        p, (v, u) = backproject(depth, fr.K, fr.T_wc, good, stride=point_stride)
        z = depth[v, u]
        s = (model.abs + model.rel * z) * np.where(conf[v, u] >= 0.99, 1.0, 2.0)
        pts.append(p)
        sig.append(s)
        fidx.append(np.full(len(p), k, np.int32))
        h, w = depth.shape
        vv, uu = np.mgrid[0:h:far_stride, 0:w:far_stride]
        vv, uu = vv.ravel(), uu.ravel()
        # see-through evidence only from rays that truly went far: no return, or beyond sensor range.
        # Low-confidence in-range pixels (edge smear) are neither geometry nor evidence.
        dz = depth[vv, uu]
        bad = (dz <= 0.15) | (dz >= model.max_depth)
        dist = np.where(dz > 0.15, dz, model.far_dist)[bad]
        far.append(pixel_rays(fr.K, fr.T_wc, vv[bad], uu[bad], dist))
        far_f.append(np.full(int(bad.sum()), k, np.int32))
    if not pts:
        raise ValueError(f"no depth in group {group}")
    P, S, F, _ = voxel_reduce(np.concatenate(pts), voxel, np.concatenate(sig), np.concatenate(fidx))
    for i, f in enumerate(frames):
        f.index = i
    return Observations(tier=tier, group=group, frames=frames, points=P, sigma=S, frame_idx=F,
                        rays_far=np.concatenate(far) if far else np.zeros((0, 3)),
                        rays_far_frame=np.concatenate(far_f) if far_f else np.zeros(0, np.int32),
                        warnings=list(warnings or []), meta=dict(meta or {}))
