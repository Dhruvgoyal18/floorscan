"""Core data types shared by every module.

Conventions (used everywhere, never re-decided locally):
  * World frame: metric, right-handed, **z up** (gravity = -z).
  * Camera frame: OpenCV (x right, y down, z forward).
  * ``T_wc``: 4x4 camera-to-world transform.
  * Intrinsics ``K`` are for the resolution of the array they accompany.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

TIERS = ("lidar", "video", "photo")


@dataclass
class Estimate:
    """A measured value with a standard error, before calibration."""

    value: float
    se: float
    observed: bool = True

    def __add__(self, other: "Estimate") -> "Estimate":
        return Estimate(self.value + other.value, float(np.hypot(self.se, other.se)), self.observed and other.observed)

    def __sub__(self, other: "Estimate") -> "Estimate":
        return Estimate(self.value - other.value, float(np.hypot(self.se, other.se)), self.observed and other.observed)

    def scaled(self, k: float) -> "Estimate":
        return Estimate(self.value * k, abs(self.se * k), self.observed)


@dataclass
class Frame:
    """One posed view. ``depth`` may be None (photo views before reconstruction)."""

    index: int
    T_wc: np.ndarray                      # 4x4, z-up world
    K: np.ndarray                         # 3x3 for ``depth`` resolution
    depth: Optional[np.ndarray] = None    # (h, w) float32 metres, 0 = invalid
    conf: Optional[np.ndarray] = None     # (h, w) float32 in [0, 1]
    load_image: Optional[Callable[[], np.ndarray]] = None  # -> (H, W, 3) uint8 RGB, any resolution
    timestamp: float = 0.0
    name: str = ""

    @property
    def center(self) -> np.ndarray:
        return self.T_wc[:3, 3]


@dataclass
class Observations:
    """Tier-agnostic intermediate: everything the geometry core needs.

    points      (N, 3) world points used for geometry
    sigma       (N,)   1-sigma range noise of each point (metres)
    frame_idx   (N,)   index into ``frames`` of the view that produced the point
    rays_far    (M, 3) world points far along rays that returned nothing / were dropped
                       (used only as see-through evidence for openings)
    rays_far_frame (M,) frame index for each far ray
    """

    tier: str
    group: str
    frames: list[Frame]
    points: np.ndarray
    sigma: np.ndarray
    frame_idx: np.ndarray
    rays_far: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    rays_far_frame: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int32))
    warnings: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def cam_centers(self) -> np.ndarray:
        if not self.frames:
            return np.zeros((0, 3))
        return np.stack([f.center for f in self.frames])

    def origins(self) -> np.ndarray:
        """Camera centre for every point."""
        return self.cam_centers[self.frame_idx]

    def subset_frames(self, keep: np.ndarray) -> "Observations":
        """Observations restricted to (a multiset of) frames — used by the bootstrap.

        ``keep`` is an array of frame indices, repeats allowed. Frame indices are preserved.
        """
        counts = np.bincount(keep, minlength=len(self.frames))
        rep = counts[self.frame_idx]
        idx = np.repeat(np.arange(len(self.points)), rep)
        rrep = counts[self.rays_far_frame] if len(self.rays_far) else np.zeros(0, dtype=int)
        ridx = np.repeat(np.arange(len(self.rays_far)), rrep)
        return Observations(
            tier=self.tier, group=self.group, frames=self.frames,
            points=self.points[idx], sigma=self.sigma[idx], frame_idx=self.frame_idx[idx],
            rays_far=self.rays_far[ridx], rays_far_frame=self.rays_far_frame[ridx],
            warnings=list(self.warnings), meta=dict(self.meta),
        )

    def transformed(self, T: np.ndarray) -> "Observations":
        """Apply a rigid world transform to everything (points, rays, frame poses)."""
        R, t = T[:3, :3], T[:3, 3]
        frames = [
            Frame(f.index, T @ f.T_wc, f.K, f.depth, f.conf, f.load_image, f.timestamp, f.name)
            for f in self.frames
        ]
        return Observations(
            tier=self.tier, group=self.group, frames=frames,
            points=self.points @ R.T + t, sigma=self.sigma, frame_idx=self.frame_idx,
            rays_far=self.rays_far @ R.T + t if len(self.rays_far) else self.rays_far,
            rays_far_frame=self.rays_far_frame, warnings=list(self.warnings), meta=dict(self.meta),
        )
