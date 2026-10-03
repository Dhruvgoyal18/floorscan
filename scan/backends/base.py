"""The ReconstructionBackend seam: posed images in, metric depth + poses out.

Video and photo tiers have no depth sensor, so a learned model supplies metric depth, camera poses
and (refined) intrinsics. Everything downstream of this seam is shared with the LiDAR tier.

Adapters:
  * ``mapanything``  - MapAnything (Meta, Apache-2.0 variant available). Primary.
  * ``depthanything`` - Depth-Anything-V2 metric (indoor) per view + RGB-D PnP for poses. Lighter.
  * ``oracle``       - synthetic benchmark only: reads renderer sidecars and perturbs them with a
                       documented error model. Never used on real captures (it refuses).
Outputs are cached on disk keyed by the input bytes, so reruns replay deterministically.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from ..types import Frame


@dataclass
class View:
    name: str
    load_image: Callable[[], np.ndarray]   # RGB uint8 (H, W, 3)
    K: np.ndarray | None                   # intrinsics guess at image resolution (EXIF / device table)
    timestamp: float = 0.0
    source: str = ""                       # file path (photo) or "video:<path>#<frame>"
    group: str = ""


class ReconstructionBackend(Protocol):
    name: str

    def reconstruct(self, views: list[View], tier: str) -> list[Frame]:
        """Return one Frame per *usable* view, all in one (arbitrary, metric) world frame.

        Frames carry depth (m), conf (0..1), K at depth resolution, T_wc (OpenCV camera), and
        load_image for the original image. Views that could not be registered are omitted."""
        ...


def get_backend(name: str, **kw) -> ReconstructionBackend:
    if name == "oracle":
        from .oracle import OracleBackend
        return CachedBackend(OracleBackend(**kw))
    if name == "mapanything":
        from .mapanything import MapAnythingBackend
        return CachedBackend(MapAnythingBackend(**kw))
    if name == "depthanything":
        from .depthanything import DepthAnythingBackend
        return CachedBackend(DepthAnythingBackend(**kw))
    if name == "auto":
        for cand in ("mapanything", "depthanything"):
            try:
                return get_backend(cand, **kw)
            except BackendUnavailable:
                continue
        raise BackendUnavailable(
            "no learned backend available for the video/photo tiers. Run `make weights` (needs network once) "
            "or pass --backend oracle for synthetic captures.")
    raise ValueError(f"unknown backend {name}")


class BackendUnavailable(RuntimeError):
    pass


def cache_dir() -> Path:
    return Path(os.environ.get("FLOORSCAN_CACHE", Path.cwd() / ".cache" / "backends"))


class CachedBackend:
    """Deterministic replay of model outputs, keyed by backend name + version + image bytes + intrinsics."""

    def __init__(self, inner):
        self.inner = inner
        self.name = inner.name

    def _key(self, views: list[View], tier: str) -> str:
        h = hashlib.sha256()
        h.update(f"{self.inner.name}|{getattr(self.inner, 'version', '0')}|{tier}".encode())
        for v in views:
            h.update(v.name.encode())
            h.update(v.source.encode())
            src = v.source.split("#")[0]
            if src and Path(src).is_file():
                st = Path(src).stat()
                h.update(f"{st.st_size}:{int(st.st_mtime)}".encode())
            if v.K is not None:
                h.update(np.round(v.K, 3).tobytes())
        return h.hexdigest()[:24]

    def reconstruct(self, views: list[View], tier: str) -> list[Frame]:
        if os.environ.get("FLOORSCAN_NO_CACHE"):
            return self.inner.reconstruct(views, tier)
        d = cache_dir() / self.inner.name
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{self._key(views, tier)}.npz"
        by_name = {v.name: v for v in views}
        if f.exists():
            z = np.load(f, allow_pickle=False)
            meta = json.loads(str(z["meta"]))
            frames = []
            for i, m in enumerate(meta):
                v = by_name[m["name"]]
                frames.append(Frame(index=i, T_wc=z[f"T{i}"], K=z[f"K{i}"], depth=z[f"D{i}"].astype(np.float32),
                                    conf=z[f"C{i}"].astype(np.float32), load_image=v.load_image,
                                    timestamp=m["t"], name=m["name"]))
            return frames
        frames = self.inner.reconstruct(views, tier)
        arrs = {}
        meta = []
        for i, fr in enumerate(frames):
            arrs[f"T{i}"], arrs[f"K{i}"] = fr.T_wc, fr.K
            arrs[f"D{i}"] = fr.depth.astype(np.float16)
            arrs[f"C{i}"] = (fr.conf if fr.conf is not None else np.ones_like(fr.depth)).astype(np.float16)
            meta.append({"name": fr.name, "t": fr.timestamp})
        np.savez_compressed(f, meta=json.dumps(meta), **arrs)
        # replay from what was stored, so the first run and every rerun see identical (float16) data
        return self.reconstruct(views, tier)
