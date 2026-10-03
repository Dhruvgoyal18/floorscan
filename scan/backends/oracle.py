"""Oracle backend: synthetic benchmark only.

Reads the renderer's sidecar (true depth + true pose per view) and degrades it with an error model
meant to resemble a feed-forward metric reconstruction model:
  * per-reconstruction metric scale error   (photo 2.5%, video 1.0%, 1 sigma)
  * per-view depth noise: smooth (low-frequency) 1.5% + white 0.5% of depth
  * per-view pose noise: 0.4 deg rotation, 1.5 cm translation (photo); half for video
  * the whole reconstruction is expressed in an arbitrary world frame (random rotation incl. tilt),
    so gravity must be recovered downstream exactly as for a real model.
These numbers are assumptions, not measurements of any model; real-model accuracy is reported only
from real captures. The oracle exists so the downstream pipeline can be developed and tested.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation

from ..types import Frame
from .base import View

ERRORS = {
    "photo": {"scale": 0.025, "rel_lf": 0.015, "rel_w": 0.005, "rot_deg": 0.4, "trans": 0.015},
    "video": {"scale": 0.010, "rel_lf": 0.012, "rel_w": 0.004, "rot_deg": 0.2, "trans": 0.008},
}


class OracleBackend:
    name = "oracle"
    version = "1"

    def __init__(self, errors: dict | None = None):
        self.errors = errors or ERRORS

    def reconstruct(self, views: list[View], tier: str) -> list[Frame]:
        err = self.errors[tier]
        # seed from the capture-relative names (folder/file, frame index), never from absolute paths,
        # so the same capture gives the same draw on any machine / checkout location
        key = "|".join("/".join(Path(v.source.split("#")[0].replace("video:", "")).parts[-2:]) +
                       ("#" + v.source.split("#")[1] if "#" in v.source else "") for v in views)
        seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        scale = 1.0 + rng.normal(0, err["scale"])
        G = np.eye(4)
        G[:3, :3] = Rotation.from_euler("xyz", [rng.normal(0, 8), rng.normal(0, 8), rng.uniform(0, 360)],
                                        degrees=True).as_matrix()
        G[:3, 3] = rng.normal(0, 2, 3)
        frames = []
        video_cache: dict[str, dict] = {}
        for i, v in enumerate(views):
            depth, K, T = self._truth(v, video_cache)
            if depth is None:
                raise RuntimeError(f"oracle backend: no synthetic sidecar for {v.source}. The oracle only runs on "
                                   "synthetic captures; use --backend mapanything for real data.")
            h, w = depth.shape
            lf = ndimage.zoom(rng.normal(0, 1, (4, 5)), (h / 4, w / 5), order=1)[:h, :w]
            noisy = depth * scale * (1 + err["rel_lf"] * lf + err["rel_w"] * rng.normal(0, 1, depth.shape))
            noisy[depth <= 0] = 0
            noisy[depth > 12] = 0
            dR = Rotation.from_rotvec(rng.normal(0, np.radians(err["rot_deg"]), 3)).as_matrix()
            Tn = T.copy()
            Tn[:3, :3] = dR @ T[:3, :3]
            Tn[:3, 3] = T[:3, 3] * scale + rng.normal(0, err["trans"], 3)
            conf = np.where(noisy > 0, 0.9, 0.0).astype(np.float32)
            frames.append(Frame(index=i, T_wc=G @ Tn, K=K, depth=noisy.astype(np.float32), conf=conf,
                                load_image=v.load_image, timestamp=v.timestamp, name=v.name))
        return frames

    @staticmethod
    def _truth(v: View, cache):
        src = v.source
        if src.startswith("video:"):
            path, idx = src[len("video:"):].rsplit("#", 1)
            side = Path(path).parent / "_synthetic" / "depth.npz"
            if not side.exists():
                return None, None, None
            if str(side) not in cache:
                z = np.load(side)
                cache[str(side)] = {"depth": z["depth"], "K": z["K"], "stride": int(z["stride"]), "poses": z["poses"]}
            c = cache[str(side)]
            k = int(idx)
            K = c["K"].copy()
            K[:2] /= c["stride"]
            return c["depth"][k].astype(np.float32), K, c["poses"][k]
        p = Path(src)
        side = p.parent / "_synthetic" / f"{p.stem}.npz"
        if not side.exists():
            return None, None, None
        z = np.load(side)
        K = z["K"].copy()
        K[:2] /= int(z["stride"])
        return z["depth"].astype(np.float32), K, z["T_wc"]


def is_synthetic(capture: Path) -> bool:
    capture = Path(capture)
    base = capture if capture.is_dir() else capture.parent
    return (base / "_synthetic" / "meta.json").exists()


def synthetic_meta(capture: Path) -> dict:
    capture = Path(capture)
    base = capture if capture.is_dir() else capture.parent
    return json.loads((base / "_synthetic" / "meta.json").read_text())
