"""Stray Scanner loader (LiDAR tier).

Format (verified against the app's source, OdometryEncoder.swift / DatasetEncoder.swift):
  odometry.csv  timestamp, frame, x, y, z, qx, qy, qz, qw[, fx, fy, cx, cy, dcx, dcy]
                cam-to-world, OpenCV camera axes, ARKit world (y up)
  camera_matrix.csv  3x3 intrinsics at RGB resolution (last frame)
  depth/NNNNNN.png   uint16 millimetres (256x192 on current devices)
  confidence/NNNNNN.png  uint8 0/1/2
  rgb.mp4
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from ..types import Frame

M_AR2ZUP = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])


def is_stray(d: Path) -> bool:
    return (d / "odometry.csv").exists() and (d / "depth").is_dir()


def read_odometry(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        try:
            vals = [float(p) if p else np.nan for p in parts]
        except ValueError:
            continue  # header
        rows.append(vals[:13] + [np.nan] * (13 - len(vals[:13])))
    return np.array(rows, float)


class VideoFrames:
    """Random access to frames of a video with a small cache."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._cache: dict[int, np.ndarray] = {}
        cap = cv2.VideoCapture(str(self.path))
        self.size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

    def get(self, idx: int) -> np.ndarray | None:
        if idx in self._cache:
            return self._cache[idx]
        cap = cv2.VideoCapture(str(self.path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if len(self._cache) > 64:
            self._cache.pop(next(iter(self._cache)))
        self._cache[idx] = rgb
        return rgb


def load_stray_frames(d: Path, target_fps: float = 5.0, poses_override: list[np.ndarray] | None = None):
    """-> (frames, meta). Frames carry depth (m), confidence (0..1), K at depth resolution, z-up pose."""
    d = Path(d)
    odo = read_odometry(d / "odometry.csv")
    if len(odo) == 0:
        raise ValueError(f"{d}/odometry.csv has no rows")
    K_rgb_file = np.loadtxt(d / "camera_matrix.csv", delimiter=",").reshape(3, 3)
    video = VideoFrames(d / "rgb.mp4") if (d / "rgb.mp4").exists() else None
    W, H = video.size if video and video.size[0] > 0 else (1920, 1440)

    depth_files = sorted((d / "depth").glob("*.png")) or sorted((d / "depth").glob("*.npy"))
    if not depth_files:
        raise ValueError(f"no depth frames in {d}/depth")
    hd, wd = _read_depth(depth_files[0]).shape
    by_idx = {int(p.stem): p for p in depth_files}

    ts = odo[:, 0]
    chosen, last_t = [], -1e9
    for i in range(len(odo)):
        if ts[i] - last_t >= 1.0 / target_fps - 1e-6 and int(odo[i, 1]) in by_idx:
            chosen.append(i)
            last_t = ts[i]

    frames = []
    for k, i in enumerate(chosen):
        row = odo[i]
        fno = int(row[1])
        if poses_override is not None:
            T = np.array(poses_override[fno], float)
        else:
            T = np.eye(4)
            T[:3, :3] = M_AR2ZUP @ Rotation.from_quat(row[5:9]).as_matrix()
            T[:3, 3] = M_AR2ZUP @ row[2:5]
        K_rgb = K_rgb_file
        if not np.isnan(row[9:13]).any():
            K_rgb = np.array([[row[9], 0, row[11]], [0, row[10], row[12]], [0, 0, 1.0]])
        K = K_rgb.copy()
        K[0] *= wd / W
        K[1] *= hd / H
        depth = _read_depth(by_idx[fno])
        cpath = d / "confidence" / f"{fno:06d}.png"
        conf = np.array(Image.open(cpath)).astype(np.float32) / 2.0 if cpath.exists() else np.ones(depth.shape, np.float32)
        frames.append(Frame(index=k, T_wc=T, K=K, depth=depth, conf=conf,
                            load_image=(lambda fi=fno: video.get(fi)) if video else None,
                            timestamp=float(row[0]), name=f"{fno:06d}"))
    meta = {"device_res": [wd, hd], "rgb_res": [W, H], "n_frames_total": int(len(odo)), "n_frames_used": len(frames)}
    return frames, meta


def _read_depth(p: Path) -> np.ndarray:
    a = np.load(p) if p.suffix == ".npy" else np.array(Image.open(p))
    return a.astype(np.float32) / 1000.0
