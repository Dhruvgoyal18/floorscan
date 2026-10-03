"""Depth-Anything-V2 metric (indoor) + RGB-D PnP poses. Lighter fallback for CPU-only machines.

Depth: per view, from the HF transformers checkpoint
``depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf`` (metric metres).
Poses: SIFT matches between views; 3D points from view i's metric depth; PnP-RANSAC in view j gives
the metric relative pose. Video: consecutive keyframes (with a fallback to i-2). Photos: a maximum
spanning tree over pairwise inlier counts. Views that cannot be registered are dropped with the count
reported, never placed at a guessed pose.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from ..types import Frame
from .base import BackendUnavailable, View

DEFAULT_MODEL = os.environ.get("FLOORSCAN_DA_MODEL", "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")


class DepthAnythingBackend:
    name = "depthanything"
    version = "1"

    def __init__(self, model_id: str = DEFAULT_MODEL, max_side: int = 640):
        try:
            import torch  # noqa: F401
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise BackendUnavailable(f"transformers/torch not installed ({e}); run `make weights`") from e
        self.model_id, self.max_side = model_id, max_side
        self._m = None
        self.depth_fn = self._predict  # seam for tests: any callable(rgb) -> metric depth (H, W)

    def _predict(self, rgb: np.ndarray) -> np.ndarray:  # pragma: no cover - needs weights
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        if self._m is None:
            try:
                self._p = AutoImageProcessor.from_pretrained(self.model_id, local_files_only=True)
                self._m = AutoModelForDepthEstimation.from_pretrained(self.model_id, local_files_only=True).eval()
            except Exception as e:
                raise BackendUnavailable(f"Depth-Anything weights not cached ({e}); run `make weights`") from e
        inp = self._p(images=rgb, return_tensors="pt")
        with torch.no_grad():
            out = self._m(**inp).predicted_depth
        d = torch.nn.functional.interpolate(out[:, None], size=rgb.shape[:2], mode="bicubic", align_corners=False)
        return d[0, 0].numpy().astype(np.float32)

    def reconstruct(self, views: list[View], tier: str) -> list[Frame]:
        imgs, Ks, depths = [], [], []
        for v in views:
            im = v.load_image()
            s = self.max_side / max(im.shape[:2])
            if s < 1:
                im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
            K = v.K.copy() if v.K is not None else _default_K(im.shape)
            if v.K is not None and s < 1:
                K[:2] *= s
            imgs.append(im)
            Ks.append(K)
            depths.append(self.depth_fn(im))
        poses = register_rgbd(imgs, depths, Ks, sequential=(tier == "video"))
        frames = []
        for i, (v, T) in enumerate(zip(views, poses)):
            if T is None:
                continue
            frames.append(Frame(index=i, T_wc=T, K=Ks[i], depth=depths[i], conf=np.full(depths[i].shape, 0.8, np.float32),
                                load_image=(lambda im=imgs[i]: im), timestamp=v.timestamp, name=v.name))
        return frames


def _default_K(shape):
    h, w = shape[:2]
    f = 0.8 * max(h, w)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def _features(img):
    sift = cv2.SIFT_create(nfeatures=4000)
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return sift.detectAndCompute(g, None)


def relative_pose_pnp(kp_i, des_i, depth_i, K_i, kp_j, des_j, K_j, min_inliers: int = 30):
    """Pose of camera j in camera i's frame (4x4 T_ij), metric via depth_i. None if unreliable."""
    if des_i is None or des_j is None or len(kp_i) < 8 or len(kp_j) < 8:
        return None, 0
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_i, des_j, k=2)
    good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < 0.75 * b.distance]
    if len(good) < min_inliers:
        return None, 0
    pi = np.array([kp_i[g.queryIdx].pt for g in good])
    pj = np.array([kp_j[g.trainIdx].pt for g in good])
    u, v = pi[:, 0].astype(int), pi[:, 1].astype(int)
    ok = (u >= 0) & (v >= 0) & (u < depth_i.shape[1]) & (v < depth_i.shape[0])
    z = np.zeros(len(pi))
    z[ok] = depth_i[v[ok], u[ok]]
    ok &= (z > 0.2) & (z < 10)
    if ok.sum() < min_inliers:
        return None, 0
    X = np.stack([(pi[ok, 0] - K_i[0, 2]) / K_i[0, 0] * z[ok], (pi[ok, 1] - K_i[1, 2]) / K_i[1, 1] * z[ok], z[ok]], 1)
    succ, rvec, tvec, inl = cv2.solvePnPRansac(X, pj[ok], K_j, None, reprojectionError=3.0, iterationsCount=500,
                                               flags=cv2.SOLVEPNP_EPNP)
    if not succ or inl is None or len(inl) < min_inliers:
        return None, 0
    R, _ = cv2.Rodrigues(rvec)
    T_ji = np.eye(4)
    T_ji[:3, :3], T_ji[:3, 3] = R, tvec[:, 0]
    return np.linalg.inv(T_ji), len(inl)


def register_rgbd(imgs, depths, Ks, sequential: bool):
    n = len(imgs)
    feats = [_features(im) for im in imgs]
    poses: list = [None] * n
    poses[0] = np.eye(4)
    if sequential:
        for j in range(1, n):
            for back in (1, 2, 3):
                i = j - back
                if i < 0 or poses[i] is None:
                    continue
                T_ij, k = relative_pose_pnp(*feats[i], depths[i], Ks[i], *feats[j], Ks[j])
                if T_ij is not None:
                    poses[j] = poses[i] @ T_ij
                    break
        return poses
    # photos: maximum spanning tree over inlier counts
    rel = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                T, k = relative_pose_pnp(*feats[i], depths[i], Ks[i], *feats[j], Ks[j])
                if T is not None:
                    rel[(i, j)] = (T, k)
    placed = {0}
    while True:
        best = None
        for (i, j), (T, k) in rel.items():
            if i in placed and j not in placed and (best is None or k > best[2]):
                best = (i, j, k, T)
        if best is None:
            break
        i, j, _, T = best
        poses[j] = poses[i] @ T
        placed.add(j)
    return poses
