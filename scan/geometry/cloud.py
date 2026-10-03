"""Point-cloud utilities: back-projection, voxel downsampling, normals."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def backproject(depth: np.ndarray, K: np.ndarray, T_wc: np.ndarray, mask: np.ndarray | None = None, stride: int = 1):
    """Depth (h,w) z-depth -> world points (n,3) and the (v,u) pixel indices used."""
    h, w = depth.shape
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[v, u]
    m = z > 0
    if mask is not None:
        m &= mask[v, u]
    u, v, z = u[m], v[m], z[m]
    x = (u + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (v + 0.5 - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], 1)
    return pc @ T_wc[:3, :3].T + T_wc[:3, 3], (v, u)


def pixel_rays(K: np.ndarray, T_wc: np.ndarray, v: np.ndarray, u: np.ndarray, dist: np.ndarray):
    """World points at z-depth ``dist`` along the rays of pixels (v,u)."""
    x = (u + 0.5 - K[0, 2]) / K[0, 0] * dist
    y = (v + 0.5 - K[1, 2]) / K[1, 1] * dist
    pc = np.stack([x, y, dist], 1)
    return pc @ T_wc[:3, :3].T + T_wc[:3, 3]


def voxel_reduce(points: np.ndarray, voxel: float, *attrs: np.ndarray):
    """Average points per voxel; for each attr keep the value of the first point in the voxel.

    Returns (points_mean, attrs..., counts).
    """
    if len(points) == 0:
        return (points, *attrs, np.zeros(0, int))
    q = np.floor(points / voxel).astype(np.int64)
    q -= q.min(0)
    dims = q.max(0) + 1
    key = (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]
    uniq, first, inv, counts = np.unique(key, return_index=True, return_inverse=True, return_counts=True)
    mean = np.stack([np.bincount(inv, weights=points[:, k], minlength=len(uniq)) for k in range(3)], 1) / counts[:, None]
    return (mean, *[a[first] for a in attrs], counts)


def estimate_normals(points: np.ndarray, k: int = 16, origins: np.ndarray | None = None):
    """PCA normals (n,3) and planarity (n,) in [0,1]. If ``origins`` given, normals face the camera."""
    n = len(points)
    if n < 4:
        return np.zeros((n, 3)), np.zeros(n)
    k = min(k, n)
    tree = cKDTree(points)
    _, nb = tree.query(points, k=k, workers=-1)
    P = points[nb] - points[nb].mean(1, keepdims=True)
    C = np.einsum("nki,nkj->nij", P, P) / k
    w, V = np.linalg.eigh(C)
    normals = V[:, :, 0]
    planarity = 1.0 - w[:, 0] / np.maximum(w[:, 1], 1e-12)
    if origins is not None:
        flip = np.sum((origins - points) * normals, 1) < 0
        normals[flip] *= -1
    return normals, np.clip(planarity, 0, 1)
