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
    normals, w0, w1 = _smallest_eig_sym3(C)
    planarity = 1.0 - w0 / np.maximum(w1, 1e-12)
    if origins is not None:
        flip = np.sum((origins - points) * normals, 1) < 0
        normals[flip] *= -1
    return normals, np.clip(planarity, 0, 1)


def _smallest_eig_sym3(C: np.ndarray):
    """Closed-form smallest eigenpair of many symmetric 3x3 matrices (10x faster than batched eigh).

    Returns (eigenvector of the smallest eigenvalue, smallest eigenvalue, middle eigenvalue)."""
    a, b, c = C[:, 0, 0], C[:, 1, 1], C[:, 2, 2]
    d, e, f = C[:, 0, 1], C[:, 1, 2], C[:, 0, 2]
    q = (a + b + c) / 3
    p1 = d * d + e * e + f * f
    p2 = (a - q) ** 2 + (b - q) ** 2 + (c - q) ** 2 + 2 * p1
    p = np.sqrt(np.maximum(p2 / 6, 1e-30))
    B = (C - q[:, None, None] * np.eye(3)) / p[:, None, None]
    r = np.clip(np.linalg.det(B) / 2, -1, 1)
    phi = np.arccos(r) / 3
    l1 = q + 2 * p * np.cos(phi)                       # largest
    l3 = q + 2 * p * np.cos(phi + 2 * np.pi / 3)       # smallest
    l2 = 3 * q - l1 - l3
    M = C - l3[:, None, None] * np.eye(3)
    r0, r1, r2 = M[:, 0], M[:, 1], M[:, 2]
    cands = np.stack([np.cross(r0, r1), np.cross(r0, r2), np.cross(r1, r2)], 1)
    norms = np.linalg.norm(cands, axis=2)
    best = np.argmax(norms, axis=1)
    v = cands[np.arange(len(C)), best]
    nv = norms[np.arange(len(C)), best]
    v = v / np.maximum(nv, 1e-30)[:, None]
    degenerate = nv < 1e-20
    if degenerate.any():
        v[degenerate] = np.linalg.eigh(C[degenerate])[1][:, :, 0]
    return v, np.maximum(l3, 0), np.maximum(l2, 0)
