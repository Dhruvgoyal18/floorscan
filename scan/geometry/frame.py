"""Gravity and Manhattan alignment."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


def rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def manhattan_yaw(normals: np.ndarray, planarity: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float]:
    """Dominant wall direction (radians in [0, pi/2)) from horizontal normals, and the share of
    wall-normal mass within 5 degrees of the Manhattan axes (1.0 = perfectly Manhattan)."""
    m = (np.abs(normals[:, 2]) < 0.25) & (planarity > 0.6)
    if m.sum() < 50:
        return 0.0, 0.0
    th = np.mod(np.arctan2(normals[m, 1], normals[m, 0]), np.pi / 2)
    w = None if weights is None else weights[m]
    bins = 360
    h, _ = np.histogram(th, bins=bins, range=(0, np.pi / 2), weights=w)
    hs = gaussian_filter1d(h.astype(float), 2.0, mode="wrap")
    k = int(np.argmax(hs))
    center = (k + 0.5) * (np.pi / 2) / bins
    # refine with circular mean of the 4-fold angle inside +-3 deg
    d = np.angle(np.exp(4j * (th - center))) / 4
    sel = np.abs(d) < np.radians(3)
    yaw = center + (np.mean(d[sel]) if sel.any() else 0.0)
    strong = planarity[m] > 0.85
    th_s = th[strong] if strong.sum() > 50 else th
    share = float(np.mean(np.abs(np.angle(np.exp(4j * (th_s - yaw))) / 4) < np.radians(10)))
    return float(np.mod(yaw, np.pi / 2)), share


def gravity_from_normals(normals: np.ndarray, planarity: np.ndarray, cam_down: np.ndarray) -> np.ndarray:
    """Estimate the up vector for tiers without IMU gravity (photo/video backends).

    Horizontal surfaces (floor, ceiling, tables) dominate one normal direction; the sign is chosen so
    that 'up' is opposite to the cameras' mean image-down direction (phones are held upright).
    """
    m = planarity > 0.7
    N = normals[m]
    if len(N) < 100:
        u = -cam_down / np.linalg.norm(cam_down)
        return u
    # candidate = cameras' up; refine by the normal cluster closest to it
    u0 = -cam_down / np.linalg.norm(cam_down)
    for _ in range(3):
        sel = np.abs(N @ u0) > np.cos(np.radians(25))
        if sel.sum() < 30:
            break
        S = N[sel] * np.sign(N[sel] @ u0)[:, None]
        w, V = np.linalg.eigh(S.T @ S)
        u0 = V[:, -1] * np.sign(V[:, -1] @ u0)
    return u0 / np.linalg.norm(u0)


def align_up(up: np.ndarray) -> np.ndarray:
    """Rotation R with R @ up = +z."""
    z = np.array([0, 0, 1.0])
    v = np.cross(up, z)
    s, c = np.linalg.norm(v), float(up @ z)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1, -1])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)
