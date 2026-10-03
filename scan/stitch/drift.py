"""Drift correction for continuous captures (LiDAR, video): fragment pose graph with loop closure.

Odometry (ARKit VIO, or chained video poses) drifts over a multi-room walk. We:
  1. split the frame sequence into fragments (~3 s); each fragment is internally consistent;
  2. build a small point cloud per fragment in its anchor frame;
  3. add edges: sequential (odometry, refined by ICP) and loop closures (ICP between non-adjacent
     fragments whose clouds overlap, e.g. re-entering the hall or ending where the walk started);
  4. solve a 4-DoF pose graph (x, y, z, yaw — gravity from the IMU is trusted) with a robust loss;
  5. re-pose every frame by its fragment's correction.

"Poses used as-is" is never the default: ``enabled=False`` exists only for the ablation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from ..geometry.cloud import backproject, estimate_normals, voxel_reduce
from ..types import Frame


@dataclass
class DriftConfig:
    enabled: bool = True
    frag_seconds: float = 3.0
    min_frag_frames: int = 6
    voxel: float = 0.04
    loop_radius: float = 2.5
    icp_max_iter: int = 30
    accept_fitness: float = 0.35
    accept_rmse: float = 0.03
    sigma_odo_t: float = 0.03
    sigma_odo_yaw: float = np.radians(0.6)
    sigma_icp_t: float = 0.01
    sigma_icp_yaw: float = np.radians(0.25)


@dataclass
class DriftReport:
    method: str
    enabled: bool
    n_fragments: int = 0
    n_seq_edges: int = 0
    n_loop_edges: int = 0
    loop_closure_error_m: float | None = None       # median loop-edge translation disagreement before
    loop_closure_error_after_m: float | None = None
    max_correction_m: float = 0.0
    edges: list = field(default_factory=list)

    def as_dict(self):
        return {"method": self.method, "enabled": self.enabled, "n_fragments": self.n_fragments,
                "n_seq_edges": self.n_seq_edges, "n_loop_edges": self.n_loop_edges,
                "loop_closure_error_m": self.loop_closure_error_m,
                "loop_closure_error_after_m": self.loop_closure_error_after_m,
                "max_correction_m": self.max_correction_m}


# ------------------------------------------------------------------ SE3 helpers (yaw-only corrections)

def _yaw_of(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _C(x):
    """Correction transform from (tx, ty, tz, yaw)."""
    c, s = np.cos(x[3]), np.sin(x[3])
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:3, 3] = x[:3]
    return T


def _inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


# ------------------------------------------------------------------ ICP (4-DoF point-to-plane)

def icp_4dof(src: np.ndarray, dst: np.ndarray, dst_n: np.ndarray, max_iter: int = 30, init: np.ndarray | None = None):
    """Find E (yaw about z + translation) minimising point-to-plane distance of E@src to dst.

    Returns (E, fitness, rmse, min_eig) where min_eig is the smallest eigenvalue of the normalised
    information matrix (small => degenerate geometry, e.g. a single wall)."""
    E = np.eye(4) if init is None else init.copy()
    if len(src) > 4000:  # deterministic subsample: ICP needs coverage, not density
        src = src[np.linspace(0, len(src) - 1, 4000).astype(int)]
    tree = cKDTree(dst)
    cen = src.mean(0)
    H = np.eye(4)
    for it in range(max_iter):
        s = src @ E[:3, :3].T + E[:3, 3]
        d, j = tree.query(s, k=1)
        thr = 0.15 if it < 5 else (0.08 if it < 12 else 0.04)
        m = d < thr
        if m.sum() < 30:
            return E, 0.0, np.inf, 0.0
        sm, tm, nm = s[m], dst[j[m]], dst_n[j[m]]
        r = np.sum(nm * (sm - tm), 1)
        q = sm - cen
        J = np.stack([nm[:, 0] * -q[:, 1] + nm[:, 1] * q[:, 0], nm[:, 0], nm[:, 1], nm[:, 2]], 1)
        w = 1.0 / np.maximum(1.0, np.abs(r) / 0.01)  # Huber
        H = (J * w[:, None]).T @ J
        g = (J * w[:, None]).T @ r
        try:
            dx = -np.linalg.solve(H + 1e-9 * np.eye(4), g)
        except np.linalg.LinAlgError:
            break
        c, s_ = np.cos(dx[0]), np.sin(dx[0])
        D = np.eye(4)
        D[:2, :2] = [[c, -s_], [s_, c]]
        D[:3, 3] = cen - D[:3, :3] @ cen + dx[1:]
        E = D @ E
        if np.abs(dx[0]) < 1e-5 and np.linalg.norm(dx[1:]) < 1e-4:
            break
    s = src @ E[:3, :3].T + E[:3, 3]
    d, _ = tree.query(s, k=1)
    inl = d < 0.03
    fitness = float(inl.mean())
    rmse = float(np.sqrt(np.mean(d[inl] ** 2))) if inl.any() else np.inf
    Hn = H / max(1, len(src))
    scale = np.diag([1 / max(np.linalg.norm(src - cen, axis=1).mean(), 0.1), 1, 1, 1])
    min_eig = float(np.linalg.eigvalsh(scale @ Hn @ scale).min())
    return E, fitness, rmse, min_eig


# ------------------------------------------------------------------ main

def _fragments(frames: list[Frame], cfg: DriftConfig) -> list[list[int]]:
    frags, cur, t0 = [], [], None
    for i, f in enumerate(frames):
        if t0 is None:
            t0 = f.timestamp
        cur.append(i)
        if f.timestamp - t0 >= cfg.frag_seconds and len(cur) >= cfg.min_frag_frames:
            frags.append(cur)
            cur, t0 = [], None
    if cur:
        if frags and len(cur) < cfg.min_frag_frames:
            frags[-1].extend(cur)
        else:
            frags.append(cur)
    return frags


def _frag_cloud(frames, idx, anchor_inv, cfg, conf_min, max_depth):
    pts = []
    for i in idx:
        f = frames[i]
        if f.depth is None:
            continue
        conf = f.conf if f.conf is not None else np.ones_like(f.depth)
        good = (conf >= conf_min) & (f.depth > 0.15) & (f.depth < max_depth)
        p, _ = backproject(f.depth, f.K, anchor_inv @ f.T_wc, good, stride=2)
        pts.append(p)
    if not pts:
        return np.zeros((0, 3)), np.zeros((0, 3))
    P = voxel_reduce(np.concatenate(pts), cfg.voxel)[0]
    N, _ = estimate_normals(P, k=12, origins=np.zeros_like(P))
    return P, N


def correct_drift(frames: list[Frame], cfg: DriftConfig | None = None, conf_min: float = 0.5,
                  max_depth: float = 4.0) -> tuple[list[Frame], DriftReport]:
    cfg = cfg or DriftConfig()
    rep = DriftReport(method="fragment pose graph: ICP sequential + loop-closure edges, 4-DoF robust LSQ",
                      enabled=cfg.enabled)
    frags = _fragments(frames, cfg)
    rep.n_fragments = len(frags)
    if not cfg.enabled or len(frags) < 2:
        if not cfg.enabled:
            rep.method = "none (ablation: poses used as-is)"
        return frames, rep

    anchors = [frames[f[0]].T_wc.copy() for f in frags]
    clouds = [_frag_cloud(frames, f, _inv(A), cfg, conf_min, max_depth) for f, A in zip(frags, anchors)]
    centroids = [(A[:3, :3] @ c[0].mean(0) + A[:3, 3]) if len(c[0]) else A[:3, 3] for c, A in zip(clouds, anchors)]

    edges = []  # (i, j, Z_ij, sigma_t, sigma_yaw, kind)
    loop_err_before = []
    n = len(frags)
    for i in range(n):
        for j in range(i + 1, n):
            seq = j == i + 1
            if not seq and np.linalg.norm(centroids[i] - centroids[j]) > cfg.loop_radius:
                continue
            Pi, Ni = clouds[i]
            Pj, _ = clouds[j]
            if len(Pi) < 100 or len(Pj) < 100:
                if seq:
                    edges.append((i, j, _inv(anchors[i]) @ anchors[j], cfg.sigma_odo_t, cfg.sigma_odo_yaw, "odo"))
                continue
            Z0 = _inv(anchors[i]) @ anchors[j]     # odometry relative pose
            # ICP in fragment i's anchor frame; gravity-aligned axes needed for 4-DoF -> rotate to world axes
            Ri = anchors[i][:3, :3]
            src = (Pj @ Z0[:3, :3].T + Z0[:3, 3]) @ Ri.T
            dst = Pi @ Ri.T
            dstn = Ni @ Ri.T
            E, fit, rmse, mineig = icp_4dof(src, dst, dstn, cfg.icp_max_iter)
            ok = fit >= cfg.accept_fitness and rmse <= cfg.accept_rmse and mineig > 1e-4
            if ok:
                Ew = np.eye(4)
                Ew[:3, :3] = Ri.T @ E[:3, :3] @ Ri
                Ew[:3, 3] = Ri.T @ E[:3, 3]
                Z = Ew @ Z0
                edges.append((i, j, Z, cfg.sigma_icp_t, cfg.sigma_icp_yaw, "seq" if seq else "loop"))
                if not seq:
                    loop_err_before.append(float(np.linalg.norm(E[:3, 3])))
            elif seq:
                edges.append((i, j, Z0, cfg.sigma_odo_t, cfg.sigma_odo_yaw, "odo"))
    rep.n_seq_edges = sum(e[5] in ("seq", "odo") for e in edges)
    rep.n_loop_edges = sum(e[5] == "loop" for e in edges)
    rep.loop_closure_error_m = float(np.median(loop_err_before)) if loop_err_before else None
    if rep.n_loop_edges == 0 and all(e[5] == "odo" for e in edges):
        return frames, rep

    def residuals(x):
        X = np.concatenate([np.zeros(4), x]).reshape(n, 4)
        res = []
        for i, j, Z, st, sy, _ in edges:
            Ai = _C(X[i]) @ anchors[i]
            Aj = _C(X[j]) @ anchors[j]
            pred = Ai @ Z
            dt = (Aj[:3, 3] - pred[:3, 3]) / st
            dy = np.angle(np.exp(1j * (_yaw_of(Aj[:3, :3] @ pred[:3, :3].T)))) / sy
            res.extend([dt[0], dt[1], dt[2], dy])
        return np.array(res)

    sol = least_squares(residuals, np.zeros(4 * (n - 1)), loss="huber", f_scale=2.0, max_nfev=200)
    X = np.concatenate([np.zeros(4), sol.x]).reshape(n, 4)
    after = []
    for i, j, Z, *_ , kind in edges:
        if kind == "loop":
            Ai = _C(X[i]) @ anchors[i]
            Aj = _C(X[j]) @ anchors[j]
            after.append(float(np.linalg.norm((Ai @ Z)[:3, 3] - Aj[:3, 3])))
    rep.loop_closure_error_after_m = float(np.median(after)) if after else None
    out = []
    for fi, f in enumerate(frags):
        C = _C(X[fi])
        for k in f:
            fr = frames[k]
            out.append((k, Frame(fr.index, C @ fr.T_wc, fr.K, fr.depth, fr.conf, fr.load_image, fr.timestamp, fr.name)))
    out.sort(key=lambda t: t[0])
    rep.max_correction_m = float(max(np.linalg.norm(x[:3]) for x in X))
    rep.edges = [(int(i), int(j), kind) for i, j, *_, kind in edges]
    return [f for _, f in out], rep
