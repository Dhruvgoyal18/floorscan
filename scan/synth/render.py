"""Vectorised ray casting of a box scene: depth (z-buffer), material, and a simple shaded RGB.

Mirror: one specular bounce (LiDAR measures the folded path -> phantom points behind the wall, as
real dToF sensors do). Glass: transparent to depth (LiDAR mostly passes through), tints RGB.
"""
from __future__ import annotations

import numpy as np

from .scene import (CASING, CEIL, DAMAGE_RGB, FLOOR, FURN, GLASS, MIRROR, OUTDOOR, SIDES, WALL, Scene,
                    damage_mask)

_LIGHT = np.array([0.3, 0.5, 0.8]) / np.linalg.norm([0.3, 0.5, 0.8])


def _slab(o, d, lo, hi):
    """o,d: (n,3); lo,hi: (B,3) -> t (n,B) entry distance (inf if miss), axis (n,B) of entry face."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1 = (lo[None, :, :] - o[:, None, :]) * inv[:, None, :]
        t2 = (hi[None, :, :] - o[:, None, :]) * inv[:, None, :]
    tmin3 = np.fmin(t1, t2)
    tmax3 = np.fmax(t1, t2)
    tmin = np.max(tmin3, axis=2)
    tmax = np.min(tmax3, axis=2)
    hit = (tmax >= tmin) & (tmax > 1e-4) & (tmin > 1e-4)
    t = np.where(hit, tmin, np.inf)
    axis = np.argmax(tmin3, axis=2)
    return t, axis


try:  # numba makes synthetic rendering ~50x faster; optional (dev extra)
    import numba as _nb

    @_nb.njit(parallel=True, cache=True, fastmath=False)
    def _cast_nb(o, d, lo, hi):
        n, B = o.shape[0], lo.shape[0]
        t_out = np.full(n, np.inf)
        j_out = np.full(n, -1, dtype=np.int64)
        a_out = np.zeros(n, dtype=np.int64)
        for i in _nb.prange(n):
            best, bj, ba = np.inf, -1, 0
            for j in range(B):
                tmin, tmax, ax = -np.inf, np.inf, 0
                ok = True
                for k in range(3):
                    if d[i, k] != 0.0:
                        t1 = (lo[j, k] - o[i, k]) / d[i, k]
                        t2 = (hi[j, k] - o[i, k]) / d[i, k]
                        if t1 > t2:
                            t1, t2 = t2, t1
                        if t1 > tmin:
                            tmin, ax = t1, k
                        if t2 < tmax:
                            tmax = t2
                    elif o[i, k] < lo[j, k] or o[i, k] > hi[j, k]:
                        ok = False
                if ok and tmax >= tmin and tmin > 1e-4 and tmin < best:
                    best, bj, ba = tmin, j, ax
            t_out[i], j_out[i], a_out[i] = best, bj, ba
        return t_out, j_out, a_out
except Exception:  # pragma: no cover
    _cast_nb = None


def raycast(scene: Scene, o: np.ndarray, d: np.ndarray, exclude=(GLASS,), chunk: int = 4096):
    """Rays in scene-local coords. Returns t (n,), box (n,), normal (n,3). t in units of |d|."""
    keep = ~np.isin(scene.mat, exclude)
    idx = np.nonzero(keep)[0]
    lo, hi = scene.lo[idx], scene.hi[idx]
    n = len(o)
    if _cast_nb is not None:
        t_out, j, a = _cast_nb(np.ascontiguousarray(o, np.float64), np.ascontiguousarray(d, np.float64),
                               np.ascontiguousarray(lo, np.float64), np.ascontiguousarray(hi, np.float64))
        b_out = np.where(j >= 0, idx[np.maximum(j, 0)], -1)
        nrm = np.zeros((n, 3))
        nrm[np.arange(n), a] = -np.sign(d[np.arange(n), a])
        return t_out, b_out, nrm
    t_out = np.full(n, np.inf)
    b_out = np.full(n, -1, dtype=np.int64)
    nrm = np.zeros((n, 3))
    for s in range(0, n, chunk):
        oo, dd = o[s:s + chunk], d[s:s + chunk]
        t, ax = _slab(oo, dd, lo, hi)
        j = np.argmin(t, axis=1)
        tt = t[np.arange(len(oo)), j]
        a = ax[np.arange(len(oo)), j]
        t_out[s:s + chunk] = tt
        b_out[s:s + chunk] = np.where(np.isfinite(tt), idx[j], -1)
        nn = np.zeros((len(oo), 3))
        nn[np.arange(len(oo)), a] = -np.sign(dd[np.arange(len(oo)), a])
        nrm[s:s + chunk] = nn
    return t_out, b_out, nrm


def _glass_hit(scene: Scene, o, d, t_max):
    gi = np.nonzero(scene.mat == GLASS)[0]
    if len(gi) == 0:
        return np.zeros(len(o), bool)
    others = tuple(m for m in range(8) if m != GLASS)
    tg, _, _ = raycast(scene, o, d, exclude=others)
    return tg < t_max


def render(scene: Scene, T_wc: np.ndarray, K: np.ndarray, w: int, h: int, rgb: bool = True,
           rng: np.random.Generator | None = None):
    """Render a view. T_wc is in *world* coords; the scene's world placement is spec.T_world_local.

    Returns dict(depth (h,w) z-depth metres [inf = no hit], mat (h,w), rgb (h,w,3) float or None,
    phantom (h,w) bool = depth came via a mirror).
    """
    T_lw = np.linalg.inv(scene.spec.T_world_local)
    T_lc = T_lw @ T_wc
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    pix = np.stack([u.ravel(), v.ravel(), np.ones(u.size)], 1)
    d_cam = pix @ np.linalg.inv(K).T            # z = 1 => t is z-depth
    d = d_cam @ T_lc[:3, :3].T
    o = np.broadcast_to(T_lc[:3, 3], d.shape).copy()

    t, b, nrm = raycast(scene, o, d)
    mat = np.where(b >= 0, scene.mat[np.maximum(b, 0)], -1)
    depth = t.copy()
    phantom = np.zeros(len(t), bool)

    # mirror bounce
    mi = np.nonzero(mat == MIRROR)[0]
    col_b = b.copy()
    col_n = nrm.copy()
    hit_p = o + d * np.where(np.isfinite(t), t, 0)[:, None]
    if len(mi):
        p = hit_p[mi]
        dr = d[mi] - 2 * np.sum(d[mi] * nrm[mi], 1, keepdims=True) * nrm[mi]
        t2, b2, n2 = raycast(scene, p + dr * 1e-4, dr, exclude=(GLASS, MIRROR))
        depth[mi] = t[mi] + t2
        phantom[mi] = True
        col_b[mi] = b2
        col_n[mi] = n2
        hit_p[mi] = p + dr * np.where(np.isfinite(t2), t2, 0)[:, None]

    out = {"depth": depth.reshape(h, w), "mat": mat.reshape(h, w), "phantom": phantom.reshape(h, w), "rgb": None}
    if not rgb:
        return out

    base = np.where(col_b[:, None] >= 0, scene.color[np.maximum(col_b, 0)], 0.0)
    bmat = np.where(col_b >= 0, scene.mat[np.maximum(col_b, 0)], -1)
    shade = 0.6 + 0.4 * np.clip(col_n @ _LIGHT, 0, 1)
    shade = np.where(bmat == OUTDOOR, 1.0, shade)
    colr = base * shade[:, None]

    # floor planks
    fl = bmat == FLOOR
    if fl.any():
        stripe = 0.93 + 0.07 * (np.floor(hit_p[fl, 0] / 0.18) % 2)
        colr[fl] *= stripe[:, None]

    # damage on wall faces
    spec = scene.spec
    for dmg in spec.damage:
        r = spec.room(dmg.room)
        if dmg.surface == "C":
            sel = (bmat == CEIL) & (hit_p[:, 0] > r.x0) & (hit_p[:, 0] < r.x1) & (hit_p[:, 1] > r.y0) & (hit_p[:, 1] < r.y1)
            if not sel.any():
                continue
            uu, zz = hit_p[sel, 0] - r.x0, hit_p[sel, 1] - r.y0
        else:
            a, bb = r.wall_endpoints(dmg.surface)
            dd = (bb - a) / np.linalg.norm(bb - a)
            n_in = np.array([-dd[1], dd[0]])
            off = (hit_p[:, :2] - a) @ n_in
            sel = (bmat == WALL) & (np.abs(off) < 2e-3) & (np.abs(col_n[:, :2] @ n_in - 1) < 1e-6)
            if not sel.any():
                continue
            uu = (hit_p[sel, :2] - a) @ dd
            zz = hit_p[sel, 2]
        m = damage_mask(dmg, uu, zz)
        idx = np.nonzero(sel)[0][m]
        alpha = 0.75 if dmg.cls == "water_stain" else 0.9
        colr[idx] = (1 - alpha) * colr[idx] + alpha * np.array(DAMAGE_RGB[dmg.cls]) * shade[idx, None]

    # glass tint
    g = _glass_hit(scene, o, d, np.where(np.isfinite(t), t, 1e9))
    colr[g] = 0.8 * colr[g] + 0.2 * np.array([0.6, 0.75, 0.8])

    # per-room lighting: the room the camera stands in sets exposure
    light = 1.0
    cam = T_lc[:3, 3]
    for r in spec.rooms:
        if r.x0 <= cam[0] <= r.x1 and r.y0 <= cam[1] <= r.y1:
            light = r.light
    colr = colr * light
    rng = rng or np.random.default_rng(0)
    noise = 0.01 + 0.03 * (1 - light)
    colr = colr + rng.normal(0, noise, colr.shape)
    colr[col_b < 0] = 0.7  # sky
    out["rgb"] = np.clip(colr, 0, 1).reshape(h, w, 3)
    _ = (FURN, CASING, SIDES)
    return out
