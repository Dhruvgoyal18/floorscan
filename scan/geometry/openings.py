"""Opening detection (doors, windows) and measurement.

Detection uses *free-space* evidence, not holes: a camera->point ray that crosses the wall plane
inside the wall's extent proves the wall is open there. Unobserved wall (behind furniture, never
looked at) is therefore never mistaken for an opening. Rays that returned nothing (beyond sensor
range, glass) also count, so exterior windows are found.

Measurement: the jamb is the reveal surface perpendicular to the wall; its position along the wall
is estimated from reveal points (behind the wall face) at mid-height.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..types import Estimate
from .core import Line, OpeningEst, RoomEst, WallEst, _se

CELL = 0.02   # default grid cell on the wall plane; adapted to point spacing per capture (see _cell)


def _cell(ctx) -> float:
    return float(np.clip(1.5 * ctx.get("spacing", 0.015), 0.02, 0.08))
# Any point observed *behind* the wall face by a camera in front of it proves the ray crossed the
# plane at an open spot (it may land on the far room, the threshold floor, or the jamb reveal itself).
# Only noise can fake it, so the threshold is a few sigma.
REVEAL_MAX = 0.30


def _wall_frame(lines: list[Line], w: WallEst):
    l = lines[w.line]
    c = w.start[1] if w.axis == 1 else w.start[0]
    return l, c, l.axis, 1 - l.axis, l.facing


def _see_through(ctx, w: WallEst, l: Line, c: float, height: float):
    P, F, cams, far, far_f = ctx["P"], ctx["F"], ctx["cams"], ctx["far"], ctx["far_f"]
    a, b, f = l.axis, 1 - l.axis, l.facing
    t0, t1 = w.t_range
    CELL = _cell(ctx)
    nt, nz = int(np.ceil((t1 - t0) / CELL)), int(np.ceil(height / CELL))
    see = np.zeros((nt, nz), np.int32)
    for Q, fr, is_far in ((P, F, False), (far, far_f, True)):
        if len(Q) == 0:
            continue
        depth = (c - Q[:, a]) * f
        sel = depth > max(0.04, 3 * ctx["sigma_med"])
        if not sel.any():
            continue
        q = Q[sel]
        o = cams[fr[sel]]
        oin = (o[:, a] - c) * f > 0.05
        q, o = q[oin], o[oin]
        lam = (c - o[:, a]) / (q[:, a] - o[:, a])
        hit = o + lam[:, None] * (q - o)
        ti = ((hit[:, b] - t0) / CELL).astype(int)
        zi = (hit[:, 2] / CELL).astype(int)
        ok = (ti >= 1) & (ti < nt - 1) & (zi >= 0) & (zi < nz)
        np.add.at(see, (ti[ok], zi[ok]), 1)
    return see


def _occupancy(ctx, w, l, c, height, sig):
    P = ctx["P"]
    a, b, f = l.axis, 1 - l.axis, l.facing
    t0, t1 = w.t_range
    CELL = _cell(ctx)
    nt, nz = int(np.ceil((t1 - t0) / CELL)), int(np.ceil(height / CELL))
    depth = (c - P[:, a]) * f
    win = max(0.03, 2.5 * sig)
    sel = (np.abs(depth) < win) & (P[:, b] > t0) & (P[:, b] < t1) & (P[:, 2] >= 0) & (P[:, 2] < height)
    occ = np.zeros((nt, nz), np.int32)
    ti = ((P[sel, b] - t0) / CELL).astype(int).clip(0, nt - 1)
    zi = (P[sel, 2] / CELL).astype(int).clip(0, nz - 1)
    np.add.at(occ, (ti, zi), 1)
    return occ


def _jamb(ctx, l, c, t_guess, side, z_lo, z_hi, sig):
    """Jamb coordinate (Estimate) near t_guess. side=-1: left jamb (wall at smaller t), +1: right."""
    P = ctx["P"]
    a, b, f = l.axis, 1 - l.axis, l.facing
    depth = (c - P[:, a]) * f
    dmin = max(0.02, 2.5 * sig)
    s = ctx["cfg"].jamb_search
    lo, hi = (t_guess - s, t_guess + 0.10) if side < 0 else (t_guess - 0.10, t_guess + s)
    zsel = (P[:, 2] > z_lo) & (P[:, 2] < z_hi) & (P[:, b] > lo) & (P[:, b] < hi)
    # the reveal must be seen from *this* side; the wall's far face (seen from the neighbouring room)
    # lies at the same depth range and would otherwise drag the jamb outward
    front = (ctx["cams"][ctx["F"], a] - c) * f > 0.05
    rev = zsel & (depth > dmin) & (depth < REVEAL_MAX) & front
    if rev.sum() >= 15:
        t = P[rev, b]
        m = float(np.median(t))
        mad = float(np.median(np.abs(t - m))) * 1.4826
        n_eff = max(1, len(np.unique(np.floor(P[rev, 2] / 0.1))))
        return Estimate(m, _se(mad, n_eff, sig))
    face = zsel & (np.abs(depth) < dmin)
    if face.sum() >= 15:
        t = P[face, b]
        m = float(np.percentile(t, 98) if side < 0 else np.percentile(t, 2))
        return Estimate(m, max(0.01, 2 * sig))
    return Estimate(float(t_guess), 0.03, observed=False)


def _level(ctx, l, c, t_l, t_r, z_guess, sig):
    P = ctx["P"]
    a, b, f = l.axis, 1 - l.axis, l.facing
    depth = (c - P[:, a]) * f
    dmin = max(0.02, 2.5 * sig)
    front = (ctx["cams"][ctx["F"], a] - c) * f > 0.05
    sel = ((depth > dmin) & (depth < REVEAL_MAX) & (P[:, b] > t_l + 0.05) & (P[:, b] < t_r - 0.05) &
           (np.abs(P[:, 2] - z_guess) < 0.12) & front)
    if sel.sum() >= 15:
        z = P[sel, 2]
        m = float(np.median(z))
        mad = float(np.median(np.abs(z - m))) * 1.4826
        return Estimate(m, _se(mad, max(1, int((t_r - t_l) / 0.1)), sig))
    return Estimate(float(z_guess), 0.03, observed=False)


def detect_openings(ctx, lines, room: RoomEst, warnings: list[str]) -> list[OpeningEst]:
    out: list[OpeningEst] = []
    sig = ctx["sigma_med"]
    CELL = _cell(ctx)
    vclose = max(3, int(round(0.22 / CELL)))
    height = room.ceil_z.value
    for k, w in enumerate(room.walls):
        if w.line is None or w.length.value < 0.4:
            continue
        l, c, a, b, f = _wall_frame(lines, w)
        see = _see_through(ctx, w, l, c, height)
        occ = _occupancy(ctx, w, l, c, height, sig)
        open_ = (see >= 1) & (occ <= 1)
        # bridge thin horizontal gaps (rays at camera height go far and are sparse) before labelling
        open_ = ndimage.binary_closing(open_, structure=np.ones((3, vclose)), iterations=1)
        open_ = ndimage.binary_closing(open_, structure=np.ones((3, 3)), iterations=2)
        open_ = ndimage.binary_opening(open_, structure=np.ones((3, 3)), iterations=1)
        lab, n = ndimage.label(open_)
        t0 = w.t_range[0]
        for comp in range(1, n + 1):
            ti, zi = np.nonzero(lab == comp)
            ta, tb = t0 + ti.min() * CELL, t0 + (ti.max() + 1) * CELL
            za, zb = zi.min() * CELL, (zi.max() + 1) * CELL
            fill = len(ti) / max(1, (np.ptp(ti) + 1) * (np.ptp(zi) + 1))
            width_g = tb - ta
            if see[lab == comp].sum() < 30:
                continue
            # Below the see-through region: a window has wall (occupied face) under its sill; a door has
            # none -- the bottom of a doorway is often simply not looked at (or hidden by furniture).
            below = occ[ti.min():ti.max() + 1, :max(1, int(za / CELL))]
            sill_wall = (below > 1).any(axis=0).mean() if below.size else 0.0
            if 0.15 < za < 0.7 and sill_wall < 0.3 and zb >= 1.5 and width_g >= 0.45:
                za = 0.0
            if za <= 0.15 and zb >= 1.5 and width_g >= 0.45:
                kind = "door" if zb < height - 0.1 else "opening"
                z_lo, z_hi = 0.3, min(1.8, zb - 0.15)
            elif za > 0.15 and zb - za >= 0.25 and width_g >= 0.25 and zb <= height - 0.08:
                kind = "window"
                z_lo, z_hi = za + 0.08, zb - 0.08
            else:
                continue
            if fill < 0.45:
                continue
            mirror = False
            for li, mt0, mt1, mz0, mz1 in ctx.get("mirrors", []):
                if li == w.line:
                    ov_t = max(0.0, min(tb, mt1) - max(ta, mt0))
                    ov_z = max(0.0, min(zb, mz1) - max(za, mz0))
                    if ov_t * ov_z > 0.5 * (tb - ta) * (zb - za):
                        mirror = True
            if mirror:
                continue
            tl = _jamb(ctx, l, c, ta, -1, z_lo, z_hi, sig)
            tr = _jamb(ctx, l, c, tb, +1, z_lo, z_hi, sig)
            if tr.value - tl.value < 0.2:
                continue
            width = tr - tl
            head = _level(ctx, l, c, tl.value, tr.value, zb, sig) if kind != "opening" else Estimate(height, 0.02)
            hgt = head - room.floor_z
            sill = None
            if kind == "window":
                s_ = _level(ctx, l, c, tl.value, tr.value, za, sig)
                sill = s_ - room.floor_z
                hgt = head - s_
                if hgt.value < 0.2:
                    continue
            tm = (tl.value + tr.value) / 2
            center = np.array([c, tm]) if a == 0 else np.array([tm, c])
            out.append(OpeningEst(kind=kind, edge=k, t0=tl, t1=tr, width=width, height=hgt, sill=sill,
                                  center=center, confidence=float(np.clip(fill, 0, 1)),
                                  id=f"{room.id}O{len(out) + 1}"))
    return out


def remeasure_opening(ctx, lines, room: RoomEst, o: OpeningEst):
    w = room.walls[o.edge]
    if w.line is None:
        return None
    l, c, a, b, f = _wall_frame(lines, w)
    sig = ctx["sigma_med"]
    if o.kind == "window" and o.sill is not None:
        z_lo, z_hi = o.sill.value + 0.08, o.sill.value + o.height.value - 0.08
    else:
        z_lo, z_hi = 0.3, min(1.8, o.height.value - 0.15)
    tl = _jamb(ctx, l, c, o.t0.value, -1, z_lo, z_hi, sig)
    tr = _jamb(ctx, l, c, o.t1.value, +1, z_lo, z_hi, sig)
    return tr.value - tl.value
