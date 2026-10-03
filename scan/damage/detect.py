"""Damage detection on wall and ceiling surfaces, measured in metric surface coordinates.

1. For each view, lift every pixel to 3D (depth + pose) and keep pixels that lie on a fitted room
   surface (wall plane or ceiling). Furniture, pictures and anything off-plane are excluded by
   geometry, not appearance.
2. On those pixels, find appearance anomalies against a smooth background estimated from the same
   surface (normalised convolution in CIELAB): darker / discoloured blobs and thin dark lines.
   Optional: an open-vocabulary detector (OWLv2) gates regions when its weights are present.
3. Vote anomalies into a 1 cm raster per surface across all views; a cell is damaged when most of
   the views that observed it agree. Regions = connected components, measured on the raster.
4. Classify each region by shape and colour (crack: long and thin; mould: very dark, speckled;
   water stain: darker, yellow-brown shift), all thresholds in one table below.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage

from ..geometry.core import PlanEst, RoomEst
from ..types import Estimate, Observations

RES = 0.01   # surface raster (m)
TH = {
    "dL": -7.0,            # L* drop vs local background to count as anomalous
    "dE": 9.0,             # colour distance vs background
    "vote": 0.5,           # share of observing views that must flag a cell
    "min_area": 0.0025,    # m^2 for blobs (5 x 5 cm)
    "crack_elong": 6.0,    # length / width
    "crack_width": 0.03,   # m (1 cm raster: a 4 mm crack rasterises to 1-3 cells)
    "crack_min_len": 0.10,
    "mould_dL": -38.0,
    "stain_db": 4.0,       # b* (yellow) shift for water stains
}


@dataclass
class Surface:
    id: str
    kind: str           # wall | ceiling
    room: RoomEst
    wall_idx: int | None
    origin: np.ndarray  # wall: start point (plan xy); ceiling: (minx, miny)
    u_dir: np.ndarray   # plan xy unit vector along wall (ceiling: x)
    size: tuple         # (U, V) metres
    votes: np.ndarray = field(default=None)
    seen: np.ndarray = field(default=None)
    dL: np.ndarray = field(default=None)
    db: np.ndarray = field(default=None)


@dataclass
class DamageRegion:
    id: str
    cls: str
    surface: Surface
    area: Estimate
    width: Estimate
    height: Estimate
    bbox: tuple          # (u0, v0, u1, v1) on the surface
    centroid_height: float
    score: float
    n_views: int
    length: float = 0.0


def _surfaces(room: RoomEst) -> list[Surface]:
    out = []
    H = room.height.value
    for k, w in enumerate(room.walls):
        d = w.end - w.start
        L = float(np.linalg.norm(d))
        if L < 0.3:
            continue
        out.append(Surface(w.id, "wall", room, k, w.start.copy(), d / L, (L, H)))
    mn, mx = room.polygon.min(0), room.polygon.max(0)
    out.append(Surface(f"{room.id}C", "ceiling", room, None, mn.copy(), np.array([1.0, 0]), tuple(mx - mn)))
    for s in out:
        shape = (int(np.ceil(s.size[0] / RES)) + 1, int(np.ceil(s.size[1] / RES)) + 1)
        s.votes, s.seen = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
        s.dL, s.db = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    return out


def _lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32) * np.array([100 / 255, 1, 1], np.float32) - \
        np.array([0, 128, 128], np.float32)


def _background(ch: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    m = mask.astype(np.float32)
    num = cv2.GaussianBlur(ch * m, (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)
    return num / np.maximum(den, 1e-3)


def detect_damage(obs: Observations, plan: PlanEst, max_views: int = 80, work_width: int = 640,
                  gate=None) -> list[DamageRegion]:
    surfaces = [s for r in plan.rooms for s in _surfaces(r)]
    if not surfaces or not obs.frames:
        return []
    R = plan.R_plan
    sig = max(0.01, float(np.median(obs.sigma)) if len(obs.sigma) else 0.01)
    tol = max(0.03, 2.5 * sig)
    frames = [f for f in obs.frames if f.load_image is not None and f.depth is not None]
    if len(frames) > max_views:
        frames = [frames[i] for i in np.linspace(0, len(frames) - 1, max_views).astype(int)]
    for f in frames:
        img = f.load_image()
        if img is None:
            continue
        s = work_width / img.shape[1]
        if s < 1:
            img = cv2.resize(img, (work_width, int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        dh, dw = f.depth.shape
        depth = cv2.resize(f.depth, (w, h), interpolation=cv2.INTER_LINEAR)
        dmin = cv2.erode(cv2.resize(f.depth, (w, h), interpolation=cv2.INTER_NEAREST), np.ones((3, 3), np.uint8))
        depth[dmin <= 0] = 0
        K = f.K.copy()
        K[0] *= w / dw
        K[1] *= h / dh
        v, u = np.mgrid[0:h, 0:w]
        z = depth
        valid = (z > 0.2) & (z < 5.0)
        # depth discontinuities: no surface claims there (edges of furniture, casings, jambs)
        gz = np.maximum(np.abs(cv2.Sobel(z, cv2.CV_32F, 1, 0)), np.abs(cv2.Sobel(z, cv2.CV_32F, 0, 1)))
        valid &= gz < 0.15 * np.maximum(z, 0.3)
        X = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0] * z, (v + 0.5 - K[1, 2]) / K[1, 1] * z, z], -1)
        Pw = X @ f.T_wc[:3, :3].T + f.T_wc[:3, 3]
        Pp = Pw @ R.T
        Pp[..., 2] -= plan.z0
        # per-pixel normals: pixels on a perpendicular face touching the wall (a wardrobe side, a room
        # corner) are within the distance tolerance but not on the wall
        Ps = np.stack([cv2.GaussianBlur(Pp[..., i].astype(np.float32), (0, 0), 3) for i in range(3)], -1)
        gx = np.stack([cv2.Sobel(Ps[..., i], cv2.CV_32F, 1, 0, ksize=5) for i in range(3)], -1)
        gy = np.stack([cv2.Sobel(Ps[..., i], cv2.CV_32F, 0, 1, ksize=5) for i in range(3)], -1)
        nrm = np.cross(gx, gy)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-9)
        lab = _lab(img)
        sidx = np.full((h, w), -1, np.int32)
        su = np.zeros((h, w), np.float32)
        sv = np.zeros((h, w), np.float32)
        for k, srf in enumerate(surfaces):
            room = srf.room
            if srf.kind == "wall":
                wl = room.walls[srf.wall_idx]
                a = wl.axis
                c = wl.start[1] if a == 1 else wl.start[0]
                rel = Pp[..., :2] - srf.origin
                uu = rel @ srf.u_dir
                on = valid & (np.abs(Pp[..., a] - c) < tol) & (uu > 0.03) & (uu < srf.size[0] - 0.03) & \
                    (Pp[..., 2] > 0.03) & (Pp[..., 2] < room.height.value - 0.03) & (np.abs(nrm[..., a]) > 0.85)
                vv = Pp[..., 2]
            else:
                on = valid & (np.abs(Pp[..., 2] - room.height.value) < tol) & (np.abs(nrm[..., 2]) > 0.85)
                uu = Pp[..., 0] - srf.origin[0]
                vv = Pp[..., 1] - srf.origin[1]
                on &= (uu > 0.05) & (vv > 0.05) & (uu < srf.size[0] - 0.05) & (vv < srf.size[1] - 0.05)
            take = on & (sidx < 0)
            sidx[take], su[take], sv[take] = k, uu[take], vv[take]
        mask = sidx >= 0
        if mask.sum() < 500:
            continue
        sigma = 0.04 * w
        dL = lab[..., 0] - _background(lab[..., 0], mask, sigma)
        da = lab[..., 1] - _background(lab[..., 1], mask, sigma)
        db = lab[..., 2] - _background(lab[..., 2], mask, sigma)
        dE = np.sqrt(dL ** 2 + da ** 2 + db ** 2)
        core = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        anom = core & (dL < TH["dL"]) & (dE > TH["dE"])
        if gate is not None:
            anom &= gate(img)
        for k in np.unique(sidx[core]):
            srf = surfaces[k]
            sel = core & (sidx == k)
            iu = np.clip((su[sel] / RES).astype(int), 0, srf.seen.shape[0] - 1)
            iv = np.clip((sv[sel] / RES).astype(int), 0, srf.seen.shape[1] - 1)
            # count each cell once per view
            cell = iu * srf.seen.shape[1] + iv
            uc, first = np.unique(cell, return_index=True)
            srf.seen.ravel()[uc] += 1
            a_sel = anom[sel]
            if a_sel.any():
                ca = cell[a_sel]
                ua, fa = np.unique(ca, return_index=True)
                srf.votes.ravel()[ua] += 1
                np.add.at(srf.dL.ravel(), ua, dL[sel][a_sel][fa])
                np.add.at(srf.db.ravel(), ua, db[sel][a_sel][fa])
    return _regions(surfaces, obs.tier)


def _regions(surfaces: list[Surface], tier: str) -> list[DamageRegion]:
    out: list[DamageRegion] = []
    min_views = 1 if tier == "photo" else 2
    for srf in surfaces:
        seen = srf.seen
        if seen.max() == 0:
            continue
        frac = np.where(seen >= min_views, srf.votes / np.maximum(seen, 1), 0)
        if srf.kind == "wall":   # opening edges (reveals, frames, glass) are not damage: 6 cm margin
            wl = srf.room.walls[srf.wall_idx]
            b = 1 - wl.axis
            sgn = np.sign(wl.end[b] - wl.start[b]) or 1.0
            for o in srf.room.openings:
                if srf.room.walls[o.edge] is not wl:
                    continue
                u_a = (o.t0.value - wl.start[b]) * sgn
                u_b = (o.t1.value - wl.start[b]) * sgn
                z0 = o.sill.value if o.sill is not None else 0.0
                z1 = z0 + o.height.value
                i0, i1 = int((min(u_a, u_b) - 0.06) / RES), int((max(u_a, u_b) + 0.06) / RES) + 1
                j0, j1 = int((z0 - 0.06) / RES), int((z1 + 0.06) / RES) + 1
                frac[max(i0, 0):max(i1, 0), max(j0, 0):max(j1, 0)] = 0
        m = frac >= TH["vote"]
        m = ndimage.binary_closing(m, iterations=1)
        lab, n = ndimage.label(m, structure=np.ones((3, 3)))
        for comp in range(1, n + 1):
            iu, iv = np.nonzero(lab == comp)
            area = len(iu) * RES * RES
            u0, u1 = iu.min() * RES, (iu.max() + 1) * RES
            v0, v1 = iv.min() * RES, (iv.max() + 1) * RES
            comp_mask = lab[iu.min():iu.max() + 1, iv.min():iv.max() + 1] == comp
            thick = 2 * float(ndimage.distance_transform_edt(comp_mask).max()) * RES
            diag = float(np.hypot(u1 - u0, v1 - v0))
            votes = srf.votes[iu, iv]
            dLm = float(np.sum(srf.dL[iu, iv]) / max(votes.sum(), 1))
            dbm = float(np.sum(srf.db[iu, iv]) / max(votes.sum(), 1))
            fill = len(iu) / ((np.ptp(iu) + 1) * (np.ptp(iv) + 1))
            cls = None
            length = diag
            if thick <= TH["crack_width"] and diag >= TH["crack_min_len"] and fill < 0.35:
                cls = "crack"
            elif area >= TH["min_area"]:
                if dLm <= TH["mould_dL"]:
                    cls = "mould"
                elif dbm >= TH["stain_db"] or dLm < TH["dL"]:
                    cls = "water_stain"
            if cls is None:
                continue
            perim = 2 * ((u1 - u0) + (v1 - v0))
            area_se = 0.5 * perim * RES + 0.05 * area
            score = float(np.clip(frac[iu, iv].mean(), 0, 1))
            h_c = float((v0 + v1) / 2) if srf.kind == "wall" else float(srf.room.height.value)
            out.append(DamageRegion(id="", cls=cls, surface=srf, area=Estimate(area, area_se),
                                    width=Estimate(u1 - u0, RES), height=Estimate(v1 - v0, RES),
                                    bbox=(u0, v0, u1, v1), centroid_height=h_c, score=score,
                                    n_views=int(seen[iu, iv].max()), length=float(length)))
    for i, d in enumerate(out):
        d.id = f"D{i + 1}"
    return out
