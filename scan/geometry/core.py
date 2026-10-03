"""Shared geometry core: Observations -> rooms with walls, openings, ceiling heights, areas.

Pipeline (one call: :func:`extract_plan`):
  1. working cloud (voxel means) + PCA normals oriented towards the observing cameras
  2. floor / ceiling levels; Manhattan yaw; rotate into the *plan frame* (walls axis-aligned, floor z=0)
  3. wall lines: per axis and facing, peaks of the wall-point coordinate histogram
  4. rooms: cells of the wall-line arrangement; interior evidence = horizontal-surface coverage;
     cells merge across edges that lack wall evidence in the header band (above door height)
  5. per room: polygon -> walls (edges bound to lines), openings via see-through evidence,
     measurements with standard errors (structure is fixed; measurement can be bootstrapped)

Everything here is tier-agnostic. Tier differences enter only via point sigma and the calibration layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from ..types import Estimate, Observations
from .cloud import estimate_normals, voxel_reduce
from .frame import manhattan_yaw, rot_z


@dataclass
class CoreConfig:
    voxel: float = 0.015
    header_height: float = 2.12          # standard doors are <= 2.1 m; wall above this separates rooms
    min_room_area: float = 1.0
    min_room_side: float = 0.6
    raster: float = 0.025
    cover_thresh: float = 0.25
    min_wall_len: float = 0.25
    jamb_search: float = 0.15
    unobserved_se: float = 0.25          # se for a wall/edge never observed
    bootstrap: int = 0                   # frame-bootstrap resamples for the resampling term (0 = off)
    single_room: bool = False            # photo tier: one folder = one room (all cameras stand in it)
    seed: int = 0


@dataclass
class Line:
    axis: int          # 0: plane x = c ; 1: plane y = c   (plan frame)
    facing: int        # +1: normal points to +axis, i.e. the room it bounds lies on the + side
    c: float
    se: float
    n: int
    zspan: float
    header: np.ndarray  # bool coverage of the header band along the other axis (global grid)
    occupancy: np.ndarray  # bool coverage at any height along the other axis


@dataclass
class OpeningEst:
    kind: str
    edge: int
    t0: Estimate          # jamb coordinates along the line's other axis (plan frame)
    t1: Estimate
    width: Estimate
    height: Estimate
    sill: Estimate | None
    center: np.ndarray    # plan xy
    confidence: float
    connects_to: str | None = None
    id: str = ""


@dataclass
class WallEst:
    id: str
    start: np.ndarray
    end: np.ndarray
    length: Estimate
    axis: int
    facing: int
    line: int | None      # index into lines; None => unobserved
    t_range: tuple


@dataclass
class RoomEst:
    id: str
    cells: list
    polygon: np.ndarray
    walls: list[WallEst]
    floor_z: Estimate
    ceil_z: Estimate
    height: Estimate
    area: Estimate
    perimeter: Estimate
    openings: list[OpeningEst] = field(default_factory=list)
    n_cams: int = 0


@dataclass
class PlanEst:
    rooms: list[RoomEst]
    lines: list[Line]
    R_plan: np.ndarray         # 3x3, world -> plan rotation
    z0: float                  # world floor level subtracted
    adjacency: list[dict]
    warnings: list[str]
    wall_thickness: Estimate | None = None
    manhattan_share: float = 1.0


# ------------------------------------------------------------------------------------------ helpers

def _robust_level(z: np.ndarray, sigma_med: float, window: float = 0.02):
    """Mode of a 1 cm histogram, then mean of points within +-window; returns (value, std, n)."""
    if len(z) < 20:
        return np.nan, np.nan, len(z)
    lo, hi = z.min(), z.max()
    bins = np.arange(lo - 0.01, hi + 0.02, 0.005)
    h, e = np.histogram(z, bins=bins)
    hs = ndimage.gaussian_filter1d(h.astype(float), max(1.0, sigma_med / 0.005))
    m = (e[np.argmax(hs)] + e[np.argmax(hs) + 1]) / 2
    w = max(window, 2.5 * sigma_med)
    for _ in range(3):
        sel = np.abs(z - m) < w
        if sel.sum() < 5:
            break
        m = z[sel].mean()
    sel = np.abs(z - m) < w
    return float(m), float(z[sel].std()), int(sel.sum())


def surface_noise(P: np.ndarray, planarity: np.ndarray, n: int = 20000, k: int = 16) -> float:
    """Median RMS distance of points to their local tangent plane, over planar neighbourhoods."""
    sel = np.nonzero(planarity > 0.8)[0]
    if len(sel) < 100:
        return 0.01
    sel = sel[np.linspace(0, len(sel) - 1, min(n, len(sel))).astype(int)]
    tree = cKDTree(P)
    _, nb = tree.query(P[sel], k=k, workers=-1)
    Q = P[nb] - P[nb].mean(1, keepdims=True)
    C = np.einsum("nki,nkj->nij", Q, Q) / k
    from .cloud import _smallest_eig_sym3
    _, l3, _ = _smallest_eig_sym3(C)
    return float(np.median(np.sqrt(l3)))


def plane_spread(P: np.ndarray, N: np.ndarray) -> float:
    """Robust spread of points about the four strongest wall planes (captures cross-view misregistration,
    which local kNN scatter does not)."""
    vals = []
    for a in (0, 1):
        for f in (1, -1):
            x = P[N[:, a] * f > 0.9, a]
            if len(x) < 500:
                continue
            h, e = np.histogram(x, bins=np.arange(x.min(), x.max() + 0.01, 0.005))
            m = e[np.argmax(ndimage.gaussian_filter1d(h.astype(float), 2))]
            near = x[np.abs(x - m) < 0.25] - m
            if len(near) > 200:
                vals.append(1.4826 * float(np.median(np.abs(near - np.median(near)))))
    return float(np.median(vals)) if vals else 0.0


def _n_eff(a: np.ndarray, b: np.ndarray, cell: float = 0.1) -> int:
    """Number of distinct cell x cell patches: a conservative count of independent samples."""
    if len(a) == 0:
        return 1
    k = np.unique(np.stack([np.floor(a / cell), np.floor(b / cell)], 1), axis=0)
    return max(1, len(k))


def _se(std: float, n_eff: int, sigma_med: float) -> float:
    # white part shrinks with independent patches; correlated part (depth bias) does not
    return float(np.sqrt(std**2 / max(n_eff, 1) + (0.15 * sigma_med) ** 2))


# ------------------------------------------------------------------------------------------ main

def prepare_cloud(obs: Observations, cfg: CoreConfig):
    """Voxel-reduce, compute oriented normals. Returns dict of arrays in *world* frame."""
    P, S, F, cnt = voxel_reduce(obs.points, cfg.voxel, obs.sigma, obs.frame_idx)
    origins = obs.cam_centers[F]
    N, pl = estimate_normals(P, k=16, origins=origins)
    return {"P": P, "S": S, "F": F, "N": N, "pl": pl, "cnt": cnt}


def extract_plan(obs: Observations, cfg: CoreConfig | None = None) -> PlanEst:
    cfg = cfg or CoreConfig()
    warnings: list[str] = list(obs.warnings)
    cloud = prepare_cloud(obs, cfg)
    P, N, pl, S = cloud["P"], cloud["N"], cloud["pl"], cloud["S"]
    # Tolerance scale for "is this point on that surface": the measured local thickness of planar
    # surfaces, not the per-point absolute range error. Learned depth errors are mostly smooth
    # (scale/low-frequency), so absolute sigma (5-7 cm on video) would make every window swallow
    # neighbouring surfaces; local thickness stays at the cm level.
    sigma_abs = float(np.median(S)) if len(S) else 0.01

    # --- plan frame: Manhattan yaw, floor at z = 0
    yaw, share = manhattan_yaw(N, pl)
    if share < 0.6:
        warnings.append(f"only {share:.0%} of wall area is Manhattan-aligned; non-orthogonal walls are approximated")
    R = rot_z(-yaw)
    P = P @ R.T
    N = N @ R.T
    # thickness of real surfaces = max(local scatter, spread of the strongest walls across views)
    sigma_med = max(surface_noise(P, pl), plane_spread(P, N))
    sigma_med = float(np.clip(sigma_med, 0.003, sigma_abs))
    up = (N[:, 2] > 0.85) & (pl > 0.5)
    dn = (N[:, 2] < -0.85) & (pl > 0.5)
    floor_w, _, _ = _robust_level(P[up, 2], sigma_med) if up.sum() > 20 else (np.nan, 0, 0)
    if not np.isfinite(floor_w):
        floor_w = float(np.percentile(P[:, 2], 1))
        warnings.append("floor not observed; using lowest points")
    P[:, 2] -= floor_w
    ceil_g, _, _ = _robust_level(P[dn & (P[:, 2] > 1.6), 2], sigma_med) if (dn & (P[:, 2] > 1.6)).sum() > 20 else (np.nan, 0, 0)
    if not np.isfinite(ceil_g):
        ceil_g = float(np.percentile(P[:, 2], 99.5))
        warnings.append("ceiling not observed; ceiling heights unreliable")

    cams = obs.cam_centers @ R.T
    cams[:, 2] -= floor_w
    F = cloud["F"]
    far = obs.rays_far @ R.T if len(obs.rays_far) else np.zeros((0, 3))
    if len(far):
        far[:, 2] -= floor_w
    samp = P[np.linspace(0, len(P) - 1, min(len(P), 5000)).astype(int)]
    spacing = float(np.median(cKDTree(P).query(samp, k=2)[0][:, 1])) if len(P) > 10 else cfg.voxel
    ctx = dict(P=P, N=N, pl=pl, S=S, F=F, cams=cams, far=far, far_f=obs.rays_far_frame, sigma_med=sigma_med,
               sigma_abs=sigma_abs,
               ceil_g=ceil_g, cfg=cfg, spacing=spacing)

    lines, grids = _detect_lines(ctx)
    ctx["grids"] = grids
    rooms = _segment_rooms(ctx, lines, warnings)
    if not rooms:
        raise RuntimeError("no room recovered: check that the capture shows floor, ceiling and walls")
    for k, r in enumerate(rooms):
        r.id = f"R{k + 1}"
        for j, w in enumerate(r.walls):
            w.id = f"{r.id}W{j + 1}"
        _measure_room(ctx, lines, r)
    _filter_mirror_points(ctx, lines, rooms, warnings)
    _measure_all(ctx, lines, rooms, warnings)
    adjacency, thickness = _adjacency(rooms, lines)
    if cfg.bootstrap > 0:
        _bootstrap(ctx, lines, rooms, obs, cfg)
    warnings += ctx.get("warnings_extra", [])
    return PlanEst(rooms=rooms, lines=lines, R_plan=R, z0=floor_w, adjacency=adjacency, warnings=warnings,
                   wall_thickness=thickness, manhattan_share=share)


# ------------------------------------------------------------------------------------------ lines

def _detect_lines(ctx) -> tuple[list[Line], dict]:
    P, N, pl, S, cfg = ctx["P"], ctx["N"], ctx["pl"], ctx["S"], ctx["cfg"]
    sig = ctx["sigma_med"]
    ceil_g = ctx["ceil_g"]
    # Header band: above door height AND within the top 0.3 m of the ceiling, so tall furniture
    # (wardrobes ~2.0-2.2 m, plus scale error on thin tiers) does not split a room.
    header_z = max(min(cfg.header_height, ceil_g - 0.1), ceil_g - 0.3)
    lo_xy = P[:, :2].min(0) - 0.5
    hi_xy = P[:, :2].max(0) + 0.5
    gstep = 0.05
    grids = {b: np.arange(lo_xy[b], hi_xy[b] + gstep, gstep) for b in (0, 1)}
    zband = (P[:, 2] > 0.1) & (P[:, 2] < ceil_g - 0.08)
    lines: list[Line] = []
    win = max(0.02, 2.5 * sig)
    for a in (0, 1):
        b = 1 - a
        for facing in (1, -1):
            sel = (N[:, a] * facing > 0.9) & (pl > 0.5) & (P[:, 2] > 0.05) & (P[:, 2] < ceil_g + 0.05)
            if sel.sum() < 100:
                continue
            x = P[sel, a]
            bins = np.arange(x.min() - 0.05, x.max() + 0.06, 0.01)
            h, e = np.histogram(x, bins=bins)
            hs = ndimage.gaussian_filter1d(h.astype(float), max(1.0, sig / 0.01))
            # a wall patch of min_wall_len x 0.5 m at the working voxel size
            # low bar on the histogram (far, sparsely sampled walls must survive); real walls are then
            # told apart from clutter by support, vertical extent and length below
            peaks, _ = find_peaks(hs, height=max(5.0, 0.004 * hs.max()), distance=max(3, int(2 * win / 0.01)))
            for pk in peaks:
                c0 = (e[pk] + e[pk + 1]) / 2
                m = np.abs(x - c0) < win
                if m.sum() < 120:
                    continue
                c = x[m].mean()
                m = np.abs(x - c) < win
                pts = P[sel][m]
                zs = pts[:, 2]
                zspan = np.percentile(zs, 98) - np.percentile(zs, 2)
                if zspan < min(1.0, 0.45 * ceil_g):
                    continue  # furniture face, not a wall
                tb = pts[:, b]
                occ = np.zeros(len(grids[b]), bool)
                occ[np.clip(((tb - grids[b][0]) / gstep).astype(int), 0, len(occ) - 1)] = True
                if occ.sum() * gstep < cfg.min_wall_len:
                    continue
                hdr = np.zeros(len(grids[b]), bool)
                hp = tb[zs > header_z]
                hdr[np.clip(((hp - grids[b][0]) / gstep).astype(int), 0, len(hdr) - 1)] = True
                hdr = ndimage.binary_dilation(hdr, iterations=1)
                # bridge unobserved patches (<= 0.6 m) in the header band / wall coverage: a grazing-angle
                # view can miss the strip above a doorway; open-plan archways are wider than this
                gap = np.ones(int(round(0.6 / gstep)) + 1, bool)
                hdr = ndimage.binary_closing(hdr, structure=gap) | hdr
                occ = ndimage.binary_closing(occ, structure=gap) | occ
                std = float(pts[:, a].std())
                se = _se(std, _n_eff(tb, zs), sig)
                lines.append(Line(a, facing, float(c), se, int(m.sum()), float(zspan), hdr, occ))
    _ = zband
    return lines, grids


# ------------------------------------------------------------------------------------------ rooms

def _merge_coords(vals: list[float], tol: float = 0.02) -> np.ndarray:
    v = np.sort(np.array(vals))
    out = []
    for x in v:
        if out and x - out[-1][-1] < tol:
            out[-1].append(x)
        else:
            out.append([x])
    return np.array([np.mean(g) for g in out])


def _segment_rooms(ctx, lines: list[Line], warnings: list[str]) -> list[RoomEst]:
    P, N, cfg, cams = ctx["P"], ctx["N"], ctx["cfg"], ctx["cams"]
    grids = ctx["grids"]
    ceil_g = ctx["ceil_g"]
    xb = (grids[0][0], grids[0][-1])
    yb = (grids[1][0], grids[1][-1])
    xs = _merge_coords([l.c for l in lines if l.axis == 0] + list(xb))
    ys = _merge_coords([l.c for l in lines if l.axis == 1] + list(yb))

    # interior evidence: horizontal surfaces (floor, ceiling, furniture tops) seen from above/below
    hz = (np.abs(N[:, 2]) > 0.7) & (P[:, 2] > -0.1) & (P[:, 2] < ceil_g + 0.15)
    r = cfg.raster
    gx = np.arange(xb[0], xb[1] + r, r)
    gy = np.arange(yb[0], yb[1] + r, r)
    cov = np.zeros((len(gx), len(gy)), bool)
    ix = np.clip(((P[hz, 0] - gx[0]) / r).astype(int), 0, len(gx) - 1)
    iy = np.clip(((P[hz, 1] - gy[0]) / r).astype(int), 0, len(gy) - 1)
    cov[ix, iy] = True
    cov = ndimage.binary_closing(cov, iterations=2)

    nx, ny = len(xs) - 1, len(ys) - 1
    frac = np.zeros((nx, ny))
    for i in range(nx):
        a0, a1 = int((xs[i] - gx[0]) / r), int((xs[i + 1] - gx[0]) / r)
        for j in range(ny):
            b0, b1 = int((ys[j] - gy[0]) / r), int((ys[j + 1] - gy[0]) / r)
            blk = cov[a0:max(a1, a0 + 1), b0:max(b1, b0 + 1)]
            frac[i, j] = blk.mean() if blk.size else 0.0
    # thin cells (wall-thickness strips) pick up coverage from the dilation of neighbouring floors;
    # they must be covered for real to count as interior
    # A wall strip is bounded by the two faces of one wall, both actually present along the strip:
    # the face of the room on the low side (facing -axis) and of the room on the high side (+axis).
    def face_cover(axis, facing, coord, t0, t1):
        g = grids[1 - axis]
        k0, k1 = int((t0 - g[0]) / 0.05), int(np.ceil((t1 - g[0]) / 0.05))
        best = 0.0
        for l in lines:
            if l.axis == axis and l.facing == facing and abs(l.c - coord) < 0.025:
                seg = l.occupancy[max(k0, 0):max(k1, k0 + 1)]
                best = max(best, float(seg.mean()) if seg.size else 0.0)
        return best

    nx_, ny_ = len(xs) - 1, len(ys) - 1
    wall_strip = np.zeros((nx_, ny_), bool)
    for i in range(nx_):
        for j in range(ny_):
            if xs[i + 1] - xs[i] < 0.35 and face_cover(0, -1, xs[i], ys[j], ys[j + 1]) >= 0.5 \
                    and face_cover(0, 1, xs[i + 1], ys[j], ys[j + 1]) >= 0.5:
                wall_strip[i, j] = True
            elif ys[j + 1] - ys[j] < 0.35 and face_cover(1, -1, ys[j], xs[i], xs[i + 1]) >= 0.5 \
                    and face_cover(1, 1, ys[j + 1], xs[i], xs[i + 1]) >= 0.5:
                wall_strip[i, j] = True
    interior = np.where(wall_strip, frac >= 0.6, frac >= cfg.cover_thresh)
    ctx["debug_cells"] = (xs, ys, frac)

    def separated(axis, coord, t0, t1):
        g = grids[1 - axis]
        k0, k1 = int((t0 - g[0]) / 0.05), int(np.ceil((t1 - g[0]) / 0.05))
        best = 0.0
        for l in lines:
            if l.axis == axis and abs(l.c - coord) < 0.03:
                seg = l.header[max(k0, 0):max(k1, k0 + 1)]
                if seg.size:
                    best = max(best, seg.mean())
        return best >= 0.4

    # connected components over interior cells
    label = -np.ones((nx, ny), int)
    comps = []
    for i in range(nx):
        for j in range(ny):
            if not interior[i, j] or label[i, j] >= 0:
                continue
            stack = [(i, j)]
            label[i, j] = len(comps)
            cells = []
            while stack:
                a, b = stack.pop()
                cells.append((a, b))
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    c, d = a + da, b + db
                    if not (0 <= c < nx and 0 <= d < ny) or label[c, d] >= 0 or not interior[c, d]:
                        continue
                    if da != 0:
                        coord = xs[max(a, c)]
                        if separated(0, coord, ys[b], ys[b + 1]):
                            continue
                    else:
                        coord = ys[max(b, d)]
                        if separated(1, coord, xs[a], xs[a + 1]):
                            continue
                    label[c, d] = len(comps)
                    stack.append((c, d))
            comps.append(cells)

    from shapely import contains_xy
    n_frames = len(cams)
    polys = [unary_union([box(xs[a], ys[b], xs[a + 1], ys[b + 1]) for a, b in cells]) for cells in comps]
    ncam = [int(contains_xy(p, cams[:, 0], cams[:, 1]).sum()) for p in polys]
    keep = []
    for k, poly in enumerate(polys):
        minx, miny, maxx, maxy = poly.bounds
        if poly.area >= cfg.min_room_area and min(maxx - minx, maxy - miny) >= cfg.min_room_side \
                and ncam[k] >= max(2, 0.01 * n_frames):
            keep.append(k)
    if cfg.single_room:
        # every camera stands in the one room: any component holding a camera is a piece of it
        # (spurious lines in narrow, sparsely photographed spaces can cut it up)
        with_cam = [k for k in range(len(polys)) if ncam[k] > 0 and polys[k].area > 0.05]
        if with_cam:
            main = max(with_cam, key=lambda k: (ncam[k], polys[k].area))
            grow = True
            members = {main}
            while grow:
                grow = False
                for k in with_cam:
                    if k not in members and any(polys[k].intersection(polys[m]).length > 0.05 for m in members):
                        members.add(k)
                        grow = True
            comps[main] = [c for m in members for c in comps[m]]
            polys[main] = unary_union([polys[m] for m in members])
            ncam[main] = sum(ncam[m] for m in members)
            for m in members - {main}:
                ncam[m] = -1
            keep = [main] if polys[main].area >= cfg.min_room_area else []
    merged = {k: list(comps[k]) for k in keep}

    rooms = []
    for k in keep:
        cells = merged[k]
        poly = unary_union([box(xs[a], ys[b], xs[a + 1], ys[b + 1]) for a, b in cells])
        if poly.geom_type != "Polygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        n_in = ncam[k]
        if len(poly.interiors):
            warnings.append("room with an interior hole (column / unobserved island) - hole ignored")
            poly = Polygon(poly.exterior)
        poly = poly.simplify(1e-6)
        ring = np.array(poly.exterior.coords)[:-1]
        if Polygon(ring).exterior.is_ccw is False:
            ring = ring[::-1]
        room = RoomEst(id="", cells=cells, polygon=ring, walls=[], floor_z=Estimate(0, 0), ceil_z=Estimate(0, 0),
                       height=Estimate(0, 0), area=Estimate(0, 0), perimeter=Estimate(0, 0), n_cams=n_in)
        room.walls = _bind_walls(ring, lines)
        rooms.append(room)
    # deterministic order: by position (left to right, then bottom to top)
    rooms.sort(key=lambda r: (round(r.polygon[:, 0].min(), 1), round(r.polygon[:, 1].min(), 1)))
    return rooms


def _bind_walls(ring: np.ndarray, lines: list[Line]) -> list[WallEst]:
    walls = []
    n = len(ring)
    for k in range(n):
        p, q = ring[k], ring[(k + 1) % n]
        d = q - p
        if abs(d[0]) > abs(d[1]):   # edge along x => plane y = const, axis 1
            axis, coord = 1, p[1]
            facing = 1 if d[0] > 0 else -1     # CCW: interior on the left => +y when moving +x
            t_range = (min(p[0], q[0]), max(p[0], q[0]))
        else:
            axis, coord = 0, p[0]
            facing = -1 if d[1] > 0 else 1     # moving +y, interior on the left => -x side
            t_range = (min(p[1], q[1]), max(p[1], q[1]))
        best, bi = 0.03, None
        for i, l in enumerate(lines):
            if l.axis == axis and l.facing == facing and abs(l.c - coord) < best:
                best, bi = abs(l.c - coord), i
        walls.append(WallEst(id="", start=p.copy(), end=q.copy(), length=Estimate(float(np.linalg.norm(d)), 0.0),
                             axis=axis, facing=facing, line=bi, t_range=t_range))
    return walls


# ------------------------------------------------------------------------------------------ measurement

def _line_estimate(ctx, wall: WallEst, lines: list[Line], z_top: float) -> Estimate:
    """Re-estimate a wall's plane coordinate from points on this room's stretch of the line."""
    if wall.line is None:
        # never-observed wall: bound the room by where its floor/ceiling coverage ends, and say so
        c = wall.start[1] if wall.axis == 1 else wall.start[0]
        P, N = ctx["P"], ctx["N"]
        a, b = wall.axis, 1 - wall.axis
        t0, t1 = wall.t_range
        hz = (np.abs(N[:, 2]) > 0.7) & (P[:, b] > t0) & (P[:, b] < t1)
        inward = -wall.facing  # direction pointing out of the room along axis a
        side = (P[:, a] - c) * inward <= 0
        vals = P[hz & side, a]
        if len(vals) > 50:
            c = float(np.percentile(vals, 99 if inward > 0 else 1))
        return Estimate(float(c), ctx["cfg"].unobserved_se, observed=False)
    l = lines[wall.line]
    P, N = ctx["P"], ctx["N"]
    a, b = l.axis, 1 - l.axis
    win = max(0.02, 2.5 * ctx["sigma_med"])
    t0, t1 = wall.t_range
    shrink = min(0.1, 0.2 * (t1 - t0))
    sel = ((N[:, a] * l.facing > 0.85) & (np.abs(P[:, a] - l.c) < win) & (P[:, b] > t0 + shrink) &
           (P[:, b] < t1 - shrink) & (P[:, 2] > 0.1) & (P[:, 2] < z_top - 0.08))
    if sel.sum() < 30:
        return Estimate(l.c, max(l.se, 0.01), observed=True)
    x = P[sel, a]
    c = x.mean()
    for _ in range(2):
        m = np.abs(x - c) < win
        c = x[m].mean()
    m = np.abs(x - c) < win
    se = _se(float(x[m].std()), _n_eff(P[sel][m, b], P[sel][m, 2]), ctx["sigma_med"])
    return Estimate(float(c), se)


def _room_levels(ctx, poly: Polygon):
    from shapely import contains_xy
    P, N = ctx["P"], ctx["N"]
    inner = poly.buffer(-0.15)
    if inner.is_empty:
        inner = poly
    inside = contains_xy(inner, P[:, 0], P[:, 1])
    sig = ctx["sigma_med"]
    fsel = inside & (N[:, 2] > 0.85) & (np.abs(P[:, 2]) < 0.15)
    f, fstd, _ = _robust_level(P[fsel, 2], sig)
    csel = inside & (N[:, 2] < -0.85) & (P[:, 2] > 1.7)
    c, cstd, _ = _robust_level(P[csel, 2], sig)
    if not np.isfinite(f):
        f, fse = 0.0, 0.05
    else:
        sel = fsel & (np.abs(P[:, 2] - f) < max(0.02, 2.5 * sig))
        fse = _se(fstd, _n_eff(P[sel, 0], P[sel, 1]), sig)
    if not np.isfinite(c):
        c, cse, obs_c = ctx["ceil_g"], 0.15, False
    else:
        sel = csel & (np.abs(P[:, 2] - c) < max(0.02, 2.5 * sig))
        cse, obs_c = _se(cstd, _n_eff(P[sel, 0], P[sel, 1]), sig), True
    return Estimate(f, fse), Estimate(c, cse, obs_c)


def _measure_room(ctx, lines, room: RoomEst, rng: np.random.Generator | None = None):
    poly = Polygon(room.polygon)
    room.floor_z, room.ceil_z = _room_levels(ctx, poly)
    room.height = room.ceil_z - room.floor_z
    coords = [_line_estimate(ctx, w, lines, room.ceil_z.value) for w in room.walls]
    # rectilinear ring: vertex k = intersection of wall k-1 and wall k
    n = len(room.walls)
    rng = rng or np.random.default_rng(0)

    def ring_from(cs):
        pts = []
        for k in range(n):
            wp, wk = room.walls[k - 1], room.walls[k]
            x = cs[k - 1] if wp.axis == 0 else cs[k]
            y = cs[k - 1] if wp.axis == 1 else cs[k]
            pts.append((x, y))
        return np.array(pts)

    vals = np.array([c.value for c in coords])
    ses = np.array([c.se for c in coords])
    ring = ring_from(vals)
    if not Polygon(ring).is_valid:
        # refined planes crossed over (a sliver notch thinner than the fit noise): keep the structural
        # coordinates for this room and widen nothing silently - report it
        vals = np.array([(w.start[1] if w.axis == 1 else w.start[0]) for w in room.walls])
        ring = ring_from(vals)
        if not ctx.get("bootstrapping"):
            ctx.setdefault("warnings_extra", []).append(f"room {room.id}: refined walls self-intersected; structural outline kept")
    room.polygon = ring
    for k, w in enumerate(room.walls):
        w.start, w.end = ring[k], ring[(k + 1) % n]
        prev_c, next_c = coords[k - 1], coords[(k + 1) % n]
        L = abs(next_c.value - prev_c.value)
        w.length = Estimate(float(L), float(np.hypot(prev_c.se, next_c.se)), prev_c.observed and next_c.observed)
        if w.axis == 1:
            w.t_range = (min(ring[k][0], ring[(k + 1) % n][0]), max(ring[k][0], ring[(k + 1) % n][0]))
        else:
            w.t_range = (min(ring[k][1], ring[(k + 1) % n][1]), max(ring[k][1], ring[(k + 1) % n][1]))
    # area / perimeter uncertainty by Monte Carlo over wall coordinates (handles any rectilinear shape)
    samp = vals[None, :] + rng.normal(size=(200, n)) * ses[None, :]
    from shapely.validation import make_valid
    areas = [abs(make_valid(Polygon(ring_from(s))).area) for s in samp]
    perims = [Polygon(ring_from(s)).length for s in samp]
    room.area = Estimate(float(abs(Polygon(ring).area)), float(np.std(areas)), all(c.observed for c in coords))
    room.perimeter = Estimate(float(Polygon(ring).length), float(np.std(perims)), all(c.observed for c in coords))


def _measure_all(ctx, lines, rooms, warnings):
    from .openings import detect_openings
    for r in rooms:
        _measure_room(ctx, lines, r)
        r.openings = detect_openings(ctx, lines, r, warnings)


def _filter_mirror_points(ctx, lines, rooms, warnings):
    """Mirrors make the sensor see a reflected copy of the room behind the wall.

    For each room wall, take points observed *behind* it whose surface is parallel to the wall,
    reflect them across the wall plane, and check whether they land on real in-room surfaces of the
    same orientation. Floors, ceilings and perpendicular walls are reflection-invariant (a real door
    shows them too), so only parallel surfaces are used as evidence. If most of them match, the wall
    holds a mirror: every point seen *through* that region is a phantom and is dropped.
    """
    from shapely import contains_xy
    P, N, F, cams = ctx["P"], ctx["N"], ctx["F"], ctx["cams"]
    sig = ctx["sigma_med"]
    drop = np.zeros(len(P), bool)
    for r in rooms:
        poly = Polygon(r.polygon)
        inside = contains_xy(poly.buffer(0.05), P[:, 0], P[:, 1])  # includes the walls themselves
        for w in r.walls:
            if w.line is None:
                continue
            l = lines[w.line]
            a, b = l.axis, 1 - l.axis
            t0, t1 = w.t_range
            depth = (l.c - P[:, a]) * l.facing
            o = cams[F]
            oin = (o[:, a] - l.c) * l.facing > 0.05
            beh = (depth > max(0.04, 3 * sig)) & (depth < 4.0) & oin
            if beh.sum() < 50:
                continue
            lam = (l.c - o[beh, a]) / (P[beh, a] - o[beh, a])
            cross = o[beh] + lam[:, None] * (P[beh] - o[beh])
            within = (cross[:, b] > t0) & (cross[:, b] < t1) & (cross[:, 2] > 0) & (cross[:, 2] < (r.ceil_z.value if r.ceil_z.value else 3.0))
            idx = np.nonzero(beh)[0][within]
            cross = cross[within]
            # evidence = wall-parallel surfaces well behind the wall: points a few cm behind are the wall
            # itself (registration error) and reflect onto the wall, which would fake a match
            par = (np.abs(N[idx, a]) > 0.8) & (depth[idx] > 0.3)
            ref_idx = np.nonzero(inside & (np.abs(N[:, a]) > 0.8))[0]
            if par.sum() < 50 or len(ref_idx) < 50:
                continue
            Q = P[idx[par]].copy()
            Q[:, a] = 2 * l.c - Q[:, a]
            dist, _ = cKDTree(P[ref_idx]).query(Q, k=1, workers=-1)
            # phantom geometry carries the error of the folded path: allow 2 % of the distance behind
            hit = dist < np.maximum(max(0.03, 3 * sig), 0.02 * depth[idx[par]])
            ctx.setdefault("mirror_debug", []).append((w.id, int(par.sum()), float(hit.mean())))
            if hit.mean() < 0.4 or hit.sum() < 500:
                continue
            # the mirror's extent = the see-through blob(s) on this wall that the matched phantoms cross
            g = 0.04
            ti = ((cross[:, b] - t0) / g).astype(int)
            zi = (cross[:, 2] / g).astype(int)
            grid = np.zeros((ti.max() + 1, zi.max() + 1), bool)
            grid[ti, zi] = True
            grid = ndimage.binary_closing(grid, iterations=2)
            lab, _ = ndimage.label(grid)
            mlabels = np.unique(lab[ti[par][hit], zi[par][hit]])
            mlabels = mlabels[mlabels > 0]
            inreg = np.isin(lab[ti, zi], mlabels)
            if inreg.any() and _has_jamb_reveals(P, N, F, cams, l, cross[inreg], b):
                continue   # a doorway into a near-identical room, not a mirror: mirrors have no reveals
            drop[idx[inreg]] = True
            if inreg.any():
                ctx.setdefault("mirrors", []).append(
                    (w.line, float(cross[inreg, b].min()), float(cross[inreg, b].max()),
                     float(cross[inreg, 2].min()), float(cross[inreg, 2].max())))
            ext_t = np.ptp(cross[inreg, b]) if inreg.any() else 0
            ext_z = np.ptp(cross[inreg, 2]) if inreg.any() else 0
            warnings.append(f"mirror detected on wall {w.id} (~{ext_t:.2f} x {ext_z:.2f} m); "
                            f"{int(inreg.sum())} phantom points removed")
    if drop.any():
        keep = ~drop
        for k in ("P", "N", "pl", "S", "F"):
            ctx[k] = ctx[k][keep]
        ctx["n_mirror_points"] = int(drop.sum())


def _has_jamb_reveals(P, N, F, cams, l: Line, cross: np.ndarray, b: int) -> bool:
    """True if both side edges of a see-through region show jamb reveal surfaces: surfaces perpendicular to
    the wall (normal along the wall), within the wall thickness, seen from the front, at the region's left
    and right edges. Identified by orientation, so it works at photo-tier noise levels."""
    a = l.axis
    t_lo, t_hi = np.percentile(cross[:, b], [2, 98])
    z_lo, z_hi = np.percentile(cross[:, 2], [10, 90])
    depth = (l.c - P[:, a]) * l.facing
    front = (cams[F, a] - l.c) * l.facing > 0.05
    rev = front & (np.abs(N[:, b]) > 0.7) & (depth > -0.03) & (depth < 0.30) & (P[:, 2] > z_lo) & (P[:, 2] < z_hi)
    left = rev & (np.abs(P[:, b] - t_lo) < 0.06)
    right = rev & (np.abs(P[:, b] - t_hi) < 0.06)
    return bool(left.sum() >= 15 and right.sum() >= 15)


def _adjacency(rooms, lines):
    adj = []
    thick = []
    # doors: pair openings across rooms by centre distance
    used = set()
    for i, ra in enumerate(rooms):
        for oa in ra.openings:
            best = None
            for j, rb in enumerate(rooms):
                if j == i:
                    continue
                for ob in rb.openings:
                    if (j, ob.id) in used or ob.kind != oa.kind:
                        continue
                    d = np.linalg.norm(oa.center - ob.center)
                    if d < 0.45 and abs(oa.width.value - ob.width.value) < 0.15 and (best is None or d < best[0]):
                        best = (d, j, ob)
            if best is not None and (i, oa.id) not in used:
                _, j, ob = best
                oa.connects_to, ob.connects_to = rooms[j].id, ra.id
                used.add((i, oa.id))
                used.add((j, ob.id))
                adj.append({"rooms": [ra.id, rooms[j].id], "kind": "door" if oa.kind == "door" else "opening",
                            "via": [oa.id, ob.id]})
    # shared walls
    for i, ra in enumerate(rooms):
        for j in range(i + 1, len(rooms)):
            rb = rooms[j]
            for wa in ra.walls:
                for wb in rb.walls:
                    if wa.axis != wb.axis or wa.facing == wb.facing:
                        continue
                    ca = wa.start[1] if wa.axis == 1 else wa.start[0]
                    cb = wb.start[1] if wb.axis == 1 else wb.start[0]
                    gap = (cb - ca) * wa.facing * -1  # wb lies behind wa
                    ov = min(wa.t_range[1], wb.t_range[1]) - max(wa.t_range[0], wb.t_range[0])
                    if 0.04 < gap < 0.45 and ov > 0.3:
                        thick.append(gap)
                        if not any(set(e["rooms"]) == {ra.id, rb.id} for e in adj):
                            adj.append({"rooms": [ra.id, rb.id], "kind": "shared_wall", "via": [wa.id, wb.id]})
    th = Estimate(float(np.median(thick)), float(np.std(thick) / np.sqrt(len(thick)) + 0.005)) if thick else None
    return adj, th


def _bootstrap(ctx, lines, rooms, obs: Observations, cfg: CoreConfig):
    """Frame bootstrap of the measurement step (structure fixed). Adds the resampling spread to se."""
    rng = np.random.default_rng(cfg.seed)
    nF = len(obs.frames)
    if nF < 2:
        return
    base = {k: ctx[k] for k in ("P", "N", "pl", "S", "F")}
    samples: dict[tuple, list] = {}

    def rec(key, v):
        samples.setdefault(key, []).append(v)

    ctx["bootstrapping"] = True
    failures = 0
    for _ in range(cfg.bootstrap):
        pick = rng.integers(0, nF, nF)
        cnt = np.bincount(pick, minlength=nF)
        keep = cnt[base["F"]] > 0
        for k in base:
            ctx[k] = base[k][keep]
        for r in rooms:
            clone = RoomEst(r.id, r.cells, r.polygon.copy(),
                            [WallEst(w.id, w.start.copy(), w.end.copy(), w.length, w.axis, w.facing, w.line, w.t_range)
                             for w in r.walls], r.floor_z, r.ceil_z, r.height, r.area, r.perimeter)
            try:
                _measure_room(ctx, lines, clone, rng)
            except (ValueError, IndexError, FloatingPointError):
                failures += 1   # a resample lost a wall: counted, and the estimate is widened below
                continue
            rec((r.id, "height"), clone.height.value)
            rec((r.id, "area"), clone.area.value)
            for w in clone.walls:
                rec((w.id, "length"), w.length.value)
            from .openings import remeasure_opening
            for o in r.openings:
                v = remeasure_opening(ctx, lines, clone, o)
                if v is not None:
                    rec((o.id, "width"), v)
    for k in base:
        ctx[k] = base[k]
    ctx["bootstrapping"] = False
    if failures:
        ctx.setdefault("warnings_extra", []).append(f"{failures} bootstrap resamples could not be measured")

    def widen(est: Estimate, key):
        v = np.array([x for x in samples.get(key, []) if np.isfinite(x)])
        if len(v) >= 3:
            est.se = float(np.hypot(est.se, v.std(ddof=1)))
        elif cfg.bootstrap >= 3:
            est.se = float(np.hypot(est.se, 0.5 * abs(est.value) * 0.1))  # unstable under resampling
    for r in rooms:
        widen(r.height, (r.id, "height"))
        widen(r.area, (r.id, "area"))
        for w in r.walls:
            widen(w.length, (w.id, "length"))
        for o in r.openings:
            widen(o.width, (o.id, "width"))
