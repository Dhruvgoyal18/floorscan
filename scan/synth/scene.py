"""Parametric synthetic properties: rooms, walls with thickness, openings, casings, furniture,
mirror, glass, damage patches — plus their exact ground truth.

This is test/benchmark infrastructure. It exists so every pipeline stage can be checked against an
independent truth before real captures exist, and so failure sources (flying pixels, drift, glass,
mirrors, low light) can be injected deliberately.

Rooms are axis-aligned in *property-local* coordinates; the whole property is then rotated by
``yaw`` and translated, so the pipeline never gets an axis-aligned world for free.
Wall naming per room (CCW, interior on the left): S (y=y0, x0->x1), E (x=x1, y0->y1),
N (y=y1, x1->x0), W (x=x0, y1->y0).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# materials
WALL, FLOOR, CEIL, FURN, CASING, GLASS, MIRROR, OUTDOOR = range(8)
CASING_W, CASING_D = 0.07, 0.015
SIDES = ("S", "E", "N", "W")


@dataclass
class OpeningSpec:
    room: str
    side: str          # S/E/N/W of ``room``
    offset: float      # from wall start (CCW) to near jamb
    width: float
    height: float
    sill: float = 0.0  # 0 => door
    glass: bool = False
    name: str = ""

    @property
    def kind(self) -> str:
        return "door" if self.sill < 0.05 else "window"


@dataclass
class DamageSpec:
    room: str
    surface: str       # S/E/N/W or "C" (ceiling)
    cls: str           # water_stain | crack | mould
    u: float           # centre along wall from wall start (ceiling: x offset from room x0)
    z: float           # centre height above floor (ceiling: y offset from room y0)
    w: float
    h: float
    seed: int = 0
    name: str = ""


@dataclass
class RoomSpec:
    name: str
    x0: float
    y0: float
    x1: float
    y1: float
    ceiling: float
    furniture: list = field(default_factory=list)   # (lo, hi, colour) in local coords
    mirror: tuple | None = None                     # (side, u0, u1, z0, z1)
    wall_color: tuple = (0.86, 0.85, 0.80)
    light: float = 1.0                              # <1 => dim room

    def wall_endpoints(self, side: str):
        p = {"S": ((self.x0, self.y0), (self.x1, self.y0)), "E": ((self.x1, self.y0), (self.x1, self.y1)),
             "N": ((self.x1, self.y1), (self.x0, self.y1)), "W": ((self.x0, self.y1), (self.x0, self.y0))}
        return np.array(p[side][0], float), np.array(p[side][1], float)

    def wall_length(self, side: str) -> float:
        a, b = self.wall_endpoints(side)
        return float(np.linalg.norm(b - a))

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)


@dataclass
class PropertySpec:
    name: str
    rooms: list[RoomSpec]
    openings: list[OpeningSpec]
    damage: list[DamageSpec] = field(default_factory=list)
    wall_t: float = 0.12
    yaw_deg: float = 0.0
    offset: tuple = (0.0, 0.0)
    walk: list[str] = field(default_factory=list)   # room visiting order for captures

    def room(self, name: str) -> RoomSpec:
        return next(r for r in self.rooms if r.name == name)

    @property
    def T_world_local(self) -> np.ndarray:
        c, s = np.cos(np.radians(self.yaw_deg)), np.sin(np.radians(self.yaw_deg))
        T = np.eye(4)
        T[:2, :2] = [[c, -s], [s, c]]
        T[:2, 3] = self.offset
        return T


@dataclass
class Scene:
    """Box soup in property-local coordinates (z up). Rendering applies ``T_world_local``."""

    lo: np.ndarray
    hi: np.ndarray
    mat: np.ndarray
    color: np.ndarray
    room_of: np.ndarray          # index of room the box belongs to (-1 none)
    spec: PropertySpec


def _subtract(lo, hi, clo, chi):
    """AABB minus AABB -> list of up to 6 AABBs covering the difference."""
    if np.any(chi <= lo) or np.any(clo >= hi):
        return [(lo.copy(), hi.copy())]
    out = []
    lo, hi = lo.copy(), hi.copy()
    for ax in range(3):
        if clo[ax] > lo[ax]:
            a, b = lo.copy(), hi.copy()
            b[ax] = clo[ax]
            out.append((a, b))
            lo[ax] = clo[ax]
        if chi[ax] < hi[ax]:
            a, b = lo.copy(), hi.copy()
            a[ax] = chi[ax]
            out.append((a, b))
            hi[ax] = chi[ax]
    return out


def opening_box(spec: PropertySpec, op: OpeningSpec):
    """Cut volume of an opening (spans the wall thickness, plus a neighbour's coincident slab)."""
    r = spec.room(op.room)
    t = spec.wall_t
    a, b = r.wall_endpoints(op.side)
    d = (b - a) / np.linalg.norm(b - a)
    p0 = a + d * op.offset
    p1 = a + d * (op.offset + op.width)
    n_out = np.array([d[1], -d[0]])  # interior is on the left => outward normal on the right
    q0 = p0 - n_out * 0.01
    q1 = p1 + n_out * (t + 0.01)
    lo = np.array([min(q0[0], q1[0]), min(q0[1], q1[1]), op.sill])
    hi = np.array([max(q0[0], q1[0]), max(q0[1], q1[1]), op.sill + op.height])
    return lo, hi


def build_scene(spec: PropertySpec) -> Scene:
    t = spec.wall_t
    boxes: list[tuple] = []  # (lo, hi, mat, color, room)
    xs = [r.x0 for r in spec.rooms] + [r.x1 for r in spec.rooms]
    ys = [r.y0 for r in spec.rooms] + [r.y1 for r in spec.rooms]
    X0, X1, Y0, Y1 = min(xs) - 1, max(xs) + 1, min(ys) - 1, max(ys) + 1
    zmax = max(r.ceiling for r in spec.rooms)

    boxes.append((np.array([X0, Y0, -0.2]), np.array([X1, Y1, 0.0]), FLOOR, (0.55, 0.45, 0.35), -1))
    for ri, r in enumerate(spec.rooms):
        wc = r.wall_color
        walls = [
            (np.array([r.x0 - t, r.y0 - t, 0]), np.array([r.x1 + t, r.y0, r.ceiling])),  # S
            (np.array([r.x1, r.y0 - t, 0]), np.array([r.x1 + t, r.y1 + t, r.ceiling])),  # E
            (np.array([r.x0 - t, r.y1, 0]), np.array([r.x1 + t, r.y1 + t, r.ceiling])),  # N
            (np.array([r.x0 - t, r.y0 - t, 0]), np.array([r.x0, r.y1 + t, r.ceiling])),  # W
        ]
        for lo, hi in walls:
            boxes.append((lo, hi, WALL, wc, ri))
        boxes.append((np.array([r.x0 - t, r.y0 - t, r.ceiling]), np.array([r.x1 + t, r.y1 + t, r.ceiling + 0.2]),
                      CEIL, (0.93, 0.93, 0.92), ri))
        for lo, hi, col in r.furniture:
            boxes.append((np.array(lo, float), np.array(hi, float), FURN, col, ri))
        if r.mirror is not None:
            side, u0, u1, z0, z1 = r.mirror
            a, b = r.wall_endpoints(side)
            d = (b - a) / np.linalg.norm(b - a)
            n_in = np.array([-d[1], d[0]])
            p0, p1 = a + d * u0, a + d * u1 + n_in * 0.006
            boxes.append((np.array([min(p0[0], p1[0]), min(p0[1], p1[1]), z0]),
                          np.array([max(p0[0], p1[0]), max(p0[1], p1[1]), z1]), MIRROR, (0.7, 0.75, 0.8), ri))

    # walls above the tallest ceiling: avoid see-through into the void between different ceilings
    # (rooms with lower ceilings still get their own ceiling slab above them).
    # cut openings out of every wall slab they touch
    cuts = [opening_box(spec, op) for op in spec.openings]
    cut_boxes = []
    for lo, hi, mat, col, ri in boxes:
        pieces = [(lo, hi)]
        if mat == WALL:
            for clo, chi in cuts:
                nxt = []
                for a, b in pieces:
                    nxt += _subtract(a, b, clo, chi)
                pieces = nxt
        for a, b in pieces:
            if np.all(b - a > 1e-6):
                cut_boxes.append((a, b, mat, col, ri))
    boxes = cut_boxes

    # door casings (both faces of the wall), glass panes
    for op in spec.openings:
        r = spec.room(op.room)
        a, b = r.wall_endpoints(op.side)
        d = (b - a) / np.linalg.norm(b - a)
        n_in = np.array([-d[1], d[0]])
        ri = spec.rooms.index(r)
        if op.kind == "door":
            for face_shift, sign in ((0.0, 1.0), (-t, -1.0)):
                base = a + n_in * face_shift
                for u0, u1, z0, z1 in ((op.offset - CASING_W, op.offset, 0, op.height + CASING_W),
                                       (op.offset + op.width, op.offset + op.width + CASING_W, 0, op.height + CASING_W),
                                       (op.offset - CASING_W, op.offset + op.width + CASING_W, op.height, op.height + CASING_W)):
                    p0 = base + d * u0
                    p1 = base + d * u1 + sign * n_in * CASING_D
                    boxes.append((np.array([min(p0[0], p1[0]), min(p0[1], p1[1]), z0]),
                                  np.array([max(p0[0], p1[0]), max(p0[1], p1[1]), z1]), CASING, (0.97, 0.97, 0.97), ri))
        if op.glass:
            p0 = a + d * op.offset - n_in * (t / 2 - 0.003)
            p1 = a + d * (op.offset + op.width) - n_in * (t / 2 + 0.003)
            boxes.append((np.array([min(p0[0], p1[0]), min(p0[1], p1[1]), op.sill]),
                          np.array([max(p0[0], p1[0]), max(p0[1], p1[1]), op.sill + op.height]), GLASS, (0.6, 0.7, 0.75), ri))

    # outdoors: far facades well beyond LiDAR range (RGB sees them, LiDAR returns nothing)
    D = 25.0
    for lo, hi in (([X0 - D, Y0 - D, 0], [X1 + D, Y0 - D + 1, 8]), ([X0 - D, Y1 + D - 1, 0], [X1 + D, Y1 + D, 8]),
                   ([X0 - D, Y0 - D, 0], [X0 - D + 1, Y1 + D, 8]), ([X1 + D - 1, Y0 - D, 0], [X1 + D, Y1 + D, 8])):
        boxes.append((np.array(lo, float), np.array(hi, float), OUTDOOR, (0.45, 0.55, 0.45), -1))
    boxes.append((np.array([X0 - D, Y0 - D, -0.3]), np.array([X1 + D, Y1 + D, -0.2]), OUTDOOR, (0.35, 0.5, 0.3), -1))
    _ = zmax
    return Scene(
        lo=np.array([b[0] for b in boxes]), hi=np.array([b[1] for b in boxes]),
        mat=np.array([b[2] for b in boxes]), color=np.array([b[3] for b in boxes], float),
        room_of=np.array([b[4] for b in boxes]), spec=spec,
    )


# ---------------------------------------------------------------- damage texture

def damage_mask(dmg: DamageSpec, uu: np.ndarray, zz: np.ndarray) -> np.ndarray:
    """Boolean mask of a damage patch evaluated at surface coords (u along wall, z height)."""
    rng = np.random.default_rng(dmg.seed)
    du, dz = (uu - dmg.u) / (dmg.w / 2), (zz - dmg.z) / (dmg.h / 2)
    if dmg.cls in ("water_stain", "mould"):
        ang = np.arctan2(dz, du)
        rad = np.hypot(du, dz)
        k = np.arange(2, 6)
        amp = rng.uniform(0.03, 0.08, size=len(k))
        ph = rng.uniform(0, 2 * np.pi, size=len(k))
        boundary = 0.85 + np.sum(amp[:, None] * np.cos(k[:, None] * ang.ravel()[None, :] + ph[:, None]), axis=0).reshape(ang.shape)
        m = rad <= boundary
        if dmg.cls == "mould":  # speckled
            h = np.sin(uu * 311.0 + dmg.seed) * np.sin(zz * 297.0 + 2 * dmg.seed)
            m &= h > -0.2
        return m
    if dmg.cls == "crack":
        # diagonal polyline from one bbox corner to the other with a little wiggle, 4 mm wide
        s = np.clip(du * 0.5 + 0.5, 0, 1)
        wiggle = 0.12 * np.sin(s * 9 + dmg.seed)
        target = (s * 2 - 1) * 0.9 + wiggle
        width_n = 0.004 / (dmg.h / 2)
        return (np.abs(dz - target) < width_n) & (np.abs(du) <= 1)
    raise ValueError(dmg.cls)


def damage_truth(dmg: DamageSpec, res: float = 0.001) -> dict:
    uu, zz = np.meshgrid(np.arange(dmg.u - dmg.w / 2 - 0.01, dmg.u + dmg.w / 2 + 0.01, res),
                         np.arange(dmg.z - dmg.h / 2 - 0.01, dmg.z + dmg.h / 2 + 0.01, res))
    m = damage_mask(dmg, uu, zz)
    if not m.any():
        return {"area": 0.0, "width": 0.0, "height": 0.0}
    return {"area": float(m.sum() * res * res),
            "width": float(np.ptp(uu[m]) + res), "height": float(np.ptp(zz[m]) + res)}


DAMAGE_RGB = {"water_stain": (0.62, 0.50, 0.30), "mould": (0.18, 0.20, 0.14), "crack": (0.25, 0.23, 0.22)}


# ---------------------------------------------------------------- ground truth

def ground_truth(spec: PropertySpec, rooms: list[str] | None = None) -> dict:
    """GT in the same YAML-able structure as hand measurements (see data/gt/TEMPLATE.yaml)."""
    rooms = rooms or [r.name for r in spec.rooms]
    out = {"property": spec.name, "units": "m", "rooms": [], "adjacency": []}
    for r in spec.rooms:
        if r.name not in rooms:
            continue
        ops = []
        for i, op in enumerate(spec.openings):
            # an opening belongs to every room whose wall it pierces
            for side in SIDES:
                hit = _opening_on_room_wall(spec, op, r, side)
                if hit is not None:
                    ops.append({"id": op.name or f"o{i}", "type": op.kind, "wall": SIDES.index(side),
                                "offset": round(hit, 4), "width": op.width, "height": op.height,
                                **({"sill": op.sill} if op.kind == "window" else {})})
        dmg = []
        for j, d in enumerate(spec.damage):
            if d.room == r.name:
                tr = damage_truth(d)
                dmg.append({"id": d.name or f"d{j}", "class": d.cls,
                            "surface": "ceiling" if d.surface == "C" else "wall",
                            **({"wall": SIDES.index(d.surface)} if d.surface != "C" else {}),
                            "area": round(tr["area"], 5), "width": round(tr["width"], 4), "height": round(tr["height"], 4),
                            "u": d.u, "centroid_height": d.z if d.surface != "C" else r.ceiling})
        out["rooms"].append({
            "id": r.name,
            "walls": [round(r.wall_length(s), 4) for s in SIDES],
            "ceiling_height": [r.ceiling],
            "floor_area": round(r.area, 4),
            "diagonals": [round(float(np.hypot(r.x1 - r.x0, r.y1 - r.y0)), 4)],
            "openings": ops,
            "damage": dmg,
        })
    names = set(rooms)
    for op in spec.openings:
        rs = [r.name for r in spec.rooms if any(_opening_on_room_wall(spec, op, r, s) is not None for s in SIDES)]
        if len(rs) == 2 and set(rs) <= names:
            out["adjacency"].append(sorted(rs))
    out["net_floor_area"] = round(sum(spec.room(n).area for n in rooms), 4)
    return out


def _opening_on_room_wall(spec: PropertySpec, op: OpeningSpec, r: RoomSpec, side: str):
    """Offset of the opening along ``r``'s wall ``side`` if the opening pierces it, else None."""
    lo, hi = opening_box(spec, op)
    a, b = r.wall_endpoints(side)
    d = (b - a) / np.linalg.norm(b - a)
    # wall face line is fixed in one coordinate
    if abs(d[0]) > 0.5:  # wall along x at y = a[1]
        if not (lo[1] - 1e-6 <= a[1] <= hi[1] + 1e-6):
            return None
        u0, u1 = sorted(((lo[0] - a[0]) * d[0], (hi[0] - a[0]) * d[0]))
    else:
        if not (lo[0] - 1e-6 <= a[0] <= hi[0] + 1e-6):
            return None
        u0, u1 = sorted(((lo[1] - a[1]) * d[1], (hi[1] - a[1]) * d[1]))
    L = r.wall_length(side)
    if u1 <= 0.01 or u0 >= L - 0.01:
        return None
    return float(u0)
