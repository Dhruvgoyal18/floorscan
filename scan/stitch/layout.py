"""Photo-tier whole-property stitching: rooms that share no poses are placed by their doorways.

Each photo folder yields one room in its own Manhattan plan frame (unknown 90-degree rotation and
translation relative to the others). Doorways are the only shared evidence: the same doorway is seen
from both sides (the protocol requires one doorway photo from inside each room). Room j is placed
against an already-placed room i so that door q of j coincides with door p of i, across a wall of
thickness T, with j on the far side of i's wall. Folder order is the walk order, so each new room
attaches to an earlier one. A beam search scores hypotheses by doorway width/height agreement and
penalises any overlap between rooms; a final least-squares pass enforces every matched doorway
(including ones that close loops) at once.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Polygon

from ..geometry.core import OpeningEst, RoomEst

WALL_T = 0.12            # prior wall thickness (m) when no room pair constrains it
WALL_T_SE = 0.04


@dataclass
class Door:
    room: int
    idx: int
    c: np.ndarray      # centre on the wall face (room plan frame)
    n: np.ndarray      # inward normal (unit, axis-aligned)
    width: float
    width_se: float
    height: float
    kind: str
    id: str


@dataclass
class Seen:
    """Wall points a room's photos saw *through* a doorway (beyond its own polygon), per door."""
    pts: dict = field(default_factory=dict)   # door idx -> (n, 2) plan-frame points


@dataclass
class Placement:
    k: dict = field(default_factory=dict)       # room -> quarter turns
    t: dict = field(default_factory=dict)       # room -> translation
    pairs: list = field(default_factory=list)   # (door_a, door_b)
    score: float = 0.0


def _R(k: int) -> np.ndarray:
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][k % 4]
    return np.array([[c, -s], [s, c]], float)


def doors_of(ri: int, room: RoomEst) -> list[Door]:
    out = []
    for j, o in enumerate(room.openings):
        if o.kind not in ("door", "opening"):
            continue
        w = room.walls[o.edge]
        n = np.zeros(2)
        n[w.axis] = w.facing
        out.append(Door(ri, j, np.asarray(o.center, float), n, o.width.value, max(o.width.se, 0.01),
                        o.height.value, o.kind, o.id))
    return out


def _world(pl: Placement, d: Door):
    R = _R(pl.k[d.room])
    return R @ d.c + pl.t[d.room], R @ d.n


def _poly(pl: Placement, room: RoomEst, ri: int) -> Polygon:
    R = _R(pl.k[ri])
    return Polygon(room.polygon @ R.T + pl.t[ri])


def seen_through_doors(room: RoomEst, wall_pts: np.ndarray, max_pts: int = 600) -> Seen:
    """Assign wall points outside the room polygon to the doorway they were seen through."""
    seen = Seen()
    if wall_pts is None or len(wall_pts) == 0:
        return seen
    poly = Polygon(room.polygon)
    from shapely import contains_xy
    out = ~contains_xy(poly.buffer(0.25), wall_pts[:, 0], wall_pts[:, 1])
    P = wall_pts[out]
    ds = doors_of(0, room)
    if not len(P) or not ds:
        return seen
    for d in ds:
        rel = P - d.c
        along = -(rel @ d.n)                       # distance beyond the door's wall
        lateral = np.abs(rel @ np.array([-d.n[1], d.n[0]]))
        m = (along > 0.2) & (along < 5.0) & (lateral < 0.5 * d.width + 0.6 * along)
        q = P[m]
        if len(q) > max_pts:
            q = q[np.linspace(0, len(q) - 1, max_pts).astype(int)]
        if len(q) >= 20:
            seen.pts[d.idx] = q
    return seen


def solve_layout(rooms: list[RoomEst], beam: int = 40, wall_t: float = WALL_T,
                 seen: list[Seen] | None = None) -> tuple[Placement, list[str]]:
    warnings: list[str] = []
    doors = [doors_of(i, r) for i, r in enumerate(rooms)]
    seen = seen or [Seen() for _ in rooms]
    start = Placement(k={0: 0}, t={0: np.zeros(2)})
    states = [start]
    for j in range(1, len(rooms)):
        cand = []
        for st in states:
            used = {(d.room, d.idx) for pair in st.pairs for d in pair}
            for i in st.k:
                for dp in doors[i]:
                    if (dp.room, dp.idx) in used:
                        continue
                    cp, npw = _world(st, dp)
                    for dq in doors[j]:
                        if (dq.room, dq.idx) in used:
                            continue
                        # rotation that makes j's inward normal oppose i's
                        k = next(k for k in range(4) if np.allclose(_R(k) @ dq.n, -npw, atol=1e-6))
                        t = cp - npw * wall_t - _R(k) @ dq.c
                        new = Placement(k={**st.k, j: k}, t={**st.t, j: t}, pairs=st.pairs + [(dp, dq)])
                        cand.append((st.score + _attach_cost(new, rooms, j, dp, dq, seen), new))
        if not cand:
            # no doorway to attach through: place it beside the bounding box and say so
            st = states[0]
            minx = max(_poly(st, rooms[i], i).bounds[2] for i in st.k) + 1.0
            new = Placement(k={**st.k, j: 0}, t={**st.t, j: np.array([minx - rooms[j].polygon[:, 0].min(), 0.0])},
                            pairs=st.pairs, score=st.score + 50)
            states = [new]
            warnings.append(f"room {j + 1} shares no matchable doorway with earlier rooms; placed unconnected")
            continue
        cand.sort(key=lambda c: c[0])
        states = []
        for sc, pl in cand[:beam]:
            pl.score = sc
            states.append(pl)
    best = min(states, key=lambda s: s.score)
    best = _close_loops(best, rooms, doors, wall_t)
    _refine(best, rooms, wall_t)
    return best, warnings


def _boundary_consistency(pl: Placement, rooms, seen: list[Seen], src: Door, dst_room: int) -> float | None:
    """Share of wall points seen through ``src`` that land on the walls of the room placed behind it."""
    pts = seen[src.room].pts.get(src.idx)
    if pts is None:
        return None
    R = _R(pl.k[src.room])
    w = pts @ R.T + pl.t[src.room]
    from shapely import distance, points
    ring = _poly(pl, rooms[dst_room], dst_room).exterior
    d = distance(ring, points(w[:, 0], w[:, 1]))
    return float(np.mean(d < 0.2))


def _shared_wall_length(pl: Placement, rooms, j, wall_t) -> float:
    pj = _poly(pl, rooms[j], j)
    tot = 0.0
    for i in pl.k:
        if i != j:
            tot += pj.buffer(wall_t / 2 + 0.06).intersection(_poly(pl, rooms[i], i).buffer(wall_t / 2 + 0.06)).length / 2
    return tot


def _attach_cost(pl: Placement, rooms, j, dp: Door, dq: Door, seen: list[Seen]) -> float:
    # photo-tier widths carry a few cm of error each side; don't let a 5 cm disagreement dominate
    se = np.hypot(dp.width_se, dq.width_se) + 0.04
    cost = min(((dp.width - dq.width) / se) ** 2, 25.0)
    cost += min(((dp.height - dq.height) / (se + 0.03)) ** 2, 20.0) * 0.3
    if dp.kind != dq.kind:
        cost += 6.0
    pj = _poly(pl, rooms[j], j).buffer(-0.08)
    for i in pl.k:
        if i == j:
            continue
        ov = pj.intersection(_poly(pl, rooms[i], i).buffer(-0.08)).area
        cost += 400.0 * ov
    # what each room saw through this doorway must be the other room's walls
    for src, dst in ((dp, j), (dq, dp.room)):
        f = _boundary_consistency(pl, rooms, seen, src, dst)
        if f is not None:
            cost += 12.0 * (1.0 - f)
    # compact layouts: rooms in real buildings share walls
    cost -= 0.3 * min(_shared_wall_length(pl, rooms, j, WALL_T), 10.0)
    return cost


def _close_loops(pl: Placement, rooms, doors, wall_t) -> Placement:
    """Add doorway pairs not used by the tree that coincide after placement (e.g. living<->kitchen)."""
    used = {(d.room, d.idx) for pair in pl.pairs for d in pair}
    free = [d for ds in doors for d in ds if (d.room, d.idx) not in used and d.room in pl.k]
    for a in range(len(free)):
        for b in range(a + 1, len(free)):
            da, db = free[a], free[b]
            if da.room == db.room or (da.room, da.idx) in used or (db.room, db.idx) in used:
                continue
            ca, na = _world(pl, da)
            cb, nb = _world(pl, db)
            if np.allclose(na, -nb, atol=1e-6) and np.linalg.norm(ca - na * wall_t - cb) < 0.35 \
                    and abs(da.width - db.width) < 0.25:
                pl.pairs.append((da, db))
                used |= {(da.room, da.idx), (db.room, db.idx)}
    return pl


def _refine(pl: Placement, rooms, wall_t):
    """Least-squares translations (rotations fixed) so every matched doorway coincides."""
    idx = [r for r in sorted(pl.k) if r != 0]
    if not idx or not pl.pairs:
        return
    col = {r: 2 * i for i, r in enumerate(idx)}
    A, b = [], []
    for da, db in pl.pairs:
        Ra, Rb = _R(pl.k[da.room]), _R(pl.k[db.room])
        na = Ra @ da.n
        # Ra ca + ta - na T = Rb cb + tb
        rhs = Rb @ db.c - Ra @ da.c + na * wall_t
        for ax in range(2):
            row = np.zeros(2 * len(idx))
            if da.room != 0:
                row[col[da.room] + ax] += 1
            if db.room != 0:
                row[col[db.room] + ax] -= 1
            A.append(row)
            b.append(rhs[ax])
    sol, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    for r in idx:
        pl.t[r] = sol[col[r]:col[r] + 2]


def overlap_area(pl: Placement, rooms) -> float:
    total = 0.0
    keys = sorted(pl.k)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            total += _poly(pl, rooms[keys[a]], keys[a]).intersection(_poly(pl, rooms[keys[b]], keys[b])).area
    return float(total)


def transform_room(room: RoomEst, k: int, t: np.ndarray) -> RoomEst:
    """Move a room (polygon, walls, openings) into the stitched frame, in place."""
    R = _R(k)
    room.polygon = room.polygon @ R.T + t
    for w in room.walls:
        w.start = R @ w.start + t
        w.end = R @ w.end + t
        if k % 2 == 1:
            w.axis = 1 - w.axis
        n = np.zeros(2)
        n[w.axis] = 1
        # facing follows the rotated inward normal
        old = np.zeros(2)
        old[1 - w.axis if k % 2 == 1 else w.axis] = w.facing
        w.facing = int(np.sign((R @ old) @ n)) or w.facing
        lo = w.start[1 - w.axis]
        hi = w.end[1 - w.axis]
        w.t_range = (min(lo, hi), max(lo, hi))
    for o in room.openings:
        o.center = R @ np.asarray(o.center) + t
    return room


__all__ = ["solve_layout", "transform_room", "overlap_area", "doors_of", "OpeningEst"]
