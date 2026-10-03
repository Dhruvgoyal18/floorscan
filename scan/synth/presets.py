"""Synthetic properties used by tests and the synthetic benchmark."""
from __future__ import annotations

from .scene import DamageSpec, OpeningSpec, PropertySpec, RoomSpec


def single_room(name: str = "box", w: float = 4.0, d: float = 3.2, ceiling: float = 2.5, yaw: float = 17.0,
                door: bool = True, window: bool = True, furniture: bool = False) -> PropertySpec:
    furn = []
    if furniture:
        furn = [((1.0, d - 0.9, 0.0), (2.9, d - 0.1, 0.8), (0.45, 0.35, 0.55)),   # sofa under the window
                ((w - 0.7, d - 1.6, 0.0), (w - 0.1, d - 0.2, 2.0), (0.6, 0.5, 0.4))]  # wardrobe
    ops = []
    if door:
        ops.append(OpeningSpec(name, "S", 0.6, 0.82, 2.03, name="door"))
    if window:
        ops.append(OpeningSpec(name, "N", 1.1, 1.4, 1.2, sill=0.9, glass=True, name="window"))
    r = RoomSpec(name, 0.0, 0.0, w, d, ceiling, furniture=furn)
    return PropertySpec(name=f"single_{name}", rooms=[r], openings=ops, yaw_deg=yaw, offset=(1.3, -0.7), walk=[name])


def home() -> PropertySpec:
    """5 spaces: hall (connector), living, kitchen, bedroom (furnished + 2 damage classes), bathroom
    (mirror, dim). Living<->kitchen door closes a loop hall->living->kitchen->hall for drift tests."""
    t = 0.12
    hall = RoomSpec("hall", 0.0, 0.0, 6.2, 1.1, 2.55)
    living = RoomSpec("living", 0.0, 1.1 + t, 4.4, 5.0, 2.62,
                      furniture=[((0.4, 3.6, 0), (2.6, 4.6, 0.85), (0.4, 0.42, 0.6)),
                                 ((2.9, 2.0, 0), (3.7, 2.6, 0.45), (0.5, 0.3, 0.2))])
    kitchen = RoomSpec("kitchen", 4.4 + t, 1.1 + t, 7.8, 4.2, 2.62,
                       furniture=[((7.2, 1.3, 0), (7.8, 4.2, 0.9), (0.85, 0.85, 0.82))])
    bedroom = RoomSpec("bedroom", 0.0, -3.62, 3.6, -t, 2.58,
                       furniture=[((0.2, -3.5, 0), (1.8, -1.4, 0.55), (0.75, 0.75, 0.85)),      # bed
                                  ((2.9, -3.62, 0), (3.6, -2.2, 2.05), (0.55, 0.45, 0.35)),   # wardrobe
                                  ((0.0, -1.0, 0), (1.0, -0.6, 0.75), (0.5, 0.4, 0.3))])     # desk
    bath = RoomSpec("bath", 3.6 + t, -2.4, 5.6, -t, 2.40, mirror=("S", 0.5, 1.1, 1.0, 1.9), light=0.45,
                    wall_color=(0.80, 0.84, 0.86))
    ops = [
        OpeningSpec("hall", "N", 3.88, 0.82, 2.03, name="d_hall_living"),
        OpeningSpec("hall", "N", 0.44, 0.76, 2.03, name="d_hall_kitchen"),
        OpeningSpec("living", "E", 2.0, 0.86, 2.03, name="d_living_kitchen"),
        OpeningSpec("hall", "S", 2.4, 0.80, 2.03, name="d_hall_bedroom"),
        OpeningSpec("hall", "S", 4.3, 0.70, 2.03, name="d_hall_bath"),
        OpeningSpec("hall", "W", 0.1, 0.90, 2.05, name="d_front"),
        OpeningSpec("living", "N", 1.4, 1.6, 1.2, sill=0.9, glass=True, name="w_living"),
        OpeningSpec("kitchen", "E", 0.9, 1.0, 1.0, sill=1.05, glass=True, name="w_kitchen"),
        OpeningSpec("bedroom", "S", 1.1, 1.4, 1.2, sill=0.9, glass=True, name="w_bedroom"),
    ]
    damage = [
        DamageSpec("bedroom", "E", "water_stain", u=2.6, z=0.28, w=0.42, h=0.32, seed=3, name="stain_low"),
        DamageSpec("bedroom", "W", "crack", u=1.2, z=1.55, w=0.55, h=0.40, seed=5, name="crack_w"),
    ]
    return PropertySpec(name="home", rooms=[hall, living, kitchen, bedroom, bath], openings=ops, damage=damage,
                        wall_t=t, yaw_deg=23.0, offset=(2.0, -1.0),
                        walk=["hall", "living", "kitchen", "hall", "bedroom", "hall", "bath", "hall"])


def two_rooms(yaw: float = -31.0) -> PropertySpec:
    """Two rooms sharing a 0.12 m wall with a door: adjacency and wall-thickness checks."""
    t = 0.12
    a = RoomSpec("a", 0.0, 0.0, 4.2, 3.1, 2.55)
    b = RoomSpec("b", 4.2 + t, 0.0, 7.3, 3.4, 2.47)
    ops = [OpeningSpec("a", "E", 1.0, 0.84, 2.03, name="d_ab"),
           OpeningSpec("b", "N", 0.8, 1.1, 1.1, sill=1.0, glass=True, name="w_b")]
    return PropertySpec(name="two_rooms", rooms=[a, b], openings=ops, wall_t=t, yaw_deg=yaw, offset=(-0.5, 2.0),
                        walk=["a", "b", "a"])


def mirror_room() -> PropertySpec:
    r = RoomSpec("m", 0.0, 0.0, 3.0, 2.6, 2.45, mirror=("E", 0.6, 1.6, 0.9, 2.0))
    ops = [OpeningSpec("m", "S", 0.4, 0.80, 2.03, name="door")]
    return PropertySpec(name="mirror", rooms=[r], openings=ops, yaw_deg=8.0, walk=["m"])


def random_property(seed: int) -> PropertySpec:
    """Procedural hall + 2 rooms north + 1-2 rooms south. Used as the never-tuned-on holdout
    (stand-in for the walk-in test): its parameters are drawn, not chosen."""
    import numpy as np
    rng = np.random.default_rng(seed)
    t = float(rng.choice([0.10, 0.12, 0.15]))
    ceil = float(np.round(rng.uniform(2.38, 2.75), 2))
    hw = float(np.round(rng.uniform(1.0, 1.35), 2))
    north_w = [float(np.round(rng.uniform(2.8, 4.4), 2)) for _ in range(2)]
    hall_L = float(np.round(sum(north_w) + t, 2))
    hall = RoomSpec("hall", 0.0, 0.0, hall_L, hw, ceil)
    rooms, ops = [hall], []
    x = 0.0
    for i, w in enumerate(north_w):
        d = float(np.round(rng.uniform(3.0, 4.3), 2))
        r = RoomSpec(f"north{i + 1}", x, hw + t, x + w, hw + t + d, float(np.round(ceil + rng.choice([0, 0, -0.05]), 2)),
                     furniture=[((x + 0.3, hw + t + d - 0.9, 0), (x + 0.3 + 0.5 * w, hw + t + d - 0.1, 0.8),
                                 (0.5, 0.4, 0.35))])
        rooms.append(r)
        dw = float(rng.choice([0.76, 0.81, 0.86]))
        ops.append(OpeningSpec(r.name, "S", float(np.round(rng.uniform(0.25, w - dw - 0.25), 2)), dw, 2.03,
                               name=f"d_hall_{r.name}"))
        ww = float(np.round(rng.uniform(0.9, min(1.8, w - 0.8)), 2))
        ops.append(OpeningSpec(r.name, "N", float(np.round(rng.uniform(0.3, w - ww - 0.3), 2)), ww, 1.2, sill=0.9,
                               glass=True, name=f"w_{r.name}"))
        x += w + t
    # door between the two north rooms (closes a loop for drift / stitching)
    n1 = rooms[1]
    dd = float(np.round(rng.uniform(0.3, (n1.y1 - n1.y0) - 1.2), 2))
    ops.append(OpeningSpec("north1", "E", dd, 0.81, 2.03, name="d_north1_north2"))
    n_south = int(rng.integers(1, 3))
    x = 0.0
    for i in range(n_south):
        w = float(np.round(rng.uniform(2.4, hall_L / n_south - t), 2))
        d = float(np.round(rng.uniform(2.6, 3.8), 2))
        r = RoomSpec(f"south{i + 1}", x, -t - d, x + w, -t, ceil)
        rooms.append(r)
        dw = float(rng.choice([0.71, 0.76, 0.81]))
        xs = float(np.round(rng.uniform(x + 0.25, x + w - dw - 0.25), 2))
        ops.append(OpeningSpec(r.name, "N", float(np.round(x + w - (xs + dw), 2)), dw, 2.03, name=f"d_hall_{r.name}"))
        ww = float(np.round(rng.uniform(0.8, min(1.5, w - 0.8)), 2))
        ops.append(OpeningSpec(r.name, "S", float(np.round(rng.uniform(0.3, w - ww - 0.3), 2)), ww, 1.1, sill=1.0,
                               glass=True, name=f"w_{r.name}"))
        x += w + t
    walk = ["hall", "north1", "north2", "hall"] + sum([[f"south{i + 1}", "hall"] for i in range(n_south)], [])
    return PropertySpec(name=f"holdout{seed}", rooms=rooms, openings=ops, wall_t=t,
                        yaw_deg=float(rng.uniform(0, 90)), offset=(float(rng.uniform(-3, 3)), float(rng.uniform(-3, 3))),
                        walk=walk)
