"""Concealed-damage rules and scope line items. Every flag names its rule; every scope item is keyed
to a surface id and derives its quantity (with interval) from measured geometry."""
from __future__ import annotations

import numpy as np

from ..types import Estimate
from .detect import DamageRegion

RULES = {
    "wall_stain_within_30cm_of_floor":
        "Water stain whose lower edge is within 0.30 m of the floor: moisture wicking from a plumbing leak or "
        "rising damp; wet insulation / bottom plate likely behind the board.",
    "ceiling_stain":
        "Water stain on the ceiling: leak from the roof or the floor above; wet insulation likely above the stain.",
    "stain_near_window_corner":
        "Water stain within 0.30 m of a window corner on the same wall: failed window seal or flashing; water "
        "tracking inside the wall cavity below the window.",
    "visible_mould":
        "Visible mould: growth on the cavity side is typically larger than the visible patch.",
    "crack_from_opening_corner":
        "Crack starting within 0.25 m of a door/window corner: movement or settlement concentrated at the opening.",
}


def concealed_flags(regions: list[DamageRegion]) -> list[dict]:
    flags = []
    for d in regions:
        s = d.surface
        u0, v0, u1, v1 = d.bbox
        if d.cls in ("water_stain", "mould") and s.kind == "wall" and v0 <= 0.30:
            flags.append(_flag(s.id, "wall_stain_within_30cm_of_floor", [d.id]))
        if d.cls == "water_stain" and s.kind == "ceiling":
            flags.append(_flag(s.id, "ceiling_stain", [d.id]))
        if d.cls == "mould":
            flags.append(_flag(s.id, "visible_mould", [d.id]))
        if s.kind == "wall":
            for o in s.room.openings:
                if o.edge != s.wall_idx:
                    continue
                corners = _opening_corners(s, o)
                dist = min(_rect_point_dist(d.bbox, c) for c in corners)
                if d.cls == "water_stain" and o.kind == "window" and dist <= 0.30:
                    flags.append(_flag(s.id, "stain_near_window_corner", [d.id, o.id]))
                if d.cls == "crack" and dist <= 0.25:
                    flags.append(_flag(s.id, "crack_from_opening_corner", [d.id, o.id]))
    return flags


def _flag(surface_id, rule, evidence):
    return {"surface_id": surface_id, "rule": rule, "rationale": RULES[rule], "evidence": evidence}


def _opening_corners(s, o):
    w = s.room.walls[s.wall_idx]
    # opening jambs in surface u coordinates
    a = 1 - w.axis
    u_l = (o.t0.value - w.start[a]) * np.sign(w.end[a] - w.start[a])
    u_r = (o.t1.value - w.start[a]) * np.sign(w.end[a] - w.start[a])
    u_lo, u_hi = min(u_l, u_r), max(u_l, u_r)
    z_lo = o.sill.value if o.sill is not None else 0.0
    z_hi = z_lo + o.height.value
    return [(u_lo, z_lo), (u_hi, z_lo), (u_lo, z_hi), (u_hi, z_hi)]


def _rect_point_dist(b, p):
    u0, v0, u1, v1 = b
    du = max(u0 - p[0], 0, p[0] - u1)
    dv = max(v0 - p[1], 0, p[1] - v1)
    return float(np.hypot(du, dv))


def scope_items(regions: list[DamageRegion], flags: list[dict]) -> list[dict]:
    items = []
    fired = {(f["surface_id"], f["rule"]) for f in flags}
    done_repaint = set()

    def add(surface_id, action, q: Estimate, unit, reason):
        items.append({"surface_id": surface_id, "action": action, "quantity": q, "unit": unit, "reason": reason})

    for d in regions:
        s = d.surface
        room = s.room
        if s.kind == "wall":
            w = room.walls[s.wall_idx]
            wall_area = Estimate(w.length.value * room.height.value,
                                 float(np.hypot(w.length.se * room.height.value, room.height.se * w.length.value)))
            open_area = sum(o.width.value * o.height.value for o in room.openings if o.edge == s.wall_idx)
            net = Estimate(max(wall_area.value - open_area, 0.0), wall_area.se)
        else:
            net = room.area
        if d.cls == "water_stain":
            if (s.id, "wall_stain_within_30cm_of_floor") in fired:
                wdt = Estimate(d.width.value + 0.6, d.width.se)
                add(s.id, "cut out and replace plasterboard, lower 1.2 m, affected width + 0.3 m each side",
                    Estimate(wdt.value * 1.2, wdt.se * 1.2), "m2", f"{d.id} + rule wall_stain_within_30cm_of_floor")
                add(s.id, "open up and inspect bottom plate / plumbing behind wall", Estimate(1, 0), "each",
                    f"{d.id} + rule wall_stain_within_30cm_of_floor")
            if s.id not in done_repaint:
                add(s.id, "stain-blocking primer + repaint full surface", net, "m2", d.id)
                done_repaint.add(s.id)
        elif d.cls == "mould":
            a = Estimate((d.width.value + 0.6) * (d.height.value + 0.6), d.area.se + 0.05)
            add(s.id, "mould remediation (HEPA, biocide) incl. 0.3 m margin", a, "m2", d.id)
            add(s.id, "replace plasterboard over remediated area", a, "m2", d.id)
        elif d.cls == "crack":
            add(s.id, "rake out, fill and tape crack", Estimate(d.length, 0.03 + 0.1 * d.length), "m", d.id)
            if s.id not in done_repaint:
                add(s.id, "repaint full surface", net, "m2", d.id)
                done_repaint.add(s.id)
        elif d.cls == "hole":
            add(s.id, "patch plasterboard", Estimate(1, 0), "each", d.id)
    for i, it in enumerate(items):
        it["id"] = f"S{i + 1}"
    return items
