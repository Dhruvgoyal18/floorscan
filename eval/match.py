"""Match a pipeline result to ground truth and produce per-item error records.

GT format (data/gt/<property>.yaml, same for synthetic and tape/laser measurements):
  rooms: - id, walls: [lengths CCW starting anywhere], ceiling_height: [..], floor_area,
           openings: [{id, type, wall (index into walls), offset (from that wall's start, CCW), width, height, sill?}],
           damage:   [{id, class, surface: wall|ceiling, wall, u, centroid_height, area, width, height}]
  adjacency: [[room_a, room_b], ...]   (rooms connected by a doorway)

Matching rules (fixed before looking at results, so they cannot be tuned to flatter):
  * rooms: Hungarian on wall-length signature (+ area), rejected if area differs by > 50 %
  * walls: cyclic alignment of the CCW wall sequence; ties broken by where the openings are
  * openings: on the aligned wall, Hungarian on |dwidth| + 0.5 |doffset|; accepted if |dwidth| < 0.25 m and
    |doffset| < 0.6 m; an unmatched GT opening is a *miss*, an unmatched prediction a *phantom*
  * damage: same surface, compatible class, centre within 0.4 m
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Match:
    capture: str
    tier: str
    property: str
    records: list = field(default_factory=list)       # scored measurements
    openings: dict = field(default_factory=lambda: {"gt": 0, "matched": 0, "missed": [], "phantom": []})
    rooms: dict = field(default_factory=dict)          # pred room id -> gt room id
    walls: dict = field(default_factory=dict)          # (gt room, gt wall idx) -> pred wall length
    adjacency: dict = field(default_factory=dict)
    damage: dict = field(default_factory=dict)
    footprint: dict = field(default_factory=dict)


def _compatible(gt_type, pred_type):
    return gt_type == pred_type or {gt_type, pred_type} <= {"door", "opening"}


def _room_cost(pr, gr):
    pw = [w["length_m"]["value"] for w in pr["walls"]]
    gw = gr["walls"]
    pa, ga = pr["floor_area_m2"]["value"], gr["floor_area"]
    if abs(pa - ga) / ga > 0.5:
        return 1e6
    c = abs(pa - ga) / ga
    if len(pw) == len(gw):
        c += float(np.mean(np.abs(np.sort(pw) - np.sort(gw)) / np.sort(gw)))
    else:
        c += 0.5
    return c


def _align_walls(pr, gr):
    """Return mapping gt wall idx -> pred wall index."""
    pw = [w["length_m"]["value"] for w in pr["walls"]]
    gw = gr["walls"]
    pred_open = {}
    for o in pr["openings"]:
        pred_open.setdefault(o["wall_id"], []).append(o)
    ids = [w["id"] for w in pr["walls"]]
    if len(pw) == len(gw):
        best = None
        for s in range(len(gw)):
            cost = sum(abs(pw[(i + s) % len(pw)] - gw[i]) for i in range(len(gw)))
            for go in gr.get("openings", []):
                cands = pred_open.get(ids[(go["wall"] + s) % len(pw)], [])
                if not any(_compatible(go["type"], o["type"]) for o in cands):
                    cost += 0.5
            if best is None or cost < best[0]:
                best = (cost, s)
        s = best[1]
        return {i: (i + s) % len(pw) for i in range(len(gw))}
    used, out = set(), {}
    for i in np.argsort(gw)[::-1]:
        j = min((j for j in range(len(pw)) if j not in used), key=lambda j: abs(pw[j] - gw[i]), default=None)
        if j is not None:
            out[int(i)] = j
            used.add(j)
    return out


def match(result: dict, estimates: dict, gt: dict, rooms_in_capture: list[str], capture: str, prop: str) -> Match:
    tier = result["tier"]
    m = Match(capture, tier, prop)
    se = {it["key"]: it for it in estimates.get("items", [])}
    gts = [g for g in gt["rooms"] if g["id"] in rooms_in_capture]
    preds = result["rooms"]
    C = np.array([[_room_cost(p, g) for g in gts] for p in preds]) if preds and gts else np.zeros((0, 0))
    pairs = []
    if C.size:
        ri, ci = linear_sum_assignment(C)
        pairs = [(preds[i], gts[j]) for i, j in zip(ri, ci) if C[i, j] < 1e5]
    m.rooms = {p["id"]: g["id"] for p, g in pairs}

    def rec(kind, key, gtv, pv, room, extra=None, ci90=None):
        e = se.get(key, {})
        r = {"capture": capture, "tier": tier, "property": prop, "room": room, "kind": kind, "key": key,
             "gt": float(gtv), "pred": float(pv), "err": float(pv - gtv), "se": float(e.get("se", np.nan)),
             "observed": bool(e.get("observed", True)), "ci90": ci90}
        if extra:
            r.update(extra)
        m.records.append(r)

    for pr, gr in pairs:
        rid, gid = pr["id"], gr["id"]
        amap = _align_walls(pr, gr)
        for gi, pi in amap.items():
            w = pr["walls"][pi]
            rec("wall_length", f"{rid}/{w['id']}/length", gr["walls"][gi], w["length_m"]["value"], gid,
                {"gt_wall": gi}, w["length_m"]["ci90"])
            m.walls[(gid, gi)] = w["length_m"]["value"]
        rec("ceiling_height", f"{rid}/height", float(np.mean(gr["ceiling_height"])), pr["ceiling_height_m"]["value"],
            gid, None, pr["ceiling_height_m"]["ci90"])
        rec("floor_area", f"{rid}/area", gr["floor_area"], pr["floor_area_m2"]["value"], gid, None,
            pr["floor_area_m2"]["ci90"])
        # openings
        used_pred = set()
        for gi, pi in amap.items():
            wid = pr["walls"][pi]["id"]
            gos = [o for o in gr.get("openings", []) if o["wall"] == gi]
            pos = [o for o in pr["openings"] if o["wall_id"] == wid]
            if gos and pos:
                Cm = np.array([[abs(p["width_m"]["value"] - g["width"]) + 0.5 * abs(p["offset_m"]["value"] - g["offset"])
                                + (0 if _compatible(g["type"], p["type"]) else 10) for p in pos] for g in gos])
                a, b = linear_sum_assignment(Cm)
                ok = set()
                for i, j in zip(a, b):
                    g, p = gos[i], pos[j]
                    if (abs(p["width_m"]["value"] - g["width"]) < 0.25 and abs(p["offset_m"]["value"] - g["offset"]) < 0.6
                            and _compatible(g["type"], p["type"])):
                        ok.add(i)
                        used_pred.add(p["id"])
                        rec("opening_width", f"{rid}/{p['id']}/width", g["width"], p["width_m"]["value"], gid,
                            {"opening": g["id"], "type": g["type"]}, p["width_m"]["ci90"])
                        rec("opening_height", f"{rid}/{p['id']}/height", g["height"], p["height_m"]["value"], gid,
                            {"opening": g["id"], "type": g["type"]}, p["height_m"]["ci90"])
                for i, g in enumerate(gos):
                    if i not in ok:
                        m.openings["missed"].append(f"{gid}:{g['id']}")
            else:
                m.openings["missed"] += [f"{gid}:{g['id']}" for g in gos]
            m.openings["gt"] += len(gos)
        m.openings["phantom"] += [f"{rid}:{o['id']}({o['type']} {o['width_m']['value']:.2f})"
                                  for o in pr["openings"] if o["id"] not in used_pred]
        # damage
        for gd in gr.get("damage", []):
            m.damage.setdefault("gt", 0)
            m.damage["gt"] += 1
            if gd["surface"] == "wall":
                sid = pr["walls"][amap[gd["wall"]]]["id"] if gd["wall"] in amap else None
            else:
                sid = f"{rid}C"
            hit = None
            for d in result["damage"]:
                if d["surface_id"] != sid:
                    continue
                u0, v0, u1, v1 = d["bbox_on_surface_m"]
                cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
                if gd["surface"] == "wall":
                    ok = abs(cv - gd["centroid_height"]) < 0.4 and abs(cu - gd.get("u", cu)) < 0.4
                else:   # ceiling: GT gives the centre offset from the room's corner as (u, v) when measured
                    ok = abs(cu - gd.get("u", cu)) < 0.6 and abs(cv - gd.get("v", cv)) < 0.6
                if ok:
                    hit = d
                    break
            if hit:
                m.damage.setdefault("found", []).append({"gt": gd["id"], "pred": hit["id"], "gt_class": gd["class"],
                                                          "pred_class": hit["class"], "gt_area": gd["area"],
                                                          "pred_area": hit["area_m2"]["value"],
                                                          "gt_width": gd["width"], "pred_width": hit["extent_m"]["width"]["value"]})
                if hit["class"] != "crack":   # crack area is raster-limited; cracks are scored by extent
                    rec("damage_area", f"{hit['id']}/area", gd["area"], hit["area_m2"]["value"], gid,
                        {"damage": gd["id"]}, hit["area_m2"]["ci90"])
            else:
                m.damage.setdefault("missed", []).append(gd["id"])
    matched_damage = {f["pred"] for f in m.damage.get("found", [])}
    m.damage["false_positive"] = [d["id"] for d in result["damage"] if d["id"] not in matched_damage
                                  and any(d["surface_id"] == p + "C" or re.fullmatch(re.escape(p) + r"W\d+", d["surface_id"])
                                          for p in m.rooms)]
    m.openings["matched"] = m.openings["gt"] - len(m.openings["missed"])

    # adjacency (doorway connections) among matched rooms
    gt_adj = {frozenset(p) for p in gt.get("adjacency", []) if set(p) <= set(rooms_in_capture)}
    pr_adj = set()
    for e in result["stitched"]["adjacency"]:
        if e["kind"] in ("door", "opening"):
            a, b = e["rooms"]
            if a in m.rooms and b in m.rooms:
                pr_adj.add(frozenset((m.rooms[a], m.rooms[b])))
    m.adjacency = {"gt": sorted(map(sorted, gt_adj)), "pred": sorted(map(sorted, pr_adj)),
                   "correct": gt_adj == pr_adj, "missing": sorted(map(sorted, gt_adj - pr_adj)),
                   "extra": sorted(map(sorted, pr_adj - gt_adj))}
    gt_net = sum(g["floor_area"] for g in gts)
    st = result["stitched"]
    rec("footprint", "net_area", gt_net, st["net_floor_area_m2"]["value"], "_all", None, st["net_floor_area_m2"]["ci90"])
    m.footprint = {"gt": gt_net, "pred": st["net_floor_area_m2"]["value"], "ci90": st["net_floor_area_m2"]["ci90"],
                   "rel_err": st["net_floor_area_m2"]["value"] / gt_net - 1 if gt_net else None,
                   "overlap_m2": st["overlap_m2"], "n_rooms_pred": len(preds), "n_rooms_gt": len(gts)}
    return m
