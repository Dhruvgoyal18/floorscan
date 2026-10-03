"""Render the stitched floor plan (SVG + PNG): rooms, walls, dimensioned edges, doors, windows, damage."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Arc, Polygon as MplPolygon  # noqa: E402

FILL = ["#f3efe6", "#e8f0ee", "#f0eaf2", "#eef0e4", "#f5ece6", "#e6edf5", "#f2f0e1", "#ece8f0"]


def render_plan(result: dict, out_svg: Path, out_png: Path, title: str | None = None):
    rooms = result["rooms"]
    allp = np.concatenate([np.array(r["polygon_m"]) for r in rooms])
    span = np.ptp(allp, axis=0)
    W = 3 + span[0] * 1.6
    H = 3 + span[1] * 1.6
    s = min(1.0, 18 / max(W, H))
    fig, ax = plt.subplots(figsize=(max(6, W * s), max(5, H * s)))
    ax.set_aspect("equal")
    ax.axis("off")
    for i, r in enumerate(rooms):
        poly = np.array(r["polygon_m"])
        ax.add_patch(MplPolygon(poly, closed=True, fc=FILL[i % len(FILL)], ec="none", zorder=1))
        for wl in r["walls"]:
            a, b = np.array(wl["start_m"]), np.array(wl["end_m"])
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#222", lw=3.2, solid_capstyle="projecting", zorder=3)
            _dim(ax, a, b, poly, wl["length_m"])
        for o in r["openings"]:
            _opening(ax, r, o)
        c = poly.mean(0)
        A = r["floor_area_m2"]
        Hc = r["ceiling_height_m"]
        ax.text(c[0], c[1], f"{r['name']}\n{A['value']:.2f} m² (±{(A['ci90'][1] - A['ci90'][0]) / 2:.2f})\n"
                f"h {Hc['value']:.2f} m", ha="center", va="center", fontsize=8, zorder=6, color="#333")
    for d in result.get("damage", []):
        _damage(ax, result, d)
    pad = 1.0
    ax.set_xlim(allp[:, 0].min() - pad, allp[:, 0].max() + pad)
    ax.set_ylim(allp[:, 1].min() - pad, allp[:, 1].max() + pad)
    st = result["stitched"]
    ax.set_title(title or f"{result['capture_id']} - {result['tier']} tier - net floor area "
                 f"{st['net_floor_area_m2']['value']:.1f} m²", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_svg)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def _dim(ax, a, b, poly, m):
    mid = (a + b) / 2
    d = b - a
    L = np.linalg.norm(d)
    if L < 0.3:
        return
    n = np.array([d[1], -d[0]]) / L  # right of travel = outside for CCW rings
    p = mid + n * 0.28
    half = (m["ci90"][1] - m["ci90"][0]) / 2
    rot = np.degrees(np.arctan2(d[1], d[0]))
    if rot > 90 or rot < -90:
        rot += 180
    ax.text(p[0], p[1], f"{m['value']:.2f}±{half * 100:.0f}cm" if half < 0.1 else f"{m['value']:.2f}±{half:.2f}",
            ha="center", va="center", rotation=rot, fontsize=6.5, color="#555", zorder=5)


def _opening(ax, room, o):
    wl = next(w for w in room["walls"] if w["id"] == o["wall_id"])
    a, b = np.array(wl["start_m"]), np.array(wl["end_m"])
    d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
    p0 = a + d * o["offset_m"]["value"]
    p1 = p0 + d * o["width_m"]["value"]
    if o["type"] in ("door", "opening"):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="white", lw=4.0, zorder=4)
        if o["type"] == "door":
            nin = np.array([-d[1], d[0]])
            r = o["width_m"]["value"]
            ang0 = np.degrees(np.arctan2(d[1], d[0]))
            ax.add_patch(Arc(p0, 2 * r, 2 * r, theta1=ang0, theta2=ang0 + 90, color="#888", lw=0.7, zorder=4))
            tip = p0 + nin * r
            ax.plot([p0[0], tip[0]], [p0[1], tip[1]], color="#888", lw=0.9, zorder=4)
    else:
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="white", lw=4.0, zorder=4)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#4a90c2", lw=1.6, zorder=5)


def _damage(ax, result, d):
    sid = d["surface_id"]
    for r in result["rooms"]:
        for wl in r["walls"]:
            if wl["id"] == sid:
                a, b = np.array(wl["start_m"]), np.array(wl["end_m"])
                dd = (b - a) / max(np.linalg.norm(b - a), 1e-9)
                u0, _, u1, _ = d.get("bbox_on_surface_m", [0, 0, 0.1, 0])
                nin = np.array([-dd[1], dd[0]])
                p0, p1 = a + dd * u0 + nin * 0.1, a + dd * u1 + nin * 0.1
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#c0392b", lw=3, zorder=6)
                ax.text(*(p0 + p1) / 2 + nin * 0.15, d["class"].replace("_", " "), color="#c0392b", fontsize=6,
                        ha="center", zorder=6)
                return
        if sid == f"{r['id']}C":
            c = np.array(r["polygon_m"]).mean(0)
            ax.plot(*c, marker="x", color="#c0392b", ms=10, zorder=6)
