"""Top-view debug plot of the geometry core's intermediate state (points, wall lines, cells, rooms)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..geometry.core import CoreConfig, PlanEst, prepare_cloud  # noqa: E402
from ..geometry.frame import rot_z  # noqa: E402
from ..types import Observations  # noqa: E402


def debug_plot(obs: Observations, plan: PlanEst, out: Path, cfg: CoreConfig | None = None):
    cfg = cfg or CoreConfig()
    c = prepare_cloud(obs, cfg)
    P = c["P"] @ plan.R_plan.T
    P[:, 2] -= plan.z0
    N = c["N"] @ plan.R_plan.T
    cams = obs.cam_centers @ plan.R_plan.T
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    sel = np.random.default_rng(0).choice(len(P), min(len(P), 150000), replace=False)
    hz = np.abs(N[sel, 2]) > 0.7
    ax[0].scatter(P[sel][hz, 0], P[sel][hz, 1], s=0.2, c="tab:green", alpha=0.3, label="horizontal")
    ax[0].scatter(P[sel][~hz, 0], P[sel][~hz, 1], s=0.2, c="k", alpha=0.3, label="vertical")
    ax[0].scatter(cams[:, 0], cams[:, 1], s=8, c="tab:red", label="cameras")
    for l in plan.lines:
        col = "tab:blue" if l.facing > 0 else "tab:orange"
        if l.axis == 0:
            ax[0].axvline(l.c, color=col, lw=0.6, alpha=0.7)
        else:
            ax[0].axhline(l.c, color=col, lw=0.6, alpha=0.7)
    ax[0].set_aspect("equal")
    ax[0].set_title("points (green=horizontal surfaces), wall lines (blue +facing, orange -facing)")
    for r in plan.rooms:
        poly = np.vstack([r.polygon, r.polygon[:1]])
        ax[1].plot(poly[:, 0], poly[:, 1], "k-")
        cx, cy = r.polygon.mean(0)
        ax[1].text(cx, cy, f"{r.id}\n{r.area.value:.2f} m2", ha="center")
        for o in r.openings:
            ax[1].plot(*o.center, "o", color="tab:red" if o.kind == "door" else "tab:blue")
    ax[1].scatter(cams[:, 0], cams[:, 1], s=4, c="tab:red")
    ax[1].set_aspect("equal")
    ax[1].set_xlim(ax[0].get_xlim())
    ax[1].set_ylim(ax[0].get_ylim())
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)
    _ = rot_z
