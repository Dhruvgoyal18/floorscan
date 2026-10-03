# floorscan

Phone capture → one dimensioned, stitched whole-property floor plan, with damage regions, concealed-damage
flags and scope line items. Every measurement carries a 90 % interval that widens honestly as the
sensor data thins. Three input tiers, one output contract:

| Tier | Input | Capture app | Depth + poses from |
|---|---|---|---|
| LiDAR | Stray Scanner folder | Stray Scanner (Pro devices) | LiDAR + ARKit (no learned model) |
| Video | one walkthrough .MOV/.MP4 | built-in Camera | MapAnything (or Depth-Anything + PnP) |
| Photo | parent folder with one folder of 4–8 photos per room | built-in Camera (0.5×) | same backend, per room; rooms stitched by doorways |

How to capture: [`capture/PROTOCOL.md`](capture/PROTOCOL.md) (one page). What each tier delivers: [`capture/DEVICE_MATRIX.md`](capture/DEVICE_MATRIX.md).

## Clean machine → first plan (< 15 minutes)

Requires Python ≥ 3.10 and `make` (Linux/macOS; on Windows use WSL, or run the commands inside the Makefile by hand).

```bash
git clone <repo> floorscan && cd floorscan
make setup                         # venv + package (~2 min). Enough for the LiDAR tier.
make setup-models                  # video/photo tiers: torch, MapAnything, weights (~1.3 GB, once)
.venv/bin/scan run path/to/capture # one command per capture; tier is auto-detected
```

Output in `out/<capture>/`: `result.json` (validated against [`scan/schema/output.schema.json`](scan/schema/output.schema.json)),
`plan.svg`, `plan.png`. Useful flags: `--tier`, `--backend {mapanything,depthanything,oracle}`,
`--no-drift-correction` (ablation only), `--device "iPhone 15 Pro"`.

No capture at hand? `make synthetic` renders the synthetic benchmark (same file formats as the real apps), then
`.venv/bin/scan run data/raw/synthetic/home/multiroom/lidar/t1`.

## Reproduce every number

```bash
make test            # seam tests (geometry core, drift, eval harness)  ~1 min
make test-all        # + end-to-end CLI tests on rendered captures        ~4 min
make benchmark       # render (deterministic) + run + score -> reports/synthetic/benchmark.md
make benchmark-real  # real captures listed in data/benchmark_real.yaml -> reports/real/benchmark.md
make fixloop         # regenerates the fix-loop before/after runs from git tags
make reproduce       # all of the above
```

Model outputs are cached (`.cache/backends`, keyed by input bytes) and replay deterministically; delete the cache
or set `FLOORSCAN_NO_CACHE=1` to force the live path.

## Layout

```
scan/            the package; CLI: scan run <capture>
  io/            Stray Scanner, video keyframes, photo folders -> frames; frames -> Observations
  tiers/         lidar.py, rgb.py (video + photo front-ends, gravity from normals)
  backends/      ReconstructionBackend seam: mapanything, depthanything, oracle (synthetic only)
  geometry/      the shared core: Manhattan frame, wall lines, rooms, openings, levels, intervals
  stitch/        drift.py (fragment pose graph + loop closure), layout.py (photo-tier doorway stitching)
  damage/        detection on fitted surfaces, concealed-damage rules, scope items
  calibrate/     the one place ci90 widths are set (conformal factors + per-tier floors)
  render/        plan SVG/PNG; debug top views
  synth/         synthetic property renderer that writes real capture formats (test + benchmark infra)
eval/            matching, gates, LORO calibration, report generation
capture/         PROTOCOL.md, DEVICE_MATRIX.md
data/            benchmark manifests, ground truth, raw captures (fetched/rendered, not committed)
fixloop/         declaration, before/after runs, diff
reports/         generated benchmark + technical report
docs/            brief, plan, spec, ADR-style notes;  NOTES.md = decision log
```

Design decisions and why: [`NOTES.md`](NOTES.md). Requirement coverage: [`COMPLIANCE.md`](COMPLIANCE.md).
