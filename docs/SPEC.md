# floorscan — spec

Synthesised from the case-study brief (`docs/brief.pdf`) and the execution plan. Vocabulary: see `GLOSSARY.md`.

## Problem statement

An assessor walks a property with an ordinary iPhone. They need one whole-property floor plan they
would recognise from poly.cam / magicplan — every room placed, connected and dimensioned — plus
damage regions, concealed-damage flags and repair scope keyed to surfaces. Every number must carry an
interval that is honest about how thin the input was. The same output must come from three input
tiers: a few photos per room, a handheld video, or LiDAR.

## Solution

`scan run <capture_dir>` — one command per capture. The capture directory is auto-detected as a
LiDAR (Stray Scanner) capture, a video, or a folder of per-room photo folders. Each tier front-end
turns its sensor data into the same intermediate (a gravity-aligned metric point cloud with
per-point noise and the camera centre that observed each point). One shared geometry core turns that
into rooms, walls, openings, ceiling heights and areas; one stitcher places rooms; one calibration
layer sets every interval; one renderer draws the plan. Output: `result.json` (validated against
`scan/schema/output.schema.json`), `plan.svg`, `plan.png`.

## User stories

1. As an assessor, I run one command on a capture folder and get a JSON + rendered plan, so that I never hand-tune per capture.
2. As an assessor with a non-Pro iPhone, I can capture 2–8 photos per room and still get a stitched whole-property plan, so that the floor tier is "any picture in, results out".
3. As an assessor, I can capture a handheld walkthrough video and get the same contract with tighter intervals than photos.
4. As an assessor with a Pro iPhone, I can capture LiDAR with Stray Scanner and get centimetre-level walls, ceiling heights and opening widths.
5. As a reviewer, every measurement I read has a 90% interval, and the interval is wider when the data was thinner.
6. As a reviewer, I can see which walls were never observed (interval flagged `observed: false`) rather than receive a confident guess.
7. As a reviewer, I see room adjacency (which door connects which rooms) and no overlapping rooms.
8. As a reviewer, I get damage regions with class, host surface, metric area and extent.
9. As a reviewer, I see concealed-damage flags with the exact rule that fired and the evidence ids.
10. As an estimator, I get scope line items (action, quantity, unit) keyed to surface ids with intervals propagated from geometry.
11. As the evaluator, I regenerate every reported number from raw data with `make reproduce`.
12. As the evaluator, I see gates at all three tiers, a repeatability table, a head-to-head table and timings in `reports/benchmark.md`, generated, never typed.
13. As the evaluator, I see the drift ablation (correction on/off) on the multi-room capture.
14. As the evaluator, I can check out `fixloop-before` and `fixloop-after` and regenerate both runs.
15. As a non-engineer at the walk-in, I follow `capture/PROTOCOL.md` literally and the capture works.
16. As an operator on a clean machine, I go from clone to first plan in under 15 minutes with `make setup`.
17. As an operator without a GPU or network at the walk-in, the LiDAR tier needs no learned weights at all, and the video/photo tiers run on CPU with pre-fetched weights.
18. As the evaluator, mirrors, glass, glossy floors and low light are handled or flagged, never silently wrong.

## Implementation decisions

- **Capture route 2** (stock apps): Stray Scanner for LiDAR; native Camera for video and photos.
- **One intermediate, one core.** Tier front-ends produce `Observations` (points, normals-free, per-point sigma, per-point camera origin, frames for damage). Everything downstream is tier-agnostic; the tier only changes noise models and calibration factors.
- **Manhattan-first geometry.** Walls are axis-aligned after estimating the dominant yaw; non-Manhattan wall evidence is reported as a warning. Rooms are recovered as unions of cells of the wall-line arrangement, separated where wall evidence exists above door-header height. Opening detection uses free-space (see-through) evidence by intersecting camera→point rays with the wall plane, so furniture occlusion is not confused with an opening.
- **Measure vs structure split.** Structure (which rooms, which walls, which openings) is decided once on all data; measurements are re-estimated on frame-bootstrap resamples to obtain the resampling term of the interval.
- **Drift:** per-fragment pose graph (4-DoF: x, y, z, yaw — gravity is trusted) with odometry edges, ICP-refined sequential edges and loop-closure edges, solved with robust least squares. `--no-drift-correction` produces the ablation.
- **Photo-tier stitching:** rooms are rigid Manhattan polygons; a beam search over door-to-door matches (folder order = walk order prior) places rooms so matched doors coincide across a wall-thickness gap, with overlap as a hard penalty.
- **Backends (video/photo depth+pose):** a single `ReconstructionBackend` seam with adapters: `mapanything` (primary), `depthanything` (Depth-Anything-V2 metric + RGB-D PnP odometry, lighter), `oracle` (synthetic benchmark only; reads renderer sidecars and perturbs them). Weights fetched by `scripts/fetch_weights.py`; cached model outputs replay deterministically.
- **Damage:** classical colour/texture anomaly segmentation restricted to fitted surfaces (no weights), optionally gated by an open-vocabulary detector; masks lifted to surface coordinates and merged across views.
- **Calibration:** analytic SE ⊕ bootstrap SE, scaled per (tier, measurement kind) by a conformal factor fitted leave-one-room-out on the benchmark, with a per-tier floor on relative width so thin input cannot produce confident intervals.

## Testing decisions (agreed seams)

1. CLI end-to-end: `scan run` on synthetic captures → schema-valid JSON scored against synthetic GT.
2. Geometry core: observations → rooms (known-answer synthetic rooms).
3. Stitch + drift: photo layout solver and pose-graph correction on synthetic multi-room plans with injected drift.
4. Eval harness: GT + predictions → gates, repeatability, calibration coverage, head-to-head.

Tests assert against independent truths (the synthetic scene's parameters), never against values recomputed the way the code computes them.

## Out of scope

Custom iOS app (Route 1); non-Manhattan walls beyond a warning; multi-storey; fine-tuned damage classifier; COLMAP baseline (future work).
