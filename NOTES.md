# Decision log

Running log of design decisions and why. Newest at the bottom. This doubles as the defense script.

## 2026-10-03

- **Route 2 over a custom app.** No Apple developer account / TestFlight review risk; the effort goes into the pipeline. Stray Scanner records exactly what we need: RGB, 256x192 LiDAR depth (uint16 mm PNG), confidence (0/1/2), per-frame ARKit pose, intrinsics, IMU.
- **Verified the Stray Scanner pose convention from its source** (`OdometryEncoder.swift`): it writes `q_WA * q_AC` with `q_AC` a 180° rotation about x, so `odometry.csv` is cam-to-world in the **OpenCV** camera convention (x right, y down, z forward) in the ARKit world (y up, gravity -y). Newer builds append per-frame `fx, fy, cx, cy` (at RGB resolution). Loader handles both header variants. We convert world to z-up: `(x, y, z)_arkit -> (x, -z, y)`.
- **Package layout follows the plan**, CLI is `scan run <capture_dir>`.
- **Geometry in numpy/scipy/shapely only** — no Open3D. Keeps install small (walk-in on a clean machine) and the LiDAR tier free of learned weights, so it runs with no GPU and no network.
- **Learned backends behind one seam** (`scan/backends`): MapAnything (primary), Depth-Anything-V2-metric + PnP odometry (light), oracle (synthetic only). HuggingFace is unreachable from the dev sandbox, so learned adapters are written against the upstream API (verified by reading the MapAnything source, `model.infer` / `preprocess_inputs`) and validated on the capture machine; everything downstream is validated here with the oracle backend.
- **Synthetic benchmark first.** No real captures exist yet. A ray-cast renderer writes captures in the exact on-disk formats (Stray Scanner folder, video file, photo folders) plus GT YAML from the scene parameters. It injects realistic failure sources: depth noise growing with range, flying pixels at depth edges, low confidence at edges/range, ARKit-like drift, glass and a mirror. Real captures replace it later with no code changes.

## Geometry decisions (defend these)

- **Rooms from an arrangement of wall lines, not from a free-form outline.** Every room edge lies on a fitted
  plane, so wall length = distance between two plane fits (thousands of points each), which is how LiDAR wall
  error stays at a few mm. Cells merge unless the shared edge shows wall *above door height* (header band =
  top 0.3 m below the ceiling, and above 2.12 m): a doorway never separates rooms by itself, its header does.
- **Openings from free space, not holes.** A camera ray that ends behind a wall plane crossed that plane at an
  open spot. Unobserved wall (behind a wardrobe) produces no rays, so it is never an opening. No-return rays
  (beyond range, glass) count, which is how exterior windows are found. Threshold: 6 cm behind the face
  (casing + plaster + residual registration), plus a mostly-empty face and dense evidence in the box.
- **Door vs window:** a window has wall under its sill; a door's bottom is often hidden or unvisited.
- **Mirrors:** reflect wall-parallel geometry seen >30 cm behind a wall across the wall plane; if it lands on
  the room's own surfaces, the see-through blob is a mirror and all phantom points through it are dropped.
  Floors/ceilings/perpendicular walls are reflection-invariant and are deliberately not used as evidence.
- **Tolerances from measured surface thickness**, not from the sensor's absolute range sigma: learned depth
  errors are smooth (scale-like), so absolute sigma (~6 cm on video) would make every window swallow
  neighbouring surfaces. Surface thickness = max(local kNN scatter, spread about the strongest wall planes).
- **Intervals:** analytic se (plane residual / effective patch count + correlated bias term) (+) frame-bootstrap
  spread, scaled by a leave-one-room-out conformal factor per tier and kind, with per-tier floors.
  The shipped calibration stays at the conservative prior until fitted on *real* captures; the synthetic
  fit is reported but not shipped (it would calibrate to our own error model).
- **Drift:** fragments of ~3 s, ICP (4-DoF, gravity trusted) between consecutive fragments and between any
  fragments whose clouds come within 2.5 m (re-visits, loop end), robust pose graph. Ablation flag only.
- **Photo stitching:** folder order = walk order; beam search over doorway pairs; cost = width/height
  disagreement + overlap + "what I saw through this doorway must land on your walls" + shared-wall length;
  loop-closing least squares. Rejected: SfM across folders (2-8 photos with no overlap between rooms).

## Synthetic benchmark findings that changed the code
(see git log for the commits; each started from a failing benchmark row or a red test)
- phantom openings from 4-5 cm registration error behind walls -> 6 cm threshold, empty-face and density checks
- wall-thickness strips joined rooms through dilated coverage -> strips need both faces + real coverage
- absolute-sigma windows on video cut rooms apart -> surface-thickness windows
- photo adjacency via doorway pairs, not post-hoc centre proximity

## Fix loop and review (2026-10-03, late)
- Worst gate: photo-tier stitch (2/2 failing). Declaration committed before the fix; two root causes confirmed
  (header gap leak, doorway-as-mirror); the gate still fails on one-sided doorway detection (third defect).
  See fixloop/RESULT.md.
- Two-axis code review (spec, standards) by sub-agents found path-seeded synthetic noise (numbers would differ on
  another machine), shared noise across takes, missing intervals on offsets, an unconnected-room bug in the photo
  solver and several eval matching bugs. All fixed in one commit. The fix-loop tags were rebuilt on
  content-seeded noise; the originals are kept as fixloop-v1-*.
- Shipped calibration = synthetic LORO fit, labelled interim. The prior (k=2-3) was ~4x too narrow for the photo
  tier on the synthetic benchmark.
