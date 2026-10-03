# Applied AI Case Study: End-to-End Execution Plan

Oct 3, 2026 · @Dhruv

## Executive summary

The winning strategy is one geometry core fed by three front-ends (LiDAR, video, photos), with honest calibrated intervals, a self-built benchmark, and a pipeline that runs cold on a stranger's capture. Everything else in the brief is a check that you actually built that.

The company wants a phone-to-floor-plan system: walk a property, get a dimensioned, stitched whole-property plan plus damage regions and repair scope, as JSON plus a rendered plan. You own capture, reconstruction, measurement, damage detection, evaluation, and proof.

Where the marks are: 55% is decided by two things you cannot fake. The walk-in test (30%) runs your pipeline live on a space you have never seen. The fix loop (25%) requires a shipped, regenerable before/after on your worst gate. A further 15% is reproducible benchmark accuracy.

What they are screening for, in order:

1. **Generalisation:** works on their phone, their room, your instructions followed literally.
2. **Calibration:** intervals that widen honestly on thin input. Confident garbage caps your total score.
3. **Engineering loop:** measure, diagnose, fix, re-measure, with the numbers to prove it.
4. **Reproducibility:** every number regenerable from raw data with one command.
5. **Ownership:** a commit history and a live defense showing you understand every decision.

Core recommendation: take Route 2 (stock capture protocol), make the LiDAR tier excellent first, then build video and photo tiers that degrade gracefully into the same JSON contract. Spend the first 3.5 hours capturing the benchmark with ground truth, because the fix loop and calibration both depend on it.

## Deliverables checklist

Eight deliverables, all in one repo. Each row says what "done" means so you can self-audit against the compliance matrix.

| # | Deliverable | Where it lives | Done when |
| --- | --- | --- | --- |
| 1 | Compliance matrix | `COMPLIANCE.md` | Every sentence of the brief that states a requirement is a row: requirement, file path, artifact, status (pass / partial / fail, with the number) |
| 2 | Capture route + device matrix | `capture/PROTOCOL.pdf`, `capture/DEVICE_MATRIX.md` | A non-engineer follows the one page without asking a question; matrix lists each tier x device with measured accuracy |
| 3 | Repo + README | root `README.md` | Clean machine to first plan in under 15 minutes; one command per capture, e.g. `scan run <capture_dir>` |
| 4 | Reproduction bundle | `repro/`, `make reproduce` | One command regenerates every number in every report from raw data; model weights fetched by script; cached outputs replay deterministically and live path also runs |
| 5 | Benchmark report | `reports/benchmark.md` (generated) | Gates at all three tiers, repeatability table, head-to-head table, runtime per capture |
| 6 | Fix loop bundle | `fixloop/` | Declaration page, before run, after run, readable diff, all regenerable |
| 7 | Technical report | `reports/technical_report.pdf` | 6 pages max: architecture, tiers + device matrix, drift, error budget, calibration, fix loop story, failure modes |
| 8 | Raw benchmark data | `data/raw/` (or download script) | Sensor logs, photos, video, ground-truth sheets, competitor app exports |

Two cross-cutting requirements sit outside the numbered list but are scored: a commit history that grows with the work, and coverage of mirrors, glass, wet-look surfaces and low light.

## Key decisions

**Capture route: choose Route 2 (stock protocol).** It removes the Apple developer account, TestFlight review and install-time risk, and puts your effort into the pipeline where 85% of the marks are. Route 1 only wins if you need RoomPlan's structured output live, and you can replicate most of that offline.

| Decision | Recommendation | Why | Fallback |
| --- | --- | --- | --- |
| LiDAR capture app | [Stray Scanner](https://apps.apple.com/us/app/stray-scanner/id1557051662) (free, open source) | Exports RGB video, LiDAR depth, confidence maps, per-frame ARKit poses, intrinsics and IMU to a folder you can zip from the Files app | Record3D, or the community fork updated for iOS 27 ([stray-scanner-izuba](https://github.com/globalfranck/stray-scanner-izuba)); the App Store build was last updated in 2022, so test it on a current iOS in hour 1 |
| Video tier | Native iPhone Camera, 4K 30fps, locked exposure | Zero install, any iPhone 15+; the protocol is just how to walk | None needed |
| Photo tier | Native Camera, 1x lens, one folder per room | EXIF gives focal length, so you know intrinsics even with no poses | None needed |
| Multi-view geometry model | [MapAnything](https://github.com/facebookresearch/map-anything) (Apache-licensed variant available) | Feed-forward metric reconstruction from images plus optional intrinsics, depth or poses; one model serves photo and video tiers | VGGT or Depth Anything 3, both callable through the same repo |
| Depth refinement (LiDAR tier) | Prompt Depth Anything, using LiDAR depth as the prompt | Upsamples the 256x192 LiDAR map to sharp edges, which helps opening widths; its demo is built around Stray Scanner captures | Raw LiDAR depth with TSDF fusion |
| Damage detection | Zero-shot open-vocabulary segmentation (e.g. Grounded SAM 2) | Few labels needed; gives masks you project onto surfaces for metric extent | VLM-based classification with SAM masks |
| Head-to-head competitor | Polycam, Floorplan Mode | The brief names it; claims about half-inch tolerance make it a credible bar | magicplan |

On Polycam: the free tier exports GLTF only, while floor plan exports sit on the Business plan, which has a [free 7-day trial](https://poly.cam/blog/how-to-create-floor-plans-using-your-iphone-in-10-minutes). Start the trial only on the day you capture the two head-to-head rooms, and export everything that day ([plan details](https://poly.cam/blog/the-best-floor-plan-app-for-iphone-a-professionals-guide)).

One model choice to defend at the defense: why feed-forward reconstruction over classic SfM (COLMAP). Answer: 2 to 8 photos per room with low overlap break SfM; feed-forward models degrade gracefully and give metric scale. In 24 hours, skip a COLMAP baseline and name it as future work in the report.

## System architecture

Build one pipeline with three thin front-ends, so every improvement to the shared core lifts all tiers at once and the output contract is identical by construction.

&#91;embedded content: pipeline architecture · 3 tiers, 1 shared core\]

Only the front-ends know which tier they are; everything from the point cloud down is shared, and the calibration layer is the one place that sets interval widths.

**Output contract (sketch).** Publish this as `scan/schema/output.schema.json` and validate every run against it. Every measurement is an object with a value and an interval, never a bare number.

```json
{
  "capture_id": "home1_multiroom_lidar_t1",
  "tier": "lidar | video | photo",
  "device": "iPhone 15 Pro",
  "pipeline_version": "git sha",
  "rooms": [{
    "id": "R1", "polygon_m": [[0, 0], [4.12, 0], [4.12, 3.05], [0, 3.05]],
    "walls": [{"id": "R1W1", "length_m": {"value": 4.12, "ci90": [4.10, 4.14]}}],
    "ceiling_height_m": {"value": 2.71, "ci90": [2.70, 2.72]},
    "floor_area_m2": {"value": 12.57, "ci90": [12.4, 12.7]},
    "openings": [{"id": "R1O1", "type": "door", "wall_id": "R1W2", "width_m": {"value": 0.81, "ci90": [0.80, 0.82]}}]
  }],
  "stitched": {"adjacency": [{"rooms": ["R1", "H1"], "via": "R1O1"}], "footprint_m2": {"value": 48.2, "ci90": [47.1, 49.3]}},
  "damage": [{"id": "D1", "class": "water_stain", "surface_id": "R2W3", "area_m2": {"value": 0.34, "ci90": [0.28, 0.41]}}],
  "concealed_flags": [{"surface_id": "R2W3", "rule": "stain_within_30cm_of_floor", "evidence": ["D1"]}],
  "scope": [{"surface_id": "R2W3", "action": "replace_drywall", "quantity": {"value": 1.2, "ci90": [1.0, 1.4]}, "unit": "m2"}]
}
```

The values above are illustrative placeholders, not targets.

**Repo layout.**

```text
scan/                 # the package; CLI: scan run <capture_dir>
  io/                 # loaders: stray scanner, video, photo folders
  tiers/              # lidar.py, video.py, photo.py front-ends
  geometry/           # planes, walls, openings, polygon, ceiling
  stitch/             # pose graph, loop closure, photo layout solver
  damage/             # detect, lift to surfaces, rules, scope
  calibrate/          # analytic, bootstrap, conformal intervals
  render/             # 2D plan SVG/PNG
  schema/             # output.schema.json
eval/                 # matching, gates, report generation
capture/              # PROTOCOL.pdf, DEVICE_MATRIX.md
data/                 # fetch script; raw/, gt/, exports/
fixloop/              # DECLARATION.md, before/after runs, diff
reports/              # generated benchmark + technical report
docker/               # Dockerfile, fetch_weights.sh
Makefile              # benchmark, reproduce, fixloop targets
README.md  COMPLIANCE.md
```

Ship a Docker image with weights fetched at build time, so the 15-minute README promise holds on a clean machine with or without a GPU.

## Tier design

All three tiers converge on the same intermediate: a gravity-aligned, metric point cloud per room with a per-point uncertainty. After that, one shared geometry core extracts planes, walls, openings and the room polygon. The tiers differ only in how they produce that point cloud and how wide its uncertainty is.

### LiDAR tier (Pro devices)

1. Load Stray Scanner frames; drop depth pixels with low confidence and beyond 4 m (LiDAR noise grows with range).
2. Refine depth with Prompt Depth Anything so door jambs and window edges are sharp.
3. Fuse into a TSDF volume per room (Open3D, 1 cm voxels) and extract the mesh and point cloud.
4. Align to gravity using ARKit's gravity vector, then snap to the dominant Manhattan axes.
5. Fit floor and ceiling planes with RANSAC; ceiling height is the plane-to-plane distance from thousands of points, which is how you reach under 1.5 cm.
6. Fit wall planes, project to the floor, intersect neighbours to get corners and the room polygon.
7. Detect openings as gaps in wall planes that reach the floor (doors) or sit above a sill (windows); measure width between jamb planes, not between mesh holes.

### Video tier (any iPhone 15+)

1. Sample keyframes by sharpness and parallax (roughly 2 to 4 per second of walking).
2. Look up intrinsics from EXIF and a per-model table, so the model is not guessing focal length.
3. Run MapAnything in overlapping chunks for depth, poses and metric scale; align chunks on their shared frames.
4. Feed the result into the shared geometry core with a wider depth-noise model.
5. Only if time remains: run COLMAP on the same keyframes as a baseline.

### Photo tier (2 to 8 stills per room)

Per room, run MapAnything with EXIF intrinsics on that folder, then the geometry core. Single rooms are the easy part; the gate that sinks most entries is stitching rooms that share no poses.

Stitching approach:

1. **Design the protocol for it.** Require one photo per doorway, taken from inside each room looking through the opening. Every connection is then seen from both sides.
2. **Match openings across rooms** using cross-folder image matching (MASt3R-style) on those doorway photos, plus opening width and height as a consistency check.
3. **Solve the layout** as a small optimisation: each room is a rigid polygon; matched openings must coincide; shared walls sit one wall-thickness apart (fit, not assumed); rooms may not overlap; axes stay Manhattan-aligned where the evidence supports it.
4. **Propagate uncertainty** from each room's scale estimate into the stitched footprint interval.

The connector (hallway) gets its own folder and is stitched like any room.

### Drift handling (multi-room capture)

ARKit and video poses drift over a long walk, and "poses used as-is" fails the gate automatically. Plan: split the walk into per-room fragments, register fragments pairwise on overlapping frames, add loop-closure edges when the walk returns to the start, add plane constraints (shared floor plane, shared walls parallel), and optimise the pose graph (GTSAM or Open3D's pose-graph optimiser).

The protocol must make the capture end where it started so a loop exists. The ablation reports footprint area, loop-closure error in cm, and shared-wall misalignment, each with correction on and off, plus overlaid plan renders.

### Calibrated intervals

Every number gets an interval built in three layers:

1. **Analytic:** plane-fit residuals and a per-tier depth-noise model give a raw standard error.
2. **Resampling:** rerun on bootstrap subsets of frames or photos; the spread captures model instability.
3. **Conformal scaling:** on your benchmark, scale intervals so a stated share (e.g. 90%) of ground-truth values falls inside, using leave-one-room-out so you never calibrate on the room you score.

Report coverage and median interval width per tier and per measurement type. Coverage near the target with narrow width is the goal; coverage far below target is the "confident garbage" that caps your score.

### Device matrix and honest accuracy targets

| Tier | Hardware | Wall length target | Ceiling height | Openings | Notes |
| --- | --- | --- | --- | --- | --- |
| LiDAR | iPhone 15 Pro and later Pro models, iPad Pro | about 1% | under 1.5 cm | under 2 cm on 85%+ | Fill in measured numbers from the benchmark |
| Video | iPhone 15 and later, any model | under 3% | about 3 to 5 cm | about 3 to 5 cm | Degrades in low light and on blank walls |
| Photo | iPhone 15 and later, any model | under 8% | about 5 to 10 cm | about 5 to 10 cm | Widest intervals; depends on doorway photos for stitching |

The targets above are planning estimates. Replace them with measured numbers before submission, since the matrix is scored on honesty.

## Damage detection and scope

Damage carries no numeric gate, but it is in the output contract and the compliance matrix, so it must work end to end and be measurable. Keep it simple, rule-based where possible, and traceable.

1. **Detect in 2D.** On keyframes, run open-vocabulary segmentation for a fixed class list: water stain, mould, crack, hole, peeling paint, missing tile. Skip fine-tuning; tune the prompt list and a confidence threshold on your staged examples instead.
2. **Lift to 3D.** Project each mask onto the surface it lies on (wall, floor, ceiling plane) using depth and pose; merge detections across frames.
3. **Measure.** Report each region's class, host surface ID, area in m², bounding extent in cm, and an interval from mask-boundary uncertainty.
4. **Flag concealed damage with explicit rules.** Each flag names the rule that fired, for example: a ceiling stain directly below a bathroom on the stitched plan suggests a hidden leak; a wall stain within 30 cm of the floor suggests wicking; a stain near a window corner suggests a failed seal.
5. **Generate scope line items** keyed to surface IDs: "Wall W3, Room 2: cut out and replace drywall, 1.2 m², repaint 9.8 m²". Quantities come from your geometry, so their intervals propagate.

For the benchmark, stage two damage classes in the furnished room (for instance a tea-stain "water mark" and a drawn or taped "crack"), tape-measure each region, and report detection recall and area error.

## Benchmark set

Build the benchmark in hours 1 to 3.5, before writing pipeline code, because every later decision and the fix loop depend on it. Use a home or office with at least three rooms plus a hallway, ideally including one mirror, one glass door or window, and one dim room, which also covers the hard-surface requirement.

| Capture set | Tiers | Repeats | Purpose |
| --- | --- | --- | --- |
| Multi-room: 3+ rooms + connector, one continuous walk | LiDAR, video, photo (one folder per room) | 1 each | Stitching, adjacency, drift ablation, photo-tier stitch gate |
| Furnished room with staged damage (2 classes) | LiDAR, video, photo | 1 each | Damage detection, clutter robustness |
| Repeatability room (one of the above) | All tiers | 2+ per tier | Repeatability and ceiling-spread gates |
| Two head-to-head rooms | LiDAR + Polycam | 1 each | Part 3 comparison |
| Hard-surface room (mirror, glass, low light) | All tiers | 1 each | Failure-mode evidence |

Capture more repeats than the minimum (three per tier on one room is cheap), and hold out one room you never tune on. That room stands in for the walk-in test, since 24 hours leaves no time for a second property.

**Ground truth protocol.** Use a laser measurer (about 1.5 to 2 mm accuracy class) and record into a fixed spreadsheet template:

- Every wall length, measured at floor level and at 1 m height, corner to corner
- Ceiling height at the centre and at two corners of each room
- Every opening's width (jamb to jamb) and height, and window sill height
- Room diagonals, which let you check the room polygon's shape
- Damage region extents with a tape
- Photos of each measurement point, so a disputed number can be checked

Measure each value twice and record both; the spread is your ground-truth uncertainty and belongs in the report.

**Data layout.** `data/raw/<property>/<set>/<tier>/<take>/` for sensor files, `data/gt/<property>.csv` for measurements, `data/exports/polycam/` for competitor files. Keep raw data immutable; everything downstream is generated.

## Evaluation harness and gates

One command, `make benchmark`, runs every capture, matches predictions to ground truth, scores every gate, and writes `reports/benchmark.md` plus a JSON of all numbers. Every report figure must come from this output, never typed by hand.

**Matching.** Predicted walls and openings must be matched to ground-truth items before scoring. Store a small, committed mapping file per property (predicted ID to ground-truth ID) generated by nearest-geometry matching with a manual review step. An unmatched ground-truth opening is a miss; an unmatched predicted opening is a phantom, which also counts as a miss.

| Gate | How the harness computes it | Pass |
| --- | --- | --- |
| Opening widths | Abs. width error per GT opening; missed and phantom openings count as failures | 2 cm or less on 85%+ of openings |
| Ceiling height | Abs. error per room; plus max minus min across repeat captures | Error 1.5 cm or less; spread 1 cm or less; report bias vs noise |
| Repeatability | Per matched wall across two same-tier captures | Within 1 cm or 0.5% per wall |
| Drift accountability | Footprint area and loop error with correction on vs off | Ablation present with both renders |
| Photo-tier stitch | Adjacency graph vs GT; polygon overlap area; footprint error | Correct adjacency, zero overlap, within 8% |
| Wall lengths (photo / video) | Relative error per wall | Within 8% (photo), 3% (video) |
| Calibration (all tiers) | Share of GT values inside the interval; median interval width | Coverage near the stated target |
| Runtime | Wall-clock per capture per tier on a stated machine | Reported |

For ceiling height, the brief asks you to say which failure you have. Decompose error into bias (mean signed error across repeats) and noise (spread across repeats); a constant bias usually points to a plane-fitting or depth-scale issue, while large spread points to capture or pose instability.

**Head-to-head (Part 3).** On two rooms, list every shared dimension (each wall, ceiling height, each opening, floor area). Table columns: dimension, ground truth, your LiDAR value and error, Polycam value and error, winner. Count wins plus ties against the 70% bar, and define "tie" up front (e.g. errors within 0.5 cm of each other) so it cannot look tuned after the fact. Name the app version and commit its export files.

## Fix loop (25%)

The fix loop rewards a correct diagnosis, a shipped fix, and a prediction that comes true. Treat it as a mini research cycle with its own tagged commits.

1. **Freeze a baseline.** Tag the repo `fixloop-before` once the full benchmark runs end to end. The before run is regenerated from this tag.
2. **Pick the gate honestly.** Choose the single worst-performing gate by distance from its threshold, not the one easiest to fix. Likely candidates are opening widths at the LiDAR tier or the photo-tier stitch.
3. **Write the declaration first** (`fixloop/DECLARATION.md`, one page): the failing number, the root-cause hypothesis, the evidence, the fix, and the predicted number. Commit it before writing fix code, so the history proves the prediction came first.
4. **Gather evidence with a targeted experiment.** For example, if opening widths fail, plot error vs distance-from-camera and vs depth confidence. If error grows with range and jambs show smeared depth, the cause is LiDAR edge blur, and the fix is depth refinement plus jamb-plane fitting.
5. **Ship the fix** in a small, readable diff, then tag `fixloop-after`.
6. **Rerun and compare.** `make fixloop` regenerates both runs and writes a before/after table and the diff summary.
7. **Write the post-mortem.** If you hit the gate, show why. If not, explain what remained; if the prediction was badly off, say so plainly, since honesty still earns marks.

Two practical rules: keep the fix narrow so the root cause is obvious from the diff, and check that the fix does not break other gates by rerunning the whole benchmark, not just the target gate.

## 24-hour timeline

With 24 hours, this is a triage plan: capture all data first, make the LiDAR tier solid, get video and photo to valid output with honest intervals, and protect 2.5 hours for the fix loop. Anything that does not move the walk-in test, the fix loop or a hard gate is cut or minimised. There is no slack; any overrun comes out of the damage block first.

&#91;embedded content: 4-week roadmap · phases and gates\]

**What to keep and what to cut.**

| Area | Keep | Cut or minimise |
| --- | --- | --- |
| Capture route | Route 2: Stray Scanner + native Camera | Any custom iOS app |
| LiDAR tier | TSDF, planes, walls, openings, ceiling | Prompt Depth Anything, unless openings turn out to be the worst gate |
| Video tier | MapAnything on keyframes | COLMAP baseline |
| Photo tier | Doorway photos + layout solve (a hard gate) | Anything beyond pairwise matching on doorway photos |
| Damage | Zero-shot detection, three rules, scope items | Fine-tuned classifier |
| Benchmark | One property, all tiers, 3 repeats on one room, one held-out room | A second property |
| Reproducibility | Lockfile, weights fetch script, `make reproduce` | Docker image, if it costs more than 30 minutes |
| Technical report | 3 to 4 tight pages | Polish beyond that |

Use AI coding tools hard for boilerplate (loaders, schema, report generation, plotting). Make the geometry, drift and calibration choices yourself and log why, because you defend them with tools closed.

A natural fix-loop candidate: build openings first on raw LiDAR depth. If opening widths turn out to be the worst gate, LiDAR edge blur is a single, testable cause, and depth refinement is a fix you can predict and ship in 2.5 hours. Declare it only if it really is the worst gate.

**Commit hygiene (Part 5).** Commit every 20 to 40 minutes with messages that say why, including failed experiments you later revert. Over 24 hours that is 40-plus commits, which reads as real work because it is. Tag `benchmark-v1`, `lidar-e2e`, `fixloop-before`, `fixloop-after` and `submission`. Keep `NOTES.md` as a running decision log; it doubles as your defense script.

## Risks, hard surfaces and defense prep

| Risk | Effect | Mitigation |
| --- | --- | --- |
| Mirrors | Phantom rooms and walls behind the mirror | Detect mirror regions (segmentation + depth that lies behind a fitted wall plane); drop points behind the wall; protocol says avoid pointing straight at mirrors |
| Glass doors and windows | LiDAR passes through or returns noise; openings missed | Treat a planar region with low confidence inside a wall as a window candidate; fall back on image edges for the frame |
| Wet-look and glossy floors | Spurious depth, reflected ceiling | Robust plane fitting with confidence weighting; floor plane prior from gravity |
| Low light | Video blur, failed matching, LiDAR mostly fine | Protocol: turn on all lights; reject blurred frames; widen intervals when frame quality is low |
| Stray Scanner breaks on a new iOS | LiDAR tier dead at defense | Test in hour 1; keep Record3D path and a converter for both formats |
| Pretrained model misreads scale | Systematic photo-tier bias | Bootstrap across model variants; conformal scaling; report bias in the matrix |
| Walk-in machine has no GPU or slow network | Pipeline runs too slowly or cannot fetch weights | Pre-fetched weights in a Docker image or volume; CPU fallback path timed in advance |
| Their capture is worse than yours | Gate failures live | Rehearse with a friend following the protocol cold on a new property |

**Walk-in test rehearsal.** Once before submission (hours 20.5 to 22), hand the protocol page to someone non-technical, let them capture an unfamiliar space at a tier you did not choose, and run the pipeline cold while you laser-measure. Fix every protocol ambiguity they hit; the brief warns that ambiguity shows up in the capture.

**Defense preparation.** The defense is tools closed, so you must own every choice. Prepare a one-page list of design decisions with the alternative you rejected and the evidence for the choice: Route 2 vs an app, MapAnything vs COLMAP, TSDF voxel size, conformal target, drift method, tie definition. Be ready to explain any line of the fix-loop diff and any number in the report.
