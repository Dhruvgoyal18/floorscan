# Compliance matrix

Requirement → where it lives → artifact → status. Status says **pass / partial / fail / pending** with the number.
"Synthetic" numbers come from `reports/synthetic/benchmark.md` (rendered captures; video/photo through the oracle
stand-in for the learned model). "Pending" = needs the real captures, which require the physical property, a Pro
iPhone, a laser measurer and the competitor app; every tool to produce them is in place.

| # | Requirement (brief) | File path | Artifact | Status |
|---|---|---|---|---|
| 1 | Choose a capture route; stock tool named; one-page protocol: install, how to walk, how long, what to avoid, hand-off | `capture/PROTOCOL.md` | Route 2: Stray Scanner (LiDAR) + built-in Camera (video, photo) | **pass** |
| 2 | Three input tiers, all mandatory, same output contract | `scan/tiers/`, `scan/io/`, `scan/pipeline.py` | auto-detected tier, one JSON schema | **pass** (synthetic, all tiers end to end); learned backend live path **pending** a run on a machine with the weights (`make setup-models`) |
| 3 | Photo tier: 2-8 stills per room, one folder per room, same stitched whole-property plan | `scan/stitch/layout.py`, `scan/tiers/rgb.py` | doorway-matching stitch | **partial**: home PASS, holdout FAIL (1 one-sided doorway) |
| 4 | Video tier: handheld walkthrough, any iPhone 15+ | `scan/io/media.py`, `scan/tiers/rgb.py` | keyframes → backend → drift graph | **pass** (synthetic) |
| 5 | LiDAR tier: depth, poses, intrinsics on Pro devices | `scan/io/stray.py`, `scan/tiers/lidar.py` | Stray format verified against the app's source | **pass** (synthetic) |
| 6 | Intervals widen honestly as sensor data thins | `scan/calibrate/intervals.py`, `eval/calibrate.py` | per-tier floors + LORO conformal factors | **pass**: LORO coverage 92-95 % on walls at every tier; median ci90 width wall 2.6 cm (LiDAR) / 9 cm (video) / 65 cm (photo) |
| 7 | Device matrix: tier x hardware x honest accuracy | `capture/DEVICE_MATRIX.md` (generated) | table | **pass** (synthetic columns), real columns **pending** |
| 8 | Per-room plan: walls, ceiling height, floor area, openings | `scan/geometry/`, schema | `rooms[]` | **pass** |
| 9 | Stitched multi-room plan with correct adjacency | `scan/geometry/core.py::_adjacency`, `scan/stitch/` | `stitched.adjacency` | **partial**: correct on 4/6 multi-room captures (LiDAR 2/2, video 1/2, photo 1/2) |
| 10 | Per-surface damage regions with class and metric extent | `scan/damage/detect.py` | `damage[]` | **partial**: LiDAR bedroom 3/3 takes find both regions with correct class, stain area +3-4 %; video/photo find both but misclassify or over-size (+10-28 %); 1-5 false positives per multi-room capture |
| 11 | Concealed-damage flags with the rule that fired | `scan/damage/rules.py` | `concealed_flags[]` with rule + rationale + evidence | **pass** |
| 12 | Scope line items keyed to surfaces | `scan/damage/rules.py` | `scope[]` with intervals | **pass** |
| 13 | Confidence interval on every measurement | schema `$defs/measure` | `{value, ci90}` everywhere (incl. offsets, damage height, wall thickness) | **pass** |
| 14 | One command per capture | `scan/cli.py` | `scan run <capture>` | **pass** |
| 15 | JSON to the published schema | `scan/schema/output.schema.json` | validated on every run (`jsonschema`) | **pass** |
| 16 | Rendered plan | `scan/render/plan.py` | `plan.svg`, `plan.png` | **pass** |
| 17 | Benchmark: multi-room capture, >= 3 rooms + connector | `scripts/make_synthetic_benchmark.py` | home (4 rooms + hall), holdout (4 + hall) | **pass** synthetic / **pending** real |
| 18 | Furnished room with staged damage, two classes | same | bedroom: water stain + crack, bed/wardrobe/desk | **pass** synthetic / **pending** real |
| 19 | Same rooms at all three tiers, multi-room included; photo as per-room folders | `data/benchmark_synthetic.yaml` | 3 tiers x (home, holdout) | **pass** synthetic / **pending** real |
| 20 | At least one room captured twice at the same tier | same | bedroom x3 LiDAR, x3 photo, x2 video | **pass** |
| 21 | Laser/tape ground truth on everything; raw data submitted | `data/gt/TEMPLATE.yaml`, `data/gt/synthetic/` | GT exact for synthetic; template for laser | **pending** real |
| 22 | Gate: opening widths <= 2 cm on >= 85 %, misses and phantoms count | `eval/gates.py::opening_gate` | report | **fail**: LiDAR 22/37 = 59.5 % (median 0.8 cm, p90 5.0 cm) |
| 23 | Gate: ceiling <= 1.5 cm; repeat spread <= 1 cm; say biased vs unrepeatable | `eval/gates.py::ceiling_gate` | report, per-room bias/spread | **fail (marginal)**: LiDAR worst 0.55 cm, spread 1.01 cm → "unrepeatable" |
| 24 | Gate: repeatability <= 1 cm or 0.5 % per wall | `eval/gates.py::repeatability` | repeatability table | **fail (marginal)**: LiDAR 11/12 wall pairs, worst 1.78 cm |
| 25 | Gate: drift accountability, ablation on/off, never poses as-is | `scan/stitch/drift.py`, `--no-drift-correction` | ablation table + both renders | **pass**: LiDAR footprint +0.11 % on vs -4.49 % off; loop error 24 cm → 0.2 cm |
| 26 | Gate: photo-tier whole-property stitch | `eval/gates.py::photo_stitch_gate` | report | **fail**: home PASS (+1.5 %), holdout adjacency 1 doorway missing (-7.5 %) |
| 27 | Photo walls +-8 %, video +-3 %, calibration scored at every tier | `eval/gates.py::wall_gate`, `coverage` | report | **pass**: photo 96 %, video 100 %, LiDAR 100 % within tolerance; coverage reported per tier/kind |
| 28 | Head-to-head vs consumer app on 2 rooms, beat or tie >= 70 % | `eval/head_to_head.py`, `data/exports/` | generated table | **pending** (tool ready; needs the app export; tie rule fixed in advance) |
| 29 | Fix loop: worst gate, root cause + evidence, fix, predicted number | `fixloop/DECLARATION.md` (committed before the fix) | declaration | **pass** |
| 30 | Fix shipped; before/after regenerable; readable diff | `fixloop/run_fixloop.sh`, tags `fixloop-before`/`-after`, `fixloop/fix.diff` (48 lines), `fixloop/RESULT.md` | before/after + post-mortem | **partial**: root causes confirmed, footprint 13.9 % → 7.5 %, overlap 12.4 → 0 m², gate still FAIL (third defect, explained) |
| 31 | Commit as you work | `git log` | 20 commits on main + fix-loop baseline branch, each saying why; tags `benchmark-v1`, `lidar-e2e`, `fixloop-*` | **pass** |
| 32 | README to running on a fresh capture < 15 min on a clean machine | `README.md`, `Makefile`, `docker/Dockerfile` | clean clone → first LiDAR plan in 50 s (measured) | **pass** (LiDAR); video/photo add the weights download |
| 33 | Reproduction bundle: every number from raw inputs; cached model outputs replay deterministically and live path runs | `make reproduce`, `scan/backends/base.py::CachedBackend` | synthetic captures are deterministic from seeds; noise seeded from content | **pass** synthetic / live learned path **pending** |
| 34 | Benchmark report: gates at all three tiers, repeatability, head-to-head, timing | `reports/synthetic/benchmark.md` (generated) | report | **pass** (head-to-head section pending) |
| 35 | Technical report <= 6 pages | `reports/technical_report.pdf` | PDF | **pass** |
| 36 | Raw benchmark data | `data/raw/` (rendered by `make synthetic`) | sensor files in real formats | **pass** synthetic / **pending** real |
| 37 | Pretrained models disclosed; nothing calls our infrastructure | `docs/DISCLOSURE.md` | table | **pass** |
| 38 | Weights fetched by script | `scripts/fetch_weights.py`, `make weights` | script | **pass** |
| 39 | Mirrors, glass, wet-look surfaces, low light covered | `scan/geometry/core.py` (mirror), openings (no-return rays), `capture/PROTOCOL.md`, synthetic bath | mirror test + bath mirror detected, glass windows found, dim bath | **partial**: wet-look/glossy floors not simulated |
| 40 | Walk-in: all three tiers ready to run cold | CLI + protocol | LiDAR needs no weights; video/photo need `make setup-models` once | **partial** until the learned backend is exercised on real captures |
