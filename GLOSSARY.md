# Glossary

- **Capture**: one folder produced by following the protocol at one tier (a Stray Scanner export, a video file, or a folder of per-room photo folders).
- **Tier**: `lidar`, `video` or `photo`. Determines the front-end, the noise model and the calibration factors. Nothing below the front-end branches on tier except via those.
- **Observations**: the tier-agnostic intermediate — metric, gravity-aligned (z up) points with per-point sigma and the camera centre that observed them, plus the frames (image, depth, pose, intrinsics) used for damage. A capture yields one `Observations` per **group**.
- **Group**: a set of observations sharing one world frame. LiDAR and video captures are one group; the photo tier has one group per room folder.
- **Fragment**: a contiguous run of frames used as a pose-graph node for drift correction.
- **Wall line**: an axis-aligned vertical plane recovered from wall-facing points, with a facing direction (towards the cameras that saw it).
- **Arrangement cell**: a rectangle between consecutive wall-line coordinates; rooms are unions of interior cells.
- **Header band**: heights above standard door height (floor + 2.15 m) up to the ceiling. Wall evidence in this band separates rooms even across doorways.
- **See-through evidence**: a camera→point ray that crosses a wall plane inside the wall's extent; proves the wall is open there.
- **Estimate**: value + standard error before calibration. **Measure**: value + ci90 after calibration (the only thing that leaves the pipeline).
- **Calibration factor**: per (tier, kind) multiplier on the standard error, fitted conformally on the benchmark.
- **Kind**: measurement family: `wall_length`, `ceiling_height`, `opening_width`, `opening_height`, `floor_area`, `footprint`, `damage_area`.
- **Gate**: a pass/fail criterion from the brief, computed by `eval/`.
