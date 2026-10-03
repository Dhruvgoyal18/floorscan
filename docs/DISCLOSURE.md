# Disclosure: pretrained models, datasets, external tools

Everything runs locally; nothing calls infrastructure we operate.

| Component | Used for | License | How obtained |
|---|---|---|---|
| Stray Scanner (iOS app, Stray Robots) | LiDAR capture route | MIT (app source on GitHub) | App Store |
| iPhone Camera app | video and photo capture | - | built in |
| MapAnything, `facebook/map-anything-apache` (Meta) | video/photo tiers: metric depth, poses, intrinsics | Apache-2.0 weights variant | `make setup-models` (pip from GitHub + HF download) |
| Depth-Anything-V2 Metric Indoor Small (`depth-anything/...-hf`) | light fallback backend (per-view metric depth) | Apache-2.0 (Small) | `scripts/fetch_weights.py` |
| OpenCV SIFT + PnP-RANSAC | poses for the fallback backend | Apache-2.0 | pip |
| numpy, scipy, shapely, matplotlib, pillow, pillow-heif, jsonschema | geometry, rendering, IO | BSD/MIT-style | pip |
| numba (dev only) | fast synthetic rendering | BSD | pip |

No model was trained or fine-tuned. Damage detection uses no learned weights (geometric surface restriction +
colour statistics); thresholds are listed in `scan/damage/detect.py::TH`.

Datasets: none. The synthetic benchmark is rendered by `scan/synth` from parameters in `scan/synth/presets.py`.

Development tooling: AI coding assistants were used throughout (as the brief allows); design decisions are logged
in `NOTES.md`.

Constraint not met in the development sandbox: Hugging Face downloads were blocked there, so the learned
backends were written against the upstream APIs (MapAnything's `infer`/`preprocess_inputs` read from its source)
but executed only on the capture machine. All synthetic video/photo numbers use the `oracle` backend and are
labelled as such.
