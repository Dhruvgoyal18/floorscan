# Decision log

Running log of design decisions and why. Newest at the bottom. This doubles as the defense script.

## 2026-10-03

- **Route 2 over a custom app.** No Apple developer account / TestFlight review risk; the effort goes into the pipeline. Stray Scanner records exactly what we need: RGB, 256x192 LiDAR depth (uint16 mm PNG), confidence (0/1/2), per-frame ARKit pose, intrinsics, IMU.
- **Verified the Stray Scanner pose convention from its source** (`OdometryEncoder.swift`): it writes `q_WA * q_AC` with `q_AC` a 180° rotation about x, so `odometry.csv` is cam-to-world in the **OpenCV** camera convention (x right, y down, z forward) in the ARKit world (y up, gravity -y). Newer builds append per-frame `fx, fy, cx, cy` (at RGB resolution). Loader handles both header variants. We convert world to z-up: `(x, y, z)_arkit -> (x, -z, y)`.
- **Package layout follows the plan**, CLI is `scan run <capture_dir>`.
- **Geometry in numpy/scipy/shapely only** — no Open3D. Keeps install small (walk-in on a clean machine) and the LiDAR tier free of learned weights, so it runs with no GPU and no network.
- **Learned backends behind one seam** (`scan/backends`): MapAnything (primary), Depth-Anything-V2-metric + PnP odometry (light), oracle (synthetic only). HuggingFace is unreachable from the dev sandbox, so learned adapters are written against the upstream API (verified by reading the MapAnything source, `model.infer` / `preprocess_inputs`) and validated on the capture machine; everything downstream is validated here with the oracle backend.
- **Synthetic benchmark first.** No real captures exist yet. A ray-cast renderer writes captures in the exact on-disk formats (Stray Scanner folder, video file, photo folders) plus GT YAML from the scene parameters. It injects realistic failure sources: depth noise growing with range, flying pixels at depth edges, low confidence at edges/range, ARKit-like drift, glass and a mirror. Real captures replace it later with no code changes.
