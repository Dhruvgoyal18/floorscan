"""Write synthetic captures in the exact on-disk formats the real capture route produces.

* LiDAR  -> Stray Scanner folder: rgb.mp4, depth/NNNNNN.png (uint16 mm, 256x192), confidence/NNNNNN.png
            (0/1/2), odometry.csv (ARKit world y-up, OpenCV camera, cam-to-world, with drift), camera_matrix.csv
* video  -> walkthrough.mp4 (+ ``_synthetic/`` sidecar the *oracle* backend reads; real videos have none)
* photo  -> NN_<room>/IMG_NNNN.jpg with iPhone-like EXIF (+ ``_synthetic/`` sidecar)

Failure sources are injected on purpose: range-dependent depth noise, LiDAR edge smear ("flying
pixels") at depth discontinuities, low confidence at range/edges, odometry drift, no returns beyond
5 m, mirror phantoms, glass.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from .render import render
from .scene import SIDES, PropertySpec, Scene, _opening_on_room_wall, build_scene

# z-up world <-> ARKit y-up world:  p_zup = M @ p_ar
M_AR2ZUP = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])

LIDAR_W, LIDAR_H = 256, 192
RGB_W, RGB_H = 640, 480
IPHONE_MAIN_HFOV = 67.0
IPHONE_UW_35MM = 13.0


def K_from_hfov(w, h, hfov_deg):
    f = (w / 2) / np.tan(np.radians(hfov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def look_at(eye, yaw, pitch, roll=0.0):
    """OpenCV camera at ``eye`` looking along (yaw, pitch) in z-up world. Returns T_wc."""
    f = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    dn = np.cross(f, r)
    if roll:
        c, s = np.cos(roll), np.sin(roll)
        r, dn = c * r + s * dn, -s * r + c * dn
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = r, dn, f, eye
    return T


# ---------------------------------------------------------------- trajectories

def _door_between(spec: PropertySpec, a: str, b: str):
    for op in spec.openings:
        ra, rb = spec.room(a), spec.room(b)
        ha = [(s, _opening_on_room_wall(spec, op, ra, s)) for s in SIDES]
        hb = [(s, _opening_on_room_wall(spec, op, rb, s)) for s in SIDES]
        ha = [(s, u) for s, u in ha if u is not None]
        hb = [(s, u) for s, u in hb if u is not None]
        if ha and hb:
            s, u = ha[0]
            p0, p1 = ra.wall_endpoints(s)
            d = (p1 - p0) / np.linalg.norm(p1 - p0)
            n_in = np.array([-d[1], d[0]])
            c = p0 + d * (u + op.width / 2) - n_in * spec.wall_t / 2
            return c, n_in  # door centre (mid-wall), normal pointing into a
    raise ValueError(f"no door between {a} and {b}")


def _sweep_spots(r):
    cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
    w, d = r.x1 - r.x0, r.y1 - r.y0
    if max(w, d) > 3.6:
        if w >= d:
            return [np.array([r.x0 + w / 3, cy]), np.array([r.x0 + 2 * w / 3, cy])]
        return [np.array([cx, r.y0 + d / 3]), np.array([cx, r.y0 + 2 * d / 3])]
    return [np.array([cx, cy])]


def walk_trajectory(spec: PropertySpec, walk: list[str], rng: np.random.Generator, step: float = 0.25,
                    yaw_step_deg: float = 12.0, sweep: str = "full"):
    """Local-frame camera poses for a walk. ``sweep``: full (two revolutions, up+down) or light."""
    poses = []
    visited = set()
    pos = _sweep_spots(spec.room(walk[0]))[0]

    def cam(p2, yaw, pitch):
        h = 1.45 + rng.normal(0, 0.03)
        return look_at(np.array([p2[0], p2[1], h]), yaw + rng.normal(0, 0.01), pitch + rng.normal(0, 0.01),
                       rng.normal(0, 0.015))

    def walk_to(p_from, p_to):
        seg = p_to - p_from
        L = np.linalg.norm(seg)
        if L < 1e-6:
            return
        yaw = np.arctan2(seg[1], seg[0])
        for s in np.arange(step, L + 1e-9, step):
            poses.append(cam(p_from + seg * (s / L), yaw, np.radians(-8)))

    def do_sweep(p, room):
        pitches = (np.radians(28), np.radians(-30)) if sweep == "full" else (np.radians(0),)
        yaw0 = rng.uniform(0, 2 * np.pi)
        for pitch in pitches:
            for a in np.arange(0, 360, yaw_step_deg):
                poses.append(cam(p, yaw0 + np.radians(a), pitch))

    for i, rn in enumerate(walk):
        r = spec.room(rn)
        if i > 0:
            c, n_in = _door_between(spec, walk[i - 1], rn)  # n_in points into previous room
            walk_to(pos, c + n_in * 0.6)
            walk_to(c + n_in * 0.6, c - n_in * 0.6)
            pos = c - n_in * 0.6
        if rn not in visited or i == len(walk) - 1:
            for sp in _sweep_spots(r):
                walk_to(pos, sp)
                pos = sp
                do_sweep(sp, r)
            visited.add(rn)
    return poses


def single_room_photo_poses(spec: PropertySpec, room: str, rng: np.random.Generator):
    """Protocol for photos: from each corner (0.4 m in) aim at the opposite corner, plus one shot per
    door taken from inside the room looking straight through the doorway."""
    r = spec.room(room)
    poses = []
    corners = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]
    cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
    for x, y in corners:
        p = np.array([x + np.sign(cx - x) * 0.4, y + np.sign(cy - y) * 0.4])
        tgt = np.array([cx + (cx - x), cy + (cy - y)])
        yaw = np.arctan2(tgt[1] - p[1], tgt[0] - p[0])
        poses.append(look_at(np.array([p[0], p[1], 1.4 + rng.normal(0, 0.03)]), yaw + rng.normal(0, 0.03),
                             np.radians(5) + rng.normal(0, 0.02), rng.normal(0, 0.01)))
    door_poses = []
    for op in spec.openings:
        n_rooms = sum(_opening_on_room_wall(spec, op, rr, ss) is not None for rr in spec.rooms for ss in SIDES)
        if op.kind != "door" or n_rooms < 2:   # exterior doors need no stitching photo
            continue
        for s in SIDES:
            u = _opening_on_room_wall(spec, op, r, s)
            if u is None:
                continue
            p0, p1 = r.wall_endpoints(s)
            d = (p1 - p0) / np.linalg.norm(p1 - p0)
            n_in = np.array([-d[1], d[0]])
            c = p0 + d * (u + op.width / 2)
            standoff = min(1.6, 0.45 * (r.x1 - r.x0 if abs(n_in[0]) > 0.5 else r.y1 - r.y0))
            p = c + n_in * standoff
            yaw = np.arctan2(-n_in[1], -n_in[0])
            door_poses.append(look_at(np.array([p[0], p[1], 1.4]), yaw + rng.normal(0, 0.03), np.radians(3), 0.0))
    if len(poses) + len(door_poses) > 8:   # brief: 2-8 stills per room; keep doors, drop corners
        poses = poses[::2]
    return poses + door_poses


# ---------------------------------------------------------------- sensor models

def lidar_depth(z_true: np.ndarray, phantom: np.ndarray, rng: np.random.Generator, scale_bias: float,
                smear_px: int = 2):
    """True z-depth (256x192, inf = no hit) -> (depth_m float32, conf uint8 0/1/2)."""
    z = np.where(np.isfinite(z_true), z_true, 0.0)
    valid = (z > 0.05) & (z < 5.0)
    h, w = z.shape
    low = ndimage.zoom(rng.normal(0, 1, (6, 8)), (h / 6, w / 8), order=1)[:h, :w]
    noisy = z * (1 + scale_bias) + low * (0.001 + 0.002 * z) + rng.normal(0, 1, z.shape) * (0.0015 + 0.0015 * z)
    # edge smear: ARKit depth is upsampled from a sparse dToF grid, so discontinuities are blurred
    zf = np.where(valid, z, 5.0)
    gy, gx = np.abs(np.diff(zf, axis=0, append=zf[-1:])), np.abs(np.diff(zf, axis=1, append=zf[:, -1:]))
    edge = (gx > 0.08) | (gy > 0.08)
    zone = ndimage.binary_dilation(edge, iterations=smear_px)
    blurred = ndimage.uniform_filter(np.where(valid, noisy, 0.0), size=2 * smear_px + 1)
    wsum = ndimage.uniform_filter(valid.astype(float), size=2 * smear_px + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        blurred = np.where(wsum > 0, blurred / wsum, 0.0)
    mix = zone & valid & (rng.uniform(size=z.shape) < 0.7)
    noisy = np.where(mix, blurred, noisy)
    conf = np.full(z.shape, 2, np.uint8)
    conf[(z > 3.0) | zone] = 1
    conf[(z > 4.0) | (zone & (rng.uniform(size=z.shape) < 0.3))] = 0
    noisy[~valid] = 0.0
    conf[~valid] = 0
    _ = phantom  # mirrors fool the sensor with full confidence, as on real devices
    return noisy.astype(np.float32), conf


@dataclass
class DriftModel:
    yaw_per_m: float      # radians per metre of travel (bias)
    yaw_rw: float         # random-walk sigma per sqrt(m)
    trans_frac: float     # fraction of travelled distance
    z_frac: float


def apply_drift(poses_w: list[np.ndarray], drift: DriftModel, rng: np.random.Generator):
    """Odometry = D_i @ T_i with D_i accumulating yaw + translation error along the path (D_0 = I)."""
    out = []
    theta, tau = 0.0, np.zeros(3)
    direction = rng.normal(size=3)
    direction[2] = 0
    direction /= np.linalg.norm(direction)
    p0 = poses_w[0][:3, 3].copy()
    prev = p0
    for T in poses_w:
        stepv = np.linalg.norm(T[:3, 3] - prev)
        prev = T[:3, 3]
        theta += stepv * drift.yaw_per_m + rng.normal(0, drift.yaw_rw * np.sqrt(stepv + 1e-9))
        direction = direction + rng.normal(0, 0.05, 3) * np.array([1, 1, 0])
        direction /= np.linalg.norm(direction)
        tau = tau + stepv * (drift.trans_frac * direction + np.array([0, 0, drift.z_frac]))
        c, s = np.cos(theta), np.sin(theta)
        D = np.eye(4)
        D[:2, :2] = [[c, -s], [s, c]]
        D[:3, 3] = p0 - D[:3, :3] @ p0 + tau  # rotate about the start position
        out.append(D @ T)
    return out


# ---------------------------------------------------------------- writers

def _to_world(spec: PropertySpec, poses_local):
    Tw = spec.T_world_local
    return [Tw @ T for T in poses_local]


def write_stray(out: Path, spec: PropertySpec, walk: list[str], seed: int = 0, drift: DriftModel | None = None,
                scene: Scene | None = None, sweep: str = "full") -> dict:
    rng = np.random.default_rng(seed)
    scene = scene or build_scene(spec)
    out.mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(exist_ok=True)
    (out / "confidence").mkdir(exist_ok=True)
    poses = _to_world(spec, walk_trajectory(spec, walk, rng, sweep=sweep))
    drift = drift or DriftModel(np.radians(0.06) * rng.choice([-1, 1]), np.radians(0.04), 0.008, 0.001)
    odo = apply_drift(poses, drift, rng)
    K_rgb = K_from_hfov(RGB_W, RGB_H, IPHONE_MAIN_HFOV)
    K_d = K_rgb.copy()
    K_d[:2] *= LIDAR_W / RGB_W
    scale_bias = rng.normal(0, 0.0015)
    vw = cv2.VideoWriter(str(out / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (RGB_W, RGB_H))
    rows = []
    for i, (T, To) in enumerate(zip(poses, odo)):
        r_rgb = render(scene, T, K_rgb, RGB_W, RGB_H, rgb=True, rng=rng)
        r_d = render(scene, T, K_d, LIDAR_W, LIDAR_H, rgb=False)
        depth, conf = lidar_depth(r_d["depth"], r_d["phantom"], rng, scale_bias)
        Image.fromarray(np.round(depth * 1000).astype(np.uint16)).save(out / "depth" / f"{i:06d}.png")
        Image.fromarray(conf).save(out / "confidence" / f"{i:06d}.png")
        vw.write(cv2.cvtColor((r_rgb["rgb"] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        R_ar = M_AR2ZUP.T @ To[:3, :3]
        t_ar = M_AR2ZUP.T @ To[:3, 3]
        from scipy.spatial.transform import Rotation
        q = Rotation.from_matrix(R_ar).as_quat()  # x, y, z, w
        rows.append(f"{i * 0.1:.6f}, {i:06d}, {t_ar[0]:.6f}, {t_ar[1]:.6f}, {t_ar[2]:.6f}, "
                    f"{q[0]:.7f}, {q[1]:.7f}, {q[2]:.7f}, {q[3]:.7f}, {K_rgb[0,0]:.4f}, {K_rgb[1,1]:.4f}, "
                    f"{K_rgb[0,2]:.4f}, {K_rgb[1,2]:.4f}, , ")
    vw.release()
    (out / "odometry.csv").write_text(
        "timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, distortion_center_x, distortion_center_y\n"
        + "\n".join(rows) + "\n")
    np.savetxt(out / "camera_matrix.csv", K_rgb, delimiter=",", fmt="%.6f")
    (out / "imu.csv").write_text("timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z\n")
    _sidecar(out, {"kind": "lidar", "true_poses": [T.tolist() for T in poses], "scale_bias": scale_bias})
    return {"frames": len(poses)}


def write_video(out_dir: Path, spec: PropertySpec, walk: list[str], seed: int = 0, scene: Scene | None = None) -> dict:
    """Handheld walkthrough at 10 fps (keyframe sampler is time-based, so fps is irrelevant downstream)."""
    rng = np.random.default_rng(seed)
    scene = scene or build_scene(spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    side = out_dir / "_synthetic"
    side.mkdir(exist_ok=True)
    poses = _to_world(spec, walk_trajectory(spec, walk, rng, sweep="full"))
    K = K_from_hfov(RGB_W, RGB_H, IPHONE_MAIN_HFOV)
    vw = cv2.VideoWriter(str(out_dir / "walkthrough.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (RGB_W, RGB_H))
    depths = []
    for T in poses:
        r = render(scene, T, K, RGB_W, RGB_H, rgb=True, rng=rng)
        vw.write(cv2.cvtColor((r["rgb"] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        z = r["depth"][::4, ::4]
        depths.append(np.where(np.isfinite(z), z, 0).astype(np.float16))
    vw.release()
    np.savez_compressed(side / "depth.npz", depth=np.stack(depths), K=K, stride=4, fps=10.0,
                        poses=np.stack(poses))
    _sidecar(out_dir, {"kind": "video", "fps": 10.0})
    return {"frames": len(poses)}


def _exif_bytes(focal35: float, model: str = "iPhone 15"):
    img = Image.new("RGB", (1, 1))
    ex = img.getexif()
    ex[0x010F] = "Apple"
    ex[0x0110] = model
    ex[0x0112] = 1
    ifd = ex.get_ifd(0x8769)
    ifd[0xA405] = int(round(focal35))  # FocalLengthIn35mmFilm
    ifd[0x920A] = 2.22                   # FocalLength (mm, ultra-wide)
    return ex.tobytes()


def write_photos(out_dir: Path, spec: PropertySpec, rooms: list[str], seed: int = 0, scene: Scene | None = None,
                 w: int = 1024, h: int = 768) -> dict:
    """One folder per room, numbered in walk order. 0.5x ultra-wide (13 mm equiv) as the protocol says."""
    rng = np.random.default_rng(seed)
    scene = scene or build_scene(spec)
    hfov = 2 * np.degrees(np.arctan(18.0 / IPHONE_UW_35MM))  # 35mm-equiv: 36 mm frame width
    K = K_from_hfov(w, h, hfov)
    n = 0
    for k, rn in enumerate(rooms):
        folder = out_dir / f"{k + 1:02d}_{rn}"
        side = folder / "_synthetic"
        side.mkdir(parents=True, exist_ok=True)
        poses = _to_world(spec, single_room_photo_poses(spec, rn, rng))
        for j, T in enumerate(poses):
            r = render(scene, T, K, w, h, rgb=True, rng=rng)
            name = f"IMG_{1000 + n:04d}"
            Image.fromarray((r["rgb"] * 255).astype(np.uint8)).save(folder / f"{name}.jpg", quality=92,
                                                                     exif=_exif_bytes(IPHONE_UW_35MM))
            z = r["depth"][::4, ::4]
            np.savez_compressed(side / f"{name}.npz", depth=np.where(np.isfinite(z), z, 0).astype(np.float16),
                                K=K, stride=4, T_wc=T)
            n += 1
    _sidecar(out_dir, {"kind": "photo", "rooms": rooms})
    return {"photos": n}


def _sidecar(out: Path, meta: dict):
    side = out / "_synthetic"
    side.mkdir(exist_ok=True)
    (side / "meta.json").write_text(json.dumps(meta))
