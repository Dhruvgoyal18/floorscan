"""Video keyframes and photo folders -> backend Views (with intrinsics from metadata when trustworthy)."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from ..backends.base import View

try:  # iPhone default photo format
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:  # pragma: no cover
    pass

IMG_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
VID_EXT = {".mov", ".mp4", ".m4v"}

# Main (1x) camera 35 mm-equivalent focal length by model, used only as a fallback hint.
MAIN_F35 = {"iPhone 15": 26, "iPhone 15 Plus": 26, "iPhone 15 Pro": 24, "iPhone 15 Pro Max": 24,
            "iPhone 16": 26, "iPhone 16 Plus": 26, "iPhone 16 Pro": 24, "iPhone 16 Pro Max": 24,
            "iPhone 16e": 26, "iPhone 17": 26, "iPhone Air": 26, "iPhone 17 Pro": 24, "iPhone 17 Pro Max": 24}


def K_from_f35(f35: float, w: int, h: int) -> np.ndarray:
    f = f35 * np.hypot(w, h) / 43.27   # CIPA: 35 mm equivalence on the diagonal
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


# ------------------------------------------------------------------ photos

def photo_groups(capture: Path) -> list[tuple[str, list[Path]]]:
    capture = Path(capture)
    subs = sorted(p for p in capture.iterdir() if p.is_dir() and not p.name.startswith(("_", ".")))
    groups = []
    for s in subs:
        imgs = sorted(p for p in s.iterdir() if p.suffix.lower() in IMG_EXT)
        if imgs:
            groups.append((s.name, imgs))
    if not groups:
        imgs = sorted(p for p in capture.iterdir() if p.suffix.lower() in IMG_EXT)
        if imgs:
            groups.append((capture.name, imgs))
    return groups


def _load_photo(p: Path) -> np.ndarray:
    with Image.open(p) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return np.asarray(im)


def photo_view(p: Path, group: str) -> tuple[View, str]:
    with Image.open(p) as im:
        ex = im.getexif()
        model = str(ex.get(0x0110, "")).strip()
        sub = ex.get_ifd(0x8769)
        f35 = sub.get(0xA405)
        orient = ex.get(0x0112, 1)
        w, h = im.size
    if orient in (5, 6, 7, 8):
        w, h = h, w
    K = K_from_f35(float(f35), w, h) if f35 else None
    return View(name=p.stem, load_image=lambda p=p: _load_photo(p), K=K, source=str(p), group=group), model


# ------------------------------------------------------------------ video

def find_video(capture: Path) -> Path | None:
    capture = Path(capture)
    if capture.is_file() and capture.suffix.lower() in VID_EXT:
        return capture
    if capture.is_dir():
        vids = sorted(p for p in capture.iterdir() if p.suffix.lower() in VID_EXT and p.name != "rgb.mp4")
        return vids[0] if vids else None
    return None


def probe(path: Path) -> dict:
    if not shutil.which("ffprobe"):
        return {}
    try:
        out = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                             capture_output=True, text=True, timeout=30)
        return json.loads(out.stdout or "{}")
    except Exception:
        return {}


def video_device_and_f35(meta: dict) -> tuple[str, float | None]:
    tags = (meta.get("format") or {}).get("tags") or {}
    model = tags.get("com.apple.quicktime.model", "")
    f35 = tags.get("com.apple.quicktime.camera.focal_length.35mm_equivalent")
    return model, (float(f35) if f35 else None)


def sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def video_keyframes(path: Path, workdir: Path, target_fps: float = 2.5, max_side: int = 960,
                    max_frames: int = 360) -> tuple[list[View], dict]:
    """Pick the sharpest frame in every 1/target_fps window; write them as JPEG to ``workdir``."""
    workdir.mkdir(parents=True, exist_ok=True)
    meta = probe(path)
    model, f35 = video_device_and_f35(meta)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not cap.isOpened():
        raise RuntimeError(f"cannot decode {path}; install ffmpeg with HEVC support")
    win = max(1, int(round(fps / target_fps)))
    best, best_s, idx = None, -1.0, 0
    picks = []
    stamps: dict[int, float] = {}
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        stamps[idx] = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0   # variable frame rate safe
        small = bgr
        s = max_side / max(bgr.shape[:2])
        if s < 1:
            small = cv2.resize(bgr, (int(bgr.shape[1] * s), int(bgr.shape[0] * s)), interpolation=cv2.INTER_AREA)
        sh = sharpness(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        if sh > best_s:
            best, best_s, best_idx = small, sh, idx
        idx += 1
        if idx % win == 0:
            picks.append((best_idx, best, best_s))
            best, best_s = None, -1.0
    if best is not None:
        picks.append((best_idx, best, best_s))
    cap.release()
    if len(picks) > max_frames:
        sel = np.linspace(0, len(picks) - 1, max_frames).astype(int)
        picks = [picks[i] for i in sel]
    sharp = np.array([p[2] for p in picks]) if picks else np.zeros(0)
    views = []
    for k, (fi, img, sh) in enumerate(picks):
        f = workdir / f"kf_{fi:06d}.jpg"
        cv2.imwrite(str(f), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        h, w = img.shape[:2]
        K = K_from_f35(f35, w, h) if f35 else None   # only trust real metadata for video (EIS crops the FOV)
        views.append(View(name=f"kf_{fi:06d}", load_image=lambda f=f: cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB),
                          K=K, timestamp=stamps.get(fi, fi / fps), source=f"video:{path}#{fi}", group="all"))
    info = {"device": model, "fps": fps, "n_frames": idx, "n_keyframes": len(views),
            "blurry_share": float(np.mean(sharp < 0.25 * np.median(sharp))) if len(sharp) else 0.0}
    return views, info
