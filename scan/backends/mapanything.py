"""MapAnything adapter (facebookresearch/map-anything).

API verified against the upstream source (``MapAnything.from_pretrained``, ``preprocess_inputs``,
``model.infer`` returning per-view dicts with depth_z, intrinsics, camera_poses (OpenCV cam2world),
conf, mask, img_no_norm). Intrinsics from EXIF / the device table are passed in, so the model does
not have to guess focal length.

Video: views are processed in overlapping chunks; chunk k+1 is aligned to chunk k with a similarity
transform estimated from the pointmaps of the shared frames (same pixel, two predictions), which also
absorbs any per-chunk metric-scale disagreement. Residual inter-chunk drift is then handled by the
shared fragment pose graph exactly as for ARKit poses.
"""
from __future__ import annotations

import os

import numpy as np

from ..types import Frame
from .base import BackendUnavailable, View

DEFAULT_MODEL = os.environ.get("FLOORSCAN_MAPANYTHING_MODEL", "facebook/map-anything-apache")


class MapAnythingBackend:
    name = "mapanything"
    version = "1"

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str | None = None, chunk: int = 24, overlap: int = 6):
        try:
            import torch  # noqa: F401
            from mapanything.models import MapAnything  # noqa: F401
        except Exception as e:  # pragma: no cover - depends on optional install
            raise BackendUnavailable(f"mapanything not installed ({e}); run `make weights`") from e
        self.model_id = model_id
        self.device = device
        self.chunk, self.overlap = chunk, overlap
        self._model = None

    def _load(self):  # pragma: no cover - needs weights
        import torch
        from mapanything.models import MapAnything
        if self._model is None:
            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            try:
                self._model = MapAnything.from_pretrained(self.model_id, local_files_only=True).to(dev)
            except TypeError:
                self._model = MapAnything.from_pretrained(self.model_id).to(dev)
            except Exception as e:
                raise BackendUnavailable(f"MapAnything weights not in the local cache ({e}); run `make weights`") from e
            self._model.eval()
            self._dev = dev
        return self._model

    def _infer(self, views: list[View]):  # pragma: no cover - needs weights
        import torch
        from mapanything.utils.image import preprocess_inputs
        model = self._load()
        raw = []
        for v in views:
            d = {"img": v.load_image()}
            if v.K is not None:
                d["intrinsics"] = torch.tensor(v.K, dtype=torch.float32)
            raw.append(d)
        proc = preprocess_inputs(raw)
        with torch.no_grad():
            preds = model.infer(proc, memory_efficient_inference=True, use_amp=self._dev == "cuda",
                                amp_dtype="bf16", apply_mask=True, mask_edges=True,
                                apply_confidence_mask=False)
        out = []
        for p in preds:
            depth = p["depth_z"][0, ..., 0].float().cpu().numpy()
            mask = p["mask"][0, ..., 0].cpu().numpy().astype(bool)
            conf = p["conf"][0].float().cpu().numpy()
            conf = conf / max(np.percentile(conf[mask], 95), 1e-6) if mask.any() else conf
            depth[~mask] = 0
            img = p["img_no_norm"][0].float().cpu().numpy()
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.max() <= 1.5 else img.astype(np.uint8)
            out.append({"depth": depth, "conf": np.clip(conf, 0, 1).astype(np.float32),
                        "K": p["intrinsics"][0].float().cpu().numpy(),
                        "T": p["camera_poses"][0].float().cpu().numpy().astype(np.float64),
                        "pts": p["pts3d"][0].float().cpu().numpy(), "img": img})
        return out

    def reconstruct(self, views: list[View], tier: str) -> list[Frame]:  # pragma: no cover - needs weights
        if len(views) <= self.chunk:
            res = self._infer(views)
        else:
            res = self._chunked(views)
        frames = []
        for i, (v, r) in enumerate(zip(views, res)):
            if r is None:
                continue
            frames.append(Frame(index=i, T_wc=r["T"], K=r["K"], depth=r["depth"], conf=r["conf"],
                                load_image=(lambda im=r["img"]: im), timestamp=v.timestamp, name=v.name))
        return frames

    def _chunked(self, views):  # pragma: no cover - needs weights
        step = self.chunk - self.overlap
        result: list = [None] * len(views)
        for s in range(0, len(views), step):
            idx = list(range(s, min(s + self.chunk, len(views))))
            if len(idx) < 2:
                break
            res = self._infer([views[i] for i in idx])
            shared = [k for k, i in enumerate(idx) if result[i] is not None]
            if shared:
                ms = [(res[k]["depth"] > 0) & (result[idx[k]]["depth"] > 0) for k in shared]
                src = np.concatenate([res[k]["pts"][m] for k, m in zip(shared, ms)])
                dst = np.concatenate([result[idx[k]]["pts"][m] for k, m in zip(shared, ms)])
                S = umeyama(src, dst)
                for r in res:
                    r["T"] = S @ r["T"]
                    sc = np.cbrt(np.linalg.det(S[:3, :3]))
                    r["T"][:3, :3] /= sc
                    r["depth"] = r["depth"] * sc
                    r["pts"] = r["pts"] @ S[:3, :3].T + S[:3, 3]
            for k, i in enumerate(idx):
                if result[i] is None:
                    result[i] = res[k]
            if idx[-1] == len(views) - 1:
                break
        return result


def umeyama(src: np.ndarray, dst: np.ndarray, max_pts: int = 20000) -> np.ndarray:
    """Similarity transform (4x4, rotation*scale) mapping src -> dst in the least-squares sense."""
    if len(src) > max_pts:
        sel = np.linspace(0, len(src) - 1, max_pts).astype(int)
        src, dst = src[sel], dst[sel]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    C = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(C)
    Sg = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        Sg[2, 2] = -1
    R = U @ Sg @ Vt
    var = (xs ** 2).sum() / len(src)
    s = np.trace(np.diag(D) @ Sg) / var
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = mu_d - s * R @ mu_s
    return T
