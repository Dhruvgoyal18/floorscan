"""Seam: drift correction (frames -> corrected frames + report)."""
import numpy as np

from scan.geometry.cloud import estimate_normals
from scan.stitch.drift import DriftConfig, correct_drift, icp_4dof
from scan.synth import presets
from scan.synth.capture import DriftModel, lidar_frames_as_frames, render_lidar_frames


def _corner_cloud(rng):
    """Three orthogonal planes + a box: well-constrained for 4-DoF ICP."""
    a = rng.uniform(0, 2, (3000, 2))
    pts = np.concatenate([np.c_[a[:, 0], a[:, 1], np.zeros(len(a))],          # floor
                          np.c_[a[:, 0], np.zeros(len(a)), a[:, 1]],          # wall y=0
                          np.c_[np.zeros(len(a)), a[:, 0], a[:, 1]],          # wall x=0
                          np.c_[0.8 + 0.3 * a[:500, 0] / 2, 0.9 + np.zeros(500), 0.4 * a[:500, 1] / 2]])
    return pts + rng.normal(0, 0.002, pts.shape)


def test_icp_recovers_known_yaw_and_translation():
    rng = np.random.default_rng(0)
    dst = _corner_cloud(rng)
    n, _ = estimate_normals(dst, origins=np.tile([1.0, 1.0, 1.0], (len(dst), 1)))
    th = np.radians(2.0)
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    t = np.array([0.06, -0.04, 0.01])
    src = (dst - t) @ R  # src = R^T (dst - t)  => E = [R, t] maps src back to dst
    E, fit, rmse, _ = icp_4dof(src, dst, n)
    assert np.allclose(E[:3, :3], R, atol=2e-3)
    assert np.allclose(E[:3, 3], t, atol=3e-3)
    assert fit > 0.9 and rmse < 0.005


def _position_error(frames, true_poses):
    return np.array([np.linalg.norm(f.T_wc[:3, 3] - T[:3, 3]) for f, T in zip(frames, true_poses)])


def test_pose_graph_reduces_accumulated_drift_on_a_loop():
    spec = presets.two_rooms()
    r = render_lidar_frames(spec, ["a", "b", "a"], seed=11, yaw_step_deg=20,
                            drift=DriftModel(np.radians(0.4), np.radians(0.05), 0.02, 0.002), with_rgb=False)
    frames = lidar_frames_as_frames(r)
    before = _position_error(frames, r["poses"])
    fixed, rep = correct_drift(frames, DriftConfig())
    after = _position_error(fixed, r["poses"])
    assert rep.n_loop_edges >= 1
    assert rep.loop_closure_error_after_m < rep.loop_closure_error_m
    # drift is relative to the first frame; the end of the loop must come back close to truth
    assert after[-20:].mean() < 0.5 * before[-20:].mean(), (before[-20:].mean(), after[-20:].mean())


def test_ablation_flag_leaves_poses_untouched():
    spec = presets.two_rooms()
    r = render_lidar_frames(spec, ["a", "b"], seed=12, yaw_step_deg=30, with_rgb=False)
    frames = lidar_frames_as_frames(r)
    out, rep = correct_drift(frames, DriftConfig(enabled=False))
    assert not rep.enabled and "as-is" in rep.method
    assert all(np.array_equal(a.T_wc, b.T_wc) for a, b in zip(frames, out))
