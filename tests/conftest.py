"""Shared synthetic fixtures. Every expected value comes from the scene parameters (independent truth)."""
from __future__ import annotations

import numpy as np
import pytest

from scan.io.common import LIDAR_DEPTH, build_observations
from scan.synth import presets
from scan.synth.capture import DriftModel, lidar_frames_as_frames, render_lidar_frames

NO_DRIFT = DriftModel(0.0, 0.0, 0.0, 0.0)


def lidar_obs(spec, seed=0, drift=NO_DRIFT, true_poses=True, scale_bias=0.0, yaw_step=20.0):
    r = render_lidar_frames(spec, spec.walk, seed=seed, drift=drift, yaw_step_deg=yaw_step, scale_bias=scale_bias,
                            with_rgb=False)
    frames = lidar_frames_as_frames(r, use_true_poses=true_poses)
    return build_observations(frames, "lidar", "all", LIDAR_DEPTH), r


@pytest.fixture(scope="session")
def furnished_room():
    spec = presets.single_room(furniture=True)
    obs, _ = lidar_obs(spec, seed=1)
    return spec, obs


@pytest.fixture(scope="session")
def two_rooms():
    spec = presets.two_rooms()
    obs, _ = lidar_obs(spec, seed=2)
    return spec, obs


@pytest.fixture(scope="session")
def mirror_room():
    spec = presets.mirror_room()
    obs, _ = lidar_obs(spec, seed=3)
    return spec, obs


def match_sorted(pred, truth):
    return np.abs(np.sort(pred) - np.sort(truth))
