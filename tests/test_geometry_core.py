"""Seam: geometry core (Observations -> rooms). Truth = synthetic scene parameters."""
import numpy as np
import pytest

from scan.geometry.core import extract_plan
from scan.synth.scene import ground_truth

from .conftest import match_sorted


@pytest.fixture(scope="module")
def furnished_plan(furnished_room):
    spec, obs = furnished_room
    return spec, extract_plan(obs)


def test_single_room_recovers_one_rectangular_room(furnished_plan):
    spec, plan = furnished_plan
    assert len(plan.rooms) == 1
    assert len(plan.rooms[0].walls) == 4, "furniture (wardrobe, sofa) must not notch the room polygon"


def test_wall_lengths_within_lidar_gate(furnished_plan):
    spec, plan = furnished_plan
    gt = ground_truth(spec)["rooms"][0]["walls"]
    err = match_sorted([w.length.value for w in plan.rooms[0].walls], gt)
    assert err.max() < 0.01, err


def test_ceiling_height_within_1_5cm(furnished_plan):
    spec, plan = furnished_plan
    assert abs(plan.rooms[0].height.value - spec.rooms[0].ceiling) < 0.015


def test_floor_area_within_1pct(furnished_plan):
    spec, plan = furnished_plan
    assert abs(plan.rooms[0].area.value / spec.rooms[0].area - 1) < 0.01


def test_door_and_window_detected_and_measured_no_phantoms(furnished_plan):
    spec, plan = furnished_plan
    ops = plan.rooms[0].openings
    kinds = sorted(o.kind for o in ops)
    assert kinds == ["door", "window"], kinds
    door = next(o for o in ops if o.kind == "door")
    win = next(o for o in ops if o.kind == "window")
    assert abs(door.width.value - 0.82) < 0.02
    assert abs(door.height.value - 2.03) < 0.03
    assert abs(win.width.value - 1.40) < 0.02
    assert abs(win.sill.value - 0.90) < 0.03


def test_every_measurement_has_positive_standard_error(furnished_plan):
    _, plan = furnished_plan
    r = plan.rooms[0]
    ses = [w.length.se for w in r.walls] + [r.height.se, r.area.se] + [o.width.se for o in r.openings]
    assert all(s > 0 for s in ses)


def test_two_rooms_are_separated_and_connected_by_their_door(two_rooms):
    spec, obs = two_rooms
    plan = extract_plan(obs)
    assert len(plan.rooms) == 2
    doors = [e for e in plan.adjacency if e["kind"] == "door"]
    assert len(doors) == 1
    gt_areas = sorted(r.area for r in spec.rooms)
    pred = sorted(r.area.value for r in plan.rooms)
    assert np.allclose(pred, gt_areas, rtol=0.015)
    heights = sorted(r.height.value for r in plan.rooms)
    assert np.allclose(heights, sorted(r.ceiling for r in spec.rooms), atol=0.015)
    assert plan.wall_thickness is not None and abs(plan.wall_thickness.value - spec.wall_t) < 0.02


def test_mirror_creates_no_phantom_room_or_opening(mirror_room):
    spec, obs = mirror_room
    plan = extract_plan(obs)
    assert len(plan.rooms) == 1
    r = plan.rooms[0]
    assert [o.kind for o in r.openings] == ["door"], [(o.kind, o.width.value) for o in r.openings]
    assert any("mirror" in w for w in plan.warnings)
    err = match_sorted([w.length.value for w in r.walls], ground_truth(spec)["rooms"][0]["walls"])
    assert err.max() < 0.015
