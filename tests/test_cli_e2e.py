"""Seam: the CLI, end to end, on synthetic captures written in the real on-disk formats.

Truth = scene parameters. Tolerances are the brief's tier gates (photo 8 %, video 3 %, LiDAR 1 cm)."""
import json

import numpy as np
import pytest

from scan.cli import main
from scan.synth import presets
from scan.synth.capture import write_photos, write_stray, write_video
from scan.synth.scene import ground_truth

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def caps(tmp_path_factory):
    root = tmp_path_factory.mktemp("caps")
    room = presets.single_room(furniture=True)
    two = presets.two_rooms()
    write_stray(root / "lidar", room, room.walk, seed=3, yaw_step_deg=20)
    write_video(root / "video", room, room.walk, seed=4, yaw_step_deg=20)
    write_photos(root / "photo", two, ["a", "b"], seed=5)
    return root, room, two


def _run(capture, out):
    assert main(["run", str(capture), "-o", str(out), "--bootstrap", "2"]) == 0
    res = json.loads((out / "result.json").read_text())
    assert (out / "plan.svg").exists() and (out / "plan.png").exists()
    return res


def _sorted_walls(res, k=0):
    return np.sort([w["length_m"]["value"] for w in res["rooms"][k]["walls"]])


def test_lidar_capture_one_command(caps, tmp_path):
    root, room, _ = caps
    res = _run(root / "lidar", tmp_path / "o")
    assert res["tier"] == "lidar" and len(res["rooms"]) == 1
    gt = np.sort(ground_truth(room)["rooms"][0]["walls"])
    assert np.abs(_sorted_walls(res) - gt).max() < 0.02
    assert {o["type"] for o in res["rooms"][0]["openings"]} == {"door", "window"}
    m = res["rooms"][0]["ceiling_height_m"]
    assert m["ci90"][0] <= m["value"] <= m["ci90"][1]
    assert res["stitched"]["drift"]["enabled"] is True


def test_video_capture_one_command(caps, tmp_path):
    root, room, _ = caps
    res = _run(root / "video", tmp_path / "o")
    assert res["tier"] == "video" and res["backend"] == "oracle" and len(res["rooms"]) == 1
    gt = np.sort(ground_truth(room)["rooms"][0]["walls"])
    assert np.all(np.abs(_sorted_walls(res) / gt - 1) < 0.03)


def test_photo_folders_stitch_into_one_plan(caps, tmp_path):
    root, _, two = caps
    res = _run(root / "photo", tmp_path / "o")
    assert res["tier"] == "photo" and len(res["rooms"]) == 2
    doors = [e for e in res["stitched"]["adjacency"] if e["kind"] == "door"]
    assert len(doors) == 1
    assert res["stitched"]["overlap_m2"] < 0.05
    gt_net = sum(r.area for r in two.rooms)
    assert abs(res["stitched"]["net_floor_area_m2"]["value"] / gt_net - 1) < 0.08
    lo, hi = res["stitched"]["net_floor_area_m2"]["ci90"]
    assert hi - lo > 0.02 * gt_net, "photo-tier intervals must be honestly wide"
