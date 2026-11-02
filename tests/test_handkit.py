"""Scene rescale invariance, track events on a synthetic pick-and-place, rigid registrations only."""
import json
from pathlib import Path

import numpy as np

from any2sim.geom3d import inv, project


def _scene(tmp: Path, cam_T):
    import trimesh
    from any2sim.scene_ir import save
    sd = tmp / "scene"
    (sd / "meshes").mkdir(parents=True)
    trimesh.creation.box([0.1, 0.1, 0.1]).export(sd / "meshes" / "cube.obj")
    (sd / "poses.json").write_text(json.dumps({"T_world_cam": [cam_T.tolist()]}))
    T = np.eye(4)
    T[:3, 3] = [0.2, 0.1, 0.05]
    sc = {"schema_version": "0.3", "units": {}, "frames": {},
          "cameras": {"source_cam": {"width": 640, "height": 480, "K": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
                                     "poses_uri": "poses.json", "provenance": {}}},
          "lights": [{"type": "area", "position": [0, 0, 2], "target": [0, 0, 0], "size": [1, 1], "strength_w": 100}],
          "entities": {"cube": {"capability": "rigid", "geometry": {"visual": {"obj_uri": "meshes/cube.obj"},
                                "collision": [{"type": "box", "size": [0.1, 0.1, 0.1], "T": np.eye(4).tolist()}]},
                                "material": {"base_color": [0.5, 0.5, 0.5]}, "physics": {"mass_kg": 0.2},
                                "states": {"a": {"frames": [0, 10], "T_world_obj": T.tolist()}}, "initial_state": "a",
                                "provenance": {"x": "test"}}}}
    save(sd, sc)
    (tmp / "evidence").mkdir()
    return sd


def test_scene_rescale_keeps_projections(tmp_path):
    from any2sim.scene_ir import camera_poses, load, pose_at
    from any2sim.tools import scene_rescale
    from any2sim.robot.render import look_at
    cam = look_at([0.8, -0.6, 0.7], [0.2, 0.1, 0.05])
    sd = _scene(tmp_path, cam)
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1.0]])
    P0 = pose_at(load(sd)["entities"]["cube"], None)[:3, 3]
    uv0 = project(K, inv(cam), P0[None])[0]
    r = scene_rescale.run(tmp_path, factor=0.7, reason="test")
    assert r["projection_max_px_change"] < 1e-3
    sc = load(sd)
    assert abs(sc["entities"]["cube"]["physics"]["mass_kg"] - 0.2 * 0.7 ** 3) < 1e-9
    assert abs(sc["lights"][0]["strength_w"] - 100 * 0.49) < 1e-9
    P1 = pose_at(sc["entities"]["cube"], None)[:3, 3]
    uv1 = project(K, inv(camera_poses(sd, sc)[0]), P1[None])[0]
    assert np.allclose(uv0, uv1, atol=1e-6) and np.allclose(P1, 0.7 * P0)
    assert (sd / "revisions.yaml").exists()


def test_scaled_registration_rejected(tmp_path):
    from any2sim.robot.render import look_at
    from any2sim.scene_ir import load, validate
    sd = _scene(tmp_path, look_at([0.8, -0.6, 0.7], [0, 0, 0]))
    sc = load(sd)
    T = np.eye(4)
    T[:3, :3] *= 1.6
    sc["registrations"] = {"hand_tracks": {"T_world_tracks": T.tolist()}}
    errs = validate(sd, sc)["errors"]
    assert any("rigid" in e for e in errs)


def test_track_events_pick_and_place():
    from any2sim import handkit as hk
    from any2sim.tools.hand_retarget import layout_names
    fps, n = 20, 200
    t = np.arange(n)
    # palm path: rest at A (0-39), move to object B (40-59), dwell low (60-99), transport to C (100-129),
    # lower at C (130-139), retreat (140-199)
    A, B, C = np.array([0.0, 0, 0.2]), np.array([0.3, 0, 0.05]), np.array([0.3, 0.25, 0.08])
    path = np.zeros((n, 3))
    for i in t:
        if i < 40:
            path[i] = A
        elif i < 60:
            path[i] = A + (B - A) * (i - 40) / 20
        elif i < 100:
            path[i] = B
        elif i < 130:
            path[i] = B + (C - B) * (i - 100) / 30 + [0, 0, 0.1 * np.sin(np.pi * (i - 100) / 30)]
        elif i < 140:
            path[i] = C
        else:
            path[i] = C + (A - C) * (i - 140) / 60
    base = np.zeros((21, 3))
    for c in range(5):
        for j in range(4):
            base[1 + 4 * c + j] = [0.02 * c - 0.04, 0, 0.03 + 0.025 * j]
    base[17:21] = [[0.03, 0.0, 0.02 + 0.02 * j] for j in range(4)]         # thumb chain near the wrist
    kp = path[:, None] + (base - [0, 0, 0.12])[None]
    tr = {"kp": kp, "names": layout_names("auto", kp)}
    ev = hk.track_events(tr, fps)
    assert 55 <= ev["grasp"][0] <= 64 and 95 <= ev["grasp"][1] <= 101
    assert 125 <= ev["release"][0] <= 140
    assert abs(ev["transport_m"] - 0.25) < 0.03
