"""Synthesis stage: variant overlay, trajectory helpers, sensor model, wrist distortion maps."""
import json
from pathlib import Path

import numpy as np
import pytest


def _scene(tmp: Path):
    import trimesh
    from any2sim.scene_ir import save
    sd = tmp / "scene"
    (sd / "meshes").mkdir(parents=True)
    trimesh.creation.box([0.1, 0.1, 0.1]).export(sd / "meshes" / "cube.obj")
    (sd / "poses.json").write_text(json.dumps({"T_world_cam": [np.eye(4).tolist()]}))
    T = np.eye(4)
    T[:3, 3] = [0.2, 0.1, 0.05]
    sc = {"schema_version": "0.4", "units": {}, "frames": {},
          "cameras": {"source_cam": {"width": 640, "height": 480, "K": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
                                     "poses_uri": "poses.json", "provenance": {}}},
          "lights": [{"type": "area", "position": [0, 0, 2], "target": [0, 0, 0], "size": [1, 1], "strength_w": 100}],
          "entities": {"cube": {"capability": "rigid", "geometry": {"visual": {"obj_uri": "meshes/cube.obj"},
                                "collision": [{"type": "box", "size": [0.1, 0.1, 0.1], "T": np.eye(4).tolist()}]},
                                "material": {"base_color": [0.5, 0.5, 0.5]}, "physics": {"mass_kg": 0.2, "friction": 0.8},
                                "states": {"a": {"frames": [0, 10], "T_world_obj": T.tolist()},
                                           "b": {"frames": [20, 30], "T_world_obj": T.tolist()}},
                                "initial_state": "a", "trajectory": {"motion": "m"}, "provenance": {"x": "test"}}}}
    save(sd, sc)
    return sd, sc


def test_apply_variant_overlay(tmp_path):
    from any2sim.scene_ir import apply_variant, camera_poses, core_digest, pose_at
    sd, sc = _scene(tmp_path)
    sc["render"] = {"motion": "old", "camera_isp": {"head": {}, "wrist_left": {"bitrate_kbps": 614}}}
    T = np.eye(4)
    T[:3, 3] = [0.3, 0.0, 0.05]
    D = np.eye(4)
    D[:3, 3] = [0.01, 0, 0]
    v = {"id": "ep_1", "entities": {"cube": {"T_world_obj": T.tolist(), "material": {"hsv_shift": [5, 0, 0]},
                                             "physics": {"mass_kg": 0.3}}},
         "lights": [{"type": "world", "color": [1, 1, 1], "strength": 0.5}],
         "cameras": {"source_cam": {"T_delta": D.tolist()}}, "simulation": {"control_latency_s": 0.01}}
    before = core_digest(sc)
    out = apply_variant(sc, v)
    assert core_digest(sc) == before                         # input untouched
    e = out["entities"]["cube"]
    assert np.allclose(pose_at(e, None), T) and list(e["states"]) == ["variant_initial"]
    assert e["physics"] == {"mass_kg": 0.3, "friction": 0.8} and e["material"]["base_color"] == [0.5, 0.5, 0.5]
    assert "trajectory" not in e and out["lights"][0]["type"] == "world"
    assert np.allclose(camera_poses(sd, out)[0], D) and out["simulation"]["control_latency_s"] == 0.01
    episode = apply_variant(sc, {"motions": {"achieved": {}}, "render": {"motion": "achieved"},
                                 "trajectory_entities": ["cube"]})
    assert episode["render"]["motion"] == "achieved"
    assert episode["render"]["camera_isp"] == sc["render"]["camera_isp"]
    assert episode["entities"]["cube"]["trajectory"]["motion"] == "achieved"
    assert "motion" not in apply_variant(sc, {"motions": {}})["render"]
    assert sc["render"]["motion"] == "old"




def test_selected_head_variant_keeps_reconstruction_camera(tmp_path):
    from types import SimpleNamespace
    from copy import deepcopy
    from any2sim.scene_ir import apply_variant, camera_poses
    from any2sim.synthkit import sample_variant
    sd, sc = _scene(tmp_path)
    sc["cameras"]["robot_head"] = deepcopy(sc["cameras"]["source_cam"])
    sc["render"] = {"head_camera": "robot_head"}
    ctx = SimpleNamespace(scene=sc, project=tmp_path, core="test", rigid=lambda: ["cube"],
                          demo=SimpleNamespace(rid="robot"),
                          spec={"task": {}, "camera": {"robot_head": {"trans_m": .005, "rot_deg": .5}}})
    variant = sample_variant(ctx, 0)
    assert set(variant["cameras"]) == {"robot_head"}
    result = apply_variant(sc, variant)
    assert result["cameras"]["source_cam"] == sc["cameras"]["source_cam"]
    assert not np.allclose(camera_poses(sd, result, "robot_head"), camera_poses(sd, sc, "robot_head"))


def test_trajectory_helpers():
    from any2sim import synthkit as sy
    t = np.arange(101) / 20.0
    Q = np.stack([np.sin(t), t], 1)
    tn, Qn = sy.time_warp(t, Q, 1.25)
    assert abs((tn[-1] - tn[0]) - 5.0 / 1.25) < 0.051 and np.allclose(np.diff(tn), 0.05)
    assert np.allclose(Qn[0], Q[0]) and abs(Qn[-1, 1] - 5.0) < 0.07
    A = np.eye(4)
    B = sy.planar_shift(A, 0.1, 0.2, 90)
    assert np.allclose(sy.blend_pose(A, B, 0), A) and np.allclose(sy.blend_pose(A, B, 1), B)
    M = sy.blend_pose(A, B, 0.5)
    assert np.allclose(M[:3, 3], [0.05, 0.1, 0]) and abs(np.degrees(np.arctan2(M[1, 0], M[0, 0])) - 45) < 1e-6
    assert sy.min_jerk(0) == 0 and sy.min_jerk(1) == 1 and abs(sy.min_jerk(0.5) - 0.5) < 1e-12
    warm, cold = sy.kelvin_rgb(3000), sy.kelvin_rgb(9000)
    assert warm[0] > warm[2] and cold[2] > cold[0]
    rng = np.random.default_rng(0)
    T = sy.small_transform(rng, 0.01, 2.0)
    assert np.linalg.norm(T[:3, 3]) <= 0.01 + 1e-12
    assert np.degrees(np.arccos((np.trace(T[:3, :3]) - 1) / 2)) <= 2.0 + 1e-9


def test_sensor_model_deterministic():
    from any2sim.tools.synth_render import sensor
    img = (np.random.default_rng(1).uniform(0, 255, (24, 32, 3))).astype(np.uint8)
    assert np.array_equal(sensor(img, {}, ("e", "head", 0)), img)
    s = {"exposure_ev": 0.0, "wb_gains": [1, 1, 1], "noise_std": 0.0, "blur_px": 0.0}
    assert np.abs(sensor(img, s, ("e", "head", 0)).astype(int) - img).max() <= 1
    s = {"exposure_ev": 0.3, "wb_gains": [1.05, 1, 0.95], "noise_std": 0.01, "blur_px": 0.5, "noise_seed": 7}
    a, b = sensor(img, s, ("e", "head", 3)), sensor(img, s, ("e", "head", 3))
    c = sensor(img, s, ("e", "head", 4))
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert sensor(img, {"exposure_ev": 1.0}, ("k",)).mean() > img.mean()


def test_distortion_maps_match_opencv():
    import cv2
    from any2sim.tools.synth_render import distortion_maps
    K = np.array([[400.0, 0, 320], [0, 400, 240], [0, 0, 1]])
    D = np.array([-0.36, 0.11, 0.001, 0.0005, 0.0])
    Kb, Wb, Hb, mx, my = distortion_maps(K, 640, 480, D)
    assert Wb > 640 and Hb > 480
    P = np.array([[0.1, -0.05, 1.0], [-0.3, 0.2, 1.0], [0.25, 0.18, 1.0]])
    uvd = cv2.projectPoints(P, np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)
    uvb = (P[:, :2] / P[:, 2:]) @ Kb[:2, :2].T + Kb[:2, 2]
    for (ud, vd), (ub, vb) in zip(uvd, uvb):
        i, j = int(round(vd)), int(round(ud))
        if 0 <= i < 480 and 0 <= j < 640:
            # the map at the distorted pixel points to where the pinhole canvas sees the same ray
            assert abs(mx[i, j] - ub) < 2.0 and abs(my[i, j] - vb) < 2.0






def test_support_height_preserves_contact_geometry_and_robot(tmp_path):
    from copy import deepcopy
    from types import SimpleNamespace
    from any2sim.scene_ir import apply_variant, pose_at
    from any2sim.synthkit import sample_variant
    sd, sc = _scene(tmp_path)
    sc['entities']['table'] = {'capability': 'static', 'T_world_entity': np.eye(4).tolist()}
    sc['robots'] = {'robot': {'T_world_base': np.eye(4).tolist(), 'initial_q': {'j': .1}}}
    spatial = {'support': 'table', 'objects': {'cube': {'dx': [.03, .03]}},
               'support_height': {'dz_m': [-.05, .05], 'contents': ['cube']}}
    # No mesh support polygon needed for this geometric invariant test.
    import unittest.mock
    ctx = SimpleNamespace(scene=sc, project=tmp_path, core='test', rigid=lambda: ['cube'],
                          demo=SimpleNamespace(rid='robot'), spec={'task': {}, 'spatial': spatial})
    before = deepcopy(sc)
    with unittest.mock.patch('any2sim.synthkit._support_polygon', return_value=None):
        v = sample_variant(ctx, 0)
        assert sample_variant(ctx, 0) == v
        nominal = sample_variant(ctx, -1, nominal=True)
    result = apply_variant(sc, v)
    dz = v['support_height']['dz_m']
    assert -.05 <= dz <= .05 and dz != 0
    for name in ['table', 'cube']:
        assert pose_at(result['entities'][name], None)[2,3] - pose_at(sc['entities'][name], None)[2,3] == pytest.approx(dz)
    assert pose_at(result['entities']['cube'], None)[0,3] == pytest.approx(.23)
    assert result['robots'] == sc['robots'] and result['cameras'] == sc['cameras']
    assert sc == before and nominal['entities'] == {} and 'support_height' not in nominal


def test_support_height_rejects_incomplete_contents():
    from any2sim.synthkit import support_height_poses
    with pytest.raises(ValueError, match='include all'):
        support_height_poses({}, {'support': 'table', 'objects': {'cube': {}},
                            'support_height': {'dz_m': [-.05,.05], 'contents': []}}, np.random.default_rng(0))
