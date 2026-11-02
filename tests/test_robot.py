"""Generic robot rotations, retarget layout and Scene IR motions."""
from pathlib import Path

import numpy as np
import pytest



def test_rpy_roundtrip():
    from any2sim.robot.urdf import mat_to_rpy, rpy_to_mat
    rng = np.random.default_rng(0)
    for _ in range(50):
        r = rng.uniform([-3, -1.5, -3], [3, 1.5, 3])
        assert np.allclose(rpy_to_mat(*mat_to_rpy(rpy_to_mat(*r))), rpy_to_mat(*r), atol=1e-9)


def test_rotation_formats():
    from scipy.spatial.transform import Rotation as Rot
    from any2sim.robot import datalog as DL
    R = Rot.random(20, random_state=1).as_matrix()
    cols = np.concatenate([R[:, :, 0], R[:, :, 1]], 1)
    assert np.allclose(DL.to_matrix(cols, "rot6d_cols"), R, atol=1e-9)
    rows = np.concatenate([R[:, 0, :], R[:, 1, :]], 1)
    assert np.allclose(DL.to_matrix(rows, "rot6d_rows"), R, atol=1e-9)
    inter = R[:, :, :2].reshape(20, 6)
    assert np.allclose(DL.to_matrix(inter, "rot6d_colsinterleaved"), R, atol=1e-9)








def test_layout_detection():
    from any2sim.tools.hand_retarget import layout_names
    rng = np.random.default_rng(0)
    # synthetic hand: wrist at 0, thumb base close to the wrist, fingers fanned
    order = ["index", "middle", "pinky", "ring", "thumb"]
    base = {"thumb": [0.02, 0.01, 0.03], "index": [0.02, 0.0, 0.09], "middle": [0.0, 0.0, 0.09],
            "ring": [-0.02, 0.0, 0.085], "pinky": [-0.04, 0.0, 0.075]}
    k = np.zeros((21, 3))
    for c, f in enumerate(order):
        for j in range(4):
            k[1 + 4 * c + j] = np.asarray(base[f]) + [0, 0, 0.025 * j]
    KP = np.repeat(k[None], 3, 0) + rng.normal(0, 1e-4, (3, 21, 3))
    names = layout_names("auto", KP)
    assert names["thumb_0"] == 17 and names["index_0"] == 1 and names["pinky_0"] == 9 and names["ring_0"] == 13


def test_motion_roundtrip(tmp_path):
    from any2sim.scene_ir import motion_at, save_motion
    t = np.arange(10) / 20.0
    q = np.stack([np.arange(10), -np.arange(10)], 1).astype(float)
    T = np.tile(np.eye(4), (10, 1, 1, 1))
    save_motion(tmp_path / "m.npz", t, np.arange(10), ["a", "b"], q, obj_names=["o"], obj_T=T)
    d = np.load(tmp_path / "m.npz")
    m = {k: d[k] for k in d.files}
    m["joint_names"] = [str(x) for x in m["joint_names"]]
    m["obj_names"] = [str(x) for x in m["obj_names"]]
    qq, objs = motion_at(m, 4.5)
    assert abs(qq["a"] - 4.5) < 1e-9 and abs(qq["b"] + 4.5) < 1e-9 and "o" in objs
