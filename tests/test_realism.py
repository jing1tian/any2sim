"""Realism stack: robot cables, appearance de-lighting, camera image model, appearance-only bundle revisions."""
import numpy as np
import pytest


def _T(p, R=None):
    T = np.eye(4)
    T[:3, 3] = p
    if R is not None:
        T[:3, :3] = R
    return T


def test_cable_spline_anchors_and_length():
    from any2sim.robot import cables as CB
    spec = {"points": [{"link": "a", "p": [0, 0, 0]}, {"link": "b", "p": [0, 0, 0]}], "length_m": 0.5}
    LT = {"a": _T([0, 0, 0.3]), "b": _T([0.3, 0, 0.3])}
    P = CB.centerline(spec, LT, n=64)
    assert np.allclose(P[0], [0, 0, 0.3], atol=1e-6) and np.allclose(P[-1], [0.3, 0, 0.3], atol=1e-6)
    assert abs(CB.arc_length(P) - 0.5) < 0.01                # slack solved to the cable length
    assert P[:, 2].min() < 0.3 - 0.05                        # sags downwards (gravity)
    spec["slack"] = {"link": "a", "dir": [0, 1, 0]}           # service loop bowing sideways instead
    P2 = CB.centerline(spec, LT, n=64)
    assert P2[:, 1].max() > 0.05 and abs(P2[:, 2].min() - 0.3) < 1e-6
    taut = CB.centerline({"points": spec["points"]}, LT, n=16)
    assert abs(CB.arc_length(taut) - 0.3) < 1e-6
    V, F = CB.tube(P, 0.004, sides=8)
    assert V.shape == (64 * 8, 3) and F.shape == (63 * 8 * 2, 3)
    r = np.linalg.norm(V.reshape(64, 8, 3) - P[:, None], axis=2)
    assert np.allclose(r, 0.004, atol=1e-6)


def test_cable_loop_follows_link():
    from any2sim.robot import cables as CB
    spec = {"shape": "loop", "link": "l", "center": [0.1, 0, 0], "axis": [0, 0, 1], "radius": 0.03, "turns": 2,
            "pitch": 0.01}
    Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    P = CB.centerline(spec, {"l": _T([1, 2, 3], Rz)}, n=100)
    c = np.array([1, 2, 3]) + Rz @ np.array([0.1, 0, 0])
    assert np.allclose(np.linalg.norm(P[:, :2] - c[:2], axis=1), 0.03, atol=1e-6)
    assert abs((P[:, 2].max() - P[:, 2].min()) - 0.02) < 1e-3   # two turns of 1 cm pitch


def test_sh_delighting_recovers_albedo():
    from any2sim.tools.robot_appearance import delight_sh, sh9
    rng = np.random.default_rng(0)
    n = rng.normal(size=(6000, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    g = rng.integers(0, 3, 6000)
    alb = np.array([[0.8, 0.8, 0.78], [0.3, 0.31, 0.3], [0.02, 0.02, 0.025]])
    L = np.zeros((9, 3))
    L[0], L[2] = 0.7, 0.4                                       # ambient + light from above
    E = np.clip(sh9(n) @ L, 0.05, None)
    obs = alb[g] * E
    a, Lf, err = delight_sh(obs, n, g, 3)
    ratio = a / alb                                            # albedo up to one global scale
    assert np.allclose(ratio / ratio.mean(), 1, atol=0.05) and err < 0.05


def test_camera_isp_identity_and_effects():
    from any2sim import camera_isp as ISP
    img = (np.random.default_rng(1).uniform(30, 220, (48, 64, 3))).astype(np.uint8)
    assert np.array_equal(ISP.apply(img, {}), img)
    eye = ISP.apply(img, {"color_matrix": np.eye(3).tolist(), "vignette": [0, 0]})
    assert np.abs(eye.astype(int) - img).max() <= 1
    vig = ISP.apply(np.full((48, 64, 3), 200, np.uint8), {"vignette": [-0.4, 0.0]})
    assert vig[0, 0].mean() < vig[24, 32].mean() - 20             # darker corners
    lv = ISP.apply(img, {"levels": {"in": [0.3, 0.5, 0.8], "out": [0.1, 0.5, 0.9]}})
    assert lv.std() > img.std()                                 # more contrast
    a = ISP.apply(img, {"noise": {"a": [1e-4] * 3, "b": [0, 0, 0], "corr_px": 0.6}}, key=("e", 1))
    b = ISP.apply(img, {"noise": {"a": [1e-4] * 3, "b": [0, 0, 0], "corr_px": 0.6}}, key=("e", 1))
    c = ISP.apply(img, {"noise": {"a": [1e-4] * 3, "b": [0, 0, 0], "corr_px": 0.6}}, key=("e", 2))
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert 1.5 < np.std(a.astype(float) - img) < 4.0            # sigma = 0.01 * 255


def test_isotonic_monotone():
    from any2sim.tools.camera_calibrate import _isotonic
    y = _isotonic(np.arange(6), [0.1, 0.3, 0.2, 0.5, 0.4, 0.9])
    assert np.all(np.diff(y) >= -1e-12) and abs(y.mean() - np.mean([0.1, 0.3, 0.2, 0.5, 0.4, 0.9])) < 1e-9


def test_camera_calibration_rejects_observed_sharpening_regression():
    from any2sim.tools.camera_calibrate import _nonregressing
    baseline = {"ssim": .744272, "deltaE_median": 18.826}
    assert not _nonregressing({"ssim": .731166, "deltaE_median": 18.911}, baseline)
    assert not _nonregressing({"ssim": .75, "deltaE_median": 20}, baseline)
    assert _nonregressing({"ssim": .75, "deltaE_median": 18}, baseline)


def test_camera_calibration_does_not_pair_wrist_with_head_video(tmp_path):
    from any2sim.tools.camera_calibrate import run
    with pytest.raises(ValueError, match="own paired reference"):
        run(tmp_path, "renders", camera="wrist_left")


def test_independent_camera_isp_and_explicit_identity():
    from any2sim.camera_isp import for_camera
    scene = {"cameras": {"source_cam": {"isp": {"blur_sigma_px": -1.2, "bitrate_kbps": 7500}}}}
    cams = {"wrist_left": {"isp": {"bitrate_kbps": 2300}}, "wrist_right": {}}
    assert for_camera(scene, "head", cams)["blur_sigma_px"] == -1.2
    assert for_camera(scene, "wrist_left", cams) == {"bitrate_kbps": 2300}
    assert for_camera(scene, "wrist_right", cams) == {}
    scene["render"] = {"camera_isp": {"head": {}, "wrist_left": {"blur_sigma_px": .3}}}
    assert for_camera(scene, "head", cams) == {}
    assert for_camera(scene, "wrist_left", cams) == {"blur_sigma_px": .3}


def test_dataset_head_does_not_reuse_source_calibration():
    from any2sim.scene_ir import render_head_camera
    from any2sim.camera_isp import for_camera
    scene = {"cameras": {"source_cam": {"isp": {"bitrate_kbps": 7500}},
                         "robot_head": {"isp": {"bitrate_kbps": 5240}}},
             "render": {"head_camera": "robot_head"}}
    assert render_head_camera(scene) == "robot_head"
    assert for_camera(scene, "head") == {"bitrate_kbps": 5240}
    scene["render"]["head_camera"] = "typo"
    with pytest.raises(ValueError, match="not a scene camera"):
        render_head_camera(scene)
