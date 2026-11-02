import json

import pytest
import yaml

from any2sim import render_pipeline as RP
from any2sim import realism as R


@pytest.fixture
def project(tmp_path):
    (tmp_path / "scene").mkdir()
    scene = {"cameras": {"source_cam": {"isp": {"bitrate_kbps": 7500}}}}
    (tmp_path / "scene/scene.yaml").write_text(yaml.safe_dump(scene))
    (tmp_path / "scene/albedo.png").write_bytes(b"original albedo")
    return tmp_path, scene


def approved(p, snapshot, cameras):
    (p / "render").mkdir(exist_ok=True)
    (p / "evidence").mkdir(exist_ok=True)
    (p / "evidence/comparison.json").write_text('{"reviewed": true}')
    (p / "realism").mkdir(exist_ok=True)
    scene = yaml.safe_load((p / "scene/scene.yaml").read_text())
    R.write_json(p / "realism/protected.json", R.protected_state(p, scene))
    protocol = {"schema_version": 1, "cameras": {c: {
        "native_size": [16, 16], "fit_frames": [0], "validation_frames": [1],
        "min_static_pixels": 1, "max_mae_regression_dn": 0, "reference_status": "unavailable",
        "reference_note": "No paired camera in this gate unit fixture.",
        "regions": [{"name": "surface", "box": [0, 0, 16, 16]}]}
        for c in cameras}}
    (p / "realism/protocol.yaml").write_text(yaml.safe_dump(protocol))
    R.freeze_protocol(p)
    R.write_json(p / "realism/check.json", {"status": "ready_for_review", "fingerprint": snapshot["fingerprint"],
        "cameras": {c: {} for c in cameras}, "evidence_hashes": {"evidence/comparison.json": RP.file_hash(p / "evidence/comparison.json")}})
    data = {"schema_version": 2, "stage": "realism", "builder": "builder", "evaluator": "independent",
        "diagnostics": "realism/check.json", "status": "pass", "fingerprint": snapshot["fingerprint"], "checks": {
        k: {"status": "pass", "note": "Independent fixture finding.", "evidence": ["evidence/comparison.json"]}
        for k in (*R.CHECKS, *(f"camera:{c}" for c in cameras))},
        "evidence_hashes": {f: RP.file_hash(p / f) for f in ("realism/check.json", "evidence/comparison.json", "realism/protocol.yaml", "realism/protocol.lock.json", "realism/protected.json")}}
    (p / "render/realism_review.json").write_text(json.dumps(data))


def test_fixed_processing_is_independent_of_randomization():
    scene = {"render": {"realism_baseline": {"samples": 64}}, "sensor": {}}
    assert RP.settings(scene) == dict(engine="cycles", samples=64, denoise=True,
                                    motion_blur=.5, max_bounces=12, diffuse_bounces=4, glossy_bounces=4,
                                    adaptive_threshold=.01, view_transform="Standard")
    assert RP.settings(scene, denoise=False)["denoise"] is False


def test_calibration_uses_raw_not_processed_and_metadata_is_not_processing(tmp_path):
    assert not RP.pixel_processing({"bitrate_kbps": 7500, "provenance": "calibrated"})
    assert RP.pixel_processing({"noise": {"a": [.001] * 3}})
    (tmp_path / "render.json").write_text(json.dumps({"camera_model": {
        "enabled": True, "parameters": {"saturation": .4}}}))
    with pytest.raises(ValueError, match="raw renders are missing"):
        RP.calibration_renders(tmp_path)
    (tmp_path / "raw").mkdir()
    assert RP.calibration_renders(tmp_path) == tmp_path / "raw"


def test_review_requires_all_views_and_evidence(project):
    p, scene = project
    s = RP.baseline(p, scene, RP.settings(scene))
    with pytest.raises(ValueError, match="Realism baseline not ready"):
        RP.review(p, s, ["head"], required=True)
    approved(p, s, ["head"])
    assert RP.review(p, s, ["head"], required=True)["status"] == "pass"
    with pytest.raises(ValueError, match="camera:wrist_left"):
        RP.review(p, s, ["head", "wrist_left"], required=True)
    (p / "evidence/comparison.json").unlink()
    with pytest.raises(ValueError, match="existing evidence"):
        RP.review(p, s, ["head"], required=True)


def test_asset_isp_and_settings_changes_invalidate_review(project):
    p, scene = project
    cfg = RP.settings(scene)
    s = RP.baseline(p, scene, cfg)
    approved(p, s, ["head"])
    assert RP.baseline(p, scene, {**cfg, "samples": 16})["fingerprint"] != s["fingerprint"]
    assert RP.baseline(p, scene, {**cfg, "adaptive_threshold": .003})["fingerprint"] != s["fingerprint"]
    assert RP.baseline(p, scene, {**cfg, "resolution": [640, 360]})["fingerprint"] != s["fingerprint"]
    assert RP.baseline(p, scene, {**cfg, "view_transform": "AgX"})["fingerprint"] != s["fingerprint"]
    scene["cameras"]["source_cam"]["isp"]["noise"] = {"a": [.001] * 3}
    assert RP.baseline(p, scene, cfg)["fingerprint"] != s["fingerprint"]
    scene["cameras"]["source_cam"]["isp"].pop("noise")
    (p / "scene/albedo.png").write_bytes(b"new albedo")
    with pytest.raises(ValueError, match="stale"):
        RP.review(p, RP.baseline(p, scene, cfg), ["head"], required=True)


def test_cache_includes_motion_and_render_checks(project):
    p, scene = project
    ep = p / "episode"
    ep.mkdir()
    for f in ("variant.yaml", "achieved.npz", "commanded.npz"):
        (ep / f).write_bytes(b"one")
    s = RP.baseline(p, scene, RP.settings(scene))
    key = RP.episode_fingerprint(ep, s, {"check_frames": 4, "resume": True})
    assert RP.episode_fingerprint(ep, s, {"check_frames": 4, "resume": False}) == key
    assert RP.episode_fingerprint(ep, s, {"check_frames": 8}) != key
    (ep / "achieved.npz").write_bytes(b"different motion")
    assert RP.episode_fingerprint(ep, s, {"check_frames": 4}) != key


def test_batch_gate_precedes_gpu_launch(project, monkeypatch):
    from any2sim.tools import synth_render
    p, _ = project
    (p / "synthesis").mkdir()
    (p / "synthesis/episodes.yaml").write_text('nominal: nominal\nepisodes: [ep_00000]\n')
    monkeypatch.setattr(synth_render, "_gpus", lambda _: pytest.fail("must not allocate GPUs before review"))
    with pytest.raises(ValueError, match="Realism baseline not ready"):
        synth_render.run(p)
    assert (p / "render/realism_baseline.json").exists()


def test_source_camera_override_does_not_inherit_dataset_head():
    from any2sim.camera_isp import for_camera
    scene = {"cameras": {"source_cam": {"isp": {"bitrate_kbps": 7000}}, "robot_head": {}},
             "render": {"head_camera": "robot_head", "camera_isp": {"source_cam": {"blur_sigma_px": .3}}}}
    assert for_camera(scene, "source_cam") == {"blur_sigma_px": .3}
    assert for_camera(scene, "head") == {}




def test_adaptive_sampling_control_reaches_cycles_and_can_be_disabled():
    from types import SimpleNamespace
    scene = SimpleNamespace(cycles=SimpleNamespace(), render=SimpleNamespace(), view_settings=SimpleNamespace())
    cfg = RP.settings({'render': {'realism_baseline': {'adaptive_threshold': .003}}})
    RP.apply_settings(scene, cfg)
    assert scene.cycles.use_adaptive_sampling
    assert scene.cycles.adaptive_threshold == .003
    RP.apply_settings(scene, RP.settings({}, adaptive_threshold=0))
    assert not scene.cycles.use_adaptive_sampling
    with pytest.raises(ValueError, match='adaptive_threshold'):
        RP.settings({}, adaptive_threshold=-.01)


def test_target_resolution_preserves_rays_and_centres_crop():
    import numpy as np
    from any2sim.tools.synth_render import output_intrinsics
    K = np.array([[400., 0, 320], [0, 400, 240], [0, 0, 1]])
    crop, w, h = output_intrinsics(K, 640, 480, [640, 360])
    assert (w, h) == (640, 360)
    np.testing.assert_allclose(crop, [[400, 0, 320], [0, 400, 180], [0, 0, 1]])
    head, _, _ = output_intrinsics(K * [[2], [2], [1]], 1280, 960, [640, 360])
    np.testing.assert_allclose(head, crop)
    np.testing.assert_allclose(K[1], [0, 400, 240])
    with pytest.raises(ValueError):
        output_intrinsics(K, 640, 480, [0, 360])
