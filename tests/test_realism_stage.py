"""Small real image/video fixtures exercise the stage's fail-closed behavior without GPUs."""
import copy
import hashlib
import json
import struct
import shutil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
import yaml

from any2sim import realism as R, render_pipeline as RP, scene_ir as IR, stages
from any2sim.common import write_rgb


@pytest.fixture
def project(tmp_path):
    p = tmp_path / "source"
    for d in ("scene", "robot", "work/frames", "inputs", "evaluations/001", "realism/renders/nominal/check"):
        (p / d).mkdir(parents=True)
    (p / "project.yaml").write_text("id: source\n")
    (p / "observation.yaml").write_text("videos: []\n")
    (p / "robot/robot.yaml").write_text("name: fixture\n")
    (p / "robot/robot.urdf").write_text('<robot name="fixture"><link name="base"/></robot>')
    camera = {"width": 16, "height": 16, "K": [[16, 0, 8], [0, 16, 8], [0, 0, 1]], "T_world_cam": np.eye(4).tolist()}
    scene = {"schema_version": "0.4", "units": "m", "frames": {}, "entities": {},
             "cameras": {"source_cam": camera}, "robots": {"R": {"bundle": "../robot"}},
             "motions": {"cmd": {"uri": "cmd.npz", "kind": "commanded", "robot": "R"},
                         "ach": {"uri": "ach.npz", "kind": "achieved", "robot": "R", "derived_from": "cmd"}},
             "render": {"motion": "ach", "realism_baseline": {"samples": 4}}}
    IR.save(p / "scene", scene)
    for name in ("cmd", "ach"):
        IR.save_motion(p / f"scene/{name}.npz", np.arange(4) / 20, np.arange(4), joint_names=[], q=np.zeros((4, 0)))
    R.initialize(p, ["head"])
    proto = yaml.safe_load((p / "realism/protocol.yaml").read_text())
    spec = proto["cameras"]["head"]
    spec.update(fit_frames=[0, 1], validation_frames=[2, 3], reference_status="paired", pairs={},
                regions=[{"name": "surface", "role": "surface", "box": [0, 0, 16, 16]}], min_static_pixels=16)
    for i in range(4):
        ref = f"work/frames/{i}.png"
        mask = f"work/frames/{i}_mask.png"
        write_rgb(p / ref, np.full((16, 16, 3), 60 + i * 20, np.uint8))
        write_rgb(p / mask, np.full((16, 16, 3), 255, np.uint8))
        spec["pairs"][i] = {"reference": ref, "static_mask": mask}
    (p / "realism/protocol.yaml").write_text(yaml.safe_dump(proto))
    R.freeze_protocol(p)
    return p, scene


def light_exr(path, width=16, height=16):
    # A valid uncompressed scanline EXR with RGB for every diagnostic pass.
    channels = [f"ViewLayer.{name}.{c}" for name in R.LIGHT_PASSES for c in "RGB"]
    def attr(name, kind, value):
        return name.encode() + b"\0" + kind.encode() + b"\0" + struct.pack("<I", len(value)) + value
    header = struct.pack("<II", 20000630, 2)
    header += attr("channels", "chlist", b"".join(n.encode() + b"\0" + struct.pack("<iB3xii", 2, 0, 1, 1) for n in channels) + b"\0")
    header += attr("compression", "compression", b"\0")
    for key in ("dataWindow", "displayWindow"):
        header += attr(key, "box2i", struct.pack("<4i", 0, 0, width - 1, height - 1))
    header += attr("lineOrder", "lineOrder", b"\0")
    header += attr("pixelAspectRatio", "float", struct.pack("<f", 1))
    header += attr("screenWindowCenter", "v2f", struct.pack("<ff", 0, 0))
    header += attr("screenWindowWidth", "float", struct.pack("<f", 1)) + b"\0"
    data = bytes(width * len(channels) * 4)
    offset = len(header) + height * 8
    offsets, chunks = [], []
    for y in range(height):
        chunk = struct.pack("<iI", y, len(data)) + data
        offsets.append(offset); chunks.append(chunk); offset += len(chunk)
    path.write_bytes(header + b"".join(struct.pack("<Q", o) for o in offsets) + b"".join(chunks))


def evidence(p, scene, processed_offset=0):
    settings = {**RP.settings(scene), "cameras": ["head"], "res_scale": 1.0}
    snap = RP.baseline(p, scene, settings)
    R.write_json(p / "render/realism_baseline.json", snap)
    root = p / "realism/renders/nominal"
    captures = {}
    video = root / "head.mp4"
    writer = imageio.get_writer(str(video), fps=20, codec="libx264", macro_block_size=8, quality=8)
    for i in range(4):
        row = {}
        raw = np.full((16, 16, 3), 60 + i * 20, np.uint8)
        for key in ("noisy", "denoised", "processed"):
            path = root / f"check/{i}_{key}.png"
            im = np.clip(raw.astype(int) + (processed_offset if key == "processed" else 0), 0, 255).astype(np.uint8)
            write_rgb(path, im)
            row[key] = str(path.relative_to(p))
        writer.append_data(im)
        exr = root / f"check/{i}.exr"
        light_exr(exr)
        row["light_passes"] = str(exr.relative_to(p))
        captures[str(i)] = row
    writer.close()
    reader = imageio.get_reader(str(video))
    for i, im in enumerate(reader):
        path = root / f"check/{i}_decoded.png"
        write_rgb(path, im)
        captures[str(i)]["decoded"] = str(path.relative_to(p))
    reader.close()
    R.write_json(root / "diagnostics.json", {"fingerprint": snap["fingerprint"], "settings": settings,
        "complete": True, "captures": {"head": captures}, "videos": {"head": str(video.relative_to(p))}})
    return snap


def approve(p, snapshot, check):
    finding = "evaluations/001/findings.txt"
    (p / finding).write_text("Independent visual findings fixture; automatic metrics do not establish realism.")
    hashes = {**check["evidence_hashes"], "realism/check.json": RP.file_hash(p / "realism/check.json"),
              finding: RP.file_hash(p / finding)}
    report = {"schema_version": 2, "stage": "realism", "builder": "build-session", "evaluator": "eval-session",
              "fingerprint": snapshot["fingerprint"], "status": "pass", "diagnostics": "realism/check.json",
              "checks": {k: {"status": "pass", "note": "Scoped independent inspection fixture", "evidence": [finding]}
                         for k in (*R.CHECKS, "camera:head")}, "evidence_hashes": hashes}
    R.write_json(p / "evaluations/001/realism_review.json", report)
    return report


def test_full_evidence_ready_is_not_visual_approval(project):
    p, scene = project
    snapshot = evidence(p, scene)
    checked = R.check(p)
    assert checked["status"] == "ready_for_review", checked["issues"]
    assert checked["cameras"]["head"]["temporal"]["frames"] == 4
    with pytest.raises(ValueError, match="Realism baseline not ready"):
        RP.review(p, snapshot, ["head"], required=True)
    approve(p, snapshot, checked)
    R.adopt_review(p, "evaluations/001/realism_review.json")
    assert RP.review(p, snapshot, ["head"], required=True)["status"] == "pass"
    (p / "evaluations/001/findings.txt").write_text("changed after review")
    with pytest.raises(ValueError, match="evidence changed"):
        RP.review(p, snapshot, ["head"], required=True)


def test_isp_regression_requires_explicit_identity_rerender(project):
    p, scene = project
    scene["render"]["camera_isp"] = {"head": {"saturation": 0.5}}
    IR.save(p / "scene", scene)
    evidence(p, scene, processed_offset=30)
    bad = R.check(p)
    assert bad["status"] == "blocked"
    assert any("nonidentity ISP" in s for s in bad["issues"])
    assert R.fallback_isp(p)["status"] == "rerender_required"
    scene = IR.load(p / "scene")
    assert scene["render"]["camera_isp"]["head"] == {}
    assert R.check(p)["status"] == "blocked"  # identity declaration alone cannot reuse old images
    evidence(p, scene)
    assert R.check(p)["status"] == "ready_for_review"


@pytest.mark.parametrize("changed", ["work/frames/2_mask.png", "realism/renders/nominal/check/2_processed.png"])
def test_isp_fallback_rejects_changed_diagnostic_inputs(project, changed):
    p, scene = project
    scene["render"]["camera_isp"] = {"head": {"saturation": 0.5}}
    IR.save(p / "scene", scene)
    evidence(p, scene, processed_offset=30)
    R.check(p)
    write_rgb(p / changed, np.zeros((16, 16, 3), np.uint8))
    with pytest.raises(ValueError, match="stale camera evidence"):
        R.fallback_isp(p)
    assert IR.load(p / "scene")["render"]["camera_isp"]["head"] == {"saturation": 0.5}


def test_gate_requires_camera_findings_and_evaluator_can_measure_readonly_candidate(project):
    p, scene = project
    snapshot = evidence(p, scene)
    checked = R.check(p)
    before = {str(f.relative_to(p)): RP.file_hash(f) for f in p.rglob("*") if f.is_file()}
    outside = p.parent / "evaluation/check.json"
    assert R.check(p, out=str(outside))["status"] == "ready_for_review"
    assert before == {str(f.relative_to(p)): RP.file_hash(f) for f in p.rglob("*") if f.is_file()}
    report = approve(p, snapshot, checked)
    del report["checks"]["camera:head"]["note"]
    R.write_json(p / "render/realism_review.json", report)
    with pytest.raises(ValueError, match="camera:head: independent finding"):
        RP.review(p, snapshot, ["head"], required=True)


@pytest.mark.parametrize("change", ["threshold", "reference", "mask", "motion", "decoded", "preview"])
def test_changed_or_incomplete_evidence_blocks(project, change):
    p, scene = project
    evidence(p, scene)
    if change == "threshold":
        f = p / "realism/protocol.yaml"
        f.write_text(f.read_text().replace("max_mae_regression_dn: 0.0", "max_mae_regression_dn: 99.0"))
    elif change in ("reference", "mask"):
        f = p / ("work/frames/2.png" if change == "reference" else "work/frames/2_mask.png")
        write_rgb(f, np.zeros((16, 16, 3), np.uint8))
    elif change == "motion":
        IR.save_motion(p / "scene/ach.npz", np.arange(4) / 10, np.arange(4), joint_names=[], q=np.zeros((4, 0)))
    elif change == "decoded":
        write_rgb(p / "realism/renders/nominal/check/2_decoded.png", np.zeros((16, 16, 3), np.uint8))
    else:
        path = p / "realism/renders/nominal/diagnostics.json"
        data = R.read_json(path); data["complete"] = False; R.write_json(path, data)
    result = R.check(p)
    assert result["status"] == "blocked", result


def test_protection_allows_finish_but_not_collision_changes(project):
    p, scene = project
    scene["entities"]["table"] = {"geometry": {"collision": [{"type": "mesh", "uri": "collision.obj"}],
                                                "visual": {"obj_uri": "visual.obj"}}, "material": {"roughness": 0.5}}
    (p / "scene/collision.obj").write_text("collision")
    (p / "scene/visual.obj").write_text("visual")
    before = R.protected_state(p, scene)
    scene["entities"]["table"]["material"]["roughness"] = .8
    (p / "scene/visual.obj").write_text("refined visual surface")
    assert R.protected_state(p, scene) == before
    (p / "scene/collision.obj").write_text("different contact surface")
    assert R.protected_state(p, scene)["fingerprint"] != before["fingerprint"]


def test_no_synthesis_skip_and_portable_accepted_handoff(project, monkeypatch):
    p, scene = project
    monkeypatch.setattr(stages.C, "config", lambda: {"projects_dir": str(p.parent)})
    with pytest.raises(ValueError, match="realism stage"):
        stages.new_from("premature", str(p), "synthesis")
    assert not (p.parent / "premature").exists()
    snapshot = evidence(p, scene)
    checked = R.check(p)
    approve(p, snapshot, checked)
    R.adopt_review(p, "evaluations/001/realism_review.json")
    result = stages.new_from("dataset", str(p), "synthesis")
    assert yaml.safe_load((result.root / "request.yaml").read_text())["stage"] == "synthesis"
    assert (result.root / "robot/robot.urdf").read_bytes() == (p / "robot/robot.urdf").read_bytes()
    imported = RP.baseline(result.root, IR.load(result.root / "scene"), snapshot["settings"])
    assert imported["fingerprint"] == snapshot["fingerprint"]
    assert RP.review(result.root, imported, ["head"], required=True)["status"] == "pass"


def test_realism_import_requires_trajectory_pass_and_preserves_motion(project, monkeypatch):
    p, scene = project
    monkeypatch.setattr(stages.C, "config", lambda: {"projects_dir": str(p.parent)})
    with pytest.raises(ValueError, match="passed frozen trajectory"):
        stages.new_from("appearance", str(p), "realism")
    candidate = p / "candidates/001"
    for folder in ("scene", "robot", "inputs"):
        shutil.copytree(p / folder, candidate / folder)
    (p / "evaluations/001/report.yaml").write_text(yaml.safe_dump({"stage": "trajectory", "candidate": "candidates/001", "verdict": "pass", "criteria": {"physics": {"status": "pass"}}}))
    result = stages.new_from("appearance", str(p) + "@001", "realism")
    assert (result.root / "scene/ach.npz").read_bytes() == (p / "scene/ach.npz").read_bytes()
    assert (result.root / "realism/protected.json").exists()
    assert yaml.safe_load((result.root / "request.yaml").read_text())["stage"] == "realism"


def test_ineffective_trial_stop_and_protocol_cannot_be_reset(project):
    p, scene = project
    evidence(p, scene)
    trials = [{"hypothesis": "normal amplitude", "outcome": "ineffective"}] * 2
    R.write_json(p / "realism/trials.json", trials)
    assert any("ineffective trial limit" in s for s in R.check(p)["issues"])
    trials[-1] = {**trials[-1], "next_hypothesis": "check physical wavelength"}
    R.write_json(p / "realism/trials.json", trials)
    assert R.check(p)["status"] == "ready_for_review"
    with pytest.raises(ValueError, match="already initialized"):
        R.initialize(p)


def test_exr_header_only_cannot_pass_diagnostics(project):
    p, scene = project
    evidence(p, scene)
    (p / "realism/renders/nominal/check/2.exr").write_bytes(b"\x76\x2f\x31\x01broken")
    assert R.check(p)["status"] == "blocked"


def test_candidate_reference_bytes_win_over_live_frames(project, monkeypatch):
    p, scene = project
    monkeypatch.setattr(stages.C, "config", lambda: {"projects_dir": str(p.parent)})
    snapshot = evidence(p, scene)
    checked = R.check(p)
    approve(p, snapshot, checked)
    R.adopt_review(p, "evaluations/001/realism_review.json")
    candidate = p / "candidates/001"
    for folder in ("scene", "robot", "inputs", "work", "realism", "render", "evaluations"):
        shutil.copytree(p / folder, candidate / folder)
    original = (candidate / "work/frames/2.png").read_bytes()
    write_rgb(p / "work/frames/2.png", np.zeros((16, 16, 3), np.uint8))
    result = stages.new_from("dataset", str(p) + "@001", "synthesis")
    assert (result.root / "work/frames/2.png").read_bytes() == original
