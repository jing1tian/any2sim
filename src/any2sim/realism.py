"""Evidence and invariants for the trajectory -> realism -> synthesis handoff.

Numerical checks establish provenance, coverage and ISP nonregression, never likeness.
Only an independent evaluator can approve the visual criteria.
"""
from __future__ import annotations

import copy
import hashlib
import json
import struct
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from . import render_pipeline as RP

CHECKS = ("materials", "lighting", "signal_chain", "surface_scale", "contact", "thin_surfaces",
          "temporal", "perceptual_realism", "reference_fidelity", "integrity")


def local_file(project: Path, name: str) -> Path:
    """Evidence must be portable and inside the project, including symlink targets."""
    path = Path(name)
    resolved = (project / path).resolve()
    if path.is_absolute() or not resolved.is_relative_to(project.resolve()) or not resolved.is_file():
        raise ValueError(f"missing or nonlocal evidence: {name}")
    return resolved


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def protected_state(project: Path, scene: dict) -> dict:
    """Protect dynamics, registration, cameras and motion bytes; allow material/visual edits.

Collision meshes are hashed separately: the bundle's physics_digest alone does not
include mesh contents. Visual-contact agreement remains an independent visual check.
"""
    from .scene_ir import physics_digest, resolve_bundle
    core = {k: copy.deepcopy(scene.get(k)) for k in
            ("units", "frames", "scale", "simulation", "registrations", "motions")}
    core["render_motion"] = scene.get("render", {}).get("motion")
    core["head_camera"] = scene.get("render", {}).get("head_camera")
    core["cameras"] = {k: {a: b for a, b in v.items() if a not in ("isp", "provenance")}
                       for k, v in scene.get("cameras", {}).items()}
    core["entities"] = {}
    files = {}

    def refs(value, base, prefix):
        if isinstance(value, dict):
            for v in value.values():
                refs(v, base, prefix)
        elif isinstance(value, list):
            for v in value:
                refs(v, base, prefix)
        elif isinstance(value, str) and len(value) < 1024 and "\n" not in value:
            p = Path(value)
            p = p if p.is_absolute() else base / p
            try:
                if p.is_file():
                    files[f"{prefix}:{value}"] = RP.file_hash(p)
            except OSError:
                pass

    for name, ent in scene.get("entities", {}).items():
        e = {k: copy.deepcopy(v) for k, v in ent.items() if k not in ("material", "provenance", "geometry")}
        e["collision"] = ent.get("geometry", {}).get("collision")
        core["entities"][name] = e
        refs(e, project / "scene", f"entity:{name}")
    refs(core, project / "scene", "scene")
    core["robots"] = {}
    for name, robot in scene.get("robots", {}).items():
        bd = resolve_bundle(project / "scene", robot["bundle"])
        r = {k: v for k, v in robot.items() if k not in ("bundle", "bundle_sha256", "provenance")}
        r["physics_digest"] = physics_digest(bd)
        core["robots"][name] = r
        urdf = ET.parse(bd / "robot.urdf")
        for mesh in urdf.findall(".//collision/geometry/mesh"):
            uri = mesh.attrib["filename"]
            p = bd / uri
            if not p.is_file():
                raise ValueError(f"unresolved collision mesh: {p}")
            files[f"robot:{name}/{uri}"] = RP.file_hash(p)
    return {"fingerprint": RP.digest({"core": core, "files": files}), "core": core, "files": files}


def require_achieved(project: Path, scene: dict):
    from .scene_ir import load_motion
    mid = scene.get("render", {}).get("motion")
    entry = scene.get("motions", {}).get(mid, {})
    if not scene.get("robots") or entry.get("kind") != "achieved":
        raise ValueError("realism requires render.motion to reference an achieved robot trajectory")
    motion = load_motion(project / "scene", scene, mid)
    t = np.asarray(motion.get("t", []))
    if len(t) < 2 or not np.isfinite(t).all() or not (np.diff(t) > 0).all():
        raise ValueError("achieved trajectory must have finite strictly increasing timestamps")
    if not np.allclose(np.diff(t), np.median(np.diff(t)), rtol=1e-4, atol=1e-7):
        raise ValueError("constant-rate video requires uniformly sampled achieved timestamps; do not silently retime motion")
    return motion


def initialize(project: Path, cameras: list[str] | None = None) -> dict:
    from .scene_ir import load, render_head_camera, robot_bundle
    scene = load(project / "scene")
    motion = require_achieved(project, scene)
    root = project / "realism"
    if (root / "protocol.yaml").exists() or (root / "protected.json").exists():
        raise ValueError("realism already initialized; do not reset the protected trajectory or protocol")
    rid = scene["motions"][scene["render"]["motion"]]["robot"]
    bundle = robot_bundle(project / "scene", scene, rid)
    names = cameras or ["head", *[c for c in ("wrist_left", "wrist_right") if c in bundle.profile.get("cameras", {})]]
    n = len(motion["t"])
    frames = sorted(set(np.linspace(0, n - 1, min(n, 6)).round().astype(int).tolist()))
    specs = {}
    for name in names:
        camera = scene["cameras"][render_head_camera(scene)] if name == "head" else bundle.profile["cameras"][name]
        specs[name] = {"reference_status": "unavailable", "reference_note": "Fill in paired references or explain absence.",
                       "native_size": [camera["width"], camera["height"]],
                       "fit_frames": frames[::2], "validation_frames": frames[1::2],
                       "pairs": {}, "regions": [], "min_static_pixels": 256,
                       "max_mae_regression_dn": 0.0}
    protocol = {"schema_version": 1, "cameras": specs,
                "thin_surfaces": True, "ineffective_trial_limit": 2,
                "notes": "Declare references, static masks and surface/contact regions before freezing. Frame indices are achieved-motion rows."}
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "protected.json", protected_state(project, scene))
    (root / "protocol.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False))
    return {"protocol": "realism/protocol.yaml", "protected": "realism/protected.json", "status": "draft"}


def validate_protocol(project: Path, protocol: dict):
    if protocol.get("schema_version") != 1 or not protocol.get("cameras"):
        raise ValueError("realism protocol schema_version 1 and cameras required")
    for camera, spec in protocol["cameras"].items():
        w, h = spec["native_size"]
        if w <= 0 or h <= 0 or int(w) != w or int(h) != h:
            raise ValueError(f"{camera}: invalid native_size")
        fit, held = spec["fit_frames"], spec["validation_frames"]
        if not fit or not held or set(fit) & set(held):
            raise ValueError(f"{camera}: disjoint nonempty fit/validation frames required")
        if any(type(f) is not int or f < 0 for f in fit + held):
            raise ValueError(f"{camera}: invalid frame indices")
        if spec["min_static_pixels"] < 1 or not np.isfinite(spec["max_mae_regression_dn"]) or spec["max_mae_regression_dn"] < 0:
            raise ValueError(f"{camera}: invalid camera validation limits")
        if spec["reference_status"] not in ("paired", "unavailable"):
            raise ValueError(f"{camera}: reference_status must be paired or unavailable")
        if spec["reference_status"] == "unavailable" and (not spec.get("reference_note") or spec["reference_note"].startswith("Fill in")):
            raise ValueError(f"{camera}: explain missing paired references")
        if not spec.get("regions"):
            raise ValueError(f"{camera}: declare surface/contact regions")
        for region in spec["regions"]:
            x0, y0, x1, y1 = region["box"]
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h) or any(type(x) is not int for x in region["box"]):
                raise ValueError(f"{camera}: invalid region box")
        if spec["reference_status"] == "paired":
            for frame in fit + held:
                pair = spec.get("pairs", {}).get(frame, spec.get("pairs", {}).get(str(frame), {}))
                local_file(project, pair["reference"])
                if pair.get("static_mask"):
                    local_file(project, pair["static_mask"])


def freeze_protocol(project: Path) -> dict:
    root = project / "realism"
    protocol = yaml.safe_load((root / "protocol.yaml").read_text())
    validate_protocol(project, protocol)
    from .scene_ir import load
    protected = read_json(root / "protected.json")
    if protected_state(project, load(project / "scene"))["fingerprint"] != protected["fingerprint"]:
        raise ValueError("trajectory/physics/camera changed before protocol freeze")
    references = {}
    for spec in protocol["cameras"].values():
        for pair in spec.get("pairs", {}).values():
            for name in ("reference", "static_mask"):
                if pair.get(name):
                    references[pair[name]] = RP.file_hash(local_file(project, pair[name]))
    lock = {"protocol_sha256": RP.file_hash(root / "protocol.yaml"),
            "protected_sha256": RP.file_hash(root / "protected.json"), "reference_hashes": references}
    target = root / "protocol.lock.json"
    if target.exists() and read_json(target) != lock:
        raise ValueError("protocol is frozen; changed eligibility, references or thresholds need a new declared evaluation protocol")
    write_json(target, lock)
    return lock


def protocol_issues(project: Path) -> list[str]:
    try:
        root = project / "realism"
        lock = read_json(root / "protocol.lock.json")
        issues = []
        for name in ("protocol", "protected"):
            p = root / ("protocol.yaml" if name == "protocol" else "protected.json")
            if RP.file_hash(p) != lock[f"{name}_sha256"]:
                issues.append(f"frozen {name} changed")
        for name, sha in lock["reference_hashes"].items():
            if RP.file_hash(local_file(project, name)) != sha:
                issues.append(f"frozen reference changed: {name}")
        validate_protocol(project, yaml.safe_load((root / "protocol.yaml").read_text()))
        return issues
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [f"missing/invalid frozen realism protocol: {exc}"]


LIGHT_PASSES = ("Diffuse Direct", "Diffuse Indirect", "Diffuse Color",
                "Glossy Direct", "Glossy Indirect", "Glossy Color")


def exr_metadata(path: Path) -> dict:
    """Validate scanline EXR headers/pass channels and stored chunk bounds, not visual quality.

    Supports Blender's single-part and multipart multilayer outputs without requiring
    a GPU or an optional EXR decoder in the CPU evidence environment.
    """
    size = path.stat().st_size
    channels, parts = [], []
    with path.open("rb") as stream:
        def read(n):
            if n < 0 or n > size:
                raise ValueError("invalid EXR field size")
            value = stream.read(n)
            if len(value) != n:
                raise ValueError("truncated EXR")
            return value

        def string():
            result = bytearray()
            while True:
                b = read(1)
                if b == b"\0":
                    return result.decode("utf8")
                result.extend(b)
                if len(result) > 255:
                    raise ValueError("invalid EXR attribute name")

        magic, version = struct.unpack("<II", read(8))
        if magic != 20000630 or version & 0xff != 2 or version & 0xa00:
            raise ValueError("expected a non-deep scanline EXR")
        multipart = bool(version & 0x1000)
        while True:
            attributes = {}
            while True:
                name = string()
                if not name:
                    break
                kind = string()
                value = read(struct.unpack("<I", read(4))[0])
                attributes[name] = value
                if kind == "chlist":
                    offset = 0
                    while offset < len(value) and value[offset]:
                        end = value.index(b"\0", offset)
                        channels.append(value[offset:end].decode("utf8"))
                        offset = end + 17
                    if offset != len(value) - 1:
                        raise ValueError("invalid EXR channels")
            if not attributes:
                break
            box = struct.unpack("<4i", attributes["dataWindow"])
            width, height = box[2] - box[0] + 1, box[3] - box[1] + 1
            if width <= 0 or height <= 0:
                raise ValueError("invalid EXR dimensions")
            block_lines = {0: 1, 1: 1, 2: 1, 3: 16, 4: 32, 5: 16, 6: 32, 7: 32, 8: 32, 9: 256}
            compression = attributes["compression"][0]
            chunks = struct.unpack("<i", attributes["chunkCount"])[0] if multipart else (height + block_lines[compression] - 1) // block_lines[compression]
            if chunks < 1 or chunks * 8 > size:
                raise ValueError("invalid EXR chunk count")
            parts.append({"width": width, "height": height, "chunks": chunks})
            if not multipart:
                break
        count = sum(p["chunks"] for p in parts)
        offsets = [struct.unpack("<Q", read(8))[0] for _ in range(count)]
        data_start = stream.tell()
        for offset in offsets:
            if offset < data_start or offset >= size:
                raise ValueError("missing EXR pixel chunk")
            stream.seek(offset)
            if multipart:
                part = struct.unpack("<i", read(4))[0]
                if not 0 <= part < len(parts):
                    raise ValueError("invalid EXR part")
            _, length = struct.unpack("<iI", read(8))
            if not length or stream.tell() + length > size:
                raise ValueError("truncated EXR pixel chunk")
    for light in LIGHT_PASSES:
        if not all(any(c.endswith(f".{light}.{color}") for c in channels) for color in "RGB"):
            raise ValueError(f"missing EXR light pass: {light}")
    return {"parts": parts, "channels": channels, "validation": "headers and chunk bounds; evaluator must inspect pass pixels"}


def image_stats(image: np.ndarray) -> dict:
    import cv2
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(float)
    return {"mean_rgb": image.mean(axis=(0, 1)).tolist(),
            "white_clip_fraction": float(np.mean(np.any(image >= 250, axis=2))),
            "black_clip_fraction": float(np.mean(np.all(image <= 5, axis=2))),
            "gradient_energy": float(np.mean(np.diff(gray, axis=0) ** 2) + np.mean(np.diff(gray, axis=1) ** 2))}


def check(project: Path, manifest: str = "realism/renders/nominal/diagnostics.json", out: str = "realism/check.json") -> dict:
    """Measure actual images; reject absent/stale evidence. No subjective automatic pass."""
    import cv2
    from .common import read_rgb
    from .camera_isp import for_camera
    from .scene_ir import load, robot_bundle
    issues = protocol_issues(project)
    if issues:
        result = {"status": "blocked", "issues": issues}
        write_json(project / out, result)
        return result
    scene = load(project / "scene")
    protocol = yaml.safe_load((project / "realism/protocol.yaml").read_text())
    protected = read_json(project / "realism/protected.json")
    if protected_state(project, scene)["fingerprint"] != protected["fingerprint"]:
        issues.append("protected trajectory/physics/camera changed; return to trajectory stage")
    data = read_json(local_file(project, manifest))
    if not data.get("complete"):
        issues.append("diagnostic preview is not a complete nominal video")
    snapshot = RP.baseline(project, scene, data["settings"])
    if data.get("fingerprint") != snapshot["fingerprint"]:
        issues.append("stale diagnostic renders")
    evidence = {manifest: RP.file_hash(local_file(project, manifest)),
                "realism/protocol.yaml": RP.file_hash(project / "realism/protocol.yaml"),
                "realism/protected.json": RP.file_hash(project / "realism/protected.json"),
                "realism/protocol.lock.json": RP.file_hash(project / "realism/protocol.lock.json")}
    evidence.update(read_json(project / "realism/protocol.lock.json")["reference_hashes"])
    rid = scene["motions"][scene["render"]["motion"]]["robot"]
    bundle = robot_bundle(project / "scene", scene, rid)
    cameras = {}
    motion = require_achieved(project, scene)
    for camera, spec in protocol["cameras"].items():
        measurements, seen_refs, heldout = [], {"fit": set(), "validation": set()}, []
        isp = for_camera(scene, camera, bundle.profile.get("cameras", {}))
        temporal = {}
        try:
            import imageio.v2 as imageio
            video = local_file(project, data["videos"][camera])
            evidence[data["videos"][camera]] = RP.file_hash(video)
            previous, delta, decoded, count = None, [], {}, 0
            reader = imageio.get_reader(str(video))
            try:
                fps = reader.get_meta_data()["fps"]
                for index, rgb in enumerate(reader):
                    if list(rgb.shape[1::-1]) != spec["native_size"]:
                        raise ValueError("video resolution differs from protocol")
                    if index in spec["fit_frames"] + spec["validation_frames"]:
                        decoded[str(index)] = hashlib.sha256(rgb.tobytes()).hexdigest()
                    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(float)
                    if previous is not None:
                        delta.append(float(np.abs(gray - previous).mean()))
                    previous = gray
                    count += 1
            finally:
                reader.close()
            if count != len(motion["t"]) or abs(fps - 1 / np.median(np.diff(motion["t"]))) > .01:
                raise ValueError("video frame count/fps differs from protected achieved motion")
            temporal = {"frames": count, "fps": fps, "adjacent_gray_mae_dn": delta,
                        "exact_duplicate_pairs": sum(x == 0 for x in delta),
                        "peak_change_frame": int(np.argmax(delta)) + 1 if delta else None}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            decoded = {}
            issues.append(f"{camera}: invalid continuous video: {exc}")
        for frame in spec["fit_frames"] + spec["validation_frames"]:
            try:
                row = data["captures"][camera][str(frame)]
                images = {}
                for stage in ("noisy", "denoised", "processed", "decoded"):
                    p = local_file(project, row[stage])
                    evidence[row[stage]] = RP.file_hash(p)
                    images[stage] = read_rgb(p)
                    if list(images[stage].shape[1::-1]) != spec["native_size"]:
                        raise ValueError(f"{stage}: image is not at declared native size")
                if decoded.get(str(frame)) != hashlib.sha256(images["decoded"].tobytes()).hexdigest():
                    raise ValueError("decoded checkpoint does not match the actual video frame")
                if not RP.pixel_processing(isp) and not np.array_equal(images["denoised"], images["processed"]):
                    raise ValueError("declared identity ISP changed pixels")
                exr = local_file(project, row["light_passes"])
                if exr.suffix.lower() != ".exr":
                    raise ValueError("light passes must be an EXR file")
                exr_metadata(exr)
                evidence[row["light_passes"]] = RP.file_hash(exr)
                m = {"frame": frame, "regions": {}}
                for region in spec["regions"]:
                    x0, y0, x1, y1 = region["box"]
                    m["regions"][region["name"]] = {k: image_stats(v[y0:y1, x0:x1]) for k, v in images.items()}
                m["stage_mae_dn"] = {f"{a}_to_{b}": float(np.abs(images[a].astype(float) - images[b]).mean())
                                     for a, b in (("noisy", "denoised"), ("denoised", "processed"), ("processed", "decoded"))}
                if spec["reference_status"] == "paired":
                    pair = spec["pairs"].get(frame, spec["pairs"].get(str(frame)))
                    ref_path = local_file(project, pair["reference"])
                    evidence[pair["reference"]] = RP.file_hash(ref_path)
                    ref = read_rgb(ref_path)
                    if ref.shape != images["processed"].shape:
                        raise ValueError("paired reference resolution mismatch")
                    split = "fit" if frame in spec["fit_frames"] else "validation"
                    # Content hash detects duplicate frames even when differently named/encoded.
                    seen_refs[split].add(hashlib.sha256(ref.tobytes()).hexdigest())
                    if frame in spec["validation_frames"] and RP.pixel_processing(isp):
                        mask_path = local_file(project, pair["static_mask"])
                        evidence[pair["static_mask"]] = RP.file_hash(mask_path)
                        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        if mask is None or mask.shape != ref.shape[:2]:
                            raise ValueError("invalid static mask")
                        valid = mask > 0
                        m["static_pixels"] = int(valid.sum())
                        if valid.sum() < spec["min_static_pixels"]:
                            raise ValueError("insufficient eligible static pixels for ISP validation")
                        raw = float(np.abs(ref.astype(float) - images["denoised"])[valid].mean())
                        processed = float(np.abs(ref.astype(float) - images["processed"])[valid].mean())
                        m["heldout_mae_dn"] = {"raw": raw, "processed": processed}
                        heldout.append(processed <= raw + spec["max_mae_regression_dn"])
                measurements.append(m)
            except (OSError, ValueError, KeyError, TypeError, struct.error, IndexError) as exc:
                issues.append(f"{camera} frame {frame}: {exc}")
        duplicate = bool(seen_refs["fit"] & seen_refs["validation"])
        if duplicate:
            issues.append(f"{camera}: fit/validation reference content overlaps")
        nonidentity = RP.pixel_processing(isp)
        validated = bool(heldout) and all(heldout) and len(heldout) == len(spec["validation_frames"]) and not duplicate
        if nonidentity and not validated:
            issues.append(f"{camera}: nonidentity ISP lacks passing held-out evidence; set its explicit camera_isp override to {{}} and rerender")
        cameras[camera] = {"pixel_processing": nonidentity, "isp": isp,
                           "processing_status": "validated" if nonidentity and validated else "identity" if not nonidentity else "rejected",
                           "reference_status": spec["reference_status"], "measurements": measurements, "temporal": temporal}
    trials_path = project / "realism/trials.json"
    if trials_path.exists():
        trials = json.loads(trials_path.read_text())
        counts = {}
        for trial in trials:
            hypothesis = trial["hypothesis"]
            counts[hypothesis] = counts.get(hypothesis, 0) + 1 if trial["outcome"] == "ineffective" else 0
            if counts[hypothesis] >= protocol.get("ineffective_trial_limit", 2):
                if not trial.get("next_hypothesis") and not trial.get("stop_reason"):
                    issues.append(f"ineffective trial limit: {hypothesis}; record a new hypothesis or stop_reason")
        evidence["realism/trials.json"] = RP.file_hash(trials_path)
    result = {"schema_version": 1, "status": "blocked" if issues else "ready_for_review", "issues": issues,
              "fingerprint": snapshot["fingerprint"], "cameras": cameras, "evidence_hashes": evidence,
              "note": "Coverage, integrity and image measurements only; independent visual evaluation still required."}
    write_json(project / out, result)
    return result


def review_issues(project: Path, snapshot: dict, cameras: list[str], report: dict) -> list[str]:
    """Called at render/export and stage transition; never trust status alone."""
    issues = protocol_issues(project)
    if report.get("schema_version") != 2 or report.get("stage") != "realism":
        issues.append("independent realism stage review schema_version 2 required")
    if not report.get("evaluator") or not report.get("builder") or report.get("evaluator") == report.get("builder"):
        issues.append("distinct builder/evaluator identities required")
    for key in (*CHECKS, *(f"camera:{c}" for c in cameras)):
        row = report.get("checks", {}).get(key, {})
        if row.get("status") != "pass" or not row.get("evidence") or not row.get("note"):
            issues.append(f"{key}: independent finding and evidence required")
    try:
        from .scene_ir import load
        request_path = project / "request.yaml"
        request = (yaml.safe_load(request_path.read_text()) or {}) if request_path.exists() else {}
        protocol = yaml.safe_load((project / "realism/protocol.yaml").read_text())
        if request.get("acceptance", {}).get("fidelity") == "precise":
            if any(s["reference_status"] != "paired" for s in protocol["cameras"].values()):
                issues.append("precise fidelity requires paired references for every camera")
        protected = read_json(project / "realism/protected.json")
        if protected_state(project, load(project / "scene"))["fingerprint"] != protected["fingerprint"]:
            issues.append("protected trajectory/physics/camera changed")
        check_path = report["diagnostics"]
        measured = read_json(local_file(project, check_path))
        if measured.get("status") != "ready_for_review" or measured.get("fingerprint") != snapshot["fingerprint"]:
            issues.append("missing, failed or stale realism diagnostics")
        hashes = report.get("evidence_hashes", {})
        required = {check_path, "realism/protocol.yaml", "realism/protocol.lock.json", "realism/protected.json",
                    *measured.get("evidence_hashes", {}),
                    *read_json(project / "realism/protocol.lock.json")["reference_hashes"]}
        for row in report.get("checks", {}).values():
            required.update(row.get("evidence", []))
        for name in required:
            actual = RP.file_hash(local_file(project, name))
            if hashes.get(name) != actual:
                issues.append(f"review evidence changed or unhashed: {name}")
            if name in measured.get("evidence_hashes", {}) and measured["evidence_hashes"][name] != actual:
                issues.append(f"diagnostic input changed: {name}")
        for camera in cameras:
            if camera not in measured.get("cameras", {}):
                issues.append(f"camera:{camera}: missing measured evidence")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        issues.append(f"invalid realism stage evidence: {exc}")
    return issues


def adopt_review(project: Path, path: str) -> dict:
    """Publish an evaluator's decision after validating it; never manufacture a pass."""
    if not Path(path).parts or Path(path).parts[0] != "evaluations":
        raise ValueError("review must come from evaluations/<candidate>/realism_review.json")
    report = read_json(local_file(project, path))
    from .scene_ir import load
    saved = read_json(project / "render/realism_baseline.json")
    snapshot = RP.baseline(project, load(project / "scene"), saved["settings"])
    if report.get("fingerprint") != snapshot["fingerprint"]:
        raise ValueError("evaluator reviewed a different baseline")
    # Evidence hashes must be recorded by the evaluator, not filled in after review.
    issues = review_issues(project, snapshot, saved["settings"]["cameras"], report)
    if report.get("status") != "pass" or issues:
        raise ValueError("realism review not accepted: " + "; ".join(issues or [report.get("status", "missing status")]))
    write_json(project / "render/realism_review.json", report)
    return {"status": "pass", "fingerprint": snapshot["fingerprint"], "report": "render/realism_review.json"}


def fallback_isp(project: Path, measured_path: str = "realism/check.json") -> dict:
    """Explicitly reset only rejected camera processing; retain history and require rerender."""
    from .scene_ir import load, save
    scene = load(project / "scene")
    measured = read_json(local_file(project, measured_path))
    saved = read_json(project / "render/realism_baseline.json")
    current = RP.baseline(project, scene, saved["settings"])
    if measured.get("fingerprint") != current["fingerprint"]:
        raise ValueError("stale camera diagnosis; remeasure before applying fallback")
    if protocol_issues(project) or any(
            RP.file_hash(local_file(project, name)) != expected
            for name, expected in measured.get("evidence_hashes", {}).items()):
        raise ValueError("stale camera evidence/protocol; remeasure before applying fallback")
    cameras = [c for c, result in measured.get("cameras", {}).items()
               if result.get("processing_status") == "rejected"]
    if not cameras:
        raise ValueError("no rejected camera correction to reset")
    before = protected_state(project, scene)
    history_path = project / "realism/isp_fallback_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history.append({"baseline": current["fingerprint"], "diagnostics": measured_path,
                    "diagnostics_sha256": RP.file_hash(project / measured_path),
                    "previous": {c: measured["cameras"][c]["isp"] for c in cameras},
                    "reason": "Rejected/unsupported held-out correction; explicit identity, not calibration."})
    for camera in cameras:
        scene.setdefault("render", {}).setdefault("camera_isp", {})[camera] = {}
    if protected_state(project, scene) != before:
        raise ValueError("camera fallback would alter protected execution")
    write_json(history_path, history)
    save(project / "scene", scene)
    return {"status": "rerender_required", "cameras": cameras,
            "history": "realism/isp_fallback_history.json",
            "note": "Old renders/reviews are stale. Render, measure and independently review again."}
