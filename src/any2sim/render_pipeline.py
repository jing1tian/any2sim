"""Fixed rendering baseline, independent of episode randomization.

Rendering settings do not establish realism. A reviewed nominal comparison is
required before full synthesis renders; previews and nominal renders bootstrap it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def pixel_processing(isp: dict) -> bool:
    return bool(set(isp) & {"color_matrix", "vignette", "tone_lut", "levels", "saturation", "blur_sigma_px",
                            "sharpen_amount", "noise", "chroma_shift_px"})


def calibration_renders(directory: Path) -> Path:
    """Never fit another ISP onto already processed trajectory renders."""
    manifest = directory / "render.json"
    model = json.loads(manifest.read_text()).get("camera_model", {}) if manifest.exists() else {}
    if model.get("enabled") and model.get("parameters"):
        raw = directory / "raw"
        if not raw.is_dir():
            raise ValueError("Processed renders cannot calibrate ISP: raw renders are missing")
        return raw
    return directory


def settings(scene: dict, **overrides) -> dict:
    cfg = dict(engine="cycles", samples=128, denoise=True, motion_blur=0.5,
               max_bounces=12, diffuse_bounces=4, glossy_bounces=4,
               adaptive_threshold=0.01, view_transform="Standard")
    baseline = scene.get("render", {}).get("realism_baseline", {})
    cfg.update({k: baseline[k] for k in cfg if k in baseline})
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    if cfg["engine"] not in ("cycles", "eevee") or int(cfg["samples"]) < 1:
        raise ValueError("render engine must be cycles/eevee and samples must be positive")
    if float(cfg["motion_blur"]) < 0:
        raise ValueError("motion_blur must be nonnegative")
    if not 0 <= float(cfg["adaptive_threshold"]) <= 1:
        raise ValueError("adaptive_threshold must be between 0 and 1; zero disables adaptive sampling")
    if cfg["view_transform"] not in ("Standard", "AgX"):
        raise ValueError("view_transform must be Standard or AgX")
    return cfg


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def baseline(project: Path, scene: dict, cfg: dict) -> dict:
    """Bind review/cache to assets, calibration, motion and actual render settings."""
    from .scene_ir import resolve_bundle
    roots = {"scene": project / "scene"}
    for rid, robot in scene.get("robots", {}).items():
        roots[f"robot:{rid}"] = resolve_bundle(project / "scene", robot["bundle"])
    files = {}
    for name, root in roots.items():
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(root)
            if path.is_file() and not any(x in rel.parts for x in ("source", "checks", "__pycache__")):
                files[f"{name}/{rel}"] = file_hash(path)
    # External file references in the IR (e.g. shared textures) also invalidate it.
    def refs(value):
        if isinstance(value, dict):
            for v in value.values():
                refs(v)
        elif isinstance(value, list):
            for v in value:
                refs(v)
        elif isinstance(value, str) and len(value) < 1024 and "\n" not in value:
            path = Path(value)
            path = path if path.is_absolute() else project / "scene" / path
            try:
                if path.is_file():
                    files[f"reference:{value}"] = file_hash(path)
            except OSError:
                pass  # Descriptive strings are not filesystem references.
    refs(scene)
    keys = (*settings(scene).keys(), "cameras", "res_scale", "resolution", "view_transform")
    config = {k: cfg[k] for k in keys if k in cfg}
    code = Path(__file__).parent
    for name in ("render_pipeline.py", "realism.py", "camera_isp.py", "common.py", "backends/blender_scene.py",
                 "tools/synth_render.py", "tools/realism_render.py"):
        files[f"code/{name}"] = file_hash(code / name)
    payload = {"scene": scene, "files": files, "settings": config}
    return {"fingerprint": digest(payload), "settings": config, "files": files}


def review(project: Path, snapshot: dict, cameras: list[str], required=False) -> dict:
    path = project / "render" / "realism_review.json"
    problems = []
    try:
        report = json.loads(path.read_text()) if path.exists() else {}
        if not isinstance(report, dict):
            report = {}
    except (OSError, ValueError):
        report = {}
    if report.get("fingerprint") != snapshot["fingerprint"]:
        problems.append("missing or stale nominal realism review")
    for name in ("materials", "lighting", "temporal", *(f"camera:{c}" for c in cameras)):
        check = report.get("checks", {}).get(name, {})
        evidence = check.get("evidence", [])
        if (check.get("status") != "pass" or not evidence
                or not all(isinstance(p, str) and (project / p).is_file() for p in evidence)):
            problems.append(f"{name}: passing review with existing evidence required")
    if report.get("status") != "pass":
        problems.append("nominal realism review has not passed")
    from .realism import review_issues
    problems.extend(review_issues(project, snapshot, cameras, report))
    result = {"status": "pending" if problems else "pass", "fingerprint": snapshot["fingerprint"],
              "report": str(path.relative_to(project)), "issues": problems}
    if required and problems:
        raise ValueError("Realism baseline not ready: " + "; ".join(problems)
                         + ". Render nominal/previews, fix appearance and calibrate each camera, then obtain "
                         "an independent render/realism_review.json for render/realism_baseline.json.")
    return result


def episode_fingerprint(ep: Path, snapshot: dict, cfg: dict) -> str:
    return digest({"baseline": snapshot["fingerprint"],
                   "files": {n: file_hash(ep / n) for n in ("variant.yaml", "achieved.npz", "commanded.npz")},
                   "config": {k: v for k, v in cfg.items() if k not in ("resume", "baseline", "realism_review")}})


def apply_settings(scn, cfg: dict, animated=True):
    """Apply the same physical render settings to both render entry points."""
    scn.render.engine = "CYCLES" if cfg["engine"] == "cycles" else "BLENDER_EEVEE"
    scn.cycles.samples = int(cfg["samples"])
    scn.cycles.use_denoising = bool(cfg["denoise"])
    scn.view_settings.view_transform = cfg["view_transform"]
    scn.view_settings.look = "None"
    threshold = float(cfg.get("adaptive_threshold", 0.01))
    scn.cycles.use_adaptive_sampling = threshold > 0
    if threshold > 0:
        scn.cycles.adaptive_threshold = threshold
    for name in ("max_bounces", "diffuse_bounces", "glossy_bounces"):
        setattr(scn.cycles, name, int(cfg[name]))
    shutter = float(cfg["motion_blur"]) if animated else 0
    scn.render.use_motion_blur = shutter > 0
    if shutter:
        scn.render.motion_blur_shutter = shutter
