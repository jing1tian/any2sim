"""Render the frozen achieved trajectory for the independent realism stage, without a synthesis rollout.

Uses the same RGB/camera/encoding path as synth_render. The frozen protocol selects
native cameras and diagnostic rows. Saves noisy, denoised, processed, decoded PNGs
and linear diffuse/glossy direct/indirect EXR passes at those rows. A full continuous
nominal video per camera supports temporal review. preview_frames is diagnostic only;
it cannot satisfy the encoded/temporal gate. No commands, physics or task predicates
are modified. Run realism_check init, edit protocol, then freeze before rendering.
"""
from pathlib import Path

ENV = "render"
GPU = True


def capture_noisy(scn, png: Path, exr: Path):
    """Same pose/seed/settings, only denoising disabled. EXR retains linear light passes."""
    import bpy
    from ..backends import blender_scene as B
    layer = bpy.context.view_layer
    attributes = ("use_pass_diffuse_direct", "use_pass_diffuse_indirect", "use_pass_diffuse_color",
                  "use_pass_glossy_direct", "use_pass_glossy_indirect", "use_pass_glossy_color")
    previous = {k: getattr(layer, k) for k in attributes}
    denoise = scn.cycles.use_denoising
    fmt, depth = scn.render.image_settings.file_format, scn.render.image_settings.color_depth
    media = getattr(scn.render.image_settings, "media_type", None)
    try:
        for k in attributes:
            setattr(layer, k, True)
        scn.cycles.use_denoising = False
        B.render_to(png)
        if media is not None:  # Blender 5 separates multilayer media from ordinary images.
            scn.render.image_settings.media_type = "MULTI_LAYER_IMAGE"
        scn.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
        scn.render.image_settings.color_depth = "32"
        bpy.data.images["Render Result"].save_render(str(exr), scene=scn)
    finally:
        scn.cycles.use_denoising = denoise
        if media is not None:
            scn.render.image_settings.media_type = media
        scn.render.image_settings.file_format = fmt
        scn.render.image_settings.color_depth = depth
        for k, v in previous.items():
            setattr(layer, k, v)


def run(project: Path, preview_frames: list | None = None, samples: int | None = None,
        denoise: bool | None = None, motion_blur: float | None = None,
        view_transform: str | None = None) -> dict:
    import shutil
    from types import SimpleNamespace
    import yaml
    from .. import realism as R, render_pipeline as RP
    from ..scene_ir import load
    from .synth_render import _render_episode
    scene = load(project / "scene")
    motion = R.require_achieved(project, scene)
    issues = R.protocol_issues(project)
    if issues:
        raise ValueError("; ".join(issues))
    if R.protected_state(project, scene)["fingerprint"] != R.read_json(project / "realism/protected.json")["fingerprint"]:
        raise ValueError("protected trajectory/physics/camera changed")
    protocol = yaml.safe_load((project / "realism/protocol.yaml").read_text())
    frames = sorted({f for s in protocol["cameras"].values() for f in s["fit_frames"] + s["validation_frames"]})
    if any(f >= len(motion["t"]) for f in frames):
        raise ValueError("protocol frame outside achieved motion")
    if preview_frames is not None and (not preview_frames or any(f < 0 or f >= len(motion["t"]) for f in preview_frames)):
        raise ValueError("preview_frames must be nonempty valid achieved-motion rows")
    cfg = {**RP.settings(scene, engine="cycles", samples=samples, denoise=denoise, motion_blur=motion_blur,
                        view_transform=view_transform),
           "cameras": list(protocol["cameras"]), "res_scale": 1.0, "check_frames": len(frames),
           "diagnostic_frames": frames, "preview_frames": preview_frames, "realism_nominal": True,
           "resume": False, "render_labels": False}
    snapshot = RP.baseline(project, scene, cfg)
    cfg["baseline"] = snapshot
    cfg["realism_review"] = {"status": "pending"}
    R.write_json(project / "render/realism_baseline.json", snapshot)
    mid = scene["render"]["motion"]
    rid = scene["motions"][mid]["robot"]
    ep = project / "realism/episode"
    ep.mkdir(parents=True, exist_ok=True)
    (ep / "variant.yaml").write_text("{}\n")
    shutil.copyfile(project / "scene" / scene["motions"][mid]["uri"], ep / "achieved.npz")
    command = scene["motions"][mid].get("derived_from")
    if command not in scene["motions"] or scene["motions"][command].get("kind") != "commanded":
        raise ValueError("achieved motion must identify its upstream commanded motion")
    shutil.copyfile(project / "scene" / scene["motions"][command]["uri"], ep / "commanded.npz")
    # Only the demo's label roles are needed; no synthesis predicate, IK or rollout.
    rigid = [k for k, e in scene["entities"].items() if e.get("capability") == "rigid"]
    ctx = SimpleNamespace(scene=scene, demo=SimpleNamespace(rid=rid, obj=rigid[0] if rigid else None, target=None))
    result = _render_episode(project, ctx, "nominal", cfg, {})
    if RP.baseline(project, load(project / "scene"), cfg)["fingerprint"] != snapshot["fingerprint"]:
        (project / "realism/renders/nominal/render.json").unlink(missing_ok=True)
        raise ValueError("scene/assets changed during diagnostic rendering; rerender frozen inputs")
    return {"complete": result["complete"], "frames": result["frames"], "fingerprint": snapshot["fingerprint"],
            "outputs": ["realism/renders/nominal", "render/realism_baseline.json"]}
