"""Compile the final Blender scene (blender/scene.blend) and render source-view and novel-view images.

Materials use the de-lit albedo textures (solve_albedo) when present, lights are the rig in
work/appearance/lighting.json times its light_scale. The saved .blend is reopened and the reference frame
rendered again; the delivered file must reproduce the in-memory render (max abs difference reported).
Outputs: blender/scene.blend, blender/renders/src_%05d.png, blender/renders/novel_*.png, blender/render.json
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common import read_json, read_mask, read_rgb, write_json

ENV = "render"
GPU = True


def run(project: Path, frames: list[int] | None = None, samples: int = 256, novel_views: bool = True,
        use_albedo: bool = True, all_frames: bool = False) -> dict:
    import bpy
    import yaml
    from ..backends import blender_scene as B
    scene = yaml.safe_load((project / "scene" / "scene.yaml").read_text())
    cams = read_json(project / "work" / "world" / "cameras.json")
    world = read_json(project / "work" / "world" / "world.json")
    app = project / "work" / "appearance"
    lit = read_json(app / "lighting.json")
    scale = float(lit.get("light_scale", 1.0))
    lights = []
    for L in lit["lights"]:
        L = dict(L)
        if L["type"] == "world":
            L["strength"] = float(L.get("strength", 0.3)) * scale
        else:
            L["strength_w"] = float(L.get("strength_w", 100)) * scale
        lights.append(L)
    textures = {}
    if use_albedo:
        for n in scene["entities"]:
            p = app / n / "albedo.png"
            if p.exists():
                textures[n] = p
    out = project / "blender"
    rdir = out / "renders"
    rdir.mkdir(parents=True, exist_ok=True)
    B.reset()
    obs, cam, dev = B.build(project, scene, cams, textures=textures, lights=lights, samples=samples)
    # camera rays that miss the backdrop mesh see the backdrop's mean colour (linear)
    plate = read_rgb(project / "scene" / "assets" / "background" / "plate.png").astype(np.float32) / 255
    bd = read_mask(project / "work" / "masks" / "static" / "backdrop.png") > 0
    c = plate[bd].mean(0) if bd.any() else np.array([0.5, 0.5, 0.5])
    c_lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    B.set_camera_background(tuple(float(x) for x in c_lin))
    B.keyframe_timeline(scene, obs, cam, cams)
    ref = scene["cameras"]["source_cam"]["ref_frame"]
    if frames is None:
        fs = {ref}
        for e in scene["entities"].values():
            for st in e.get("states", {}).values():
                fs |= set(st["fit_frames"]) | set(st["heldout_frames"])
        frames = sorted(fs)
    if all_frames:
        frames = list(range(len(cams["T_world_cam"])))
    sc = bpy.context.scene
    rendered = {}
    for f in frames:
        sc.frame_set(f)
        rendered[f] = B.render_to(rdir / f"src_{f:05d}.png")
    novel = {}
    if novel_views:
        rect = np.asarray(world["table"]["min_area_rect_xy"])
        c = rect.mean(0)
        objs = [np.asarray(e["T_world_entity"])[:3, 3] for e in scene["entities"].values() if e.get("capability") == "rigid"]
        tgt = np.mean(objs, 0) if objs else np.r_[c, 0]
        cam_ref = np.asarray(cams["T_world_cam"][ref])[:3, 3]
        d = np.r_[cam_ref[:2] - tgt[:2], 0]
        d /= np.linalg.norm(d) + 1e-9
        views = {"top": tgt + [0.001, 0.001, 1.1],
                 "front_low": tgt + 0.6 * d + [0, 0, 0.25],
                 "left": tgt + 0.6 * np.r_[-d[1], d[0], 0] + [0, 0, 0.35],
                 "right": tgt + 0.6 * np.r_[d[1], -d[0], 0] + [0, 0, 0.35],
                 "back": tgt - 0.6 * d + [0, 0, 0.45]}
        sc.frame_set(ref)
        ncam = B.add_camera("novel_cam", np.asarray(cams["K"]), cams["width"], cams["height"])   # not keyed
        sc.camera = ncam
        for k, eye in views.items():
            B.set_cv_pose(ncam, B.look_at(eye, tgt, up=(0, 0, 1) if k != "top" else (0, 1, 0)))
            novel[k] = B.render_to(rdir / f"novel_{k}.png")
        sc.camera = cam
        sc.frame_set(ref)
    # deliver: save, reopen, re-render the reference frame and compare with the in-memory render
    blend = out / "scene.blend"
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(blend), compress=True)
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    B.setup_cycles(samples, cams["width"], cams["height"])
    bpy.context.scene.frame_set(ref)
    chk = B.render_to(rdir / f"reopen_{ref:05d}.png")
    a = read_rgb(chk).astype(int)
    b = read_rgb(rendered.get(ref, chk)).astype(int)
    diff = np.abs(a - b)
    res = {"device": dev, "samples": samples, "frames": frames, "novel_views": list(novel), "blend": str(blend.relative_to(project)),
           "textures": {k: str(Path(v).relative_to(project)) for k, v in textures.items()}, "light_scale": scale,
           "reopen_check": {"frame": ref, "mean_abs_diff": float(diff.mean()), "p99_abs_diff": float(np.percentile(diff, 99))}}
    write_json(res, out / "render.json")
    return {**res, "outputs": ["blender"]}
