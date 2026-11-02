"""Bake per-texel lighting and geometry maps in Blender for the appearance solve.

For the table and every rigid object: object-space position, normal and UV coverage (once), and diffuse
irradiance E under the current lighting rig (white-albedo DIFFUSE bake, direct + indirect) for every rest state
of the object, with the other objects at their pose at that time. solve_albedo divides observed radiance by E.
Lighting rig: work/appearance/lighting.json if present, else a default overhead area light + ambient world
(written there, provenance "default rig").
Outputs: work/appearance/bake/<entity>/{pos,nrm,cov,E_<state>}.exr, work/appearance/bake/bake.json
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common import read_json, write_json

ENV = "render"
GPU = True


def default_rig(world: dict) -> list:
    foot = np.asarray(world["table"]["camera_foot_xy"])
    rect = np.asarray(world["table"]["min_area_rect_xy"])
    centre = rect.mean(0)
    p = 0.5 * (foot + centre)
    return [{"name": "key", "type": "area", "position": [float(p[0]), float(p[1]), 1.6],
             "target": [float(p[0]), float(p[1]), 0.0], "size": [1.2, 1.2], "strength_w": 400.0},
            {"type": "world", "color": [1.0, 1.0, 1.0], "strength": 0.35}]


def run(project: Path, lights: list | None = None, bake_samples: int = 64, table_max_px: int = 2048) -> dict:
    import bpy
    import yaml
    from ..backends import blender_scene as B
    scene = yaml.safe_load((project / "scene" / "scene.yaml").read_text())
    cams = read_json(project / "work" / "world" / "cameras.json")
    world = read_json(project / "work" / "world" / "world.json")
    app = project / "work" / "appearance"
    app.mkdir(parents=True, exist_ok=True)
    lpath = app / "lighting.json"
    if lights is None:
        lights = read_json(lpath)["lights"] if lpath.exists() else default_rig(world)
    lit = read_json(lpath) if lpath.exists() else {}
    write_json({**lit, "lights": lights, "provenance": lit.get("provenance", "default rig (not fitted)")}, lpath)
    B.reset()
    obs, cam, dev = B.build(project, scene, cams, lights=lights, samples=bake_samples)
    bpy.context.scene.cycles.samples = bake_samples
    rigid = {n: e for n, e in scene["entities"].items() if e.get("capability") == "rigid"}
    out = app / "bake"
    info = {"device": dev, "lights": lights, "entities": {}}

    def tex_size(ob):
        img = next(n.image for n in ob.data.materials[0].node_tree.nodes if n.type == "TEX_IMAGE")
        w, h = img.size
        s = min(1.0, table_max_px / max(w, h))
        return int(w * s), int(h * s)

    def set_combo(t_frame, focus=None, focus_state=None):
        for n, e in rigid.items():
            s = focus_state if n == focus else B.state_at(e, t_frame)
            obs[n].hide_render = s is None
            if s is not None:
                B.set_matrix(obs[n], e["states"][s]["T_world_obj"])

    ref = scene["cameras"]["source_cam"]["ref_frame"]
    for name in ["table"] + list(rigid):
        ob = obs[name]
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        w, h = tex_size(ob)
        rec = {"size": [w, h], "E": {}}
        set_combo(ref)
        for kind in ("POS", "NRM", "COV"):
            rec[kind.lower()] = B.bake_image(ob, f"{name}_{kind}", w, h, kind, d / f"{kind.lower()}.exr", margin=0)
        if name == "table":
            # the clean plate shows only objects present for the whole clip ("permanent"): de-light the table
            # with those present, and with none under their footprints (inpainted there)
            n_frames = len(cams["T_world_cam"])
            permanent = [n for n, e in rigid.items() if any(st["frames"][1] - st["frames"][0] + 1 >= 0.9 * n_frames
                                                          for st in e["states"].values())]
            for n in rigid:
                obs[n].hide_render = n not in permanent
            rec["E"]["static"] = B.bake_image(ob, f"{name}_Es", w, h, "E", d / "E_static.exr", margin=4)
            for n in rigid:
                obs[n].hide_render = True
            rec["E"]["empty"] = B.bake_image(ob, f"{name}_Ee", w, h, "E", d / "E_empty.exr", margin=4)
            rec["permanent_objects"] = permanent
            set_combo(ref)
        else:
            for s, st in rigid[name]["states"].items():
                set_combo(st["frames"][0], name, s)
                rec["E"][s] = B.bake_image(ob, f"{name}_E_{s}", w, h, "E", d / f"E_{s}.exr", margin=4)
        info["entities"][name] = rec
    write_json(info, out / "bake.json")
    return {"device": dev, "entities": {k: {"size": v["size"], "states": list(v["E"])} for k, v in info["entities"].items()},
            "outputs": ["work/appearance/bake", "work/appearance/lighting.json"]}
