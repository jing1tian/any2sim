"""Bake per-texel maps of Scene IR entities in Blender (Cycles), in each entity's UV layout.

Maps: E (diffuse irradiance with white albedo, direct + indirect, under the IR lights), POS (object-space position),
NRM (object-space normal), COV (UV coverage). E is what a de-lighting step divides observed radiance by; the
caller decides which other entities are present (their shadows/occlusion) for each bake.

params:
  entities: [names]                    entities to bake (must have UVs)
  kinds: ["E", "POS", "NRM", "COV"]
  sets: {set_name: {entity: state | "hidden"}}   rigid entities' states for E bakes (default {"initial": {}}:
                                       every rigid entity at its initial state, others visible)
  size: null | [w, h]                  default: the entity's texture size (or 1024)
  samples: 64
  out_dir: project-relative (default work/bake)
Writes <out_dir>/<entity>/{pos,nrm,cov}.exr and E_<set>.exr (float, linear), bake.json.
"""
from __future__ import annotations

from pathlib import Path

from ..common import write_json

ENV = "render"
GPU = True


def run(project: Path, entities: list[str], kinds: list[str] | None = None, sets: dict | None = None,
        size: list | None = None, samples: int = 64, out_dir: str = "work/bake") -> dict:
    import bpy
    from ..backends import blender_scene as B
    from ..scene_ir import load
    kinds = kinds or ["E", "POS", "NRM", "COV"]
    sets = sets or {"initial": {}}
    scene = load(project / "scene")
    B.reset()
    obs, cam, dev = B.build(project, scene, samples=samples)
    bpy.context.scene.cycles.samples = samples
    rigid = {n: e for n, e in scene["entities"].items() if e.get("capability") == "rigid"}

    def apply_set(spec):
        for n, e in rigid.items():
            s = spec.get(n, e["initial_state"])
            obs[n].hide_render = s == "hidden"
            if s != "hidden":
                B.set_matrix(obs[n], e["states"][s]["T_world_obj"])

    out = project / out_dir
    info = {"device": dev, "entities": {}, "sets": sets}
    for name in entities:
        ob = obs[name]
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        if size:
            w, h = size
        else:
            imgs = [n.image for n in ob.data.materials[0].node_tree.nodes if n.type == "TEX_IMAGE" and n.image]
            w, h = imgs[0].size if imgs else (1024, 1024)
        rec = {"size": [int(w), int(h)], "E": {}}
        apply_set(next(iter(sets.values())))
        for kind in ("POS", "NRM", "COV"):
            if kind in kinds:
                rec[kind.lower()] = str(Path(B.bake_image(ob, f"{name}_{kind}", w, h, kind, d / f"{kind.lower()}.exr")).relative_to(project))
        if "E" in kinds:
            for sn, spec in sets.items():
                apply_set(spec)
                if ob.hide_render:
                    continue
                rec["E"][sn] = str(Path(B.bake_image(ob, f"{name}_E_{sn}", w, h, "E", d / f"E_{sn}.exr", margin=4)).relative_to(project))
        info["entities"][name] = rec
    write_json(info, out / "bake.json")
    return {"device": dev, "entities": info["entities"], "outputs": [out_dir]}
