"""Smoke test of every general tool on a small project (GPU server). Not a unit test: run by hand.

  envs/base/bin/python tests/smoke_tools.py <video> [--reuse projects/<old project with world/cameras.json>]

Creates projects/_smoke, plays the agent's part with a few lines of its own code (a minimal v0.2 Scene IR), and
calls each tool once. Prints one line per tool: status, seconds, short result.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2sim import meshkit as mk  # noqa: E402
from any2sim.project import Project  # noqa: E402
from any2sim.scene_ir import save, validate  # noqa: E402
from any2sim.toolrun import run_tool  # noqa: E402

STATUS = []


def step(proj, tool, params):
    t = time.time()
    r = run_tool(proj, tool, params)
    res = r["result"] or {}
    short = {k: v for k, v in res.items() if k not in ("outputs",) and not isinstance(v, (dict, list))} if isinstance(res, dict) else res
    line = f"{tool:20s} {r['status']:6s} {time.time() - t:7.1f}s  {json.dumps(short, default=str)[:300]}"
    print(line, flush=True)
    if r["status"] != "ok":
        print(r.get("log_tail", "")[-1500:], flush=True)
    STATUS.append((tool, r["status"]))
    return res


def main():
    video, old = sys.argv[1], None
    if "--reuse" in sys.argv:
        old = ROOT / sys.argv[sys.argv.index("--reuse") + 1]
    root = ROOT / "projects" / "_smoke"
    if root.exists():
        shutil.rmtree(root)
    proj = Project.create(root, {"video": video})
    step(proj.root, "ingest_video", {"video": video})
    step(proj.root, "sam3_segment", {"prompts": ["basket", "yellow bell pepper", "hand", "glove"], "images": [0, 40],
                                     "out_dir": "work/sam3_seed"})
    meta = json.loads((root / "work/sam3_seed/meta.json").read_text())["images"]["frame00000"]
    seeds = [{"obj_id": 1, "frame": 0, "mask": meta["basket"][0]["mask"]},
             {"obj_id": 2, "frame": 0, "mask": meta["yellow bell pepper"][0]["mask"]}]
    step(proj.root, "sam2_track", {"seeds": seeds, "names": {"1": "basket", "2": "pepper"}})
    step(proj.root, "sam3_segment", {"prompts": ["hand", "glove", "arm", "VR controller"], "images": [0, 40, 80, 120],
                                     "out_dir": "work/dyn", "save_instances": False})
    # dynamic masks under the frame%05d naming expected by mask specs
    (root / "work/dyn/frames").mkdir(parents=True, exist_ok=True)
    for f in (root / "work/dyn/union").glob("*.png"):
        shutil.copy(f, root / "work/dyn/frames" / f.name)
    step(proj.root, "moge_geometry", {"frames": [0]})
    step(proj.root, "pi3x_geometry", {"frames": [0, 40, 80]})
    step(proj.root, "sam3d_generate", {"frame": 0, "mask": seeds[1]["mask"], "pointmap": "work/moge/frame00000.npz",
                                       "out_dir": "work/sam3d/pepper"})

    # --- the agent's part: a minimal Scene IR from an earlier project's cameras and meshes ---
    sd = root / "scene"
    (sd / "assets").mkdir(parents=True, exist_ok=True)
    shutil.copy(old / "work/world/cameras.json", sd / "cameras.json")
    cams = json.loads((sd / "cameras.json").read_text())
    oldsc = __import__("yaml").safe_load((old / "scene/scene.yaml").read_text())
    table = mk.uv_planar_metres(mk.box([1.2, 1.0, 0.03], center=(0, -0.2, -0.015)))
    mk.save_obj(table, sd / "assets/table.obj")
    shutil.copy(old / "scene/assets/background/plate.png", root / "work/plate.png")
    step(proj.root, "synthesize_texture", {"exemplar": "work/plate.png", "rect": [700, 180, 1000, 300],
                                               "size": [256, 256], "patch": 32, "overlap": 8, "out": "scene/assets/table_tex.png"})
    ents = {"table": {"category": "table", "capability": "static", "T_world_entity": np.eye(4).tolist(),
                      "geometry": {"visual": {"obj_uri": "assets/table.obj"},
                                   "collision": [{"type": "box", "size": [1.2, 1.0, 0.03],
                                                  "T": np.r_[np.c_[np.eye(3), [0, -0.2, -0.015]], [[0, 0, 0, 1]]].tolist()}]},
                      "material": {"shader": "principled", "base_color_texture": "assets/table_tex.png", "tile_m": 0.3,
                                   "roughness": 0.85}, "provenance": {"smoke": "test"}}}
    for n in ("basket", "pepper"):
        e = oldsc["entities"][n]
        for f in ("visual.obj", "visual.mtl", "visual.png"):
            shutil.copy(old / "scene/assets/objects" / n / f, sd / "assets" / f"{n}_{f}")
        txt = (sd / "assets" / f"{n}_visual.obj").read_text().replace("mtllib visual.mtl", f"mtllib {n}_visual.mtl")
        (sd / "assets" / f"{n}_visual.obj").write_text(txt)
        ents[n] = {"category": n, "capability": "rigid",
                   "geometry": {"visual": {"obj_uri": f"assets/{n}_visual.obj"}, "collision": []},
                   "material": {"shader": "principled", "base_color_texture": f"assets/{n}_visual.png", "roughness": 0.6},
                   "physics": {"mass_kg": 0.15, "friction": 0.7},
                   "states": {s: {"frames": v["frames"], "T_world_obj": v["T_world_obj"]} for s, v in e["states"].items()},
                   "initial_state": e["initial_state"], "provenance": {"smoke": "test"}}
    scene = {"schema_version": "0.2", "units": {"length": "m"}, "frames": {"world": "smoke"},
             "cameras": {"source_cam": {"model": "pinhole", "width": cams["width"], "height": cams["height"], "K": cams["K"],
                                        "poses_uri": "cameras.json", "provenance": {"smoke": "copied"}}},
             "lights": [{"name": "key", "type": "area", "position": [0, -0.25, 1.6], "target": [0, -0.25, 0],
                         "size": [2.0, 2.0], "strength_w": 150}, {"type": "world", "color": [1, 1, 1], "strength": 0.3}],
             "entities": ents}
    save(sd, scene)
    step(proj.root, "convex_decompose", {"mesh": "scene/assets/basket_visual.obj", "out_dir": "scene/assets/basket_col",
                                         "threshold": 0.05, "max_hulls": 16})
    scene["entities"]["basket"]["geometry"]["collision"] = [{"type": "mesh", "uri": f"assets/basket_col/{f}"}
                                                            for f in json.loads((sd / "assets/basket_col/decomposition.json").read_text())["files"]]
    scene["entities"]["pepper"]["geometry"]["collision"] = [{"type": "box", "size": [0.1, 0.1, 0.1],
                                                             "T": np.r_[np.c_[np.eye(3), [0, 0, 0.05]], [[0, 0, 0, 1]]].tolist()}]
    save(sd, scene)
    print("validate", validate(sd), flush=True)
    tgt = {"label_maps": "work/track/labels", "label": 1}
    ign = [{"masks": "work/dyn/frames"}]
    T_b = scene["entities"]["basket"]["states"]["rest0"]["T_world_obj"]
    step(proj.root, "render_views", {"frames": [0, 40]})
    step(proj.root, "fit_render", {"frames": [0, 40], "iters": 40, "objects": [
        {"name": "basket", "mesh": "scene/assets/basket_visual.obj", "T_init": T_b, "dof": "planar", "proxy": "hull",
         "scale": {"fit": True}, "target": tgt, "ignore": ign}], "out_dir": "work/fit_grad"})
    (root / "work/code").mkdir(parents=True, exist_ok=True)
    (root / "work/code/basket_builder.py").write_text(
        "from any2sim import meshkit as mk\n"
        "def build(p):\n"
        "    return mk.normalize(mk.lathe([(p['r_bottom'], 0.0), (p['r_top'], p['h'])], wall=0.004))\n")
    step(proj.root, "fit_render", {"frames": [0, 40], "max_evals": 40, "objects": [
        {"name": "basket", "builder": {"module": "work/code/basket_builder.py",
                                       "params": {"r_bottom": [0.11, 0.07, 0.15], "r_top": [0.15, 0.10, 0.19], "h": [0.12, 0.08, 0.16]}},
         "T_init": T_b, "dof": "planar", "target": tgt, "ignore": ign, "w_edges": 0.2}], "out_dir": "work/fit_builder"})
    T_p = scene["entities"]["pepper"]["states"]["rest0"]["T_world_obj"]
    step(proj.root, "project_texture", {"mesh": "scene/assets/pepper_visual.obj", "poses": [{"T_world": T_p, "frames": [0, 40, 80]}],
                                        "visible": {"label_maps": "work/track/labels", "label": 2}, "ignore": ign,
                                        "occluders": [{"mesh": "scene/assets/basket_visual.obj", "T_world": T_b}]})
    step(proj.root, "geometry_metrics", {"targets": {"basket": tgt, "pepper": {"label_maps": "work/track/labels", "label": 2}},
                                         "ignore": ign, "frames": [0, 40]})
    step(proj.root, "mujoco_check", {"settle_s": 0.5, "probes": [{"name": "cavity", "body": "basket",
                                                                  "points": [[0, 0, 0.06], [0, 0, 0.09]]}]})
    step(proj.root, "blender_bake", {"entities": ["pepper"], "kinds": ["E", "COV"], "size": [256, 256], "samples": 16})
    step(proj.root, "blender_render", {"frames": [0], "novel": [{"name": "side", "eye": [0.5, -0.8, 0.4], "target": [0, -0.2, 0.05]}],
                                       "turntables": ["pepper"], "background_only": True, "samples": 16, "res_scale": 0.5})
    shutil.copy(root / "work/frames/frame00000.jpg", root / "work/real0.jpg")
    step(proj.root, "image_metrics", {"pairs": [{"id": "f0", "a": "work/real0.jpg", "b": "blender/renders/src_00000.png",
                                                 "ignore": "work/dyn/frames/frame00000.png"}]})
    step(proj.root, "sam3_segment", {"prompts": ["hand", "shadow"], "images": ["blender/renders/bg_00000.png"],
                                     "out_dir": "work/artifact_check"})
    print("SMOKE", json.dumps(STATUS))


if __name__ == "__main__":
    main()
