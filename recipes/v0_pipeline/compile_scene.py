"""Assemble the Scene IR (scene/scene.yaml) from the reconstruction outputs, and build collision proxies.

The Scene IR is the single source of truth: Blender and MuJoCo files are compiled from it and never edited by
hand. Every value carries its provenance (observed / fitted / completed / prior). Objects get a convex
decomposition (CoACD) that keeps cavities (a basket stays open); its agreement with the visual mesh is reported.
Inputs: work/world, work/background, work/objects/*/registration.json
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common import read_json
from ..project import now

ENV = "base"
GPU = False


def glb_to_obj(glb: Path, out_dir: Path, name: str) -> tuple[str, str | None]:
    """Textured OBJ + PNG next to a GLB (MuJoCo cannot read GLB; Blender's OBJ importer keeps our axes)."""
    import trimesh
    m = trimesh.load(glb, force="mesh")
    out_dir.mkdir(parents=True, exist_ok=True)
    mat = getattr(m.visual, "material", None)
    img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    tex = None
    if img is not None and getattr(m.visual, "uv", None) is not None:
        tex = f"{name}.png"
        img.convert("RGB").save(out_dir / tex)
    lines = [f"mtllib {name}.mtl", "usemtl surface"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in m.vertices]
    if tex:
        lines += [f"vt {u:.6f} {v:.6f}" for u, v in m.visual.uv]
        lines += [f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}" for a, b, c in m.faces]
    else:
        lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in m.faces]
    (out_dir / f"{name}.obj").write_text("\n".join(lines) + "\n")
    (out_dir / f"{name}.mtl").write_text("newmtl surface\nKd 1 1 1\n" + (f"map_Kd {tex}\n" if tex else ""))
    return f"{name}.obj", tex


def _collision(visual_glb: Path, out_dir: Path, threshold: float, max_hulls: int, foot: dict | None) -> dict:
    """CoACD decomposition of the visual mesh; optionally a flat support foot: a thin convex prism under the
    footprint of the vertices within `band_m` of the bottom (hidden, generated bottoms are rarely flat, and an
    object observed resting on a support must rest on it in simulation too)."""
    import coacd
    import trimesh
    m = trimesh.load(visual_glb, force="mesh")
    m.merge_vertices()
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("hull_*.obj"):
        f.unlink()
    parts = coacd.run_coacd(coacd.Mesh(np.asarray(m.vertices), np.asarray(m.faces)), threshold=threshold,
                            max_convex_hull=max_hulls, preprocess_mode="on", preprocess_resolution=80, seed=0)
    hulls = [trimesh.Trimesh(v, f) for v, f in parts]
    foot_info = None
    if foot is not None:
        V = np.asarray(m.vertices)
        z0 = V[:, 2].min()
        import cv2
        com = m.convex_hull.center_mass[:2]
        height = float(V[:, 2].max() - z0)
        # smallest band whose footprint holds the centre of mass with a margin (minimal flat support that keeps
        # the observed rest pose stable), unless a band is given
        bands = [foot["band_m"]] if "band_m" in foot else list(np.arange(0.004, 0.15 * height + 1e-9, 0.002))
        low, band, hull2d = None, None, None
        for band in bands:
            low = V[V[:, 2] < z0 + band][:, :2]
            if len(low) < 3:
                continue
            hull2d = cv2.convexHull(low.astype(np.float32))[:, 0].astype(float)
            margin = cv2.pointPolygonTest(hull2d.astype(np.float32).reshape(-1, 1, 2), (float(com[0]), float(com[1])), True)
            if margin > foot.get("com_margin_m", 0.01):
                break
        if hull2d is not None and len(hull2d) >= 3:
            t = foot.get("thickness_m", 0.003)
            prism = np.r_[np.c_[hull2d, np.full(len(hull2d), z0)], np.c_[hull2d, np.full(len(hull2d), z0 + t)]]
            hulls.append(trimesh.Trimesh(prism).convex_hull)
            ext = hull2d.max(0) - hull2d.min(0)
            foot_info = {"band_m": float(band), "thickness_m": t, "footprint_extent_m": ext.tolist(),
                         "provenance": "completed (flat support foot under the generated bottom)"}
    for k, h in enumerate(hulls):
        h.export(out_dir / f"hull_{k:03d}.obj")
    vol_vis = float(abs(m.volume)) if m.is_volume else None
    vol_vis_hull = float(m.convex_hull.volume)
    # surface agreement: distance from visual-surface samples to the nearest collision hull
    samp = m.sample(3000, return_index=False) if len(m.faces) else np.zeros((0, 3))
    dist = np.min(np.stack([np.abs(trimesh.proximity.signed_distance(h, samp)) for h in hulls]), 0) if len(samp) else np.zeros(1)
    vol_col = float(sum(h.volume for h in hulls))
    # agreement on a voxel grid: fraction of collision voxels inside the visual mesh's closed hull and vice versa
    pts = trimesh.sample.volume_mesh(m.convex_hull, 4000)
    in_col = np.zeros(len(pts), bool)
    for h in hulls:
        in_col |= h.contains(pts)
    in_vis = m.contains(pts) if m.is_watertight else in_col
    return {"n_hulls": len(hulls), "files": [f"hull_{k:03d}.obj" for k in range(len(hulls))],
            "volume_visual_m3": vol_vis, "volume_visual_convex_hull_m3": vol_vis_hull, "volume_collision_m3": vol_col,
            "visual_to_collision_dist_m": {"p50": float(np.percentile(dist, 50)), "p90": float(np.percentile(dist, 90)),
                                           "p99": float(np.percentile(dist, 99))},
            "provenance": "derived (CoACD of the visual mesh, whose hidden parts are completed)" +
                          ("; flat support foot completed: stability on the table depends on it" if foot_info else ""),
            "visual_watertight": bool(m.is_watertight),
            "agreement": float((in_col == in_vis).mean()) if m.is_watertight else None,
            "threshold": threshold, "support_foot": foot_info}


def run(project: Path, physics_priors: dict | None = None, categories: dict | None = None, size_priors: dict | None = None,
        coacd_threshold: float = 0.02, max_hulls: int = 64, initial_states: dict | None = None,
        support_foot: dict | None = None) -> dict:
    """support_foot: {entity: {band_m, thickness_m}}; default: every object whose initial state rests on the table."""
    import yaml
    world = read_json(project / "work" / "world" / "world.json")
    cams = read_json(project / "work" / "world" / "cameras.json")
    bg = read_json(project / "work" / "background" / "background.json")
    labels = read_json(project / "work" / "masks" / "labels.json")["labels"]
    physics_priors = physics_priors or {}
    categories = categories or {}
    obs = yaml.safe_load((project / "observation.yaml").read_text())
    t = bg["table"]
    corners = np.asarray(t["corners_xy"])
    c = corners.mean(0)
    e1, e2 = corners[1] - corners[0], corners[3] - corners[0]
    yaw = float(np.arctan2(e1[1], e1[0]))
    entities = {
        "table": {
            "category": "table", "capability": "static",
            "geometry": {
                "visual": {"uri": "assets/background/table.obj", "T_entity_mesh": np.eye(4).tolist()},
                "collision": [{"type": "box", "size": [float(np.linalg.norm(e1)), float(np.linalg.norm(e2)), t["thickness_m"]],
                               "pos": [float(c[0]), float(c[1]), -t["thickness_m"] / 2], "yaw": yaw}],
            },
            "materials": {"top": {"base_color_texture": "assets/background/table_top.png", "roughness": 0.85,
                                  "provenance": "texture observed (clean plate), roughness prior"}},
            "physics": {"friction": {"value": 0.8, "kind": "prior"}},
            "T_world_entity": np.eye(4).tolist(),
            "provenance": {"top_plane": "fitted (MoGe-2 points, RANSAC)", "observed_sides": t["observed_sides"],
                           "unseen_sides": f"completed: extended {t['extended_unseen_sides_by_m']} m",
                           "thickness": "prior", "texture_observed_fraction": t["texture_observed_fraction"],
                           "legs/height": f"not modelled; height prior {t['height_prior_m']} m"},
            "affordances": [{"type": "support_surface", "z": 0.0}],
        },
        "backdrop": {
            "category": "background", "capability": "visual_only",
            "geometry": {"visual": {"uri": "assets/background/backdrop.obj", "T_entity_mesh": np.eye(4).tolist()}},
            "materials": {"backdrop": {"emission_texture": "assets/background/plate.png", "uv": "reference-camera projection"}},
            "T_world_entity": np.eye(4).tolist(),
            "provenance": {"geometry": "observed (MoGe-2 depth grid of the reference frame)",
                           "texture": "observed (clean plate)", "hidden": "nothing behind/outside the reference view"},
        },
    }
    for _, name in labels.items():
        reg_p = project / "work" / "objects" / name / "registration.json"
        if not reg_p.exists():
            continue
        reg = read_json(reg_p)
        init_state = min(reg["states"], key=lambda st: st["frames"][0])
        foot = (support_foot or {}).get(name, {} if init_state.get("support") == "table" else None)
        col = _collision(project / "scene" / "assets" / "objects" / name / "visual.glb",
                         project / "scene" / "assets" / "objects" / name / "collision", coacd_threshold, max_hulls,
                         foot if foot is not None else None)
        objf, tex = glb_to_obj(project / "scene" / "assets" / "objects" / name / "visual.glb",
                               project / "scene" / "assets" / "objects" / name, "visual")
        pp = physics_priors.get(name, {})
        states = {st["state"]: {"frames": st["frames"], "T_world_obj": st["T_world_obj"], "support": st.get("support"),
                                "fit_frames": st["fit"], "heldout_frames": st["heldout"],
                                "eval_summary": st.get("eval_summary")} for st in reg["states"]}
        init = (initial_states or {}).get(name, min(states, key=lambda s: states[s]["frames"][0]))
        size_check = None
        if (size_priors or {}).get(name):
            pr = size_priors[name]
            ext = max(reg["extent_m"][:2])
            size_check = {"final_mesh_max_xy_extent_m": ext, "prior_range_m": pr["range_m"],
                          "within_prior": bool(pr["range_m"][0] <= ext <= pr["range_m"][1]),
                          "note": "registered mesh incl. generated parts (handles/lugs)"}
        entities[name] = {
            "category": categories.get(name, name), "capability": "rigid", "size_check": size_check,
            "geometry": {
                "visual": {"uri": f"assets/objects/{name}/visual.glb", "obj_uri": f"assets/objects/{name}/{objf}",
                           "texture_uri": f"assets/objects/{name}/{tex}" if tex else None,
                           "T_entity_mesh": np.eye(4).tolist(), "extent_m": reg["extent_m"]},
                "collision": {"method": "coacd", "dir": f"assets/objects/{name}/collision", **col},
            },
            "materials": {"surface": {"base_color_texture": f"assets/objects/{name}/{tex}" if tex else None, "roughness": 0.6,
                                      "provenance": "generated texture (SAM 3D); roughness prior"}},
            "physics": {"mass_kg": {"value": pp.get("mass_kg", 0.1), "kind": "prior" if "mass_kg" in pp else "default"},
                        "friction": {"value": pp.get("friction", 0.8), "kind": "prior"},
                        "inertia": "from collision volume, uniform density"},
            "states": states, "initial_state": init,
            "T_world_entity": states[init]["T_world_obj"],
            "provenance": {"shape": "generated (SAM 3D Objects, single view) - hidden sides are completed",
                           "scale": f"fitted (x{reg['scale_vs_sam3d_moge']:.3f} vs SAM 3D layout on MoGe-2)",
                           "poses": "fitted (multi-frame silhouettes + support), per rest state"},
        }
    scene = {
        "schema_version": "0.1",
        "generated": now(),
        "units": {"length": "m", "time": "s", "angle": "rad"},
        "frames": {"world": world["frame"], "quaternion_order": "wxyz", "T_a_b": "maps points from b into a",
                   "camera_optical": "OpenCV (x right, y down, z forward)"},
        "source": {"video": obs["videos"][0]["uri"], "sha256": obs["videos"][0]["sha256"], "withheld": obs["withheld"]},
        "scale": {"source": world["scale_source"], "cues": world["scale_cues"],
                  "uncertainty_rel": world["scale_uncertainty_rel"], "findings": world["findings"]},
        "cameras": {"source_cam": {"model": "pinhole", "width": cams["width"], "height": cams["height"],
                                   "K": cams["K"], "distortion": None, "fps": cams["fps"],
                                   "poses_uri": "../work/world/cameras.json", "ref_frame": world["ref_frame"],
                                   "provenance": {"K": "estimated (MoGe-2, reference frame; Pi3X cross-check)",
                                                  "distortion": "assumed none (not estimated)",
                                                  "poses": "estimated (Pi3X, PnP cross-check, plane-ECC refinement)"},
                                   "motion": world["camera_motion"]["class"]}},
        "lights": [],
        "entities": entities,
        "bindings": {},
    }
    (project / "scene" / "scene.yaml").write_text(yaml.safe_dump(scene, sort_keys=False, allow_unicode=True))
    (project / "scene" / "patches.jsonl").write_text("")    # a new IR version starts with no patches applied
    return {"entities": {k: {"capability": v["capability"], **({"states": list(v["states"]), "collision_hulls":
                                                                  v["geometry"]["collision"]["n_hulls"]} if "states" in v else {})}
                         for k, v in entities.items()},
            "outputs": ["scene/scene.yaml", "scene/assets"]}
