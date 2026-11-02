"""Rescale the whole Scene IR uniformly (a declared scene revision): new = factor * old about the world origin.

For a scene whose metric scale was estimated wrongly (e.g. video-only depth priors) once a metric reference is
available (hand tracks, a robot, a measured object). Everything with a length scales: camera positions (intrinsics
unchanged), entity poses and rest states, visual and collision meshes (written as new files), box sizes, metre-UV
texture tiling (UVs and tile_m together, so textures stay put), light positions/targets/sizes (area/point light power * factor^2 so irradiance is unchanged), robot base
translations and object poses of motions; masses * factor^3 (same density) unless mass_mode "keep". A uniform
scale about the cameras changes no projection: the tool verifies that every entity's vertices project to the same
pixels from every camera pose (max px difference), so all pixel-based scene metrics stay valid.

params:
  factor: <float>                     true metres per current scene unit (e.g. 0.70)
  reason: <text>, cues: {..}          why (the evidence); recorded in scene/revisions.yaml
  mass_mode: "density" | "keep"
Writes the revised scene in place (old mesh files kept), appends scene/revisions.yaml.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..common import write_json

ENV = "base"
GPU = False


def _scaleT(T, f):
    T = np.asarray(T, float).copy()
    T[:3, 3] *= f
    return T.tolist()


def run(project: Path, factor: float, reason: str, cues: dict | None = None, mass_mode: str = "density") -> dict:
    import trimesh
    import yaml
    from ..geom3d import inv, project as proj
    from ..scene_ir import camera_poses, core_digest, load, pose_at, save, validate
    f = float(factor)
    if not (0.2 < f < 5):
        raise ValueError(f"implausible factor {f}")
    sd = project / "scene"
    sc = load(sd)
    before = core_digest(sc)
    rev = len((yaml.safe_load((sd / "revisions.yaml").read_text()) or []) if (sd / "revisions.yaml").exists() else []) + 1
    tag = f"r{rev}"
    # reference projections before
    c = sc["cameras"]["source_cam"]
    K = np.asarray(c["K"], float)
    poses0 = camera_poses(sd, sc)
    sample = {}
    for n, e in sc["entities"].items():
        m = trimesh.load(sd / e["geometry"]["visual"]["obj_uri"], force="mesh", process=False)
        V = np.asarray(m.vertices)
        V = V[np.random.default_rng(0).choice(len(V), min(300, len(V)), replace=False)]
        T = pose_at(e, None)
        sample[n] = V @ T[:3, :3].T + T[:3, 3]
    idx = np.linspace(0, len(poses0) - 1, min(len(poses0), 12)).astype(int)
    W_, H_ = c["width"], c["height"]
    uv0 = {}
    for n, P in sample.items():
        uv0[n] = []
        for i in idx:
            uv, z = proj(K, inv(poses0[i]), P)
            ok = (z > 0.05) & (uv[:, 0] > -W_) & (uv[:, 0] < 2 * W_) & (uv[:, 1] > -H_) & (uv[:, 1] < 2 * H_)
            uv0[n].append((uv, ok))                      # points well in front of the camera, near the image
    # cameras
    for cn, cam in sc["cameras"].items():
        if "poses_uri" in cam:
            src = sd / cam["poses_uri"]
            d = json.loads(src.read_text())
            Ts = np.asarray(d["T_world_cam"] if isinstance(d, dict) else d, float)
            Ts[:, :3, 3] *= f
            new = Path(cam["poses_uri"]).with_name(f"{Path(cam['poses_uri']).stem}_{tag}.json")
            out = dict(d) if isinstance(d, dict) else {}
            out["T_world_cam"] = Ts.tolist()
            (sd / new).write_text(json.dumps(out))
            cam["poses_uri"] = str(new)
        if "T_world_cam" in cam:
            cam["T_world_cam"] = _scaleT(cam["T_world_cam"], f)
    # entities
    mdir = sd / "meshes" / tag
    mdir.mkdir(parents=True, exist_ok=True)
    def scaled_mesh(uri, scale_uv=False):
        """Vertices * f; UVs too when they are metres (tile_m materials), so the texture stays on the surface."""
        src = sd / uri
        dst = mdir / (Path(uri).stem + ("_uvm" if scale_uv else "") + Path(uri).suffix)
        if not dst.exists():
            txt = src.read_text().splitlines()
            out = []
            for ln in txt:
                if ln.startswith("v "):
                    p = ln.split()
                    out.append("v " + " ".join(f"{float(x) * f:.7f}" for x in p[1:4]) + (" " + " ".join(p[4:]) if len(p) > 4 else ""))
                elif ln.startswith("vt ") and scale_uv:
                    p = ln.split()
                    out.append("vt " + " ".join(f"{float(x) * f:.7f}" for x in p[1:3]))
                elif ln.startswith("mtllib"):
                    out.append(ln)
                    mtl = src.with_name(ln.split(maxsplit=1)[1])
                    if mtl.exists() and not (mdir / mtl.name).exists():
                        (mdir / mtl.name).write_text(mtl.read_text().replace("map_Kd ", "map_Kd ../"))
                else:
                    out.append(ln)
            dst.write_text("\n".join(out) + "\n")
        return str(dst.relative_to(sd))
    for n, e in sc["entities"].items():
        g = e["geometry"]
        g["visual"]["obj_uri"] = scaled_mesh(g["visual"]["obj_uri"], bool((e.get("material") or {}).get("tile_m")))
        for col in g.get("collision") or []:
            if col.get("T") is not None:
                col["T"] = _scaleT(col["T"], f)
            if col["type"] == "mesh":
                col["uri"] = scaled_mesh(col["uri"])
            elif col["type"] == "box":
                col["size"] = [float(x) * f for x in col["size"]]
        if "T_world_entity" in e:
            e["T_world_entity"] = _scaleT(e["T_world_entity"], f)
        for st in (e.get("states") or {}).values():
            st["T_world_obj"] = _scaleT(st["T_world_obj"], f)
        mat = e.get("material") or {}
        if mat.get("tile_m"):
            mat["tile_m"] = float(mat["tile_m"]) * f
        ph = e.get("physics") or {}
        if "mass_kg" in ph and mass_mode == "density":
            mv = ph["mass_kg"]
            if isinstance(mv, dict):
                mv["value"] = float(mv["value"]) * f ** 3
                mv["kind"] = f"{mv.get('kind', '')} (scaled f^3 by revision {tag})".strip()
            else:
                ph["mass_kg"] = float(mv) * f ** 3
        e.setdefault("provenance", {})["scale_revision"] = f"{tag}: x{f:.4f} ({reason})"
    for L in sc.get("lights") or []:
        for k in ("position", "target"):
            if k in L:
                L[k] = [float(x) * f for x in L[k]]
        if "size" in L:
            L["size"] = [float(x) * f for x in L["size"]]
        if L.get("type") in ("area", "point") and "strength_w" in L:
            L["strength_w"] = float(L["strength_w"]) * f ** 2
        if "T_world_light" in L:
            L["T_world_light"] = _scaleT(L["T_world_light"], f)
    for r in (sc.get("robots") or {}).values():
        r["T_world_base"] = _scaleT(r["T_world_base"], f)
        r.setdefault("provenance", {})["scale_revision"] = f"{tag}: base translation x{f:.4f}"
    for mid, m in (sc.get("motions") or {}).items():
        p = sd / m["uri"]
        d = dict(np.load(p))
        if "obj_T" in d:
            d["obj_T"] = d["obj_T"].copy()
            d["obj_T"][..., :3, 3] *= f
            new = Path(m["uri"]).with_name(f"{Path(m['uri']).stem}_{tag}.npz")
            np.savez_compressed(sd / new, **d)
            m["uri"] = str(new)
    sch = sc.get("scale") or {}
    sc["scale"] = {**sch, "source": f"revision {tag}: {reason}", "revision_factor": f,
                   "previous": {k: v for k, v in sch.items() if k != "previous"}, "cues": cues or sch.get("cues")}
    save(sd, sc)
    # invariance check
    sc2 = load(sd)
    poses1 = camera_poses(sd, sc2)
    worst, per_ent = 0.0, {}
    for n, e in sc2["entities"].items():
        m = trimesh.load(sd / e["geometry"]["visual"]["obj_uri"], force="mesh", process=False)
        V = np.asarray(m.vertices)
        V = V[np.random.default_rng(0).choice(len(V), min(300, len(V)), replace=False)]
        T = pose_at(e, None)
        P = V @ T[:3, :3].T + T[:3, 3]
        w = 0.0
        for k, i in enumerate(idx):
            uv1 = proj(K, inv(poses1[i]), P)[0]
            u0, ok = uv0[n][k]
            if ok.any():
                w = max(w, float(np.abs(uv1[ok] - u0[ok]).max()))
        per_ent[n] = w
        worst = max(worst, w)
    after = core_digest(sc2)
    revs = (yaml.safe_load((sd / "revisions.yaml").read_text()) or []) if (sd / "revisions.yaml").exists() else []
    entry = {"id": tag, "kind": "uniform_scale", "factor": f, "reason": reason, "cues": cues,
             "mass_mode": mass_mode, "core_sha256_before": before, "core_sha256_after": after,
             "projection_max_px_change": worst, "projection_max_px_change_per_entity": per_ent}
    revs.append(entry)
    (sd / "revisions.yaml").write_text(yaml.safe_dump(json.loads(json.dumps(revs, default=float)), sort_keys=False))
    v = validate(sd)
    res = {**entry, "validate": {"ok": v["ok"], "errors": v["errors"][:5]}}
    write_json(res, project / "evidence" / f"scene_rescale_{tag}.json")
    return {**res, "outputs": ["scene/scene.yaml", "scene/revisions.yaml", f"scene/meshes/{tag}"]}
