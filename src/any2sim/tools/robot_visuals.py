"""Attach render visuals to robot links (USD/GLB/glTF/OBJ/STL/FBX/blend), check their alignment, optionally derive
collision hulls from them.

For every link: import its source file(s) in Blender, keep the selected objects, apply an optional link-frame
transform, export <bundle>/visual/<link>/<link>.glb (standard glTF, y-up; materials, textures and normal maps embedded; what Blender
renders) and per-material OBJ parts + parts.json (what MuJoCo shows). Alignment check against the link's current URDF
visual meshes: distance from every URDF visual vertex to the new surface (a cover/glove sits a few mm outside the
bare mesh; a wrong frame shows as centimetres) and the bounding boxes. The profile's `visuals` (and `collisions`)
entries are updated and robot.xml recompiled.

params:
  bundle: "robot"                           bundle folder (project-relative) or an absolute path
  links: {link: {file, objects: [glob] | null, exclude: [glob] | null, T: 4x4 | null}}   explicit sources, or
  pattern: "/dir/{link}.usdc"               one file per link (links: every URDF link with a file present)
  link_filter: ["*finger*", "*palm*"]       which links the pattern applies to (default all)
  exclude: null | [glob]                    objects dropped from pattern sources (e.g. parts routed for another arm)
  collision_from_visual: null | {links: [glob], objects: [glob]}   convex hull of the selected visual objects
                                            per link -> collision/<link>.obj (replaces the URDF collision)
  hide: null | [glob]                       links whose URDF visual is dropped (e.g. fixed tips already contained
                                            in the parent link's new visual)
  align: null | {objects: ["{link}"], min_improvement_mm: 1.0} | {reference: <urdf|mjcf> | [..], anchors: [link, ...]}
                                            visual sources in another link-frame convention. ICP form: rigidly
                                            register the selected objects (e.g. the bare part shipped with the cover)
                                            to the URDF visual (24 axis-aligned starts). Reference form (exact): the
                                            source's own kinematic description; per link T_ours_src from FK of both
                                            relative to the first anchor link that is an ancestor in both
  max_offset_mm: 15                         alignment warning threshold (P95 URDF-vertex distance)
Writes <bundle>/checks/visuals.json and checks/visuals_sheet.jpg (old vs new per link).
"""
from __future__ import annotations

import fnmatch
import json
from pathlib import Path

import numpy as np

from ..common import write_json

ENV = "render"
GPU = False


def _import_any(path: Path):
    import bpy
    before = set(bpy.data.objects)
    ext = path.suffix.lower()
    if ext in (".usd", ".usdc", ".usda", ".usdz"):
        bpy.ops.wm.usd_import(filepath=str(path))
    elif ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path), forward_axis="Y", up_axis="Z")
    elif ext == ".stl":
        bpy.ops.wm.stl_import(filepath=str(path), forward_axis="Y", up_axis="Z")
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif ext == ".blend":
        with bpy.data.libraries.load(str(path)) as (src, dst):
            dst.objects = src.objects
        for o in dst.objects:
            if o is not None:
                bpy.context.scene.collection.objects.link(o)
    else:
        raise ValueError(f"unsupported visual format {ext}")
    return [o for o in bpy.data.objects if o not in before]


def _mesh_world(o):
    """(V, F) of an evaluated mesh object in world coordinates."""
    import bpy
    dg = bpy.context.evaluated_depsgraph_get()
    oe = o.evaluated_get(dg)
    me = oe.to_mesh()
    me.calc_loop_triangles()
    M = np.array(o.matrix_world)
    V = np.array([M[:3, :3] @ np.array(v.co) + M[:3, 3] for v in me.vertices]).reshape(-1, 3)
    F = np.array([t.vertices[:] for t in me.loop_triangles], int).reshape(-1, 3)
    oe.to_mesh_clear()
    return V, F


def _material_params(mat):
    out = {"material": mat.name if mat else None, "base_color": [0.5, 0.5, 0.5], "roughness": 0.5, "metallic": 0.0}
    if mat is None or not mat.node_tree:
        return out
    for n in mat.node_tree.nodes:
        if n.type == "BSDF_PRINCIPLED":
            bc = n.inputs["Base Color"]
            out["base_color"] = list(bc.default_value)[:3]
            out["roughness"] = float(n.inputs["Roughness"].default_value)
            out["metallic"] = float(n.inputs["Metallic"].default_value)
            out["textured"] = bool(bc.is_linked)
            out["normal_map"] = bool(n.inputs["Normal"].is_linked)
    return out


def _ref_vertices(model, link):
    import trimesh
    pts = []
    for g in model["links"][link]["visuals"]:
        if g["type"] != "mesh":
            continue
        m = trimesh.load(g["uri"], force="mesh", process=False)
        V = np.asarray(m.vertices) * np.asarray(g.get("scale", [1, 1, 1]))
        pts.append(V @ g["T"][:3, :3].T + g["T"][:3, 3])
    return np.concatenate(pts) if pts else None


def _icp(src, ref, iters=40):
    """Rigid T (4x4) moving points src onto points ref; returns (T, mean distance)."""
    import itertools
    from scipy.spatial import cKDTree
    tree = cKDTree(ref)
    rots = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            R = np.zeros((3, 3))
            for i, (p, sg) in enumerate(zip(perm, signs)):
                R[i, p] = sg
            if np.linalg.det(R) > 0:
                rots.append(R)
    best = None
    c_s, c_r = src.mean(0), ref.mean(0)
    for R0 in rots:
        T = np.eye(4)
        T[:3, :3] = R0
        T[:3, 3] = c_r - R0 @ c_s
        for _ in range(iters):
            P = src @ T[:3, :3].T + T[:3, 3]
            d, i = tree.query(P)
            Q = ref[i]
            mp, mq = P.mean(0), Q.mean(0)
            U_, _, Vt = np.linalg.svd((P - mp).T @ (Q - mq))
            Rk = (U_ @ Vt).T
            if np.linalg.det(Rk) < 0:
                Vt[-1] *= -1
                Rk = (U_ @ Vt).T
            dT = np.eye(4)
            dT[:3, :3] = Rk
            dT[:3, 3] = mq - Rk @ mp
            T = dT @ T
        d = tree.query(src @ T[:3, :3].T + T[:3, 3])[0].mean()
        if best is None or d < best[1]:
            best = (T, d)
    return best


def _reference_offsets(model, ref_path, anchors, links, n=3):
    """{link: T_ours_src} from FK of our URDF and the source description (URDF or MJCF) at random joint values."""
    from ..robot import urdf as U
    from ..geom3d import inv
    rng = np.random.default_rng(0)
    ref_path = Path(ref_path)
    if ref_path.suffix == ".urdf":
        ref = U.load(ref_path)
        names = U.movable_joints(ref)
        def ref_fk(q):
            return U.fk(ref, q)
    else:
        import mujoco
        m = mujoco.MjModel.from_xml_path(str(ref_path))
        d = mujoco.MjData(m)
        names = [m.joint(j).name for j in range(m.njnt)]
        def ref_fk(q):
            d.qpos[:] = 0
            for j, v in q.items():
                try:
                    d.qpos[m.jnt_qposadr[m.joint(j).id]] = v
                except KeyError:
                    pass
            mujoco.mj_kinematics(m, d)
            out = {}
            for b in range(1, m.nbody):
                T = np.eye(4)
                T[:3, :3] = d.xmat[b].reshape(3, 3)
                T[:3, 3] = d.xpos[b]
                out[m.body(b).name] = T
            return out
    ours_j = set(U.movable_joints(model))
    out = {}
    for _ in range(n):
        q = {}
        for j in names:
            if j in ours_j:
                L = model["joints"][j].get("limit") or {}
                q[j] = rng.uniform(L.get("lower", -0.5), L.get("upper", 0.5))
        To, Tr = U.fk(model, q), ref_fk(q)
        for ln in links:
            if ln not in Tr:
                continue
            for a in anchors:
                if a in To and a in Tr:
                    out.setdefault(ln, []).append(inv(inv(To[a]) @ To[ln]) @ (inv(Tr[a]) @ Tr[ln]))
                    break
    res = {}
    for ln, L in out.items():
        L = np.asarray(L)
        res[ln] = (L[0], float(max(np.abs(x - L[0]).max() for x in L)))     # offset, spread over samples
    return res


def run(project: Path, bundle: str = "robot", links: dict | None = None, pattern: str | None = None,
        link_filter: list | None = None, collision_from_visual: dict | None = None, hide: list | None = None,
        align: dict | None = None, max_offset_mm: float = 15.0, exclude: list | None = None) -> dict:
    import bpy
    import trimesh
    import yaml
    from scipy.spatial import cKDTree
    from ..robot import urdf as U
    from ..robot.bundle import Bundle
    from ..robot import mjcf
    broot = Path(bundle) if Path(bundle).is_absolute() else project / bundle
    b = Bundle(broot)
    model = b.model
    src = dict(links or {})
    if pattern:
        for ln in model["links"]:
            if link_filter and not any(fnmatch.fnmatch(ln, p) for p in link_filter):
                continue
            f = Path(pattern.format(link=ln))
            if f.exists() and ln not in src:
                src[ln] = {"file": str(f), **({"exclude": exclude} if exclude else {})}
    if not src and not hide:
        raise ValueError("no visual sources found (links / pattern)")
    prof = b.profile
    prof.setdefault("visuals", {})
    hidden = [ln for ln in model["links"] if any(fnmatch.fnmatch(ln, p) for p in hide or [])]
    ref_off = {}
    if align and align.get("reference"):
        refs = align["reference"] if isinstance(align["reference"], list) else [align["reference"]]
        for rp in refs:                                  # e.g. one description per hand
            for k, v in _reference_offsets(model, rp, align.get("anchors") or [model["root"]], list(src)).items():
                ref_off.setdefault(k, v)
    for ln in hidden:
        prof["visuals"][ln] = {"hidden": True, "source": "hidden (contained in another link's visual)"}
    report = {}
    cfv = collision_from_visual or {}
    for ln, spec in src.items():
        if ln not in model["links"]:
            report[ln] = {"error": "not a URDF link"}
            continue
        bpy.ops.wm.read_factory_settings(use_empty=True)
        objs = _import_any(Path(spec["file"]))
        meshes = [o for o in objs if o.type == "MESH"]
        if spec.get("objects"):
            meshes = [o for o in meshes if any(fnmatch.fnmatch(o.name, p) for p in spec["objects"])]
        if spec.get("exclude"):
            meshes = [o for o in meshes if not any(fnmatch.fnmatch(o.name, p) for p in spec["exclude"])]
        for o in objs:
            if o not in meshes and o.type == "MESH":
                bpy.data.objects.remove(o, do_unlink=True)
        if not meshes:
            report[ln] = {"error": "no mesh objects selected"}
            continue
        if spec.get("T") is not None:
            import mathutils
            Tm = mathutils.Matrix(np.asarray(spec["T"], float).tolist())
            for o in meshes:
                if o.parent is None or o.parent not in meshes:
                    o.matrix_world = Tm @ o.matrix_world
        bpy.context.view_layer.update()
        align_info = None
        if align and align.get("reference"):
            import mathutils
            if ln in ref_off:
                Ta, spread = ref_off[ln]
                Tm = mathutils.Matrix(Ta.tolist())
                for o in meshes:
                    if o.parent is None or o.parent not in meshes:
                        o.matrix_world = Tm @ o.matrix_world
                bpy.context.view_layer.update()
                align_info = {"method": "reference", "applied": True, "T": Ta.round(6).tolist(),
                              "spread": spread, "consistent": spread < 1e-3}
            else:
                align_info = {"method": "reference", "applied": False, "note": "link not in reference"}
        elif align:
            import mathutils
            pats = [p.format(link=ln) for p in align.get("objects", ["{link}"])]
            sel = [o for o in meshes if any(fnmatch.fnmatch(o.name, p) for p in pats)]
            ref = _ref_vertices(model, ln)
            if sel and ref is not None:
                srcV = np.concatenate([_mesh_world(o)[0] for o in sel])
                sub = srcV[np.random.default_rng(0).choice(len(srcV), min(len(srcV), 4000), replace=False)]
                d0 = cKDTree(ref).query(sub)[0].mean()
                Ta, d1 = _icp(sub, ref)
                align_info = {"objects": [o.name for o in sel], "mean_mm_before": float(d0 * 1e3),
                              "mean_mm_after": float(d1 * 1e3), "applied": False}
                if (d0 - d1) * 1e3 >= align.get("min_improvement_mm", 1.0):
                    Tm = mathutils.Matrix(Ta.tolist())
                    for o in meshes:
                        if o.parent is None or o.parent not in meshes:
                            o.matrix_world = Tm @ o.matrix_world
                    bpy.context.view_layer.update()
                    align_info.update(applied=True, T=Ta.round(6).tolist())
        od = broot / "visual" / ln
        od.mkdir(parents=True, exist_ok=True)
        for f in od.glob("part_*"):
            f.unlink()
        bpy.ops.object.select_all(action="DESELECT")
        for o in meshes:
            o.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.export_scene.gltf(filepath=str(od / f"{ln}.glb"), export_format="GLB", use_selection=True,
                                  export_apply=True, export_yup=True, export_materials="EXPORT", export_image_format="AUTO")
        # per-material parts for MuJoCo (z-up, link frame)
        by_mat = {}
        allV, allF, hullV = [], [], []
        for o in meshes:
            V, F = _mesh_world(o)
            mats = [s.material for s in o.material_slots] or [None]
            me = o.data
            me.calc_loop_triangles()
            fm = np.array([t.material_index for t in me.loop_triangles], int) if len(F) else np.zeros(0, int)
            for mi in np.unique(fm) if len(fm) else []:
                mat = mats[mi] if mi < len(mats) else None
                key = mat.name if mat else "default"
                by_mat.setdefault(key, {"mat": mat, "V": [], "F": [], "n": 0})
                sel = F[fm == mi]
                ent = by_mat[key]
                ent["V"].append(V)
                ent["F"].append(sel + ent["n"])
                ent["n"] += len(V)
            allV.append(V)
            allF.append(F + sum(len(v) for v in allV[:-1]))
            if cfv and any(fnmatch.fnmatch(ln, p) for p in cfv.get("links", [])) and \
                    any(fnmatch.fnmatch(o.name, p) for p in cfv.get("objects", ["*"])):
                hullV.append(V)
        parts = []
        for k, (key, ent) in enumerate(by_mat.items()):
            m = trimesh.Trimesh(np.concatenate(ent["V"]), np.concatenate(ent["F"]), process=True)
            m.remove_unreferenced_vertices()
            fn = f"part_{k}.obj"
            m.export(od / fn)
            parts.append({"obj": fn, **_material_params(ent["mat"]), "faces": int(len(m.faces))})
        (od / "parts.json").write_text(json.dumps(parts, indent=1))
        rel = od.relative_to(broot)
        prof["visuals"][ln] = {"glb": str(rel / f"{ln}.glb"), "parts": str(rel / "parts.json"),
                               "source": spec["file"], "objects": [o.name for o in meshes]}
        # alignment against the URDF visual mesh
        Vn = np.concatenate(allV)
        Fn = np.concatenate(allF)
        new = trimesh.Trimesh(Vn, Fn, process=False)
        samp = new.sample(min(200000, max(20000, len(Fn) * 3)), return_index=False)
        ref = _ref_vertices(model, ln)
        r = {"objects": [o.name for o in meshes], "parts": len(parts), "faces": int(len(Fn)), "align": align_info,
             "bounds_new": [Vn.min(0).round(4).tolist(), Vn.max(0).round(4).tolist()]}
        if ref is not None:
            d = cKDTree(samp).query(ref)[0]
            r.update(bounds_urdf=[ref.min(0).round(4).tolist(), ref.max(0).round(4).tolist()],
                     urdf_to_new_mm={"median": float(np.median(d) * 1e3), "p95": float(np.percentile(d, 95) * 1e3),
                                     "max": float(d.max() * 1e3)})
            r["aligned"] = bool(r["urdf_to_new_mm"]["p95"] <= max_offset_mm)
        if hullV:
            hull = trimesh.Trimesh(np.concatenate(hullV), process=True).convex_hull
            cd = broot / "collision"
            cd.mkdir(exist_ok=True)
            hull.export(cd / f"{ln}.obj")
            prof.setdefault("collisions", {})[ln] = [f"collision/{ln}.obj"]
            r["collision_hull"] = {"faces": int(len(hull.faces)), "volume_cm3": float(hull.volume * 1e6)}
        report[ln] = r
    prof.setdefault("provenance", {})["visuals"] = "robot_visuals: " + ", ".join(sorted({Path(s["file"]).parent.name for s in src.values()}))
    (broot / "robot.yaml").write_text(yaml.safe_dump(json.loads(json.dumps(prof, default=float)), sort_keys=False,
                                                     allow_unicode=True))
    mjcf.compile_bundle(Bundle(broot))
    bad = [k for k, v in report.items() if v.get("aligned") is False or "error" in v]
    res = {"links": len(report), "hidden": hidden, "misaligned_or_failed": bad, "per_link": report}
    write_json(res, broot / "checks" / "visuals.json")
    return {**{k: v for k, v in res.items() if k != "per_link"},
            "summary": {k: (v.get("urdf_to_new_mm"), v.get("error")) for k, v in report.items()},
            "outputs": [str(broot.relative_to(project)) if broot.is_relative_to(project) else str(broot)]}
