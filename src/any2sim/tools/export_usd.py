"""Export the Scene IR to USD for Isaac Sim (UsdPhysics / UsdShade; metres, z-up): isaac/stage.usda (physics-ready
initial state) and isaac/playback.usda (sublayers the stage; time-sampled robot-link and object transforms of the
render motion at the video frame rate, for kinematic playback / rendering with physics disabled).

Contents: entities as meshes with UsdPreviewSurface materials (albedo textures copied to isaac/textures),
colliders (convex hulls / boxes), rigid bodies with mass and friction; robots as articulations (one rigid body per
link with mass/inertia, revolute/prismatic/fixed joints with limits, angular/linear drives from the bundle's
actuators; base fixed to the world at T_world_base); the source camera (intrinsics + per-frame poses).

Validation (no Isaac needed): the stage opens; every joint's two frames coincide at the authored state (or differ
only by the joint's own motion); every rigid body has mass and a collider; articulation roots exist; physics
material present. Isaac execution itself is not tested here.

params:
  out_dir: "isaac"
  playback: true
  frames: null | [a, b]         playback frame range (default whole motion)
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from ..common import write_json

ENV = "sam3d"
GPU = False


def _quat(R):
    from pxr import Gf
    from ..geom3d import mat_to_quat
    w, x, y, z = mat_to_quat(np.asarray(R)[:3, :3])
    return Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z)))


def _mat(T):
    from pxr import Gf
    return Gf.Matrix4d(np.asarray(T, float).T.tolist())      # USD: row vectors


def _xform(prim_or_xf, T):
    from pxr import UsdGeom
    xf = UsdGeom.Xformable(prim_or_xf)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(_mat(T))


def _mesh(stage, path, V, F, N=None, UV=None):
    from pxr import UsdGeom, Vt, Gf, Sdf
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, v)) for v in V]))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(F)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray(np.asarray(F, int).ravel().tolist()))
    m.CreateSubdivisionSchemeAttr("none")
    if UV is not None:
        pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec2fArray([Gf.Vec2f(*map(float, u)) for u in UV]))
    lo, hi = np.asarray(V).min(0), np.asarray(V).max(0)
    m.CreateExtentAttr([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))])
    return m


def _material(stage, path, color=(0.8, 0.8, 0.8), roughness=0.5, metallic=0.0, texture=None, emission=False):
    from pxr import UsdShade, Sdf, Gf
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    if texture:
        st = UsdShade.Shader.Define(stage, path + "/st")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tx = UsdShade.Shader.Define(stage, path + "/albedo")
        tx.CreateIdAttr("UsdUVTexture")
        tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture)
        tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        out = tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        sh.CreateInput("emissiveColor" if emission else "diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(out)
        if emission:
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0, 0, 0))
    else:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, color[:3])))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def _bind(prim, mat):
    from pxr import UsdShade
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def _collider(stage, path, V, F, approx="convexHull"):
    from pxr import UsdPhysics, UsdGeom
    m = _mesh(stage, path, V, F)
    UsdGeom.Imageable(m).CreatePurposeAttr("guide")
    UsdPhysics.CollisionAPI.Apply(m.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr(approx)
    return m


def _box_collider(stage, path, size, T):
    from pxr import UsdPhysics, UsdGeom, Gf
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(c)
    xf.AddTransformOp().Set(_mat(T))
    xf.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
    UsdGeom.Imageable(c).CreatePurposeAttr("guide")
    UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    return c


def _prim_collider(stage, path, g):
    """URDF cylinder (along z) / sphere collider at its origin."""
    from pxr import UsdPhysics, UsdGeom
    if g["type"] == "cylinder":
        c = UsdGeom.Cylinder.Define(stage, path)
        c.CreateRadiusAttr(float(g["radius"]))
        c.CreateHeightAttr(float(g["length"]))
        c.CreateAxisAttr("Z")
    else:
        c = UsdGeom.Sphere.Define(stage, path)
        c.CreateRadiusAttr(float(g["radius"]))
    UsdGeom.Xformable(c).AddTransformOp().Set(_mat(g["T"]))
    UsdGeom.Imageable(c).CreatePurposeAttr("guide")
    UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    return c


def _load(path):
    import trimesh
    m = trimesh.load(path, force="mesh", process=False)
    uv = getattr(m.visual, "uv", None)
    return np.asarray(m.vertices), np.asarray(m.faces), (np.asarray(uv) if uv is not None and len(uv) == len(m.vertices) else None)


def _name(s):
    import re
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    return s if not s[0].isdigit() else "_" + s


def _mass_props(I, R_body_inertial):
    """URDF inertial (mass, inertia in the inertial frame) -> (com, diag, principal axes quat) in the body frame."""
    ixx, iyy, izz, ixy, ixz, iyz = I["inertia"]
    Ii = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    Ib = R_body_inertial @ Ii @ R_body_inertial.T
    w, V = np.linalg.eigh(Ib)
    if np.linalg.det(V) < 0:
        V[:, 0] *= -1
    return np.maximum(w, 1e-9), V


def run(project: Path, out_dir: str = "isaac", playback: bool = True, frames: list | None = None) -> dict:
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf, Sdf
    from ..robot import urdf as U
    from ..scene_ir import camera_poses, load, load_motion, motion_at, phys, pose_at, robot_bundle
    scene = load(project / "scene")
    sdir = project / "scene"
    od = project / out_dir
    (od / "textures").mkdir(parents=True, exist_ok=True)
    path = od / "stage.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    sim = scene.get("simulation") or {}
    ps = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    g = np.asarray(sim.get("gravity", [0, 0, -9.81]), float)
    ps.CreateGravityDirectionAttr(Gf.Vec3f(*map(float, g / np.linalg.norm(g))))
    ps.CreateGravityMagnitudeAttr(float(np.linalg.norm(g)))
    report = {"entities": [], "robots": {}, "checks": {}}
    ent_prims = {}
    # ---- entities
    for name, e in scene["entities"].items():
        pn = f"/World/Scene/{_name(name)}"
        xf = UsdGeom.Xform.Define(stage, pn)
        _xform(xf, pose_at(e, None))
        V, F, UV = _load(sdir / e["geometry"]["visual"]["obj_uri"])
        vm = _mesh(stage, pn + "/visual", V, F, UV=UV)
        mat = e.get("material") or {}
        tex = None
        if mat.get("base_color_texture"):
            src = sdir / mat["base_color_texture"]
            dst = od / "textures" / f"{_name(name)}{src.suffix}"
            shutil.copy(src, dst)
            tex = f"./textures/{dst.name}"
        bc = mat.get("base_color", [0.7, 0.7, 0.7])
        m = _material(stage, f"/World/Looks/{_name(name)}", bc, mat.get("roughness", 0.6), 0.0, tex,
                      mat.get("shader") == "emission")
        _bind(vm.GetPrim(), m)
        cap = e.get("capability")
        ph = e.get("physics") or {}
        if cap in ("static", "rigid"):
            pm = UsdShade.Material.Define(stage, f"/World/PhysicsMaterials/{_name(name)}")
            pma = UsdPhysics.MaterialAPI.Apply(pm.GetPrim())
            fr = float(phys(ph.get("friction"), 0.8))
            pma.CreateStaticFrictionAttr(fr)
            pma.CreateDynamicFrictionAttr(fr)
            pma.CreateRestitutionAttr(0.0)
            for k, c in enumerate(e["geometry"].get("collision") or []):
                T = np.asarray(c.get("T", np.eye(4)), float)
                if c["type"] == "mesh":
                    Vc, Fc, _ = _load(sdir / c["uri"])
                    Vc = Vc @ T[:3, :3].T + T[:3, 3]
                    col = _collider(stage, f"{pn}/collision_{k}", Vc, Fc)
                else:
                    col = _box_collider(stage, f"{pn}/collision_{k}", c["size"], T)
                UsdShade.MaterialBindingAPI.Apply(col.GetPrim()).Bind(pm, UsdShade.Tokens.weakerThanDescendants, "physics")
        if cap == "rigid":
            UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())
            UsdPhysics.MassAPI.Apply(xf.GetPrim()).CreateMassAttr(float(phys(ph.get("mass_kg"), 0.1)))
        ent_prims[name] = pn
        report["entities"].append(name)
    # ---- robots
    link_prims = {}
    for rid, r in (scene.get("robots") or {}).items():
        b = robot_bundle(sdir, scene, rid)
        M = b.model
        Twb = np.asarray(r["T_world_base"], float)
        q0 = {**b.init_q(), **(r.get("initial_q") or {})}
        Tl = U.fk(M, q0, Twb)
        rp = f"/World/Robots/{_name(rid)}"
        UsdPhysics.ArticulationRootAPI.Apply(UsdGeom.Xform.Define(stage, rp).GetPrim())
        act = b.profile.get("actuators") or {}
        jgroup = {j: g for g, G in b.profile["groups"].items() for j in G["joints"]}
        vis = b.profile.get("visuals") or {}
        cols = b.profile.get("collisions") or {}
        nl = 0
        for ln, L in M["links"].items():
            lp = f"{rp}/{_name(ln)}"
            xf = UsdGeom.Xform.Define(stage, lp)
            _xform(xf, Tl[ln])
            link_prims[(rid, ln)] = lp
            prim = xf.GetPrim()
            I = L.get("inertial")
            UsdPhysics.RigidBodyAPI.Apply(prim)                    # every articulation link is a rigid body
            if not (I and I["mass"] > 0):
                UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(1e-3)
            else:
                ma = UsdPhysics.MassAPI.Apply(prim)
                ma.CreateMassAttr(float(I["mass"]))
                ma.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, I["T"][:3, 3])))
                d_, Vp = _mass_props(I, I["T"][:3, :3])
                ma.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, d_)))
                ma.CreatePrincipalAxesAttr(_quat(Vp))
            if ln == M["root"]:
                fj = UsdPhysics.FixedJoint.Define(stage, f"{rp}/root_joint")
                fj.CreateBody1Rel().SetTargets([Sdf.Path(lp)])
                fj.CreateLocalPos0Attr(Gf.Vec3f(*map(float, Tl[ln][:3, 3])))
                fj.CreateLocalRot0Attr(_quat(Tl[ln]))
                fj.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
                fj.CreateLocalRot1Attr(Gf.Quatf(1, Gf.Vec3f(0, 0, 0)))
            v = vis.get(ln) or {}
            if not v.get("hidden"):
                if v.get("parts"):
                    import json
                    parts = json.loads((b.root / v["parts"]).read_text())
                    for k, p in enumerate(parts):
                        Vv, Fv, _ = _load(b.root / Path(v["parts"]).parent / p["obj"])
                        mm = _mesh(stage, f"{lp}/visual_{k}", Vv, Fv)
                        _bind(mm.GetPrim(), _material(stage, f"/World/Looks/{_name(rid)}_{_name(ln)}_{k}",
                                                      p.get("base_color", [0.5] * 3), p.get("roughness", 0.5), p.get("metallic", 0.0)))
                else:
                    import fnmatch
                    a = next((x for pat, x in (b.profile.get("appearance") or {}).items() if fnmatch.fnmatch(ln, pat)), None)
                    for k, gv in enumerate(L["visuals"]):
                        if gv["type"] != "mesh":
                            continue
                        Vv, Fv, _ = _load(gv["uri"])
                        Vv = Vv * np.asarray(gv.get("scale", [1, 1, 1])) @ gv["T"][:3, :3].T + gv["T"][:3, 3]
                        mm = _mesh(stage, f"{lp}/visual_{k}", Vv, Fv)
                        col = (a or {}).get("base_color") or [c ** 2.2 for c in (gv.get("rgba") or [0.8] * 3)[:3]]
                        _bind(mm.GetPrim(), _material(stage, f"/World/Looks/{_name(rid)}_{_name(ln)}_{k}", col,
                                                      (a or {}).get("roughness", 0.5), (a or {}).get("metallic", 0.0)))
            if ln in cols:
                for k, f in enumerate(cols[ln]):
                    Vc, Fc, _ = _load(b.root / f)
                    _collider(stage, f"{lp}/collision_{k}", Vc, Fc)
            else:
                for k, c in enumerate(L["collisions"]):
                    if c["type"] == "mesh":
                        Vc, Fc, _ = _load(c["uri"])
                        Vc = Vc * np.asarray(c.get("scale", [1, 1, 1])) @ c["T"][:3, :3].T + c["T"][:3, 3]
                        _collider(stage, f"{lp}/collision_{k}", Vc, Fc)
                    elif c["type"] == "box":
                        _box_collider(stage, f"{lp}/collision_{k}", c["size"], c["T"])
                    else:
                        _prim_collider(stage, f"{lp}/collision_{k}", c)
            nl += 1
        nj = 0
        for jn, j in M["joints"].items():
            p0, p1 = link_prims[(rid, j["parent"])], link_prims[(rid, j["child"])]
            jp = f"{rp}/joints/{_name(jn)}"
            a = np.asarray(j["axis"], float)
            a = a / (np.linalg.norm(a) + 1e-12)
            # joint frame with its x axis along the joint axis
            x = a
            y = np.cross([0, 0, 1], x) if abs(x[2]) < 0.9 else np.cross([0, 1, 0], x)
            y /= np.linalg.norm(y)
            Ra = np.stack([x, y, np.cross(x, y)], 1)
            T0 = j["T"].copy()
            T0[:3, :3] = T0[:3, :3] @ Ra                          # in the parent link frame
            if j["type"] == "fixed":
                J = UsdPhysics.FixedJoint.Define(stage, jp)
            elif j["type"] == "prismatic":
                J = UsdPhysics.PrismaticJoint.Define(stage, jp)
                J.CreateAxisAttr("X")
            else:
                J = UsdPhysics.RevoluteJoint.Define(stage, jp)
                J.CreateAxisAttr("X")
            J.CreateBody0Rel().SetTargets([Sdf.Path(p0)])
            J.CreateBody1Rel().SetTargets([Sdf.Path(p1)])
            J.CreateLocalPos0Attr(Gf.Vec3f(*map(float, T0[:3, 3])))
            J.CreateLocalRot0Attr(_quat(T0))
            J.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
            J.CreateLocalRot1Attr(_quat(Ra))
            lim = j.get("limit") or {}
            if j["type"] in ("revolute", "prismatic") and "lower" in lim:
                conv = np.degrees if j["type"] == "revolute" else (lambda v: v)
                J.CreateLowerLimitAttr(float(conv(lim["lower"])))
                J.CreateUpperLimitAttr(float(conv(lim["upper"])))
            if j["type"] in ("revolute", "continuous", "prismatic") and not j.get("mimic"):
                gname = jgroup.get(jn)
                ac = {**(act.get("default") or {}), **(act.get(gname) or {})}
                kind = "angular" if j["type"] != "prismatic" else "linear"
                dr = UsdPhysics.DriveAPI.Apply(J.GetPrim(), kind)
                per = np.pi / 180 if kind == "angular" else 1.0      # PhysX angular drive gains are per degree
                dr.CreateTypeAttr("force")
                dr.CreateStiffnessAttr(float(ac.get("kp", 100) * per))
                dr.CreateDampingAttr(float(ac.get("kv", 5) * per))
                if lim.get("effort"):
                    dr.CreateMaxForceAttr(float(lim["effort"]))
                dr.CreateTargetPositionAttr(float(np.degrees(q0.get(jn, 0.0)) if kind == "angular" else q0.get(jn, 0.0)))
            nj += 1
        report["robots"][rid] = {"links": nl, "joints": nj, "prim": rp}
    # ---- camera
    c = scene["cameras"]["source_cam"]
    K = np.asarray(c["K"], float)
    cam = UsdGeom.Camera.Define(stage, "/World/Cameras/source_cam")
    ap = 36.0
    cam.CreateHorizontalApertureAttr(ap)
    cam.CreateVerticalApertureAttr(ap * c["height"] / c["width"])
    cam.CreateFocalLengthAttr(float(K[0, 0] * ap / c["width"]))
    cam.CreateHorizontalApertureOffsetAttr(float((K[0, 2] - c["width"] / 2) * ap / c["width"]))
    cam.CreateVerticalApertureOffsetAttr(float(-(K[1, 2] - c["height"] / 2) * ap / c["width"]))
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    poses = camera_poses(sdir, scene)
    CVGL = np.diag([1.0, -1.0, -1.0, 1.0])
    op = UsdGeom.Xformable(cam).AddTransformOp()
    fps = float(c.get("fps") or 20.0)
    stage.SetTimeCodesPerSecond(fps)
    if len(poses) > 1:
        for f, T in enumerate(poses):
            op.Set(_mat(T @ CVGL), Usd.TimeCode(f))
        stage.SetStartTimeCode(0)
        stage.SetEndTimeCode(len(poses) - 1)
    else:
        op.Set(_mat(poses[0] @ CVGL))
    stage.GetRootLayer().Save()
    outs = [f"{out_dir}/stage.usda"]
    # ---- playback layer
    rm = (scene.get("render") or {}).get("motion")
    if playback and rm:
        mo = load_motion(sdir, scene, rm)
        f0, f1 = frames or [0, int(np.floor(mo["frame"][-1]))]
        pb = Usd.Stage.CreateNew(str(od / "playback.usda"))
        pb.GetRootLayer().subLayerPaths.append("./stage.usda")
        pb.SetTimeCodesPerSecond(fps)
        pb.SetStartTimeCode(f0)
        pb.SetEndTimeCode(f1)
        UsdGeom.SetStageUpAxis(pb, UsdGeom.Tokens.z)
        rid = scene["motions"][rm].get("robot")
        ops = {}
        for f in range(f0, f1 + 1):
            q, objs = motion_at(mo, f)
            if rid:
                b = robot_bundle(sdir, scene, rid)
                Tl = U.fk(b.model, {**b.init_q(), **q}, np.asarray(scene["robots"][rid]["T_world_base"], float))
                for ln, T in Tl.items():
                    pp = link_prims[(rid, ln)]
                    if pp not in ops:
                        pr = pb.OverridePrim(pp)
                        xf = UsdGeom.Xformable(pr)
                        xf.ClearXformOpOrder()
                        ops[pp] = xf.AddTransformOp(opSuffix="playback")
                    ops[pp].Set(_mat(T), Usd.TimeCode(f))
            for n, T in objs.items():
                pp = ent_prims[n]
                if pp not in ops:
                    pr = pb.OverridePrim(pp)
                    xf = UsdGeom.Xformable(pr)
                    xf.ClearXformOpOrder()
                    ops[pp] = xf.AddTransformOp(opSuffix="playback")
                ops[pp].Set(_mat(T), Usd.TimeCode(f))
        pb.GetRootLayer().Save()
        outs.append(f"{out_dir}/playback.usda")
        report["playback"] = {"motion": rm, "frames": [f0, f1], "animated_prims": len(ops)}
    # ---- validation
    st = Usd.Stage.Open(str(path))
    bad_joint, n_joint = [], 0
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    for prim in st.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            J = UsdPhysics.Joint(prim)
            n_joint += 1
            b0 = J.GetBody0Rel().GetTargets()
            b1 = J.GetBody1Rel().GetTargets()
            def frame(btargets, pos, rot):
                T = np.eye(4)
                q = rot.Get()
                from ..geom3d import quat_to_mat
                T[:3, :3] = quat_to_mat([q.GetReal(), *q.GetImaginary()])
                T[:3, 3] = list(pos.Get())
                if btargets:
                    W = np.array(xc.GetLocalToWorldTransform(st.GetPrimAtPath(btargets[0]))).T
                    T = W @ T
                return T
            A = frame(b0, J.GetLocalPos0Attr(), J.GetLocalRot0Attr())
            B = frame(b1, J.GetLocalPos1Attr(), J.GetLocalRot1Attr())
            dp = np.linalg.norm(A[:3, 3] - B[:3, 3])
            Rr = A[:3, :3].T @ B[:3, :3]
            if prim.IsA(UsdPhysics.RevoluteJoint):
                off_axis = np.linalg.norm(Rr[:, 0] - [1, 0, 0])       # rotation must be about x only
            elif prim.IsA(UsdPhysics.FixedJoint):
                off_axis = np.linalg.norm(Rr - np.eye(3))
            else:
                off_axis = 0.0
            if dp > 1e-4 or off_axis > 1e-4:
                bad_joint.append({"joint": str(prim.GetPath()), "pos_err_m": float(dp), "rot_err": float(off_axis)})
    bodies = [p for p in st.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    no_col = [str(p.GetPath()) for p in bodies
              if not any(c.HasAPI(UsdPhysics.CollisionAPI) for c in Usd.PrimRange(p))]   # e.g. massless frames
    report["checks"] = {"opens": True, "joints": n_joint, "joint_frame_mismatch": bad_joint[:10],
                        "rigid_bodies": len(bodies), "bodies_without_collider": no_col[:20],
                        "articulation_roots": sum(1 for p in st.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)),
                        "ok": not bad_joint}
    write_json(report, od / "export.json")
    return {**report, "outputs": outs + [f"{out_dir}/export.json"]}
