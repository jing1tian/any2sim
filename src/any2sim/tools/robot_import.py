"""Import a robot description (URDF) into a RobotBundle and check it (any embodiment: arms, hands, grippers).

Copies every referenced mesh into <out>/meshes/ (relative paths), writes robot.urdf, a robot.yaml profile (groups
detected from the kinematic tree unless given: serial chains are arms, short branching subtrees are hands/grippers),
compiles robot.xml (MuJoCo) and checks it: loads, masses/inertias, joint limits, self-contact at the init pose,
mimic joints. Renders a preview sheet (init pose and every named pose, 4 views).
Re-running on an existing bundle keeps the profile fields you set (groups, poses, actuators, visuals, cameras...)
unless `reset_profile: true`.

params:
  urdf: path (absolute, or project-relative)          source URDF
  package_dirs: {pkg: dir}                             for package:// mesh paths
  out: "robot"                                         bundle folder (project-relative)
  name: null                                           default: URDF robot name
  base_link: null                                      default: URDF root
  groups: null | {group: {kind, joints, tip_link, side}}
  init_pose: null | {joint: value}                     default: 0 clipped into the limits
  named_poses: null | {name: {joint: value}}
  actuators: null | {default|group: {kp, kv, armature, damping}}
  auto_exclude_hops: 2                                 self-contacts already present at the init pose between links at
                                                       most this many joints apart are excluded (mesh-hull artefacts)
  render: true
Returns groups, checks and the preview path.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from ..common import contact_sheet, write_json, write_rgb

ENV = "base"
GPU = False

KEEP = ("groups", "init_pose", "named_poses", "actuators", "controller", "contact", "collision_exclude", "visuals",
        "collisions", "appearance", "cameras", "mount", "data_mapping", "retarget", "provenance", "checks", "frames")


def detect_groups(model: dict, max_hand_depth: int = 5) -> dict:
    """Arms = serial chains of movable joints; a branching subtree whose branches are all short (<= max_hand_depth
    movable joints) is a hand/gripper. Names: <side>_<kind>[_k]."""
    from ..robot.urdf import MOVABLE
    ch = {}
    for jn, j in model["joints"].items():
        ch.setdefault(j["parent"], []).append(jn)

    def movable_below(link):
        out = []
        for jn in ch.get(link, []):
            j = model["joints"][jn]
            if j["type"] in MOVABLE and not j.get("mimic"):
                out.append(jn)
            out += movable_below(j["child"])
        return out

    def depth(link):
        best = 0
        for jn in ch.get(link, []):
            j = model["joints"][jn]
            best = max(best, depth(j["child"]) + (1 if j["type"] in MOVABLE and not j.get("mimic") else 0))
        return best

    groups = []

    def walk(link, chain, last_link):
        branches = [jn for jn in ch.get(link, []) if movable_below(model["joints"][jn]["child"]) or
                    model["joints"][jn]["type"] in MOVABLE]
        if len(branches) >= 2:
            if all(depth(model["joints"][b]["child"]) + 1 <= max_hand_depth for b in branches) and chain:
                groups.append(("arm", chain, last_link))
                groups.append(("hand", movable_below(link), link))
                return
            if chain:
                groups.append(("arm", chain, last_link))
            for b in branches:
                j = model["joints"][b]
                walk(j["child"], [b] if j["type"] in MOVABLE and not j.get("mimic") else [], j["child"])
            return
        if not branches:
            if chain:
                groups.append(("arm", chain, last_link))
            return
        j = model["joints"][branches[0]]
        mov = j["type"] in MOVABLE and not j.get("mimic")
        walk(j["child"], chain + ([branches[0]] if mov else []), j["child"] if mov else last_link)

    walk(model["root"], [], model["root"])
    out = {}
    for kind, joints, tip in groups:
        if not joints:
            continue
        s = " ".join(joints).lower()
        side = "left" if "left" in s else "right" if "right" in s else None
        if kind == "hand" and len(joints) <= 2:
            kind = "gripper"
        name = f"{side}_{kind}" if side else kind
        k = 1
        while name in out:
            k += 1
            name = f"{side}_{kind}_{k}" if side else f"{kind}_{k}"
        out[name] = {"kind": kind, "side": side, "joints": joints, "tip_link": tip}
    return out


def _copy_meshes(model: dict, out: Path) -> None:
    mdir = out / "meshes"
    mdir.mkdir(parents=True, exist_ok=True)
    seen = {}
    for ln, l in model["links"].items():
        for g in l["visuals"] + l["collisions"]:
            if g["type"] != "mesh":
                continue
            src = Path(g["uri"])
            if str(src) not in seen:
                try:
                    rel = Path(src).resolve().relative_to(out.resolve())     # already inside the bundle
                    seen[str(src)] = out / rel
                except ValueError:
                    dst = mdir / src.parent.name / src.name
                    k = 1
                    while dst.exists() and str(dst) not in seen.values() and dst.read_bytes() != src.read_bytes():
                        dst = mdir / src.parent.name / f"{src.stem}_{k}{src.suffix}"
                        k += 1
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(src, dst)
                    seen[str(src)] = dst
            g["uri"] = str(seen[str(src)])


def default_actuators(groups: dict) -> dict:
    return {"default": {"kp": 200.0, "kv": 10.0, "armature": 0.01, "damping": 0.5},
            **{g: ({"kp": 400.0, "kv": 40.0, "armature": 0.05, "damping": 1.0} if G["kind"] == "arm"
                   else {"kp": 5.0, "kv": 0.1, "armature": 0.0005, "damping": 0.01}) for g, G in groups.items()}}


def check_bundle(bundle, poses: dict, render: bool, out: Path) -> dict:
    """Load robot.xml in MuJoCo; mass/inertia, limits, self-contacts per pose; optional preview images."""
    import mujoco
    from ..robot.render import CalibRenderer, add_calib_camera, look_at
    from ..robot import mjcf
    res = {}
    fr = mjcf.robot_fragment(bundle)
    xml = (f'<mujoco>{mjcf.compiler_xml()}<option timestep="0.002"/><asset>{"".join(fr["assets"])}</asset>'
           f'<worldbody><light pos="0 0 3" directional="true"/><light pos="1 -1 2" dir="-1 1 -2"/>'
           f'{add_calib_camera(fr["body"], 640, 480)}</worldbody>'
           f'<actuator>{"".join(fr["actuators"])}</actuator>'
           + (f'<equality>{"".join(fr["equality"])}</equality>' if fr["equality"] else "")
           + (f'<contact>{"".join(fr["contact"])}</contact>' if fr["contact"] else "") + "</mujoco>")
    m = mujoco.MjModel.from_xml_string(xml)
    res["mujoco"] = {"nbody": m.nbody, "njnt": m.njnt, "nu": m.nu, "ngeom": m.ngeom,
                     "total_mass_kg": float(m.body_subtreemass[0])}
    light = [m.body(b).name for b in range(1, m.nbody) if 0 < m.body_mass[b] < 1e-4]
    res["tiny_mass_bodies"] = light
    d = mujoco.MjData(m)
    contacts = {}
    imgs, labels = [], []
    ren = CalibRenderer(m, 640, 480, groups=(1,)) if render else None

    def set_pose(q):
        d.qpos[:] = 0
        for jn, v in q.items():
            try:
                d.qpos[m.jnt_qposadr[m.joint(jn).id]] = v
            except KeyError:
                pass
        mujoco.mj_forward(m, d)

    set_pose(next(iter(poses.values())))
    P = d.xpos[1:]
    c0, R = (P.min(0) + P.max(0)) / 2, max(np.linalg.norm(P.max(0) - P.min(0)), 0.3) * 1.3
    K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1.0]])
    for pname, q in poses.items():
        set_pose(q)
        pairs = {}
        for c in d.contact[:d.ncon]:
            b1, b2 = m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name
            k = "|".join(sorted([b1, b2]))
            pairs[k] = min(pairs.get(k, 0.0), float(c.dist))
        contacts[pname] = dict(sorted(pairs.items(), key=lambda kv: kv[1]))
        if ren is not None:
            for vn, eye in (("front", [1, 0, 0.4]), ("left", [0, 1, 0.4]), ("top", [0.3, 0, 1]), ("back", [-1, -0.4, 0.5])):
                e = c0 + R * np.asarray(eye) / np.linalg.norm(eye)
                imgs.append(ren.rgb(d, K, look_at(e, c0)))
                labels.append(f"{pname} {vn}")
    res["self_contacts"] = contacts
    if ren is not None:
        ren.close()
        write_rgb(out / "checks" / "preview.jpg", contact_sheet(imgs, cols=4, width=320, labels=labels))
        res["preview"] = "checks/preview.jpg"
    return res


def _auto_exclude(b, contacts: dict, hops: int) -> list:
    M = b.model
    parent = {j["child"]: j["parent"] for j in M["joints"].values()}
    nmov = {j["child"]: j["type"] != "fixed" for j in M["joints"].values()}

    def chain(l):
        out = [l]
        while l in parent:
            l = parent[l]
            out.append(l)
        return out

    def dist(a, b_):
        ca, cb = chain(a), chain(b_)
        common = next(x for x in ca if x in cb)
        n = 0
        for c in (ca, cb):
            for x in c[:c.index(common)]:
                n += nmov.get(x, False)
        return n

    excl = [list(x) for x in (b.profile.get("collision_exclude") or [])]
    have = {tuple(sorted(x)) for x in excl}
    added = []
    for k in contacts:
        a, c = k.split("|")
        if tuple(sorted((a, c))) not in have and dist(a, c) <= hops:
            excl.append([a, c])
            added.append([a, c])
    b.profile["collision_exclude"] = excl
    return added


def run(project: Path, urdf: str, package_dirs: dict | None = None, out: str = "robot", name: str | None = None,
        base_link: str | None = None, groups: dict | None = None, init_pose: dict | None = None,
        named_poses: dict | None = None, actuators: dict | None = None, render: bool = True,
        reset_profile: bool = False, auto_exclude_hops: int = 2) -> dict:
    import yaml
    from ..robot import mjcf
    from ..robot import urdf as U
    from ..robot.bundle import Bundle, validate
    out_dir = project / out
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(urdf) if Path(urdf).is_absolute() else project / urdf
    model = U.load(src, package_dirs)
    if base_link and base_link != model["root"]:
        raise ValueError(f"base_link {base_link} is not the URDF root {model['root']} (re-root the URDF first)")
    _copy_meshes(model, out_dir)
    U.write(model, out_dir / "robot.urdf")
    old = {}
    if (out_dir / "robot.yaml").exists() and not reset_profile:
        old = yaml.safe_load((out_dir / "robot.yaml").read_text()) or {}
    G = groups or old.get("groups") or detect_groups(model)
    movable = U.movable_joints(model)
    lim = {j: (model["joints"][j].get("limit") or {}) for j in movable}
    q0 = {j: float(np.clip(0.0, lim[j].get("lower", -np.inf), lim[j].get("upper", np.inf))) for j in movable}
    q0.update(old.get("init_pose") or {})
    q0.update(init_pose or {})
    prof = {"schema_version": "0.1", "name": name or old.get("name") or model["name"], "urdf": "robot.urdf",
            "mjcf": "robot.xml", "base_link": model["root"], "source": {"urdf": str(src)}}
    for k in KEEP:
        if k in old:
            prof[k] = old[k]
    prof["groups"] = G
    prof["init_pose"] = q0
    if named_poses:
        prof["named_poses"] = {**(prof.get("named_poses") or {}), **named_poses}
    prof["actuators"] = actuators or prof.get("actuators") or default_actuators(G)
    prof.setdefault("controller", {"gravity_compensation": True, "rate_hz": None})
    prof.setdefault("provenance", {})
    prof["provenance"].setdefault("kinematics", f"supplied URDF {src.name}")
    (out_dir / "robot.yaml").write_text(yaml.safe_dump(prof, sort_keys=False, allow_unicode=True))
    b = Bundle(out_dir)
    mjcf.compile_bundle(b)
    poses = {"init": b.init_q()}
    for pn, pq in (b.profile.get("named_poses") or {}).items():
        poses[pn] = {**b.init_q(), **pq}
    chk = check_bundle(b, poses, False, out_dir)
    added = _auto_exclude(b, chk["self_contacts"]["init"], auto_exclude_hops)
    if added:
        b.save_profile()
        b = Bundle(out_dir)
        mjcf.compile_bundle(b)
    chk = check_bundle(b, poses, render, out_dir)
    chk["auto_excluded"] = added
    v = validate(out_dir)
    res = {"bundle": out, "groups": {g: {"kind": x["kind"], "n_joints": len(x["joints"]), "tip_link": x.get("tip_link")}
                                      for g, x in G.items()},
           "n_movable": len(movable), "validate": v, **chk}
    write_json(res, out_dir / "checks" / "import.json")
    return {**res, "outputs": [out]}
