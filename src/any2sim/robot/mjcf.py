"""RobotBundle -> MJCF: body tree with joints, inertials, visual geoms (group 1, no contact), collision geoms
(group 3), position actuators, mimic equalities and contact excludes.

`robot_fragment()` returns the pieces (assets, body xml, actuators, equality, contact) so a scene compiler can
embed the robot under any mount transform; `compile_bundle()` writes a standalone robot.xml into the bundle.
"""
from __future__ import annotations

import fnmatch
import json
from pathlib import Path

import numpy as np

from ..geom3d import mat_to_quat


def _p(T):
    return " ".join(f"{x:.7g}" for x in np.asarray(T)[:3, 3])


def _q(T):
    return " ".join(f"{x:.8g}" for x in mat_to_quat(np.asarray(T)[:3, :3]))


def _merge(*ds):
    out = {}
    for d in ds:
        out.update({k: v for k, v in (d or {}).items() if v is not None})
    return out


def _group_of(profile: dict) -> dict:
    out = {}
    for g, G in (profile.get("groups") or {}).items():
        for j in G.get("joints", []):
            out[j] = g
    return out


def _appearance(profile: dict, link: str):
    for pat, a in (profile.get("appearance") or {}).items():
        if fnmatch.fnmatch(link, pat):
            return a
    return None


def _linear_to_srgb(c):
    c = np.clip(np.asarray(c, float), 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def robot_fragment(bundle, prefix: str = "", mesh_ref=None, T_world_base=None, with_visuals=True) -> dict:
    """MJCF pieces for a bundle. mesh_ref(path) -> file attribute (default: absolute path).
    Names are prefixed with `prefix` (empty for a single robot)."""
    P, M = bundle.profile, bundle.model
    mesh_ref = mesh_ref or (lambda p: str(Path(p).resolve()))
    jgroup = _group_of(P)
    act = P.get("actuators") or {}
    contact = P.get("contact") or {}
    gravcomp = (P.get("controller") or {}).get("gravity_compensation", True)
    assets, actuators, equality = [], [], []
    done_mesh = set()

    def mesh_asset(name, path, scale=None):
        if name not in done_mesh:
            sc = "" if scale is None or np.allclose(scale, 1) else f' scale="{" ".join(map(str, scale))}"'
            assets.append(f'<mesh name="{name}" file="{mesh_ref(path)}"{sc}/>')
            done_mesh.add(name)
        return name

    ch = {}
    for jn, j in M["joints"].items():
        ch.setdefault(j["parent"], []).append(jn)
    vis_over = P.get("visuals") or {}
    col_over = P.get("collisions") or {}
    link_group = {}
    for jn, j in M["joints"].items():                       # a link belongs to the group of its nearest joint
        if jn in jgroup:
            link_group[j["child"]] = jgroup[jn]
    changed = True
    while changed:
        changed = False
        for jn, j in M["joints"].items():
            if j["child"] not in link_group and j["parent"] in link_group:
                link_group[j["child"]] = link_group[j["parent"]]
                changed = True

    def geoms(ln):
        out = []
        L = M["links"][ln]
        g = link_group.get(ln)
        cp = _merge(contact.get("default"), contact.get(g))
        fr = cp.get("friction", [1.0, 0.005, 0.0001])
        cattr = (f' friction="{" ".join(map(str, fr))}" condim="{cp.get("condim", 4)}"'
                 + (f' solref="{" ".join(map(str, cp["solref"]))}"' if cp.get("solref") else "")
                 + (f' solimp="{" ".join(map(str, cp["solimp"]))}"' if cp.get("solimp") else "")
                 + (f' margin="{cp["margin"]}"' if cp.get("margin") else ""))
        if with_visuals and not (vis_over.get(ln) or {}).get("hidden"):
            if ln in vis_over and vis_over[ln].get("parts"):
                parts = json.loads((bundle.root / vis_over[ln]["parts"]).read_text())
                for k, part in enumerate(parts):
                    mn = mesh_asset(f"{prefix}{ln}_v{k}", bundle.root / Path(vis_over[ln]["parts"]).parent / part["obj"])
                    rgba = list(_linear_to_srgb(part.get("base_color", [0.5] * 3))) + [1]
                    out.append(f'<geom name="{prefix}{ln}_v{k}" type="mesh" mesh="{mn}" rgba="{" ".join(f"{c:.3f}" for c in rgba)}" '
                               f'contype="0" conaffinity="0" group="1" density="0"/>')
            else:
                for k, v in enumerate(L["visuals"]):
                    a = _appearance(P, ln)
                    rgba = (list(_linear_to_srgb(a["base_color"])) + [1]) if a else v.get("rgba", [0.8, 0.8, 0.8, 1])
                    rgba_s = " ".join(f"{c:.3f}" for c in rgba)
                    out.append(_prim(f"{prefix}{ln}_v{k}", v, mesh_asset, f'rgba="{rgba_s}" contype="0" conaffinity="0" group="1" density="0"'))
        if ln in col_over:
            for k, f in enumerate(col_over[ln]):
                mn = mesh_asset(f"{prefix}{ln}_c{k}", bundle.root / f)
                out.append(f'<geom name="{prefix}{ln}_c{k}" type="mesh" mesh="{mn}" group="3" rgba="0.9 0.4 0.3 0.5" density="0"{cattr}/>')
        else:
            for k, c in enumerate(L["collisions"]):
                out.append(_prim(f"{prefix}{ln}_c{k}", c, mesh_asset, f'group="3" rgba="0.9 0.4 0.3 0.5" density="0"{cattr}'))
        return out

    def body(ln, T_parent_link, joint_xml, depth):
        L = M["links"][ln]
        attrs = f'name="{prefix}{ln}" pos="{_p(T_parent_link)}" quat="{_q(T_parent_link)}"'
        if gravcomp and depth > 0:
            attrs += ' gravcomp="1"'
        xs = [f"<body {attrs}>"]
        if joint_xml:
            xs.append(joint_xml)
        I = L.get("inertial")
        if I and I["mass"] > 0:
            ixx, iyy, izz, ixy, ixz, iyz = I["inertia"]
            R = np.asarray(I["T"])[:3, :3]                   # inertia expressed in the body frame
            Ib = R @ np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]) @ R.T
            xs.append(f'<inertial pos="{_p(I["T"])}" mass="{I["mass"]:.7g}" fullinertia="{Ib[0, 0]:.7g} {Ib[1, 1]:.7g} '
                      f'{Ib[2, 2]:.7g} {Ib[0, 1]:.7g} {Ib[0, 2]:.7g} {Ib[1, 2]:.7g}"/>')
        xs += geoms(ln)
        xs.append(f'<site name="{prefix}{ln}_frame" size="0.002" group="5"/>')
        for jn in ch.get(ln, []):
            j = M["joints"][jn]
            jx = ""
            if j["type"] in ("revolute", "continuous", "prismatic"):
                g = jgroup.get(jn)
                a = _merge(act.get("default"), act.get(g))
                lim = j.get("limit") or {}
                rng = (f' range="{lim["lower"]} {lim["upper"]}" limited="true"'
                       if j["type"] != "continuous" and "lower" in lim else "")
                jx = (f'<joint name="{prefix}{jn}" type="{"slide" if j["type"] == "prismatic" else "hinge"}" '
                      f'axis="{" ".join(map(str, j["axis"]))}"{rng} armature="{a.get("armature", 0.001)}" '
                      f'damping="{a.get("damping", 0.0)}" frictionloss="{a.get("frictionloss", 0.0)}"'
                      + (' actuatorgravcomp="true"' if gravcomp else "") + "/>")
                if j.get("mimic"):
                    m = j["mimic"]
                    equality.append(f'<joint joint1="{prefix}{jn}" joint2="{prefix}{m["joint"]}" '
                                    f'polycoef="{m["offset"]} {m["multiplier"]} 0 0 0"/>')
                else:
                    eff = lim.get("effort", a.get("forcerange", 100.0))
                    ctrl = f' ctrlrange="{lim["lower"]} {lim["upper"]}" ctrllimited="true"' if "lower" in lim else ""
                    actuators.append(f'<position name="{prefix}{jn}" joint="{prefix}{jn}" kp="{a.get("kp", 100)}" '
                                     f'kv="{a.get("kv", 5)}" forcerange="{-eff} {eff}" forcelimited="true"{ctrl}/>')
            xs.append(body(j["child"], j["T"], jx, depth + 1))
        xs.append("</body>")
        return "".join(xs)

    root = P.get("base_link", M["root"])
    T0 = np.eye(4) if T_world_base is None else np.asarray(T_world_base, float)
    if root != M["root"]:
        raise ValueError("base_link must be the URDF root")
    tree = body(root, T0, "", 0)
    excl = [f'<exclude body1="{prefix}{a}" body2="{prefix}{b}"/>' for a, b in (P.get("collision_exclude") or [])]
    return {"assets": assets, "body": tree, "actuators": actuators, "equality": equality, "contact": excl,
            "joints": [prefix + j for j in bundle.joints() if not M["joints"][j].get("mimic")]}


def _prim(name, g, mesh_asset, extra):
    T = g["T"]
    pose = f'pos="{_p(T)}" quat="{_q(T)}"'
    if g["type"] == "mesh":
        mn = mesh_asset(name, g["uri"], g.get("scale"))
        return f'<geom name="{name}" type="mesh" mesh="{mn}" {pose} {extra}/>'
    if g["type"] == "box":
        s = " ".join(str(x / 2) for x in g["size"])
        return f'<geom name="{name}" type="box" size="{s}" {pose} {extra}/>'
    if g["type"] == "cylinder":
        return f'<geom name="{name}" type="cylinder" size="{g["radius"]} {g["length"] / 2}" {pose} {extra}/>'
    return f'<geom name="{name}" type="sphere" size="{g["radius"]}" {pose} {extra}/>'


def compiler_xml() -> str:
    return ('<compiler angle="radian" autolimits="true" balanceinertia="true" boundmass="1e-5" '
            'boundinertia="1e-10" fusestatic="false" discardvisual="false"/>')


def compile_bundle(bundle, timestep: float = 0.002) -> str:
    """Standalone robot.xml in the bundle (paths relative to the bundle), with a floor and a light for previews."""
    fr = robot_fragment(bundle, mesh_ref=lambda p: str(Path(p).resolve().relative_to(bundle.root)))
    xml = (f'<mujoco model="{bundle.name}">\n  {compiler_xml()}\n'
           f'  <option timestep="{timestep}" cone="elliptic" impratio="10" integrator="implicitfast"/>\n'
           f'  <visual><global offwidth="1280" offheight="960"/></visual>\n'
           '  <asset>\n    ' + "\n    ".join(fr["assets"]) + '\n  </asset>\n'
           '  <worldbody>\n    <light pos="0 0 3" dir="0 0 -1" directional="true"/>\n    ' + fr["body"] + '\n  </worldbody>\n'
           '  <actuator>\n    ' + "\n    ".join(fr["actuators"]) + '\n  </actuator>\n'
           + ('  <equality>\n    ' + "\n    ".join(fr["equality"]) + '\n  </equality>\n' if fr["equality"] else "")
           + ('  <contact>\n    ' + "\n    ".join(fr["contact"]) + '\n  </contact>\n' if fr["contact"] else "")
           + '</mujoco>\n')
    out = bundle.root / "robot.xml"
    out.write_text(xml)
    return str(out)
