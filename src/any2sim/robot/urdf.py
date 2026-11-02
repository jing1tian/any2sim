"""URDF reading/writing and forward kinematics (numpy only; no ROS).

Model: links {name: {inertial, visuals, collisions}}, joints {name: {type, parent, child, T (parent->joint origin),
axis, limit, mimic}}. Mesh paths are resolved to absolute paths at load time (package:// is resolved against
`package_dirs`), so a model can be re-written anywhere.
"""
from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

MOVABLE = ("revolute", "continuous", "prismatic")


def rpy_to_mat(r, p, y) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def mat_to_rpy(R) -> np.ndarray:
    sp = -R[2, 0]
    if abs(sp) < 1 - 1e-9:
        return np.array([np.arctan2(R[2, 1], R[2, 2]), np.arcsin(sp), np.arctan2(R[1, 0], R[0, 0])])
    return np.array([np.arctan2(-R[1, 2], R[1, 1]), np.pi / 2 * np.sign(sp), 0.0])


def origin_T(el) -> np.ndarray:
    T = np.eye(4)
    if el is None:
        return T
    xyz = [float(v) for v in (el.get("xyz") or "0 0 0").split()]
    rpy = [float(v) for v in (el.get("rpy") or "0 0 0").split()]
    T[:3, :3] = rpy_to_mat(*rpy)
    T[:3, 3] = xyz
    return T


def axis_angle(axis, a) -> np.ndarray:
    k = np.asarray(axis, float)
    k = k / (np.linalg.norm(k) + 1e-12)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def _resolve(fn: str, base: Path, package_dirs: dict) -> str:
    if fn.startswith("package://"):
        pkg, rest = fn[len("package://"):].split("/", 1)
        root = Path(package_dirs.get(pkg, base))
        return str((root / rest).resolve())
    if fn.startswith("file://"):
        return fn[len("file://"):]
    return str((base / fn).resolve())


def _geom(el, base, package_dirs):
    g = el.find("geometry")
    if g is None or len(g) == 0:
        return None
    s = g[0]
    out = {"T": origin_T(el.find("origin"))}
    if s.tag == "mesh":
        out.update(type="mesh", uri=_resolve(s.get("filename"), base, package_dirs),
                   scale=[float(v) for v in (s.get("scale") or "1 1 1").split()])
    elif s.tag == "box":
        out.update(type="box", size=[float(v) for v in s.get("size").split()])
    elif s.tag == "cylinder":
        out.update(type="cylinder", radius=float(s.get("radius")), length=float(s.get("length")))
    elif s.tag == "sphere":
        out.update(type="sphere", radius=float(s.get("radius")))
    else:
        return None
    m = el.find("material")
    if m is not None and m.find("color") is not None:
        out["rgba"] = [float(v) for v in m.find("color").get("rgba").split()]
    return out


def load(path: str | Path, package_dirs: dict | None = None) -> dict:
    path = Path(path).resolve()
    root = ET.parse(path).getroot()
    base = path.parent
    links, joints = {}, {}
    for l in root.findall("link"):
        inert = l.find("inertial")
        I = None
        if inert is not None and inert.find("mass") is not None:
            ii = inert.find("inertia")
            I = {"T": origin_T(inert.find("origin")), "mass": float(inert.find("mass").get("value")),
                 "inertia": [float(ii.get(k, 0)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")] if ii is not None else [0] * 6}
        links[l.get("name")] = {
            "inertial": I,
            "visuals": [g for g in (_geom(v, base, package_dirs or {}) for v in l.findall("visual")) if g],
            "collisions": [g for g in (_geom(c, base, package_dirs or {}) for c in l.findall("collision")) if g]}
    for j in root.findall("joint"):
        lim = j.find("limit")
        mim = j.find("mimic")
        ax = j.find("axis")
        joints[j.get("name")] = {
            "type": j.get("type"), "parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
            "T": origin_T(j.find("origin")),
            "axis": [float(v) for v in ax.get("xyz").split()] if ax is not None else [1.0, 0, 0],
            "limit": None if lim is None else {k: float(lim.get(k)) for k in ("lower", "upper", "effort", "velocity") if lim.get(k) is not None},
            "mimic": None if mim is None else {"joint": mim.get("joint"), "multiplier": float(mim.get("multiplier", 1)),
                                               "offset": float(mim.get("offset", 0))}}
    children = {j["child"] for j in joints.values()}
    roots = [l for l in links if l not in children]
    return {"name": root.get("name"), "links": links, "joints": joints, "root": roots[0] if roots else None,
            "source": str(path)}


def movable_joints(model: dict) -> list[str]:
    return [n for n, j in model["joints"].items() if j["type"] in MOVABLE and not j.get("mimic")]


def child_joints(model: dict) -> dict:
    out = {}
    for n, j in model["joints"].items():
        out.setdefault(j["parent"], []).append(n)
    return out


def fk(model: dict, q: dict | None = None, T_world_root=None) -> dict:
    """World pose (4x4) of every link for joint values q {joint: value} (missing joints = 0; mimic joints follow)."""
    q = dict(q or {})
    for n, j in model["joints"].items():
        if j.get("mimic"):
            m = j["mimic"]
            q[n] = q.get(m["joint"], 0.0) * m["multiplier"] + m["offset"]
    out = {model["root"]: np.eye(4) if T_world_root is None else np.asarray(T_world_root, float)}
    ch = child_joints(model)
    todo = [model["root"]]
    while todo:
        l = todo.pop()
        for jn in ch.get(l, []):
            j = model["joints"][jn]
            T = out[l] @ j["T"]
            a = q.get(jn, 0.0)
            M = np.eye(4)
            if j["type"] in ("revolute", "continuous"):
                M[:3, :3] = axis_angle(j["axis"], a)
            elif j["type"] == "prismatic":
                M[:3, 3] = np.asarray(j["axis"], float) / np.linalg.norm(j["axis"]) * a
            out[j["child"]] = T @ M
            todo.append(j["child"])
    return out


def _fmt(v):
    return " ".join(f"{float(x):.9g}" for x in v)


def _origin(T):
    return f'<origin xyz="{_fmt(T[:3, 3])}" rpy="{_fmt(mat_to_rpy(T[:3, :3]))}"/>'


def _geom_xml(g, rel_to: Path):
    if g["type"] == "mesh":
        try:
            fn = str(Path(g["uri"]).resolve().relative_to(rel_to.resolve()))
        except ValueError:
            fn = g["uri"]
        sc = g.get("scale", [1, 1, 1])
        s = f'<mesh filename="{fn}"' + ("" if np.allclose(sc, 1) else f' scale="{_fmt(sc)}"') + "/>"
    elif g["type"] == "box":
        s = f'<box size="{_fmt(g["size"])}"/>'
    elif g["type"] == "cylinder":
        s = f'<cylinder radius="{g["radius"]}" length="{g["length"]}"/>'
    else:
        s = f'<sphere radius="{g["radius"]}"/>'
    return f'{_origin(g["T"])}<geometry>{s}</geometry>'


def write(model: dict, path: str | Path) -> str:
    """Write a URDF; mesh paths become relative to the output folder when possible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = [f'<?xml version="1.0"?>\n<robot name="{model["name"]}">']
    for ln, l in model["links"].items():
        out.append(f'  <link name="{ln}">')
        if l.get("inertial"):
            I = l["inertial"]
            ii = " ".join(f'{k}="{v:.9g}"' for k, v in zip(("ixx", "iyy", "izz", "ixy", "ixz", "iyz"), I["inertia"]))
            out.append(f'    <inertial>{_origin(I["T"])}<mass value="{I["mass"]:.9g}"/><inertia {ii}/></inertial>')
        for g in l.get("visuals", []):
            mat = ""
            if g.get("rgba"):
                mat = f'<material name="{ln}_mat"><color rgba="{_fmt(g["rgba"])}"/></material>'
            out.append(f'    <visual>{_geom_xml(g, path.parent)}{mat}</visual>')
        for g in l.get("collisions", []):
            out.append(f'    <collision>{_geom_xml(g, path.parent)}</collision>')
        out.append("  </link>")
    for jn, j in model["joints"].items():
        s = f'  <joint name="{jn}" type="{j["type"]}"><parent link="{j["parent"]}"/><child link="{j["child"]}"/>{_origin(j["T"])}'
        if j["type"] != "fixed":
            s += f'<axis xyz="{_fmt(j["axis"])}"/>'
        if j.get("limit"):
            s += "<limit " + " ".join(f'{k}="{v:.9g}"' for k, v in j["limit"].items()) + "/>"
        if j.get("mimic"):
            m = j["mimic"]
            s += f'<mimic joint="{m["joint"]}" multiplier="{m["multiplier"]}" offset="{m["offset"]}"/>'
        out.append(s + "</joint>")
    out.append("</robot>\n")
    path.write_text("\n".join(out))
    return str(path)


def copy_model(model: dict) -> dict:
    return copy.deepcopy(model)
