"""Scene IR -> MJCF (mujoco/scene.xml): static/rigid/visual entities, robots (RobotBundles at T_world_base with
position actuators), simulation settings, the source camera and a calibrated "calib" camera for rendering.

Every asset is copied under mujoco/assets/ so the delivered folder is self-contained (copied only when missing or
changed, so parallel builds of scene variants into one folder never rewrite a file another worker is reading).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from ..geom3d import cv_to_gl, mat_to_quat


def _q(T):
    return " ".join(f"{x:.8f}" for x in mat_to_quat(np.asarray(T)[:3, :3]))


def _p(T):
    return " ".join(f"{x:.6f}" for x in np.asarray(T)[:3, 3])


def _copy(src, dst):
    """Copy unless dst already has src's size and is not older (atomic replace otherwise)."""
    src, dst = Path(src), Path(dst)
    if dst.exists() and dst.stat().st_size == src.stat().st_size and dst.stat().st_mtime >= src.stat().st_mtime:
        return
    tmp = dst.with_name(f".{dst.name}.{__import__('os').getpid()}.tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)


def entity_xml(project: Path, scene: dict, adir: Path) -> tuple[list, list]:
    from ..scene_ir import phys, pose_at
    sdir = project / "scene"
    assets, world = [], []
    for name, e in scene["entities"].items():
        vis = e["geometry"]["visual"]
        od = adir / name
        od.mkdir(parents=True, exist_ok=True)
        objf = f"{name}_vis.obj"
        _copy(sdir / vis["obj_uri"], od / objf)
        mat = e.get("material") or {}
        tex = mat.get("base_color_texture")
        if tex and not mat.get("tile_m"):
            texf = f"{name}_{Path(tex).stem}.png"
            _copy(sdir / tex, od / texf)
            assets += [f'<texture name="{name}_tex" type="2d" file="assets/{name}/{texf}"/>',
                       f'<material name="{name}_mat" texture="{name}_tex"' +
                       (' emission="1"' if mat.get("shader") == "emission" else "") + "/>"]
        else:
            c = mat.get("base_color", [0.7, 0.7, 0.7])
            assets.append(f'<material name="{name}_mat" rgba="{c[0]} {c[1]} {c[2]} 1"/>')
        assets.append(f'<mesh name="{name}_vis" file="assets/{name}/{objf}"/>')
        geoms = [f'<geom name="{name}_vis" type="mesh" mesh="{name}_vis" material="{name}_mat" contype="0" '
                 f'conaffinity="0" group="1" mass="0"/>']
        ph = e.get("physics") or {}
        fr = phys(ph.get("friction"), 0.8)
        cols = (e["geometry"].get("collision") or []) if e.get("capability") != "visual_only" else []
        mass = phys(ph.get("mass_kg"), 0.1)
        vols = []
        for c in cols:
            if c["type"] == "mesh":
                import trimesh
                vols.append(abs(trimesh.load(sdir / c["uri"], force="mesh").volume))
            else:
                vols.append(float(np.prod(c["size"])))
        dens = mass / max(sum(vols), 1e-7)
        extra = f' solref="{" ".join(map(str, ph.get("solref") or [0.004, 1]))}"'
        if ph.get("solimp"):
            extra += f' solimp="{" ".join(map(str, ph["solimp"]))}"'
        for k, c in enumerate(cols):
            T = np.asarray(c.get("T", np.eye(4)), float)
            if c["type"] == "mesh":
                f = f"{name}_c{k}.obj"
                _copy(sdir / c["uri"], od / f)
                assets.append(f'<mesh name="{name}_c{k}" file="assets/{name}/{f}"/>')
                g = f'type="mesh" mesh="{name}_c{k}"'
            else:
                g = f'type="box" size="{c["size"][0] / 2} {c["size"][1] / 2} {c["size"][2] / 2}"'
            geoms.append(f'<geom name="{name}_c{k}" {g} pos="{_p(T)}" quat="{_q(T)}" density="{dens:.3f}" '
                         f'friction="{fr} {phys(ph.get("torsional_friction"), 0.005)} 0.0001" group="3" '
                         f'rgba="0.8 0.3 0.3 0.4" condim="{ph.get("condim", 4)}"{extra}/>')
        T = pose_at(e, None)
        if e.get("capability") == "rigid":
            world.append(f'<body name="{name}" pos="{_p(T)}" quat="{_q(T)}"><freejoint name="{name}_free"/>' + "".join(geoms) + "</body>")
        else:
            world.append(f'<body name="{name}" pos="{_p(T)}" quat="{_q(T)}">' + "".join(geoms) + "</body>")
    return assets, world


def build_xml(project: Path, scene: dict, robots: bool = True, timestep: float | None = None,
              calib_size: tuple | None = None, mdir: Path | None = None) -> tuple[str, dict]:
    """(xml, info) for the scene; info: robots {rid: {joints, actuators}}, entities, rigid. mdir: folder the xml
    will be written to (its assets/ next to it; default <project>/mujoco)."""
    from ..robot import mjcf
    from ..robot.render import add_calib_camera
    from ..scene_ir import camera_poses, robot_bundle
    sdir = project / "scene"
    mdir = Path(mdir) if mdir else project / "mujoco"
    adir = mdir / "assets"
    adir.mkdir(parents=True, exist_ok=True)
    assets, world = entity_xml(project, scene, adir)
    acts, eqs, excl = [], [], []
    info = {"robots": {}, "entities": list(scene["entities"]),
            "rigid": [n for n, e in scene["entities"].items() if e.get("capability") == "rigid"]}
    if robots:
        for rid, r in (scene.get("robots") or {}).items():
            b = robot_bundle(sdir, scene, rid)
            rdir = adir / "robots" / rid
            copied = {}

            def ref(p, rdir=rdir, copied=copied, broot=b.root):
                p = Path(p).resolve()
                if str(p) not in copied:
                    try:
                        rel = p.relative_to(broot)
                    except ValueError:
                        rel = Path("external") / p.name
                    dst = rdir / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    _copy(p, dst)
                    copied[str(p)] = str(dst.relative_to(mdir))
                return copied[str(p)]

            prefix = "" if len(scene["robots"]) == 1 else f"{rid}/"
            fr = mjcf.robot_fragment(b, prefix=prefix, mesh_ref=ref, T_world_base=np.asarray(r["T_world_base"], float))
            assets += fr["assets"]
            world.append(fr["body"])
            acts += fr["actuators"]
            eqs += fr["equality"]
            excl += fr["contact"]
            info["robots"][rid] = {"prefix": prefix, "joints": fr["joints"], "bundle": str(b.root)}
    sim = scene.get("simulation") or {}
    dt = timestep or sim.get("timestep", 0.002)
    c = scene["cameras"]["source_cam"]
    K = np.asarray(c["K"])
    Tgl = cv_to_gl(camera_poses(sdir, scene)[0])
    world.append(f'<camera name="source_cam" pos="{_p(Tgl)}" quat="{_q(Tgl)}" '
                 f'fovy="{np.degrees(2 * np.arctan(c["height"] / (2 * K[1, 1]))):.4f}"/>')
    W, H = calib_size or (c["width"], c["height"])
    world_xml = add_calib_camera("\n    ".join(world), W, H)
    lights = '<light name="top" pos="0 0 2.5" dir="0 0 -1" diffuse="0.6 0.6 0.6" castshadow="true"/>' \
             '<light name="fill" pos="1 -1 2" dir="-1 1 -2" diffuse="0.3 0.3 0.3" castshadow="false"/>'
    g = sim.get("gravity", [0, 0, -9.81])
    xml = (f'<mujoco model="{project.name}">\n  {mjcf.compiler_xml()}\n'
           f'  <option timestep="{dt}" gravity="{g[0]} {g[1]} {g[2]}" cone="{sim.get("cone", "elliptic")}" '
           f'impratio="{sim.get("impratio", 10)}" integrator="{sim.get("integrator", "implicitfast")}" '
           f'noslip_iterations="{sim.get("noslip_iterations", 0)}"/>\n'
           f'  <size memory="{sim.get("memory", "256M")}"/>\n'
           f'  <visual><global offwidth="{max(W, c["width"])}" offheight="{max(H, c["height"])}"/>'
           f'<quality shadowsize="4096"/></visual>\n'
           "  <asset>\n    " + "\n    ".join(assets) + "\n  </asset>\n"
           f"  <worldbody>\n    {lights}\n    {world_xml}\n  </worldbody>\n"
           + ("  <actuator>\n    " + "\n    ".join(acts) + "\n  </actuator>\n" if acts else "")
           + ("  <equality>\n    " + "\n    ".join(eqs) + "\n  </equality>\n" if eqs else "")
           + ("  <contact>\n    " + "\n    ".join(excl) + "\n  </contact>\n" if excl else "")
           + "</mujoco>\n")
    return xml, info


def write_scene(project: Path, scene: dict, robots: bool = True, timestep: float | None = None,
                name: str = "scene.xml", mdir: Path | None = None) -> tuple[Path, dict]:
    mdir = Path(mdir) if mdir else project / "mujoco"
    xml, info = build_xml(project, scene, robots, timestep, mdir=mdir)
    p = mdir / name
    p.write_text(xml)
    return p, info
