"""Compile scene/scene.yaml to MuJoCo (mujoco/scene.xml) and run physical checks per object state set.

Checks for each state set (initial, and every other rest state of each object with the others at the state
valid at that time): initial penetration (deepest contact), drift and rotation after `settle_s` of free
simulation, and a source-camera segmentation render compared with the tracked masks (geometry diagnostic).
Outputs: mujoco/{scene.xml, assets/}, mujoco/checks.json, evidence/mujoco_*.jpg
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from ..common import contact_sheet, frame_files, read_json, read_mask, read_rgb, write_json, write_rgb
from ..geom3d import cv_to_gl, mat_to_quat

ENV = "base"
GPU = False


def _q(T):
    return " ".join(f"{x:.8f}" for x in mat_to_quat(np.asarray(T)[:3, :3]))


def _p(T):
    return " ".join(f"{x:.6f}" for x in np.asarray(T)[:3, 3])


def build_xml(project: Path, scene: dict, state_of: dict, cam_T: np.ndarray) -> str:
    sdir = project / "scene"
    mdir = project / "mujoco"
    adir = mdir / "assets"
    adir.mkdir(parents=True, exist_ok=True)
    assets, world = [], []
    ents = scene["entities"]
    # table
    tb = ents["table"]
    shutil.copy(sdir / tb["geometry"]["visual"]["uri"], adir / "table.obj")
    shutil.copy(sdir / tb["materials"]["top"]["base_color_texture"], adir / "table_top.png")
    assets += ['<texture name="table_top" type="2d" file="assets/table_top.png"/>',
               '<material name="table_top" texture="table_top" texuniform="false"/>',
               '<mesh name="table_vis" file="assets/table.obj"/>']
    box = tb["geometry"]["collision"][0]
    fr = tb["physics"]["friction"]["value"]
    world += ['<geom name="table_vis" type="mesh" mesh="table_vis" material="table_top" contype="0" conaffinity="0" group="1"/>',
              f'<geom name="table_col" type="box" size="{box["size"][0] / 2} {box["size"][1] / 2} {box["size"][2] / 2}" '
              f'pos="{" ".join(map(str, box["pos"]))}" euler="0 0 {box["yaw"]}" friction="{fr} 0.005 0.0001" group="3" rgba="0 0 0 0"/>']
    # backdrop (visual only)
    if "backdrop" in ents:
        shutil.copy(sdir / ents["backdrop"]["geometry"]["visual"]["uri"], adir / "backdrop.obj")
        shutil.copy(sdir / ents["backdrop"]["materials"]["backdrop"]["emission_texture"], adir / "plate.png")
        assets += ['<texture name="plate" type="2d" file="assets/plate.png"/>',
                   '<material name="backdrop" texture="plate" emission="1"/>',
                   '<mesh name="backdrop" file="assets/backdrop.obj"/>']
        world += ['<geom name="backdrop" type="mesh" mesh="backdrop" material="backdrop" contype="0" conaffinity="0" group="1"/>']
    # objects
    for name, e in ents.items():
        if e.get("capability") != "rigid":
            continue
        od = adir / name
        od.mkdir(parents=True, exist_ok=True)
        vis = e["geometry"]["visual"]
        objf = Path(vis["obj_uri"]).name
        shutil.copy(sdir / vis["obj_uri"], od / objf)
        tex = None
        if vis.get("texture_uri"):
            tex = Path(vis["texture_uri"]).name
            shutil.copy(sdir / vis["texture_uri"], od / tex)
        mat = ""
        if tex:
            assets += [f'<texture name="{name}_tex" type="2d" file="assets/{name}/{tex}"/>',
                       f'<material name="{name}_mat" texture="{name}_tex"/>']
            mat = f'material="{name}_mat"'
        assets += [f'<mesh name="{name}_vis" file="assets/{name}/{objf}"/>']
        col = e["geometry"]["collision"]
        vol = max(col["volume_collision_m3"], 1e-7)
        dens = e["physics"]["mass_kg"]["value"] / vol
        fr = e["physics"]["friction"]["value"]
        geoms = [f'<geom name="{name}_vis" type="mesh" mesh="{name}_vis" {mat} contype="0" conaffinity="0" group="1" mass="0"/>']
        for k, f in enumerate(col["files"]):
            shutil.copy(sdir / col["dir"] / f, od / f)
            assets.append(f'<mesh name="{name}_c{k}" file="assets/{name}/{f}"/>')
            geoms.append(f'<geom name="{name}_c{k}" type="mesh" mesh="{name}_c{k}" density="{dens:.3f}" friction="{fr} 0.005 0.0001" '
                         f'group="3" rgba="0.8 0.3 0.3 0.4" solref="0.004 1" condim="4"/>')
        T = e["states"][state_of[name]]["T_world_obj"]
        world.append(f'<body name="{name}" pos="{_p(T)}" quat="{_q(T)}"><freejoint name="{name}_free"/>' + "".join(geoms) + "</body>")
    cam = scene["cameras"]["source_cam"]
    K = np.asarray(cam["K"])
    fovy = float(np.degrees(2 * np.arctan(cam["height"] / (2 * K[1, 1]))))
    Tgl = cv_to_gl(cam_T)
    world.append(f'<camera name="source_cam" pos="{_p(Tgl)}" quat="{_q(Tgl)}" fovy="{fovy:.4f}"/>')
    world.append('<light name="top" pos="0 0 2" dir="0 0 -1" diffuse="0.6 0.6 0.6" castshadow="true"/>')
    return f"""<mujoco model="{project.name}">
  <compiler angle="radian" meshdir="." texturedir="."/>
  <option timestep="0.002" cone="elliptic" impratio="10"/>
  <visual><global offwidth="{cam['width']}" offheight="{cam['height']}"/><headlight ambient="0.4 0.4 0.4"/></visual>
  <asset>
    {chr(10).join('    ' + a for a in assets).strip()}
  </asset>
  <worldbody>
    {chr(10).join('    ' + w for w in world).strip()}
  </worldbody>
</mujoco>
"""


def run(project: Path, settle_s: float = 2.0, drift_tol_m: float = 0.01, rot_tol_deg: float = 5.0,
        penetration_tol_m: float = 0.003) -> dict:
    import cv2
    import mujoco
    import yaml
    scene = yaml.safe_load((project / "scene" / "scene.yaml").read_text())
    cams = read_json(project / "work" / "world" / "cameras.json")
    Ts = np.asarray(cams["T_world_cam"])
    ref = scene["cameras"]["source_cam"]["ref_frame"]
    rigid = {n: e for n, e in scene["entities"].items() if e.get("capability") == "rigid"}
    (project / "mujoco").mkdir(exist_ok=True)
    init = {n: e["initial_state"] for n, e in rigid.items()}
    xml = build_xml(project, scene, init, Ts[ref])
    (project / "mujoco" / "scene.xml").write_text(xml)
    model = mujoco.MjModel.from_xml_path(str(project / "mujoco" / "scene.xml"))

    # state sets: the initial one, then each non-initial state with the others at their state at that time
    sets = [("initial", init, ref)]
    for n, e in rigid.items():
        for s, st in e["states"].items():
            if s == init[n]:
                continue
            t = st["frames"][0]
            combo = {}
            for m, f in rigid.items():
                cur = [k for k, v in f["states"].items() if v["frames"][0] <= t <= v["frames"][1]]
                combo[m] = s if m == n else (cur[0] if cur else None)
            sets.append((f"{n}:{s}", combo, int(np.mean(st["frames"]))))
    labels = read_json(project / "work" / "masks" / "labels.json")["labels"]
    name_to_label = {v: int(k) for k, v in labels.items()}
    files = frame_files(project / "work" / "frames")
    checks, tiles, caps = [], [], []
    for set_name, combo, frame in sets:
        data = mujoco.MjData(model)
        for n, s in combo.items():
            jid = model.joint(f"{n}_free").id
            adr = model.jnt_qposadr[jid]
            if s is None:                                       # moving at that time: park it far away
                data.qpos[adr:adr + 3] = [10 + 2 * len(n), 10, 1]
                continue
            T = np.asarray(rigid[n]["states"][s]["T_world_obj"])
            data.qpos[adr:adr + 3] = T[:3, 3]
            data.qpos[adr + 3:adr + 7] = mat_to_quat(T[:3, :3])
        mujoco.mj_forward(model, data)
        pen = {}
        for c in data.contact[:data.ncon]:
            b1, b2 = model.body(model.geom_bodyid[c.geom1]).name, model.body(model.geom_bodyid[c.geom2]).name
            key = "|".join(sorted([b1 or "world", b2 or "world"]))
            pen[key] = min(pen.get(key, 0.0), float(c.dist))
        p0 = {n: data.body(n).xpos.copy() for n, s in combo.items() if s is not None}
        R0 = {n: data.body(n).xmat.reshape(3, 3).copy() for n in p0}
        for _ in range(int(settle_s / model.opt.timestep)):
            mujoco.mj_step(model, data)
        res = {}
        for n in p0:
            dR = R0[n].T @ data.body(n).xmat.reshape(3, 3)
            ang = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
            drift = float(np.linalg.norm(data.body(n).xpos - p0[n]))
            res[n] = {"drift_m": drift, "rotation_deg": ang, "stable": drift < drift_tol_m and ang < rot_tol_deg}
        worst = min(pen.values()) if pen else 0.0
        # source-camera segmentation render at a frame of this set (geometry diagnostic)
        data2 = mujoco.MjData(model)
        data2.qpos[:] = data.qpos
        for n, s in combo.items():
            if s is None:
                continue
            adr = model.jnt_qposadr[model.joint(f"{n}_free").id]
            T = np.asarray(rigid[n]["states"][s]["T_world_obj"])
            data2.qpos[adr:adr + 3] = T[:3, 3]
            data2.qpos[adr + 3:adr + 7] = mat_to_quat(T[:3, :3])
        cam_id = model.camera("source_cam").id
        Tgl = cv_to_gl(Ts[frame])
        model.cam_pos[cam_id] = Tgl[:3, 3]
        model.cam_quat[cam_id] = mat_to_quat(Tgl[:3, :3])
        mujoco.mj_forward(model, data2)
        H, W = scene["cameras"]["source_cam"]["height"], scene["cameras"]["source_cam"]["width"]
        with mujoco.Renderer(model, H, W) as r:
            r.update_scene(data2, camera="source_cam")
            rgb = r.render()
            r.enable_segmentation_rendering()
            r.update_scene(data2, camera="source_cam")
            seg = r.render()[..., 0]
        ious = {}
        lab = read_mask(project / "work" / "masks" / "objects" / f"frame{frame:05d}.png")
        dyn = read_mask(project / "work" / "masks" / "dynamic" / f"frame{frame:05d}.png") > 0
        for n, s in combo.items():
            if s is None:
                continue
            gids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == model.body(n).id and model.geom_group[g] == 1]
            others = [g for g in range(model.ngeom) if model.geom_group[g] == 1 and model.geom_bodyid[g] not in (0, model.body(n).id)]
            m_sim = np.isin(seg, gids).astype(np.uint8)          # openings filled, other objects in front removed
            cn, _ = cv2.findContours(m_sim, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            m_sim = (cv2.drawContours(np.zeros_like(m_sim), cn, -1, 1, -1) > 0) & ~np.isin(seg, others)
            m_src = lab == name_to_label[n]
            v = ~dyn
            ious[n] = float((m_sim & m_src & v).sum() / max(((m_sim | m_src) & v).sum(), 1))
        src = read_rgb(files[frame])
        blend = (0.5 * src + 0.5 * rgb).astype(np.uint8)
        tiles += [src, rgb, blend]
        caps += [f"real f{frame} ({set_name})", "mujoco", "blend"]
        checks.append({"set": set_name, "states": combo, "frame": frame, "deepest_contact_m": worst,
                       "penetration_ok": worst > -penetration_tol_m, "contacts": pen, "settle": res,
                       "source_view_iou": ious})
    write_rgb(project / "evidence" / "mujoco_source_view.jpg", contact_sheet(tiles, cols=3, width=480, labels=caps))
    ok = all(c["penetration_ok"] and all(v["stable"] for v in c["settle"].values()) for c in checks)
    out = {"model": {"nbody": model.nbody, "ngeom": model.ngeom, "nq": model.nq}, "checks": checks, "all_pass": ok,
           "mujoco_version": mujoco.__version__}
    write_json(out, project / "mujoco" / "checks.json")
    return {**out, "outputs": ["mujoco", "evidence/mujoco_source_view.jpg"]}
