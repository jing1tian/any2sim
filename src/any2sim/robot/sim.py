"""MuJoCo model of a bundle for kinematic rendering: compile with a calibrated camera, set joint values by name."""
from __future__ import annotations

import numpy as np


def kinematic_model(bundle, width: int, height: int, T_world_base=None, extra_world: str = ""):
    import mujoco
    from . import mjcf
    from .render import add_calib_camera
    fr = mjcf.robot_fragment(bundle, T_world_base=T_world_base)
    xml = (f'<mujoco>{mjcf.compiler_xml()}<option timestep="0.002"/>'
           f'<visual><global offwidth="{width}" offheight="{height}"/></visual>'
           f'<asset>{"".join(fr["assets"])}</asset><worldbody><light pos="0 0 3" directional="true"/>'
           f'<light pos="1 -1 2" dir="-1 1 -2"/>{add_calib_camera(fr["body"] + extra_world, width, height)}</worldbody>'
           + (f'<equality>{"".join(fr["equality"])}</equality>' if fr["equality"] else "") + "</mujoco>")
    m = mujoco.MjModel.from_xml_string(xml)
    return m, mujoco.MjData(m)


def set_q(model, data, q: dict, forward=True):
    import mujoco
    for jn, v in q.items():
        try:
            data.qpos[model.jnt_qposadr[model.joint(jn).id]] = v
        except KeyError:
            pass
    if forward:
        mujoco.mj_kinematics(model, data)


def link_pose(model, data, link: str) -> np.ndarray:
    T = np.eye(4)
    b = model.body(link)
    T[:3, :3] = data.xmat[b.id].reshape(3, 3)
    T[:3, 3] = data.xpos[b.id]
    return T
