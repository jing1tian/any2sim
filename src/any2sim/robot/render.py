"""MuJoCo offscreen rendering with calibrated (OpenCV) cameras: RGB, depth and per-body segmentation.

A model meant for calibrated rendering contains one world camera named "calib" (add_calib_camera); its pose and
intrinsics are set per call, so one compiled model serves every camera and frame. Needs MUJOCO_GL=egl (set by the
tool runner from configs/any2sim.yaml).
"""
from __future__ import annotations

import numpy as np

from ..geom3d import mat_to_quat


def add_calib_camera(worldbody_xml: str, width: int = 1280, height: int = 720) -> str:
    return worldbody_xml + (f'<camera name="calib" mode="fixed" pos="0 0 1" quat="1 0 0 0" '
                            f'resolution="{width} {height}" focalpixel="{width} {width}" principalpixel="0 0" '
                            f'sensorsize="0.01 0.01"/>')


CV_TO_GL = np.diag([1.0, -1.0, -1.0])


class CalibRenderer:
    """Render a MuJoCo model through camera "calib" with OpenCV intrinsics K and pose T_world_cam."""

    def __init__(self, model, width: int, height: int, groups=(0, 1, 2)):
        import mujoco
        self.mj = mujoco
        self.model = model
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self.r = mujoco.Renderer(model, height, width)
        self.W, self.H = width, height
        self.cid = model.camera("calib").id
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        for g in groups:
            self.opt.geomgroup[g] = 1
        self.opt.sitegroup[:] = 0

    def set_camera(self, data, K, T_world_cam):
        m, c = self.model, self.cid
        K = np.asarray(K, float)
        T = np.asarray(T_world_cam, float)
        m.cam_pos[c] = T[:3, 3]
        m.cam_quat[c] = mat_to_quat(T[:3, :3] @ CV_TO_GL)
        # intrinsic camera: sensor size in "pixels" units so focal/principal are in pixels
        m.cam_resolution[c] = [self.W, self.H]
        m.cam_sensorsize[c] = [self.W, self.H]
        # OpenCV pixel centres are integers; MuJoCo pixel i spans [i, i+1) (sign/offset verified by a projection test)
        m.cam_intrinsic[c] = [K[0, 0], K[1, 1], self.W / 2 - (K[0, 2] + 0.5), self.H / 2 - (K[1, 2] + 0.5)]
        m.cam_fovy[c] = np.degrees(2 * np.arctan(self.H / (2 * K[1, 1])))
        self.mj.mj_camlight(m, data)                       # cam_xpos/xmat from the new cam_pos/quat

    def rgb(self, data, K, T_world_cam) -> np.ndarray:
        self.set_camera(data, K, T_world_cam)
        self.r.disable_segmentation_rendering()
        self.r.disable_depth_rendering()
        self.r.update_scene(data, camera=self.cid, scene_option=self.opt)
        return self.r.render().copy()

    def depth(self, data, K, T_world_cam) -> np.ndarray:
        self.set_camera(data, K, T_world_cam)
        self.r.enable_depth_rendering()
        self.r.update_scene(data, camera=self.cid, scene_option=self.opt)
        d = self.r.render().copy()
        self.r.disable_depth_rendering()
        return d

    def segment(self, data, K, T_world_cam) -> np.ndarray:
        """(H, W) body id per pixel (-1: background)."""
        self.set_camera(data, K, T_world_cam)
        self.r.enable_segmentation_rendering()
        self.r.update_scene(data, camera=self.cid, scene_option=self.opt)
        s = self.r.render().copy()
        self.r.disable_segmentation_rendering()
        gid, typ = s[..., 0], s[..., 1]
        out = np.full(gid.shape, -1, np.int32)
        ok = (gid >= 0) & (typ == int(self.mj.mjtObj.mjOBJ_GEOM))
        out[ok] = self.model.geom_bodyid[gid[ok]]
        return out

    def close(self):
        try:
            self.r.close()
        except Exception:                                   # noqa: BLE001  (EGL teardown noise)
            pass


def body_subtree(model, root_name: str) -> np.ndarray:
    """Ids of a body and all its descendants."""
    rid = model.body(root_name).id
    ids = [b for b in range(model.nbody) if _ancestor(model, b, rid)]
    return np.asarray(ids, int)


def _ancestor(model, b, rid):
    while True:
        if b == rid:
            return True
        if b == 0:
            return False
        b = model.body_parentid[b]


def look_at(eye, target, up=(0, 0, 1)) -> np.ndarray:
    """T_world_cam (OpenCV) looking from eye at target."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, [0, 1, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], 1)
    T[:3, 3] = eye
    return T
