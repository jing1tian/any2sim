"""Fit a robot-mounted camera (head, wrist, external-but-fixed-to-a-link) from images of the robot, robot masks and
the joint values at those images: the camera pose relative to a link (and optionally the focal length) that makes
the rendered robot silhouette match the masks. Also the evidence that the kinematics, mounting and visuals agree
with the real robot: per-frame IoU and outline overlays.

params:
  bundle: "robot"
  camera: {name, link, width, height, K: 3x3 | null, fit_focal: false, fit_principal: false}
          (K null: f = 0.8 * width, principal at the centre, focal fitted)
  frames: [{image, mask, row}]  project-relative images (undistorted) and binary robot masks; row = log row
          or {images: dir, masks: dir, rows: "frame" | [..]}   (frame%05d names; row = frame index)
  log: path, joints: [{column, slice?, joints}]       joint values per row (datalog.joint_series)
  q_fixed: {joint: value}                            joints not in the log
  init: {eye: [x,y,z] | [[..]..], target: [x,y,z], up: [0,0,1], roll_steps: 1} in the link frame
        | null (search around the robot); several eyes and roll_steps > 1 give several starts
  bodies: null | [link globs]                        rendered links (default: all)
  scale: 0.25                                        render/mask resolution factor for fitting
  min_depth_m: 0.02                                  penalise poses that see robot geometry closer than this
                                                     (a camera inside the robot can still match a silhouette)
  iters: 1500
  write: false                                       store the result in robot.yaml cameras.<name>
Writes <bundle>/checks/camera_<name>.json and camera_<name>_overlay.jpg (mask outline magenta, render green).
"""
from __future__ import annotations

import fnmatch
from pathlib import Path

import numpy as np

from ..common import contact_sheet, read_mask, read_rgb, write_json, write_rgb

ENV = "base"
GPU = False


def _frames(project, frames):
    if isinstance(frames, list):
        return [(project / f["image"], project / f["mask"], int(f["row"])) for f in frames]
    imgs = sorted((project / frames["images"]).glob("frame*.jpg")) + sorted((project / frames["images"]).glob("frame*.png"))
    out = []
    for i, p in enumerate(imgs):
        mk = project / frames["masks"] / (p.stem + ".png")
        if not mk.exists():
            continue
        row = int(p.stem[5:]) if frames.get("rows", "frame") == "frame" else int(frames["rows"][i])
        out.append((p, mk, row))
    return out


def _T(x):
    from scipy.spatial.transform import Rotation as Rot
    T = np.eye(4)
    T[:3, :3] = Rot.from_rotvec(x[:3]).as_matrix()
    T[:3, 3] = x[3:6]
    return T


def _x(T):
    from scipy.spatial.transform import Rotation as Rot
    return np.r_[Rot.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]]


def run(project: Path, camera: dict, frames, log: str, joints: list, bundle: str = "robot", q_fixed: dict | None = None,
        init: dict | None = None, bodies: list | None = None, scale: float = 0.25, iters: int = 1500,
        write: bool = False, max_frames: int = 24, min_depth_m: float = 0.02) -> dict:
    import cv2
    from scipy.optimize import minimize
    from ..robot.bundle import Bundle
    from ..robot import datalog as DL
    from ..robot.render import CalibRenderer, look_at
    from ..robot.sim import kinematic_model, link_pose, set_q
    broot = Path(bundle) if Path(bundle).is_absolute() else project / bundle
    b = Bundle(broot)
    W, H = int(camera["width"]), int(camera["height"])
    w, h = max(16, round(W * scale)), max(16, round(H * scale))
    K0 = np.asarray(camera["K"], float) if camera.get("K") is not None else \
        np.array([[0.8 * W, 0, (W - 1) / 2], [0, 0.8 * W, (H - 1) / 2], [0, 0, 1.0]])
    fit_f = camera.get("fit_focal", camera.get("K") is None)
    fit_c = camera.get("fit_principal", False)
    tab = DL.load_table(Path(log) if Path(log).is_absolute() else project / log)
    qs = DL.joint_series(tab, joints)
    fr = _frames(project, frames)
    if len(fr) > max_frames:
        fr = [fr[i] for i in np.linspace(0, len(fr) - 1, max_frames).astype(int)]
    m, d = kinematic_model(b, W, H)
    ren = CalibRenderer(m, w, h, groups=(1,))
    keep_body = np.zeros(m.nbody, bool)
    for bi in range(1, m.nbody):
        keep_body[bi] = not bodies or any(fnmatch.fnmatch(m.body(bi).name, p) for p in bodies)
    data = []
    for img, mk, row in fr:
        q = dict(b.init_q())
        q.update(q_fixed or {})
        q.update({j: float(v[row]) for j, v in qs.items()})
        set_q(m, d, q)
        M = cv2.resize((read_mask(mk) > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        data.append({"img": img, "mask": M, "q": q, "row": row,
                     "T_link": link_pose(m, d, camera["link"]), "xpos": d.xpos.copy(), "xmat": d.xmat.copy()})

    def render_sil(fd, T_link_cam, K):
        d.xpos[:], d.xmat[:] = fd["xpos"], fd["xmat"]
        set_q(m, d, fd["q"])
        Ks = K.copy()
        Ks[:2] *= [[w / W], [h / H]]
        Ks[0, 2], Ks[1, 2] = (K[0, 2] + 0.5) * w / W - 0.5, (K[1, 2] + 0.5) * h / H - 0.5
        seg = ren.segment(d, Ks, fd["T_link"] @ T_link_cam)
        return (seg >= 0) & keep_body[np.clip(seg, 0, None)]

    def unpack(x):
        T = _T(x[:6])
        K = K0.copy()
        k = 6
        if fit_f:
            K[0, 0] = K[1, 1] = K0[0, 0] * np.exp(x[k])
            k += 1
        if fit_c:
            K[0, 2], K[1, 2] = K0[0, 2] + x[k] * W, K0[1, 2] + x[k + 1] * H
        return T, K

    def too_close(fd, T_link_cam, K):
        Ks = K.copy()
        Ks[:2] *= [[w / W], [h / H]]
        dep = ren.depth(d, Ks, fd["T_link"] @ T_link_cam)
        return float(((dep > 0) & (dep < min_depth_m)).mean())

    def cost(x, sub=None):
        T, K = unpack(x)
        ious, pen = [], 0.0
        for fd in (data if sub is None else [data[i] for i in sub]):
            r = render_sil(fd, T, K)
            u = (r | fd["mask"]).sum()
            ious.append((r & fd["mask"]).sum() / max(u, 1))
        if min_depth_m:
            pen = 5.0 * too_close(data[0], T, K)
        return 1 - float(np.mean(ious)) + pen, ious

    nx = 6 + int(fit_f) + 2 * int(fit_c)
    # initial pose(s)
    starts = []
    if init:
        eyes = np.atleast_2d(np.asarray(init["eye"], float))
        for e in eyes:
            T0 = look_at(e, init["target"], init.get("up", (0, 0, 1)))
            for k in range(int(init.get("roll_steps", 1))):
                a = 2 * np.pi * k / int(init.get("roll_steps", 1))
                Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
                Tk = T0.copy()
                Tk[:3, :3] = T0[:3, :3] @ Rz
                starts.append(np.r_[_x(Tk), np.zeros(nx - 6)])
    else:
        P = data[0]["xpos"][1:]
        T0inv = np.linalg.inv(data[0]["T_link"])
        c = T0inv[:3, :3] @ P.mean(0) + T0inv[:3, 3]                # robot centre in the link frame
        ext = np.ptp(P, axis=0).max()
        for az in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            for el in (0.3, 0.7, 1.1):
                for dist in (0.6, 1.0, 1.6):
                    e = c + dist * ext * np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
                    starts.append(np.r_[_x(look_at(e, c)), np.zeros(nx - 6)])
    sub = list(range(0, len(data), max(1, len(data) // 6)))
    scored = sorted(((cost(s, sub)[0], i) for i, s in enumerate(starts)))[:4]
    best = None
    for _, i in scored:
        r = minimize(lambda x: cost(x, sub)[0], starts[i], method="Powell",
                     options={"maxfev": iters // 2, "xtol": 1e-4, "ftol": 1e-5})
        if best is None or r.fun < best.fun:
            best = r
    r = minimize(lambda x: cost(x)[0], best.x, method="Powell", options={"maxfev": iters, "xtol": 1e-5, "ftol": 1e-6})
    T, K = unpack(r.x)
    c_final, ious = cost(r.x)
    # overlays at full resolution of the mask grid
    tiles, labels = [], []
    order = np.argsort(ious)
    for i in list(order[:4]) + list(order[-2:]):
        fd = data[i]
        im = cv2.resize(read_rgb(fd["img"]), (w, h))
        sil = render_sil(fd, T, K)
        for msk, col in ((fd["mask"], (255, 0, 255)), (sil, (0, 255, 0))):
            cn, _ = cv2.findContours(msk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(im, cn, -1, col, 1)
        tiles.append(im)
        labels.append(f"row {fd['row']} IoU {ious[i]:.3f}")
    ov = f"checks/camera_{camera['name']}_overlay.jpg"
    write_rgb(broot / ov, contact_sheet(tiles, cols=3, width=480, labels=labels))
    ren.close()
    res = {"camera": camera["name"], "link": camera["link"], "T_link_cam": T.round(6).tolist(), "K": K.round(3).tolist(),
           "width": W, "height": H, "iou_mean": float(np.mean(ious)), "iou_min": float(np.min(ious)),
           "per_frame": [{"row": fd["row"], "iou": float(v)} for fd, v in zip(data, ious)], "overlay": ov,
           "frames": len(data), "fit_focal": fit_f}
    write_json(res, broot / "checks" / f"camera_{camera['name']}.json")
    if write:
        b.profile.setdefault("cameras", {})[camera["name"]] = {
            "link": camera["link"], "T_link_cam": res["T_link_cam"], "K": res["K"], "width": W, "height": H,
            "distortion": camera.get("distortion"),
            "provenance": f"robot_fit_camera: silhouette fit on {len(data)} frames, IoU mean {res['iou_mean']:.3f}"}
        b.save_profile()
    return {k: v for k, v in res.items() if k != "per_frame"} | {"outputs": [str(Path(bundle) / ov)]}
