"""Render a robot bundle at logged joint values through a calibrated camera and overlay it on real images: the
visual check of kinematics, joint conventions, hand mounting and visuals (side by side, blend, outline).

params:
  bundle: "robot"
  camera: name in robot.yaml cameras | {link, T_link_cam, K, width, height}
  images: [project-relative paths] | {dir, rows: "frame" | [..]}     (frame%05d names -> row = frame index)
  log: path, joints: [{column, slice?, joints}], q_fixed: {joint: value}
  q_override: null | {row: {joint: value}}       (e.g. try a different sign/mapping)
  mode: "side_by_side" | "blend" | "outline"
  crop: null | [x0, y0, x1, y1]                  region of interest in full-resolution pixels
  out: project-relative jpg (default <bundle>/checks/overlay_<camera>.jpg)
  video: null | project-relative mp4             all images as a video
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common import contact_sheet, read_rgb, write_rgb, write_video

ENV = "base"
GPU = False


def run(project: Path, camera, images, log: str, joints: list, bundle: str = "robot", q_fixed: dict | None = None,
        q_override: dict | None = None, mode: str = "side_by_side", crop: list | None = None, out: str | None = None,
        video: str | None = None, max_images: int = 12) -> dict:
    import cv2
    from ..robot.bundle import Bundle
    from ..robot import datalog as DL
    from ..robot.render import CalibRenderer
    from ..robot.sim import kinematic_model, link_pose, set_q
    broot = Path(bundle) if Path(bundle).is_absolute() else project / bundle
    b = Bundle(broot)
    cam = b.profile["cameras"][camera] if isinstance(camera, str) else camera
    cname = camera if isinstance(camera, str) else cam.get("name", "cam")
    W, H = cam["width"], cam["height"]
    K = np.asarray(cam["K"], float)
    if isinstance(images, dict):
        files = sorted((project / images["dir"]).glob("frame*.jpg"))
        rows = [int(p.stem[5:]) for p in files] if images.get("rows", "frame") == "frame" else images["rows"]
    else:
        files = [project / p for p in images]
        rows = [int(Path(p).stem[5:]) for p in images]
    tab = DL.load_table(Path(log) if Path(log).is_absolute() else project / log)
    qs = DL.joint_series(tab, joints)
    m, d = kinematic_model(b, W, H)
    ren = CalibRenderer(m, W, H, groups=(1,))
    frames_out, tiles, labels = [], [], []
    sel = np.linspace(0, len(files) - 1, min(max_images, len(files))).astype(int) if not video else range(len(files))
    for i in sel:
        f, row = files[i], rows[i]
        q = dict(b.init_q())
        q.update(q_fixed or {})
        q.update({j: float(v[row]) for j, v in qs.items()})
        q.update((q_override or {}).get(str(row), {}))
        set_q(m, d, q)
        T = link_pose(m, d, cam["link"]) @ np.asarray(cam["T_link_cam"], float)
        rgb = ren.rgb(d, K, T)
        seg = ren.segment(d, K, T) >= 0
        real = read_rgb(f)
        if real.shape[:2] != (H, W):
            real = cv2.resize(real, (W, H))
        if mode == "blend":
            img = np.where(seg[..., None], (0.5 * rgb + 0.5 * real).astype(np.uint8), real)
            pair = [img]
        elif mode == "outline":
            img = real.copy()
            cn, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cn, -1, (0, 255, 0), 2)
            pair = [img]
        else:
            ren_img = np.where(seg[..., None], rgb, (real * 0.35).astype(np.uint8))
            pair = [real, ren_img]
        if crop:
            x0, y0, x1, y1 = crop
            pair = [p[y0:y1, x0:x1] for p in pair]
        if video:
            frames_out.append(np.hstack(pair))
        else:
            tiles += pair
            labels += [f"row {row} real", f"row {row} render"] if len(pair) == 2 else [f"row {row}"]
    ren.close()
    outs = []
    if video:
        write_video(frames_out, project / video, fps=10)
        outs.append(video)
    else:
        o = out or str((broot / "checks" / f"overlay_{cname}.jpg").relative_to(project))
        write_rgb(project / o, contact_sheet(tiles, cols=4 if mode == "side_by_side" else 3, width=480, labels=labels))
        outs.append(o)
    return {"images": len(sel), "outputs": outs}
