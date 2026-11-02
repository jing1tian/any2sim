"""Multi-frame geometry (Pi3X): camera poses, intrinsics, per-frame point maps and confidences, jointly.

params:
  frames: [frame ids]                 the first one is the reference (its camera = identity)
  out_dir: project-relative folder (default work/pi3x)
  pixel_limit: 255000                 Pi3X input size (aspect kept, multiples of 14)
Writes <out_dir>/pi3x.npz: frame_ids, c2w (N,4,4 camera -> reference camera, OpenCV; translations in Pi3X metric
units), local_points (N,h,w,3 float16), conf (N,h,w), K_full (3x3 at video resolution), K_all_lowres, size (w,h),
metric (Pi3X metric factor). Pi3X's metric scale is approximate: compare it with other cues before trusting it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common import frame_files, read_rgb, write_json

ENV = "geom"
GPU = True


def run(project: Path, frames: list[int], out_dir: str = "work/pi3x", pixel_limit: int = 255000) -> dict:
    from ..config import model_path
    from ..geom3d import rot_angle_deg
    from ..models.geometry import run_pi3x
    files = frame_files(project / "work" / "frames")
    out = project / out_dir
    out.mkdir(parents=True, exist_ok=True)
    imgs = [read_rgb(files[int(i)]) for i in frames]
    H, W = imgs[0].shape[:2]
    pi = run_pi3x(imgs, str(model_path("pi3_repo")), str(model_path("pi3x_ckpt")), pixel_limit)
    w, h = pi["size"]
    K = pi["K"].copy()
    K[0] *= W / w
    K[1] *= H / h
    np.savez_compressed(out / "pi3x.npz", frame_ids=np.asarray(frames), c2w=pi["c2w"],
                        local_points=pi["local_points"].astype(np.float16), conf=pi["conf"].astype(np.float16),
                        K_full=K, K_all_lowres=pi["K_all"], size=np.asarray([w, h]), metric=np.float32(pi["metric"] or np.nan))
    rot = [rot_angle_deg(T[:3, :3]) for T in pi["c2w"]]
    tr = [float(np.linalg.norm(T[:3, 3])) for T in pi["c2w"]]
    s = {"frames": len(frames), "K_full": K.tolist(), "metric_factor": pi["metric"],
         "max_rotation_deg": max(rot), "max_translation_units": max(tr)}
    write_json(s, out / "summary.json")
    return {**s, "outputs": [out_dir]}
