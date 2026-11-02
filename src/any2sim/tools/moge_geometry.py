"""Single-image metric geometry (MoGe-2): point map, depth, normals, validity mask and intrinsics per image.

params:
  frames: [frame ids] | [image paths relative to the project]
  out_dir: project-relative folder (default work/moge)
  fov_x_deg: optional known horizontal FOV (otherwise estimated)
Writes <out_dir>/<stem>.npz with points (H,W,3, OpenCV camera frame, metres), depth (H,W), normal (H,W,3),
mask (H,W), K (3x3 pixels); <out_dir>/<stem>_depth.jpg preview; <out_dir>/summary.json (focal, median depth).
Metric scale is a model estimate (can differ between frames by 10-20 %): treat it as one scale cue.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common import frame_files, read_rgb, write_json, write_rgb

ENV = "geom"
GPU = True


def run(project: Path, frames: list, out_dir: str = "work/moge", fov_x_deg: float | None = None) -> dict:
    import cv2
    from ..config import model_path
    from ..models.geometry import run_moge2
    files = frame_files(project / "work" / "frames")
    out = project / out_dir
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for x in frames:
        f = files[int(x)] if isinstance(x, int) or str(x).isdigit() else project / x
        r = run_moge2(read_rgb(f), str(model_path("moge2_ckpt")), fov_x=fov_x_deg)
        stem = Path(f).stem
        np.savez_compressed(out / f"{stem}.npz", **r)
        d = np.where(r["mask"], r["depth"], np.nan)
        lo, hi = np.nanpercentile(d, [2, 98])
        vis = cv2.applyColorMap((255 * (1 - np.clip((np.nan_to_num(d, nan=hi) - lo) / max(hi - lo, 1e-6), 0, 1))).astype(np.uint8),
                                cv2.COLORMAP_TURBO)[..., ::-1]
        write_rgb(out / f"{stem}_depth.jpg", vis)
        summary[stem] = {"fx": float(r["K"][0, 0]), "fy": float(r["K"][1, 1]), "cx": float(r["K"][0, 2]),
                         "cy": float(r["K"][1, 2]), "median_depth_m": float(np.nanmedian(d))}
    write_json(summary, out / "summary.json")
    return {"images": summary, "outputs": [out_dir]}
