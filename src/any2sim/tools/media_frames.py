"""Extract frames from any video (extra views, robot cameras, reference clips) into a project folder.

params:
  video: path (absolute or project-relative)
  out_dir: project-relative folder (frames named frame%05d.jpg by source frame index)
  frames: null | [ids] ; step: 1         which frames (default every `step`-th)
  undistort: null | {K: 3x3, coeffs: [k1, k2, p1, p2, k3]}   OpenCV undistortion to the same K
  max_side: null                          downscale so the longer side is at most this
Returns n, size, fps.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common import write_json

ENV = "base"
GPU = False


def run(project: Path, video: str, out_dir: str, frames: list | None = None, step: int = 1,
        undistort: dict | None = None, max_side: int | None = None) -> dict:
    import cv2
    src = Path(video) if Path(video).is_absolute() else project / video
    cap = cv2.VideoCapture(str(src))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    want = set(frames) if frames is not None else set(range(0, n, step))
    out = project / out_dir
    out.mkdir(parents=True, exist_ok=True)
    maps = None
    i, k, size = 0, 0, None
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if i in want:
            if undistort:
                if maps is None:
                    K = np.asarray(undistort["K"], float)
                    D = np.asarray(undistort["coeffs"], float)
                    maps = cv2.initUndistortRectifyMap(K, D, None, K, (im.shape[1], im.shape[0]), cv2.CV_32FC1)
                im = cv2.remap(im, maps[0], maps[1], cv2.INTER_LINEAR)
            if max_side and max(im.shape[:2]) > max_side:
                s = max_side / max(im.shape[:2])
                im = cv2.resize(im, (round(im.shape[1] * s), round(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out / f"frame{i:05d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            size = [im.shape[1], im.shape[0]]
            k += 1
        i += 1
    meta = {"video": str(src), "n_video_frames": i, "n_extracted": k, "fps": fps, "size": size,
            "undistorted": bool(undistort)}
    write_json(meta, out / "meta.json")
    return {**meta, "outputs": [out_dir]}
