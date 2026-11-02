"""Propagate object masks through the video (SAM 2.1), keeping identities.

params:
  seeds: [{"obj_id": 1.., "frame": i, "mask": <project-relative png>}, ...]   several seeds per object allowed
  names: {"1": "basket", ...}                (optional, for the overlay)
  out_dir: project-relative folder           (default work/track)
Writes <out_dir>/labels/frame%05d.png (label map: 0 none, k = obj_id; smaller objects drawn on top),
<out_dir>/areas.json (pixels per object per frame), <out_dir>/overlay.mp4 and overview.jpg.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common import contact_sheet, frame_files, overlay_labels, read_mask, read_rgb, write_json, write_rgb, write_video

ENV = "vision"
GPU = True


def run(project: Path, seeds: list[dict], names: dict | None = None, out_dir: str = "work/track") -> dict:
    import cv2
    from ..config import model_path
    from ..models.sam import sam2_propagate
    files = frame_files(project / "work" / "frames")
    out = project / out_dir
    (out / "labels").mkdir(parents=True, exist_ok=True)
    sd = [{"obj_id": int(s["obj_id"]), "frame": int(s["frame"]), "mask": read_mask(project / s["mask"]) > 0} for s in seeds]
    tracks = sam2_propagate(files, sd, str(model_path("sam2_checkpoint")))       # (N, K, H, W)
    n, K = tracks.shape[:2]
    areas = tracks.reshape(n, K, -1).sum(-1)
    order = np.argsort(-areas.mean(0))
    names = {int(k): v for k, v in (names or {}).items()}
    ov = []
    for i in range(n):
        lab = np.zeros(tracks.shape[2:], np.uint8)
        for k in order:
            lab[tracks[i, k]] = k + 1
        cv2.imwrite(str(out / "labels" / f"frame{i:05d}.png"), lab)
        ov.append(overlay_labels(read_rgb(files[i]), lab, names=names or None))
    write_json({str(k + 1): areas[:, k].tolist() for k in range(K)}, out / "areas.json")
    write_video(ov, out / "overlay.mp4", 20.0)
    pick = np.linspace(0, n - 1, 6).round().astype(int)
    write_rgb(out / "overview.jpg", contact_sheet([ov[k] for k in pick], cols=3, width=480, labels=[f"frame {k}" for k in pick]))
    return {"frames": n, "objects": K, "nonempty_frames": {str(k + 1): int((areas[:, k] > 0).sum()) for k in range(K)},
            "outputs": [out_dir]}
