"""Copy a video into the project, extract its frames and write a video-only ObservationBundle."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from ..common import contact_sheet, read_rgb, sha256_file, write_rgb

ENV = "base"
GPU = False


def run(project: Path, video: str, task: str | None = None, frame_step: int = 1, view_id: str = "ego") -> dict:
    import cv2
    import yaml
    src = Path(video)
    dst = project / "inputs" / f"{view_id}{src.suffix.lower()}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dst)
    cap = cv2.VideoCapture(str(dst))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = project / "work" / "frames"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("frame*.jpg"):
        f.unlink()
    src_index, i = [], 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if i % frame_step == 0:
            cv2.imwrite(str(out / f"frame{len(src_index):05d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 97])
            src_index.append(i)
        i += 1
    n_src = i
    ts = np.asarray(src_index, float) / fps if fps else np.arange(len(src_index), dtype=float)
    np.save(project / "work" / "frames" / "timestamps.npy", ts)
    np.save(project / "work" / "frames" / "source_index.npy", np.asarray(src_index))
    obs = {
        "schema_version": "1.0",
        "actor_type": "human",
        "task_hint": task,
        "videos": [{
            "view_id": view_id, "uri": str(dst.relative_to(project)), "sha256": sha256_file(dst),
            "source_uri": str(src), "width": W, "height": H, "fps": fps, "n_source_frames": n_src,
            "frames_dir": "work/frames", "frame_step": frame_step, "n_frames": len(src_index),
            "timestamps_uri": "work/frames/timestamps.npy", "time_base": "video frame index / fps (no device clock)",
            "camera_id": f"{view_id}_camera", "time_offset_s": 0.0, "time_scale": 1.0,
        }],
        "camera_calibration": None,
        "head_camera_trajectory": None,
        "hand_tracks": None,
        "robot_state": None,
        "robot_description": None,
        "measurements": [],
        "withheld": ["camera_calibration", "head_camera_trajectory", "hand_tracks", "robot_state",
                     "other video views"],
        "notes": ("Video-only run: intrinsics, camera poses, metric scale and hand motion must be estimated "
                  "from pixels; scale is a hypothesis with an uncertainty interval."),
    }
    (project / "observation.yaml").write_text(yaml.safe_dump(obs, sort_keys=False, allow_unicode=True))
    files = sorted(out.glob("frame*.jpg"))
    pick = np.linspace(0, len(files) - 1, 6).round().astype(int)
    write_rgb(project / "evidence" / "frames_overview.jpg",
              contact_sheet([read_rgb(files[k]) for k in pick], cols=3, width=480, labels=[f"frame {k}" for k in pick]))
    return {"n_frames": len(files), "fps": fps, "width": W, "height": H, "duration_s": n_src / fps if fps else None,
            "outputs": ["inputs", "work/frames", "observation.yaml", "evidence/frames_overview.jpg"]}
