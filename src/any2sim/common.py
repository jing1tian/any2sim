"""Small dependency-light helpers shared by tools in every worker env (numpy + opencv + stdlib only)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def write_json(obj, path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=_default))
    return str(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def sha256_file(path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return "sha256:" + h.hexdigest()


def digest(path) -> str:
    """Content hash of a file, or of a folder's (relative path, size, file hash) listing."""
    p = Path(path)
    if p.is_file():
        return sha256_file(p)
    h = hashlib.sha256()
    for f in sorted(x for x in p.rglob("*") if x.is_file()):
        h.update(f"{f.relative_to(p)}:{f.stat().st_size}:{sha256_file(f)}\n".encode())
    return "sha256:" + h.hexdigest()


def frame_files(frames_dir) -> list[Path]:
    d = Path(frames_dir)
    return sorted(list(d.glob("frame*.jpg")) + list(d.glob("frame*.png")))


def read_rgb(path) -> np.ndarray:
    import cv2
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def write_rgb(path, rgb, quality=95) -> str:
    import cv2
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, quality] if path.suffix.lower() == ".jpg" else [])
    return str(path)


def read_mask(path) -> np.ndarray:
    import cv2
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(path)
    return m if m.ndim == 2 else m[..., 0]


def write_mask(path, m) -> str:
    import cv2
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = np.asarray(m)
    cv2.imwrite(str(path), (m.astype(np.uint8) * 255) if m.dtype == bool else m.astype(np.uint8))
    return str(path)


PALETTE = np.array([[0, 0, 0], [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
                    [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212]], np.uint8)


def overlay_labels(rgb, labels, alpha=0.5, names=None) -> np.ndarray:
    """Label map coloured over an image, with outlines and optional names at the label centroids."""
    import cv2
    out = rgb.copy()
    for lab in np.unique(labels):
        if lab == 0:
            continue
        m = labels == lab
        c = PALETTE[int(lab) % len(PALETTE)]
        out[m] = (out[m] * (1 - alpha) + c * alpha).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, tuple(int(v) for v in c), 2)
        if names and int(lab) in names:
            ys, xs = np.nonzero(m)
            cv2.putText(out, str(names[int(lab)]), (int(xs.mean()), int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def write_video(frames, path, fps=20.0, bitrate_kbps=None) -> str:
    import imageio.v2 as imageio
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {"ffmpeg_params": ["-b:v", f"{max(200, int(bitrate_kbps))}k"]} if bitrate_kbps else {"quality": 8}
    with imageio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=8, **encoding) as w:
        for f in frames:
            w.append_data(np.asarray(f, np.uint8))
    return str(path)


def contact_sheet(images, cols=3, width=480, labels=None) -> np.ndarray:
    """Images (RGB) resized to one width, tiled in rows, optional captions."""
    import cv2
    tiles = []
    for k, im in enumerate(images):
        h = int(round(im.shape[0] * width / im.shape[1]))
        t = cv2.resize(np.asarray(im, np.uint8), (width, h), interpolation=cv2.INTER_AREA)
        if t.ndim == 2:
            t = np.stack([t] * 3, -1)
        if labels:
            cv2.rectangle(t, (0, 0), (width, 24), (0, 0, 0), -1)
            cv2.putText(t, str(labels[k]), (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    H = max(t.shape[0] for t in tiles)
    tiles = [np.pad(t, ((0, H - t.shape[0]), (0, 0), (0, 0))) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
