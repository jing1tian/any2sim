"""Pi3X (multi-frame cameras + metric-ish point maps) and MoGe-2 (single-image metric point map) adapters.
Runs in the `geom` env."""
from __future__ import annotations

import sys

import numpy as np


def pi3_size(W: int, H: int, pixel_limit: int = 255000, patch: int = 14) -> tuple[int, int]:
    """Largest (w, h), multiples of the patch size, with w*h <= pixel_limit and the source aspect ratio."""
    s = (pixel_limit / (W * H)) ** 0.5
    w, h = int(W * s // patch) * patch, int(H * s // patch) * patch
    while w * h > pixel_limit:
        w -= patch
    return w, h


def run_pi3x(images: list[np.ndarray], repo: str, ckpt: str, pixel_limit: int = 255000) -> dict:
    """images: RGB uint8 frames of one size. Returns local points (N,h,w,3) in each camera (OpenCV frame, metric
    scale from Pi3X's metric head), c2w poses (N,4,4) in the first frame's camera frame, confidence (N,h,w),
    intrinsics (3,3) at (h,w), and the metric factor."""
    import cv2
    import torch
    sys.path.insert(0, repo)
    from pi3.models.pi3x import Pi3X
    from pi3.utils.geometry import recover_intrinsic_from_rays_d
    H, W = images[0].shape[:2]
    w, h = pi3_size(W, H, pixel_limit)
    x = np.stack([cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA) for im in images]).astype(np.float32) / 255
    dev = "cuda"
    imgs = torch.from_numpy(x).permute(0, 3, 1, 2)[None].to(dev)
    from safetensors.torch import load_file
    model = Pi3X(use_multimodal=False).eval()
    missing, unexpected = model.load_state_dict(load_file(ckpt), strict=False)
    if missing:
        raise ValueError(f"Pi3X checkpoint lacks {len(missing)} weights, e.g. {missing[:3]}")
    model = model.to(dev)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        res = model(imgs=imgs)
    lp = res["local_points"].float()
    rays = torch.nn.functional.normalize(lp, dim=-1)
    K = recover_intrinsic_from_rays_d(rays, force_center_principal_point=True)[0].cpu().numpy()  # (N,3,3)
    poses = res["camera_poses"][0].float().cpu().numpy()
    T0inv = np.linalg.inv(poses[0])
    poses = np.einsum("ij,njk->nik", T0inv, poses)              # first frame = identity
    return {"local_points": lp[0].cpu().numpy(), "c2w": poses,
            "conf": torch.sigmoid(res["conf"][0, ..., 0].float()).cpu().numpy(),
            "K": np.median(K, axis=0), "K_all": K, "size": (w, h),
            "metric": float(res["metric"].reshape(-1)[0]) if res.get("metric") is not None else None}


def run_moge2(image: np.ndarray, ckpt: str, fov_x: float | None = None) -> dict:
    """Metric point map (H,W,3, OpenCV camera frame), depth, mask, normals and pixel intrinsics of one image."""
    import torch
    from moge.model.v2 import MoGeModel
    dev = "cuda"
    model = MoGeModel.from_pretrained(ckpt).to(dev).eval()
    t = torch.from_numpy(image.astype(np.float32) / 255).permute(2, 0, 1).to(dev)
    with torch.no_grad():
        out = model.infer(t, fov_x=fov_x)
    H, W = image.shape[:2]
    K = out["intrinsics"].float().cpu().numpy()
    K = K * np.array([[W], [H], [1]], np.float32)               # normalized -> pixels
    r = {"points": out["points"].float().cpu().numpy(), "depth": out["depth"].float().cpu().numpy(),
         "mask": out["mask"].cpu().numpy().astype(bool), "K": K}
    if "normal" in out:
        r["normal"] = out["normal"].float().cpu().numpy()
    return r
