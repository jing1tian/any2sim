"""Camera image model (ISP) applied to renders so they look like the real camera's frames.

    from any2sim import camera_isp as ISP
    out = ISP.apply(render_uint8, isp, rng)        # isp: Scene IR cameras.<cam>.isp (camera_calibrate) or bundle

isp (all optional; missing = identity):
  color_matrix: 3x3                 linear-light colour transform (render -> real), near identity
  vignette: [a1, a2]                gain 1 + a1 r^2 + a2 r^4 (r = distance from the centre / half diagonal), linear
  tone_lut: {x: [...], y: [[r..], [g..], [b..]]}   monotone per-channel curves on sRGB values in [0, 1]
  levels: {in: [p5, p50, p95], out: [p5, p50, p95]}   luminance curve (sRGB 0-1) through matched percentiles,
                                    applied as a per-pixel gain so hue is kept
  saturation: 1.0                   chroma gain around the pixel's luminance
  blur_sigma_px: 0.0                optics/demosaic softness (Gaussian); negative = unsharp mask of that radius
  sharpen_amount: 0.0               (with negative blur_sigma_px) unsharp strength
  noise: {a: [..3], b: [..3], corr_px: 0.6}   per-channel variance a + b * I (sRGB 0-1), spatially correlated
  chroma_shift_px: [dr, db]         lateral chromatic aberration: R / B scaled radially by these px at the corners
  bitrate_kbps: <int>               the real video's bitrate (dataset videos are encoded with it)
Order: linearise -> colour matrix -> vignette -> sRGB -> tone LUT -> levels -> saturation -> chroma shift ->
blur/sharpen -> noise.
"""
from __future__ import annotations

import numpy as np


def for_camera(scene: dict, camera: str, bundle_cameras: dict | None = None) -> dict:
    """Resolve a camera's own calibration; an explicit empty override means identity."""
    overrides = scene.get("render", {}).get("camera_isp") or {}
    if camera in overrides:
        return dict(overrides[camera] or {})
    from .scene_ir import render_head_camera
    config = (scene["cameras"][render_head_camera(scene)] if camera == "head"
              else scene.get("cameras", {}).get(camera, (bundle_cameras or {}).get(camera, {})))
    return dict(config.get("isp") or {})


def srgb_to_lin(x):
    x = np.asarray(x, np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def radius2(H, W):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return ((xx - (W - 1) / 2) ** 2 + (yy - (H - 1) / 2) ** 2) / (((W - 1) / 2) ** 2 + ((H - 1) / 2) ** 2)


def apply_lut(x, lut):
    xs = np.asarray(lut["x"], np.float32)
    out = np.empty_like(x)
    for c in range(3):
        ys = np.asarray(lut["y"][c], np.float32)
        out[..., c] = np.interp(x[..., c], xs, ys)
    return out


def apply(img, isp: dict | None, rng=None, key=None) -> np.ndarray:
    """uint8 RGB render -> uint8 RGB as the real camera would record it."""
    import cv2
    if not isp:
        return img
    x = img.astype(np.float32) / 255.0
    H, W = x.shape[:2]
    if isp.get("color_matrix") is not None or isp.get("vignette"):
        lin = srgb_to_lin(x)
        if isp.get("color_matrix") is not None:
            lin = lin @ np.asarray(isp["color_matrix"], np.float32).T
        if isp.get("vignette"):
            a1, a2 = isp["vignette"]
            r2 = radius2(H, W)
            lin = lin * (1 + a1 * r2 + a2 * r2 * r2)[..., None]
        x = lin_to_srgb(lin)
    if isp.get("tone_lut"):
        x = apply_lut(np.clip(x, 0, 1), isp["tone_lut"])
    if isp.get("levels"):
        xi = np.r_[0.0, isp["levels"]["in"], 1.0]
        xo = np.r_[0.0, isp["levels"]["out"], 1.0]
        Y = np.clip(x @ np.array([0.299, 0.587, 0.114], np.float32), 1e-4, 1)
        Y2 = np.interp(Y, xi, xo)
        x = x * (Y2 / Y)[..., None]
    if isp.get("saturation") and abs(float(isp["saturation"]) - 1) > 1e-3:
        Y = (x @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
        x = Y + float(isp["saturation"]) * (x - Y)
    cs = isp.get("chroma_shift_px")
    if cs and any(abs(v) > 1e-3 for v in cs):
        diag = np.hypot(W / 2, H / 2)
        for c, px in ((0, cs[0]), (2, cs[1])):
            s = 1 + px / diag
            M = np.array([[s, 0, (1 - s) * (W - 1) / 2], [0, s, (1 - s) * (H - 1) / 2]], np.float32)
            x[..., c] = cv2.warpAffine(x[..., c], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    sb = float(isp.get("blur_sigma_px") or 0.0)
    if sb > 0.05:
        x = cv2.GaussianBlur(x, (0, 0), sb)
    elif sb < -0.05:
        bl = cv2.GaussianBlur(x, (0, 0), -sb)
        x = x + float(isp.get("sharpen_amount", 0.5)) * (x - bl)
    nz = isp.get("noise")
    if nz:
        if rng is None:
            import zlib
            rng = np.random.default_rng(0 if key is None else zlib.crc32(str(key).encode()))
        var = np.asarray(nz.get("a", [0, 0, 0]), np.float32) + np.asarray(nz.get("b", [0, 0, 0]), np.float32) * np.clip(x, 0, 1)
        n = rng.standard_normal(x.shape).astype(np.float32)
        cp = float(nz.get("corr_px", 0.0))
        if cp > 0.05:
            n = cv2.GaussianBlur(n, (0, 0), cp)
            n /= max(float(n.std()), 1e-6)
        x = x + n * np.sqrt(np.clip(var, 0, None))
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)
