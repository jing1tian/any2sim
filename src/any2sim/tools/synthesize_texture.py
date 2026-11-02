"""Synthesize a clean, tileable texture from an exemplar (image quilting, Efros & Freeman 2001).

For surfaces whose observed texture is contaminated (shadows, hands, equipment, inpainting, seams): pick a clean
exemplar region, synthesize a texture of any size with the same statistics, and use it as a (tiled) material.

params:
  exemplar: project-relative image (e.g. a rectified surface texture, a frame, an observed.png from project_texture)
  rect: [x0, y0, x1, y1] | null      exemplar region in pixels (default: whole image)
  mask: project-relative png | null   exemplar pixels allowed (patches must lie fully inside)
  size: [w, h] = [1024, 1024]          output size
  patch: 48, overlap: 12               pixels (patch must be smaller than the exemplar region)
  tileable: true                       wrap-around constraints so the output tiles seamlessly
  match_mean: null | [r,g,b]           optional mean colour of the result, sRGB in 0-1 (not 0-255)
  seed: 0
  out: project-relative png (default work/textures/<exemplar stem>_synth.png)
Writes the texture and <out stem>_preview.jpg (exemplar region | result | 2x2 tiling), returns mean/std and an
exemplar cleanliness hint (fraction of outlier pixels: dark/bright blobs vs local median).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common import contact_sheet, read_mask, read_rgb, write_json, write_rgb

ENV = "base"
GPU = False


def _min_cut_vertical(err):
    """Minimum-error path through err (h x ov), top to bottom; returns per-row cut column."""
    h, ov = err.shape
    E = err.copy()
    for i in range(1, h):
        prev = np.pad(E[i - 1], 1, constant_values=np.inf)
        E[i] += np.minimum(np.minimum(prev[:-2], prev[1:-1]), prev[2:])
    cut = np.zeros(h, int)
    cut[-1] = int(np.argmin(E[-1]))
    for i in range(h - 2, -1, -1):
        c = cut[i + 1]
        lo, hi = max(c - 1, 0), min(c + 2, ov)
        cut[i] = lo + int(np.argmin(E[i, lo:hi]))
    return cut


def run(project: Path, exemplar: str, rect: list | None = None, mask: str | None = None, size: list | None = None,
        patch: int = 48, overlap: int = 12, tileable: bool = True, match_mean: list | None = None, seed: int = 0,
        out: str | None = None, candidates: int = 400) -> dict:
    import cv2
    rng = np.random.default_rng(seed)
    ex_full = read_rgb(project / exemplar).astype(np.float32)
    x0, y0, x1, y1 = rect or [0, 0, ex_full.shape[1], ex_full.shape[0]]
    ex = ex_full[y0:y1, x0:x1]
    allow = np.ones(ex.shape[:2], bool)
    if mask:
        allow = (read_mask(project / mask) > 0)[y0:y1, x0:x1]
    eh, ew = ex.shape[:2]
    if eh <= patch or ew <= patch:
        raise ValueError("exemplar region smaller than the patch")
    # candidate patch positions fully inside the allowed region
    ok = cv2.erode(allow.astype(np.uint8), np.ones((patch, patch), np.uint8), anchor=(0, 0)) > 0
    ys, xs = np.nonzero(ok[: eh - patch, : ew - patch])
    if not len(ys):
        raise ValueError("no patch fits inside the exemplar mask")
    W, H = size or [1024, 1024]
    step = patch - overlap
    nx, ny = int(np.ceil((W - overlap) / step)) + 1, int(np.ceil((H - overlap) / step)) + 1
    canvas = np.zeros((ny * step + overlap, nx * step + overlap, 3), np.float32)
    for j in range(ny):
        for i in range(nx):
            yy, xx = j * step, i * step
            pick = rng.choice(len(ys), size=min(candidates, len(ys)), replace=False)
            best, best_e = None, np.inf
            for k in pick:
                p = ex[ys[k]:ys[k] + patch, xs[k]:xs[k] + patch]
                e = 0.0
                if i > 0:
                    e += ((p[:, :overlap] - canvas[yy:yy + patch, xx:xx + overlap]) ** 2).sum()
                if j > 0:
                    e += ((p[:overlap] - canvas[yy:yy + overlap, xx:xx + patch]) ** 2).sum()
                if tileable and i == nx - 1:
                    e += ((p[:, -overlap:] - canvas[yy:yy + patch, :overlap]) ** 2).sum()
                if tileable and j == ny - 1:
                    e += ((p[-overlap:] - canvas[:overlap, xx:xx + patch]) ** 2).sum()
                if e < best_e:
                    best, best_e = p, e
            p = best.copy()
            if i > 0:                                   # vertical seam: min-error boundary cut
                cut = _min_cut_vertical(((p[:, :overlap] - canvas[yy:yy + patch, xx:xx + overlap]) ** 2).sum(-1))
                for r in range(patch):
                    p[r, :cut[r]] = canvas[yy + r, xx:xx + cut[r]]
            if j > 0:                                   # horizontal seam
                cut = _min_cut_vertical(((p[:overlap] - canvas[yy:yy + overlap, xx:xx + patch]) ** 2).sum(-1).T)
                for c in range(patch):
                    p[:cut[c], c] = canvas[yy:yy + cut[c], xx + c]
            canvas[yy:yy + patch, xx:xx + patch] = p
    res = canvas[:H, :W]
    if tileable:                                         # blend the wrap-around band so edges meet
        band = overlap
        a = np.linspace(0, 1, band)[None, :, None]
        res[:, -band:] = (1 - a) * res[:, -band:] + a * res[:, :band]
        a = np.linspace(0, 1, band)[:, None, None]
        res[-band:] = (1 - a) * res[-band:] + a * res[:band]
    if match_mean is not None:
        mm = np.asarray(match_mean, np.float32)
        if mm.shape != (3,) or mm.min() < 0 or mm.max() > 1:
            raise ValueError(f"match_mean must be 3 sRGB values in 0-1, got {match_mean}")
        res = res + (mm * 255 - res.reshape(-1, 3).mean(0))
    res = np.clip(res, 0, 255).astype(np.uint8)
    dst = project / (out or f"work/textures/{Path(exemplar).stem}_synth.png")
    write_rgb(dst, res)
    lum = ex @ np.array([0.299, 0.587, 0.114], np.float32)
    med = cv2.medianBlur(np.clip(lum, 0, 255).astype(np.uint8), 31).astype(np.float32)
    outlier = (np.abs(lum - med) > 25) & allow
    tile = np.vstack([np.hstack([res, res]), np.hstack([res, res])])
    write_rgb(dst.with_name(dst.stem + "_preview.jpg"), contact_sheet(
        [ex.clip(0, 255).astype(np.uint8), res, tile], cols=3, width=400,
        labels=["exemplar region", "synthesized", "2x2 tiling (seams?)"]))
    info = {"out": str(dst.relative_to(project)), "size": [W, H], "mean_srgb": (res.reshape(-1, 3).mean(0) / 255).tolist(),
            "std": (res.reshape(-1, 3).std(0) / 255).tolist(), "exemplar_outlier_fraction": float(outlier.mean())}
    write_json(info, dst.with_suffix(".json"))
    return {**info, "outputs": [info["out"], str(dst.with_name(dst.stem + "_preview.jpg").relative_to(project))]}
