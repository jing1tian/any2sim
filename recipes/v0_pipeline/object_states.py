"""Rest intervals of each tracked object (when it is static and not handled), with fit and held-out frames.

Per frame the object's mask is reduced to a world-frame anchor: the ray through its lowest-centre pixel meets the
table plane. Frames where that anchor stays within `still_m` of the interval median, the mask area is stable and
the outline touches no dynamic pixels form rest intervals (>= min_len frames). Each interval gets up to `n_fit`
spread fit frames and `n_heldout` held-out frames (never used for fitting).
Output: work/objects/states.json  {name: [{state, frames: [a, b], fit: [...], heldout: [...], anchor_xy}]}
The agent may edit states.json (e.g. merge, drop or relabel intervals) before register_objects.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common import frame_files, read_json, read_mask, write_json
from ..geom3d import apply, inv, pixel_rays, ray_plane

ENV = "base"
GPU = False


def run(project: Path, still_m: float = 0.012, min_len: int = 6, max_touch: float = 0.02, n_fit: int = 6,
        n_heldout: int = 2, area_tol: float = 0.25, dup_tol: float = 0.5, min_gap: int = 4) -> dict:
    import cv2
    from ..common import read_rgb
    labels = read_json(project / "work" / "masks" / "labels.json")["labels"]
    files = frame_files(project / "work" / "frames")
    # frames that repeat the previous image (static camera, encoder duplicates) carry no new evidence
    small = [cv2.resize(cv2.cvtColor(read_rgb(f), cv2.COLOR_RGB2GRAY), (320, 180)).astype(np.float32) for f in files]
    dup = np.array([False] + [float(np.abs(small[i] - small[i - 1]).mean()) < dup_tol for i in range(1, len(small))])
    vis = read_json(project / "work" / "masks" / "visibility.json")
    cams = read_json(project / "work" / "world" / "cameras.json")
    K = np.asarray(cams["K"])
    Ts = np.asarray(cams["T_world_cam"])
    n = len(frame_files(project / "work" / "frames"))
    out = {}
    for k, name in labels.items():
        anchors = np.full((n, 2), np.nan)
        area = np.array([v["area_px"] for v in vis[name]], float)
        touch = np.array([v["outline_touching_dynamic"] for v in vis[name]], float)
        for i in range(n):
            m = read_mask(project / "work" / "masks" / "objects" / f"frame{i:05d}.png") == int(k)
            ys, xs = np.nonzero(m)
            if len(xs) < 50:
                continue
            y0 = np.percentile(ys, 98)
            sel = ys >= y0 - 2
            uv = np.array([[xs[sel].mean(), y0]])
            T = Ts[i]
            o = T[:3, 3][None]
            d = (T[:3, :3] @ pixel_rays(K, uv).T).T
            anchors[i] = ray_plane(o, d, np.array([0, 0, 1.0]), 0.0)[0, :2]
        ok = np.isfinite(anchors[:, 0]) & (touch <= max_touch)
        states, i = [], 0
        while i < n:
            if not ok[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and ok[j + 1]:
                seg = anchors[i:j + 2]
                med = np.nanmedian(seg, 0)
                a_med = np.median(area[i:j + 2])
                if np.nanmax(np.linalg.norm(seg - med, axis=1)) > still_m or abs(area[j + 1] - a_med) > area_tol * a_med:
                    break
                j += 1
            if j - i + 1 >= min_len:
                states.append((i, j))
            i = j + 1
        # merge intervals separated by short gaps with the same anchor (brief touches / mask flicker)
        merged = []
        for a, b in states:
            if merged and np.linalg.norm(np.nanmedian(anchors[a:b + 1], 0) - np.nanmedian(anchors[merged[-1][0]:merged[-1][1] + 1], 0)) < still_m:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        recs = []
        for s, (a, b) in enumerate(merged):
            fr = [f for f in range(a, b + 1) if ok[f] and not dup[f]]       # distinct images only
            fit = sorted({fr[int(round(x))] for x in np.linspace(0, len(fr) - 1, min(n_fit, len(fr)))}) if fr else []
            # held-out: distinct frames as far as possible from every fit frame (never duplicates of them)
            cand = [f for f in fr if f not in fit and min(abs(f - g) for g in fit) >= min_gap]
            ho = []
            for _ in range(n_heldout):
                if not cand:
                    break
                best = max(cand, key=lambda f: min(abs(f - g) for g in fit + ho))
                ho.append(best)
                cand.remove(best)
            ho = sorted(ho)
            recs.append({"state": f"rest{s}", "frames": [a, b], "n_frames": b - a + 1, "distinct_frames": len(fr),
                         "fit": fit, "heldout": ho,
                         "anchor_xy": np.nanmedian(anchors[a:b + 1], 0).tolist()})
        out[name] = recs
    write_json(out, project / "work" / "objects" / "states.json")
    write_json({"duplicate_frames": np.nonzero(dup)[0].tolist(), "dup_tol": dup_tol}, project / "work" / "objects" / "duplicates.json")
    return {"states": out, "outputs": ["work/objects/states.json"]}
