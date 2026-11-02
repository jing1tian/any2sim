"""Geometry of the Scene IR versus the video: every rigid entity, every (or every k-th) frame of each rest state.

Per entity and frame (edge_f: orientation-aware edge F-score at 2 px, the structure measure that dense
patterns cannot game; edge_chamfer_px is reported too): silhouette IoU (rendered silhouette with openings filled, minus other entities in front, vs
the tracked mask; ignored pixels excluded), outline distance mean/P95 (px), interior edge alignment (Canny edges
of the real frame inside the object vs geometric edges of the render: slots, rims, handles; symmetric chamfer px),
and optional depth agreement against metric depth maps (median ratio and abs-rel error on object pixels).
Static entities can be included with `static_targets`.

params:
  targets: {entity: {label_maps: dir, label: k} | {masks: dir}}
  static_targets: {entity: {...}}        optional (e.g. a table mask): IoU of static entities on the same frames
  ignore: [{masks: dir} | {label_maps: dir, labels: [..]}]
  frames: "states" (all frames of every rest state, duplicates of the previous frame skipped) | [ids]
  step: 1                                 use every k-th frame
  depth: null | {dir: project-relative folder of npz with "depth", "mask" named frame%05d.npz}
  out: project-relative json (default evidence/geometry_metrics.json)
Writes the json (per frame + per entity/state summary: mean/min IoU, P95 boundary, edge chamfer) and
<out stem>_worst.jpg (the worst frames, outlines and edges).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common import contact_sheet, frame_files, read_rgb, write_json, write_rgb

ENV = "sam3d"
GPU = True


def run(project: Path, targets: dict, ignore: list | None = None, frames="states", step: int = 1,
        static_targets: dict | None = None, depth: dict | None = None,
        out: str = "evidence/geometry_metrics.json") -> dict:
    import cv2
    import torch
    from ..meshio import (any_mask, boundary_stats, edge_alignment, edge_fscore, frame_mask, hull_front_depth,
                          load_mesh, visible_filled)
    from ..models.raster import Rasterizer
    from ..scene_ir import camera_poses, load, pose_at, state_at
    sc = load(project / "scene")
    poses = camera_poses(project / "scene", sc)
    c = sc["cameras"]["source_cam"]
    K, W, H = np.asarray(c["K"], float), c["width"], c["height"]
    files = frame_files(project / "work" / "frames")
    R = Rasterizer()
    meshes = {n: load_mesh(project / "scene" / e["geometry"]["visual"]["obj_uri"])[:2] for n, e in sc["entities"].items()}
    if frames == "states":
        fs = set()
        for n in targets:
            for st in sc["entities"][n].get("states", {}).values():
                fs |= set(range(st["frames"][0], st["frames"][1] + 1))
        prev, keep = None, []
        for f in sorted(fs):
            g = cv2.resize(cv2.cvtColor(read_rgb(files[f]), cv2.COLOR_RGB2GRAY), (320, 180)).astype(np.float32)
            if prev is None or np.abs(g - prev).mean() >= 0.5 or f - (keep[-1] if keep else -10) > 1:
                keep.append(f)
            prev = g
        frames = keep[::step]
    names = list(sc["entities"])
    per = []
    worst = []
    for f in frames:
        Twc = poses[f] if len(poses) > 1 else poses[0]
        Tcw = np.linalg.inv(Twc)
        Vs, Fs, idx = [], [], []
        for n in names:
            T = pose_at(sc["entities"][n], f)
            if T is None:
                continue
            V, F = meshes[n]
            Vs.append(torch.tensor((V @ T[:3, :3].T + T[:3, 3]) @ Tcw[:3, :3].T + Tcw[:3, 3], dtype=torch.float32, device="cuda"))
            Fs.append(torch.tensor(F, dtype=torch.int64, device="cuda"))
            idx.append(n)
        if not Vs:
            continue
        obj, dep, nrm, tri = R.render_full(Vs, Fs, K, W, H)
        ig = any_mask(project, ignore, f, (H, W), dilate_px=5)
        rgb = None
        for n, spec in list(targets.items()) + list((static_targets or {}).items()):
            if n not in idx:
                continue
            e = sc["entities"][n]
            if e.get("capability") == "rigid" and state_at(e, f) is None:
                continue
            g = frame_mask(project, spec, f)
            if g is None:
                continue
            k = idx.index(n)
            obj_a, _, _, _ = R.render_full([Vs[k]], [Fs[k]], K, W, H)
            movable = [j for j, m in enumerate(idx) if sc["entities"][m].get("capability") == "rigid"]
            pr = visible_filled(obj, dep, k, obj_a, hull_front_depth(R, Vs[k], K, W, H), movable)
            iou = float((pr & g & ~ig).sum() / max(((pr | g) & ~ig).sum(), 1))
            bm, b95 = boundary_stats(pr, g, ig)
            row = {"entity": n, "frame": int(f), "state": state_at(e, f), "iou": round(iou, 4),
                   "boundary_mean_px": bm, "boundary_p95_px": b95}
            if n in targets:
                if rgb is None:
                    rgb = read_rgb(files[f])
                region = cv2.dilate(g.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
                ed, er, en = edge_alignment(rgb, np.where(obj == k, tri, -1), dep, nrm, region, ig)
                row["edge_chamfer_px"] = ed
                ef = edge_fscore(rgb, er, en, dep) if ed is not None else None
                row["edge_f"] = ef["f"] if ef else None
                row["edge_precision"] = ef["precision"] if ef else None
                row["edge_recall"] = ef["recall"] if ef else None
                dp = project / depth["dir"] / f"frame{f:05d}.npz" if depth else None
                if dp is not None and dp.exists():
                    dd = np.load(dp)
                    sel = (obj == k) & g & ~ig & dd["mask"] & (dep > 0)
                    if sel.sum() > 50:
                        r = dd["depth"][sel] / dep[sel]
                        row["depth_ratio_median"] = float(np.median(r))
                        row["depth_absrel_after_ratio"] = float(np.median(np.abs(r / np.median(r) - 1)))
                worst.append((iou, f, n, pr, g, er, en))
            per.append(row)
    summary = {}
    for n in list(targets) + list(static_targets or {}):
        for s in sorted({r["state"] for r in per if r["entity"] == n}, key=str):
            rr = [r for r in per if r["entity"] == n and r["state"] == s]
            eds = [r["edge_chamfer_px"] for r in rr if r.get("edge_chamfer_px") is not None]
            efs = [r["edge_f"] for r in rr if r.get("edge_f") is not None]
            summary[f"{n}:{s}"] = {"frames": len(rr), "iou_mean": float(np.mean([r["iou"] for r in rr])),
                                   "iou_min": float(np.min([r["iou"] for r in rr])),
                                   "boundary_p95_px_median": float(np.median([r["boundary_p95_px"] or 0 for r in rr])),
                                   "boundary_p95_px_max": float(np.max([r["boundary_p95_px"] or 0 for r in rr])),
                                   "edge_chamfer_px_mean": float(np.mean(eds)) if eds else None,
                                   "edge_f_mean": float(np.mean(efs)) if efs else None}
    worst.sort(key=lambda x: x[0])
    tiles, caps = [], []
    for iou, f, n, pr, g, er, en in worst[:4]:
        im = read_rgb(files[f])
        for msk, col in ((g, (255, 0, 255)), (pr, (0, 255, 0))):
            cn, _ = cv2.findContours(msk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(im, cn, -1, col, 2)
        ys, xs = np.nonzero(g | pr)
        y0, y1, x0, x1 = max(ys.min() - 30, 0), min(ys.max() + 30, H), max(xs.min() - 30, 0), min(xs.max() + 30, W)
        ec = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
        if er is not None:
            ec[er[y0:y1, x0:x1]] = (255, 0, 255)
            ec[en[y0:y1, x0:x1]] = (0, 255, 0)
        tiles += [im[y0:y1, x0:x1], ec]
        caps += [f"{n} f{f} IoU {iou:.3f}", "edges: real magenta / render green"]
    if tiles:
        write_rgb((project / out).with_name(Path(out).stem + "_worst.jpg"), contact_sheet(tiles, cols=4, width=360, labels=caps))
    write_json({"summary": summary, "per_frame": per, "frames": len(frames)}, project / out)
    return {"summary": summary, "frames": len(frames), "outputs": [out]}
