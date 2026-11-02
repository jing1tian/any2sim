"""Per-frame masks: tracked object instances (SAM 3 seeds + SAM 2.1), dynamic distractors and static surfaces.

Outputs (work/masks/):
  objects/frame%05d.png   label map, 0 = none, k = objects[k-1] (identity kept through the video by SAM 2)
  dynamic/frame%05d.png   255 where any dynamic prompt matched (hands, arms, controllers, cables, ...)
  static/<name>.png       static surfaces (table, backdrop, ...) on the reference frame (SAM 3)
  labels.json             {k: name}, seeds, scores
  visibility.json         per object per frame: area, bbox, fraction of its outline touching dynamic pixels
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common import (contact_sheet, frame_files, overlay_labels, read_rgb, write_json, write_mask, write_rgb,
                      write_video)

ENV = "vision"
GPU = True

DEFAULT_DYNAMIC = ["hand", "glove", "arm", "VR controller", "cable", "wrist tracker", "person"]


def _touch_fraction(m, dyn):
    import cv2
    if not m.any():
        return 0.0
    k = np.ones((5, 5), np.uint8)
    edge = cv2.dilate(m.astype(np.uint8), k) & ~m
    edge = edge.astype(bool)
    return float((edge & dyn).sum() / max(edge.sum(), 1))


def run(project: Path, objects: list[dict], dynamic_prompts: list[str] | None = None,
        static: list[dict] | None = None, ref_frame: int = 0, scan_step: int = 5, min_score: float = 0.3,
        seed_min_score: float = 0.5, dynamic_min_score: float = 0.35) -> dict:
    import cv2
    from ..config import model_path
    from ..models.sam import Sam3, sam2_propagate
    files = frame_files(project / "work" / "frames")
    n = len(files)
    out = project / "work" / "masks"
    for sub in ("objects", "dynamic", "static"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    dyn_prompts = dynamic_prompts or DEFAULT_DYNAMIC
    static = static or [{"name": "table", "prompt": "table"}]
    sam3 = Sam3(min_score=min(min_score, dynamic_min_score))

    # 1. dynamic masks on every frame; object detections on scan frames (+ first/last)
    scan = sorted(set(range(0, n, scan_step)) | {0, n - 1, ref_frame})
    dets = {o["name"]: {} for o in objects}
    dyn = np.zeros((n,) + read_rgb(files[0]).shape[:2], bool)
    for i, f in enumerate(files):
        rgb = read_rgb(f)
        prompts = list(dyn_prompts) + ([o["prompt"] for o in objects] if i in scan else [])
        r = sam3.segment(rgb, prompts)
        for p in dyn_prompts:
            for inst in r.get(p, []):
                if inst["score"] >= dynamic_min_score:
                    dyn[i] |= inst["mask"]
        if i in scan:
            for o in objects:
                inst = [x for x in r.get(o["prompt"], []) if x["score"] >= min_score]
                if inst:
                    dets[o["name"]][i] = inst[0]

    # 2. seeds: best unoccluded detection early and late in the clip (object state can change in between)
    seeds, seed_info = [], {}
    for k, o in enumerate(objects, 1):
        d = dets[o["name"]]
        if not d:
            seed_info[o["name"]] = {"error": "never detected"}
            continue
        def quality(i):
            m = d[i]["mask"]
            occl = _touch_fraction(m, cv2.dilate(dyn[i].astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
            return d[i]["score"] * (1.0 - occl), occl
        chosen = []
        for lo, hi in ((0, n // 4), (3 * n // 4, n)):
            cand = [i for i in d if lo <= i < hi and d[i]["score"] >= seed_min_score]
            if cand:
                best = max(cand, key=lambda i: quality(i)[0])
                if quality(best)[1] < 0.25:
                    chosen.append(best)
        if not chosen:
            chosen = [max(d, key=lambda i: quality(i)[0])]
        for i in chosen:
            seeds.append({"obj_id": k, "frame": int(i), "mask": d[i]["mask"]})
        seed_info[o["name"]] = [{"frame": int(i), "score": round(d[i]["score"], 3),
                                 "outline_touching_dynamic": round(quality(i)[1], 3)} for i in chosen]
    tracks = sam2_propagate(files, seeds, str(model_path("sam2_checkpoint")))
    # dynamic prompts also fire on the tracked objects (a basket reads as "cable", a held pepper as "hand")
    dyn &= ~tracks.any(1)
    for i in range(n):
        write_mask(out / "dynamic" / f"frame{i:05d}.png", dyn[i])

    # 3. label maps (smaller objects drawn last, so they win overlaps) and visibility
    K = len(objects)
    areas = tracks.reshape(n, K, -1).sum(-1)
    order = np.argsort(-areas.mean(0))
    vis = {o["name"]: [] for o in objects}
    overlays = []
    names = {k + 1: o["name"] for k, o in enumerate(objects)}
    for i in range(n):
        lab = np.zeros(tracks.shape[2:], np.uint8)
        for k in order:
            lab[tracks[i, k]] = k + 1
        cv2.imwrite(str(out / "objects" / f"frame{i:05d}.png"), lab)
        for k, o in enumerate(objects):
            m = lab == k + 1
            ys, xs = np.nonzero(m)
            vis[o["name"]].append({
                "area_px": int(m.sum()),
                "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None,
                "outline_touching_dynamic": round(_touch_fraction(m, dyn[i]), 3)})
        both = lab.copy()
        both[dyn[i] & (lab == 0)] = K + 1
        overlays.append(overlay_labels(read_rgb(files[i]), both, names={**names, K + 1: "dynamic"}))

    # 4. static surfaces on the reference frame
    ref = read_rgb(files[ref_frame])
    rs = sam3.segment(ref, [s["prompt"] for s in static])
    static_info = {}
    for s in static:
        inst = [x for x in rs.get(s["prompt"], []) if x["score"] >= min_score]
        m = np.zeros(ref.shape[:2], bool)
        for x in inst[: s.get("max_instances", 1)]:
            m |= x["mask"]
        write_mask(out / "static" / f"{s['name']}.png", m)
        static_info[s["name"]] = {"prompt": s["prompt"], "n_instances": len(inst), "coverage": round(float(m.mean()), 4),
                                  "scores": [round(x["score"], 3) for x in inst[:3]]}

    fps = 20.0
    write_video(overlays, project / "evidence" / "masks_overlay.mp4", fps)
    pick = np.linspace(0, n - 1, 6).round().astype(int)
    write_rgb(project / "evidence" / "masks_overview.jpg",
              contact_sheet([overlays[k] for k in pick], cols=3, width=480, labels=[f"frame {k}" for k in pick]))
    st = np.zeros(ref.shape[:2], np.uint8)
    for j, s in enumerate(static, 1):
        st[cv2.imread(str(out / "static" / f"{s['name']}.png"), 0) > 0] = j
    write_rgb(project / "evidence" / "static_masks.jpg",
              overlay_labels(ref, st, names={j: s["name"] for j, s in enumerate(static, 1)}))
    write_json({"labels": names, "objects": objects, "dynamic_prompts": dyn_prompts, "seeds": seed_info,
                "static": static_info, "ref_frame": ref_frame, "model": sam3.identity}, out / "labels.json")
    write_json(vis, out / "visibility.json")
    return {"labels": names, "seeds": seed_info, "static": static_info,
            "dynamic_mean_coverage": round(float(dyn.mean()), 4),
            "object_nonempty_frames": {o["name"]: int((areas[:, k] > 0).sum()) for k, o in enumerate(objects)},
            "outputs": ["work/masks", "evidence/masks_overlay.mp4", "evidence/masks_overview.jpg",
                        "evidence/static_masks.jpg"]}
