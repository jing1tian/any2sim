"""Text-prompted instance segmentation (SAM 3) on video frames or any images (e.g. renders, to look for artifacts).

params:
  images: "all" | [frame ids] | [image paths relative to the project]   (default "all" video frames)
  prompts: [text, ...]                       e.g. ["hand", "glove", "basket"]
  out_dir: project-relative folder           (default work/sam3/<first prompt>)
  min_score: 0.3
  save_union: true    -> <out_dir>/union/<image stem>.png  (all prompts, all instances)
  save_instances: true -> <out_dir>/instances/<prompt>/<image stem>_<k>.png (best score first)
Writes <out_dir>/meta.json: per image, per prompt: [{mask, score, box [x0,y0,x1,y1], area_px}].
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..common import frame_files, read_rgb, write_json, write_mask

ENV = "vision"
GPU = True


def _slug(p):
    return re.sub(r"[^a-z0-9]+", "_", p.lower()).strip("_") or "prompt"


def _images(project, images):
    files = frame_files(project / "work" / "frames")
    if images in (None, "all"):
        return files
    out = []
    for x in images:
        out.append(files[int(x)] if isinstance(x, int) or str(x).isdigit() else project / x)
    return out


def run(project: Path, prompts: list[str], images="all", out_dir: str | None = None, min_score: float = 0.3,
        save_union: bool = True, save_instances: bool = True) -> dict:
    from ..models.sam import Sam3
    out = project / (out_dir or f"work/sam3/{_slug(prompts[0])}")
    out.mkdir(parents=True, exist_ok=True)
    sam = Sam3(min_score=min_score)
    meta, counts = {}, {p: 0 for p in prompts}
    for f in _images(project, images):
        rgb = read_rgb(f)
        r = sam.segment(rgb, prompts)
        stem = Path(f).stem
        union = np.zeros(rgb.shape[:2], bool)
        meta[stem] = {}
        for p in prompts:
            rows = []
            for k, inst in enumerate(x for x in r.get(p, []) if x["score"] >= min_score):
                union |= inst["mask"]
                row = {"score": round(inst["score"], 4), "box": inst["box"], "area_px": int(inst["mask"].sum())}
                if save_instances:
                    mp = out / "instances" / _slug(p) / f"{stem}_{k}.png"
                    write_mask(mp, inst["mask"])
                    row["mask"] = str(mp.relative_to(project))
                rows.append(row)
            counts[p] += len(rows)
            meta[stem][p] = rows
        if save_union:
            write_mask(out / "union" / f"{stem}.png", union)
    write_json({"prompts": prompts, "min_score": min_score, "model": sam.identity, "images": meta}, out / "meta.json")
    return {"images": len(meta), "instances_per_prompt": counts, "out_dir": str(out.relative_to(project)),
            "outputs": [str(out.relative_to(project))]}
