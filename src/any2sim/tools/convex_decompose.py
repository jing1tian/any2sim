"""Convex decomposition (CoACD) of a mesh into collision hulls, with the visual-to-collision surface distance.

params:
  mesh: project-relative mesh (object frame)
  out_dir: project-relative folder for hull_XXX.obj
  threshold: 0.02 (lower keeps more concavity: cavities, slots), max_hulls: 64, preprocess_resolution: 80
Returns hull files, volumes, and visual->collision distance percentiles (how faithfully collision follows the
visual surface); a cavity that should stay open must be checked by the caller (e.g. probe points with mujoco_check
or trimesh containment).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common import write_json

ENV = "base"
GPU = False


def run(project: Path, mesh: str, out_dir: str, threshold: float = 0.02, max_hulls: int = 64,
        preprocess_resolution: int = 80, seed: int = 0) -> dict:
    import coacd
    import trimesh
    m = trimesh.load(project / mesh, force="mesh")
    m.merge_vertices()
    out = project / out_dir
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("hull_*.obj"):
        f.unlink()
    parts = coacd.run_coacd(coacd.Mesh(np.asarray(m.vertices), np.asarray(m.faces)), threshold=threshold,
                            max_convex_hull=max_hulls, preprocess_mode="on",
                            preprocess_resolution=preprocess_resolution, seed=seed)
    hulls = [trimesh.Trimesh(v, f) for v, f in parts]
    files = []
    for k, h in enumerate(hulls):
        h.export(out / f"hull_{k:03d}.obj")
        files.append(f"hull_{k:03d}.obj")
    samp = m.sample(3000)
    dist = np.min(np.stack([np.abs(trimesh.proximity.signed_distance(h, samp)) for h in hulls]), 0)
    info = {"n_hulls": len(hulls), "files": files, "dir": out_dir, "threshold": threshold,
            "volume_collision_m3": float(sum(h.volume for h in hulls)),
            "volume_visual_m3": float(m.volume) if m.is_volume else None,
            "visual_watertight": bool(m.is_watertight),
            "visual_to_collision_dist_m": {q: float(np.percentile(dist, int(q[1:]))) for q in ("p50", "p90", "p99")}}
    write_json(info, out / "decomposition.json")
    return {**info, "outputs": [out_dir]}
