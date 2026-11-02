"""Textured object meshes with SAM 3D Objects, placed in the metric world frame.

For each object: anchor frame = the frame where it is largest and least touched by hands (or given); SAM 3D gets
the RGB frame, the object mask and the MoGe-2 metric point map of that frame (converted to PyTorch3D camera axes),
so its layout (rotation, translation, scale) is already metric in that camera. The mesh's local axes convention
relative to the layout is resolved by silhouette IoU against the anchor mask (recorded).
Outputs: work/objects/<name>/{sam3d.glb, layout.json, anchor.png, mask.png}, scene/assets/objects/<name>/init_world.glb
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from ..common import frame_files, read_json, read_mask, read_rgb, write_json, write_mask, write_rgb
from ..geom3d import apply

ENV = "sam3d"
GPU = True

P3D_FROM_CV = np.diag([-1.0, -1.0, 1.0])     # PyTorch3D camera (x left, y up, z fwd) <-> OpenCV (x right, y down)
AXES = {                                      # mesh-local -> layout-local rotations (checked, see run())
    "identity": np.eye(3),
    "yup_to_zup": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float),
    "zup_to_yup": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float),
}


def silhouette(K, V_cam, F, H, W):
    import cv2
    z = V_cam[:, 2]
    uv = V_cam[:, :2] / np.maximum(z, 1e-6)[:, None] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    m = np.zeros((H, W), np.uint8)
    tri = np.round(uv[F] * 4).astype(np.int32)                   # 2 bits of sub-pixel precision
    ok = (z[F] > 0).all(1)
    for t in tri[ok]:
        cv2.fillConvexPoly(m, t, 1, lineType=cv2.LINE_8, shift=2)
    return m > 0


def pick_anchor(vis, n_frames, min_area_frac=0.85):
    areas = np.array([v["area_px"] for v in vis], float)
    touch = np.array([v["outline_touching_dynamic"] for v in vis], float)
    big = areas >= min_area_frac * np.percentile(areas, 95)
    score = np.where(big, 1.0 - touch, -1.0) - 1e-4 * np.arange(n_frames)   # earliest among equals
    return int(np.argmax(score))


def run(project: Path, objects: list[str] | None = None, anchors: dict | None = None, seed: int = 42,
        use_moge_pointmap: bool = True, reuse: bool = False) -> dict:
    """reuse=True skips SAM 3D and re-places existing work/objects/<name>/{sam3d.glb, layout.json}."""
    import cv2
    import trimesh
    from scipy.spatial.transform import Rotation
    from ..config import model_path
    labels = read_json(project / "work" / "masks" / "labels.json")["labels"]
    vis = read_json(project / "work" / "masks" / "visibility.json")
    cams = read_json(project / "work" / "world" / "cameras.json")
    K = np.asarray(cams["K"])
    files = frame_files(project / "work" / "frames")
    names = objects or list(labels.values())
    name_to_label = {v: int(k) for k, v in labels.items()}
    raw = {}
    inf = None
    for name in names:
        out = project / "work" / "objects" / name
        out.mkdir(parents=True, exist_ok=True)
        if reuse and (out / "layout.json").exists() and (out / "sam3d.glb").exists():
            lay = read_json(out / "layout.json")
            raw[name] = {"anchor": lay["anchor_frame"], "pm_src": lay["pointmap"], **lay["layout"]}
            continue
        if inf is None:
            repo = model_path("sam3d_repo")
            sys.path.insert(0, str(repo / "notebook"))
            os.environ.setdefault("CONDA_PREFIX", os.environ.get("CUDA_HOME", ""))
            from inference import Inference
            inf = Inference(str(repo / "checkpoints" / "hf" / "pipeline.yaml"), compile=False)
        import torch
        i = int((anchors or {}).get(name, pick_anchor(vis[name], len(files))))
        rgb = read_rgb(files[i])
        mask = read_mask(project / "work" / "masks" / "objects" / f"frame{i:05d}.png") == name_to_label[name]
        write_mask(out / "mask.png", mask)
        pm, pm_src = None, "sam3d internal depth model"
        mf = project / "work" / "geometry" / f"moge_{i:05d}.npz"
        if use_moge_pointmap and mf.exists():
            P = np.load(mf)["points"].astype(np.float32)
            P[~np.isfinite(P).all(-1)] = np.nan
            pm = torch.from_numpy(P @ P3D_FROM_CV.T.astype(np.float32))
            pm_src = f"MoGe-2 metric point map of frame {i}"
        o = inf._pipeline.run(inf.merge_mask_to_rgba(rgb, mask.astype(np.uint8)), None, int(seed), stage1_only=False,
                              with_mesh_postprocess=True, with_texture_baking=True, with_layout_postprocess=False,
                              use_vertex_color=False, pointmap=pm)
        o["glb"].export(str(out / "sam3d.glb"))
        raw[name] = {"anchor": i, "pm_src": pm_src,
                     "quat_wxyz_local_to_p3dcam": np.asarray(o["rotation"].detach().cpu()).reshape(-1).tolist(),
                     "translation_p3dcam": np.asarray(o["translation"].detach().cpu()).reshape(-1).tolist(),
                     "scale": np.asarray(o["scale"].detach().cpu()).reshape(-1).tolist()}

    # mesh-axes convention (from SAM 3D's code, not fitted): postprocessing_utils.to_glb exports
    # V_glb = V_local @ [[1,0,0],[0,0,-1],[0,1,0]] and the layout is a PyTorch3D Transform3d (row vectors:
    # p_cam = s * p_local @ R + t). Every candidate's silhouette IoU is still recorded as a check.
    cand = {}
    meshes = {}
    for name, r in raw.items():
        out = project / "work" / "objects" / name
        i = r["anchor"]
        rgb = read_rgb(files[i])
        H, W = rgb.shape[:2]
        mask = read_mask(out / "mask.png") > 0
        glb = trimesh.load(out / "sam3d.glb")
        mesh = glb.dump(concatenate=True) if isinstance(glb, trimesh.Scene) else glb
        meshes[name] = mesh
        q, t, s = np.asarray(r["quat_wxyz_local_to_p3dcam"]), np.asarray(r["translation_p3dcam"]), np.asarray(r["scale"])
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        for ax_name, A in AXES.items():
            for rowvec in (True, False):
                Vl = V @ A.T
                Vp = (Vl * s) @ R + t if rowvec else (Vl * s) @ R.T + t
                Vc = Vp @ P3D_FROM_CV.T
                sil = silhouette(K, Vc, F, H, W)
                iou = float((sil & mask).sum() / max((sil | mask).sum(), 1))
                cand.setdefault((ax_name, rowvec), {})[name] = (iou, Vc, sil)
    conv = ("yup_to_zup", True)
    best_alt = max(cand, key=lambda c: sum(v[0] for v in cand[c].values()))
    results = {}
    for name, r in raw.items():
        out = project / "work" / "objects" / name
        i = r["anchor"]
        iou, Vc, sil = cand[conv][name]
        rgb = read_rgb(files[i])
        mask = read_mask(out / "mask.png") > 0
        T_wc = np.asarray(cams["T_world_cam"][i])
        Vw = apply(T_wc, Vc)
        mw = meshes[name].copy()
        mw.vertices = Vw
        dst = project / "scene" / "assets" / "objects" / name
        dst.mkdir(parents=True, exist_ok=True)
        mw.export(str(dst / "init_world.glb"))
        ov = rgb.copy()
        ov[sil] = (0.5 * ov[sil] + 0.5 * np.array([0, 255, 255])).astype(np.uint8)
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cnts, -1, (255, 0, 255), 2)
        write_rgb(out / "init_overlay.png", ov)
        info = {"anchor_frame": i, "pointmap": r["pm_src"],
                "layout": {k: r[k] for k in ("quat_wxyz_local_to_p3dcam", "translation_p3dcam", "scale")},
                "mesh_axes": conv[0], "pytorch3d_row_vector": conv[1], "anchor_silhouette_iou": iou,
                "iou_by_convention": {f"{a}/{'row' if b else 'col'}": round(v[name][0], 4) for (a, b), v in cand.items()},
                "n_vertices": int(len(meshes[name].vertices)), "n_faces": int(len(meshes[name].faces)),
                "world_bbox_min": Vw.min(0).tolist(), "world_bbox_max": Vw.max(0).tolist(),
                "world_extent_m": (Vw.max(0) - Vw.min(0)).tolist(), "bottom_z_m": float(Vw[:, 2].min())}
        write_json(info, out / "layout.json")
        results[name] = info
    return {"convention": {"mesh_axes": conv[0], "pytorch3d_row_vector": conv[1],
                           "best_scoring_alternative": f"{best_alt[0]}/{'row' if best_alt[1] else 'col'}"},
            "objects": results, "outputs": ["work/objects", "scene/assets/objects"]}
