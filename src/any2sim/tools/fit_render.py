"""Render-and-compare fitting of meshes to the video (nvdiffrast): pose, scale, vertex offsets, or the parameters of
an agent-written shape builder, against per-frame object masks (+ optional depth and interior-edge terms).

Two modes per object:
  gradient (default)  `mesh` given: Adam on pose (dof), log-scale and optional per-vertex offsets
  builder             `builder` given: a module with build(params: dict) -> trimesh.Trimesh (object frame: z up,
                      origin at the bottom centre, metres; see any2sim.meshkit). Derivative-free (Powell /
                      Nelder-Mead) over the builder parameters and the pose dof. Use it to rebuild objects whose
                      generated mesh is wrong (slots, rims, handles, proportions).

params:
  frames: [source frame ids]                 targets (pick distinct frames where the object is static)
  cameras: "scene" | project-relative json {K, width, height, T_world_cam: [N x 4x4]}
  objects: [{
     name,
     mesh: path | builder: {module: path.py, function: "build", params: {p: [init, lo, hi]}, fixed: {..}},
     T_init: 4x4 world pose of the mesh frame,
     dof: "6dof" | "planar" | "yaw" | "fixed"     planar: yaw + x, y with the lowest points at support_z
     support_z: 0.0,
     scale: {fit: false, init: 1.0, bounds: [0.5, 2.0]},
     vertex_offsets: {enabled: false, w_laplacian: 50.0, max_m: 0.03}   (gradient mode, smooth local deformation)
     target: {label_maps: dir, label: k} | {masks: dir},
     ignore: [{masks: dir} | {label_maps: dir, labels: [..]}],   pixels not judged (hands, ...)
     proxy: "mesh" | "hull"          silhouette of the convex hull (meshes with openings vs solid masks)
     fill_render: true               fill holes of the rendered silhouette before comparing (solid masks)
     w_edges: 0.0                    builder mode: weight of (1 - interior edge F-score) (real Canny vs geometry edges)
     depth: {npz: {frame: path}, w: 0.0}   optional metric depth targets ("depth" key in npz, camera z, metres)
  }]
  occluders: [{mesh, T, movable: false}]   fixed geometry (depth-tested, never fitted); movable: true for other
                                   objects (their visible pixels never count for this object's silhouette)
  sequential: true                 objects fitted in order; fitted ones occlude later ones
  iters: 300, lr: 0.01, res_scale: 0.5, max_evals: 300 (builder), seed: 0
  out_dir: project-relative (default work/fit/<first object>)
Outputs per object in <out_dir>/<name>/: fitted.obj/.glb (object frame, scale baked; textured if the input was),
result.json (T_world, scale, builder params, per-frame IoU / boundary P95 / edge distance, loss history),
overlay.jpg (green = render outline, magenta = target, grey = ignored).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from ..common import contact_sheet, frame_files, read_rgb, write_json, write_rgb

ENV = "sam3d"
GPU = True


def _so3(w):
    import torch
    th = torch.linalg.norm(w) + 1e-12
    k = w / th
    Kx = torch.zeros(3, 3, device=w.device)
    Kx[0, 1], Kx[0, 2], Kx[1, 0], Kx[1, 2], Kx[2, 0], Kx[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    return torch.eye(3, device=w.device) + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx


def _laplacian(F, n, dev):
    import torch
    e = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]
    e = np.unique(np.sort(np.r_[e, e[:, ::-1]], 1), axis=0)
    i = torch.tensor(np.r_[e[:, 0], e[:, 1]], device=dev)
    j = torch.tensor(np.r_[e[:, 1], e[:, 0]], device=dev)
    deg = torch.zeros(n, device=dev).index_add_(0, i, torch.ones(len(i), device=dev))
    return i, j, deg


def _load_builder(project, spec):
    path = project / spec["module"]
    s = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    return getattr(mod, spec.get("function", "build"))


def run(project: Path, frames: list[int], objects: list[dict], cameras="scene", occluders: list | None = None,
        sequential: bool = True, iters: int = 300, lr: float = 0.01, res_scale: float = 0.5, max_evals: int = 300,
        seed: int = 0, out_dir: str | None = None) -> dict:
    import cv2
    import torch
    import trimesh
    from scipy.optimize import minimize
    from ..meshio import (any_mask, boundary_stats, edge_alignment, fill_holes, frame_mask, hull_front_depth,
                          edge_fscore, load_cameras, load_mesh, visible_filled)
    from ..models.raster import Rasterizer
    torch.manual_seed(seed)
    dev = "cuda"
    R = Rasterizer()
    cams = load_cameras(project, cameras)
    K0, W0, H0 = cams["K"], cams["width"], cams["height"]
    W, H = int(W0 * res_scale), int(H0 * res_scale)
    K = K0.copy()
    K[:2] *= res_scale
    Twc = cams["T_world_cam"]
    files = frame_files(project / "work" / "frames")
    out_root = project / (out_dir or f"work/fit/{objects[0]['name']}")
    out_root.mkdir(parents=True, exist_ok=True)

    def cam_T(f):
        return np.linalg.inv(Twc[f] if len(Twc) > 1 else Twc[0])

    def to_cam(Vw, f):
        Tcw = torch.tensor(cam_T(f), dtype=torch.float32, device=dev)
        return Vw @ Tcw[:3, :3].T + Tcw[:3, 3]

    fixed = []                                            # (V world torch, F torch)
    for o in occluders or []:
        V, F, *_ = load_mesh(project / o["mesh"])
        T = np.asarray(o.get("T", np.eye(4)), float)
        fixed.append((torch.tensor(V @ T[:3, :3].T + T[:3, 3], dtype=torch.float32, device=dev),
                      torch.tensor(F, dtype=torch.int64, device=dev), bool(o.get("movable", False))))

    def small(m):
        return cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0

    results = {}
    for ob in objects:
        name = ob["name"]
        od = out_root / name
        od.mkdir(parents=True, exist_ok=True)
        tgt = {f: frame_mask(project, ob["target"], f) for f in frames}
        frs = [f for f in frames if tgt[f] is not None and tgt[f].any()]
        ign = {f: any_mask(project, ob.get("ignore"), f, (H0, W0), dilate_px=5) for f in frs}
        tgt_s = {f: torch.tensor(small(tgt[f]), dtype=torch.float32, device=dev) for f in frs}
        val_s = {f: torch.tensor(~small(ign[f]), dtype=torch.float32, device=dev) for f in frs}
        T0 = np.asarray(ob.get("T_init", np.eye(4)), float)
        dof = ob.get("dof", "6dof")
        sz = float(ob.get("support_z", 0.0))
        sc_spec = ob.get("scale", {})
        builder = ob.get("builder")
        hist = []

        def iou_loss(Vw, Fo, f, fill=False):
            others = [(to_cam(V, f), Fx) for V, Fx, _ in fixed]
            sil, _ = R.render_ids([to_cam(Vw, f)] + [o[0] for o in others], [Fo] + [o[1] for o in others], K, W, H)
            p = sil[0]
            if fill:
                with torch.no_grad():
                    pf = torch.tensor(fill_holes(p.detach().cpu().numpy() > 0.5), dtype=torch.float32, device=dev)
                p = torch.maximum(p, pf)
            g, v = tgt_s[f], val_s[f]
            inter = (p * g * v).sum()
            union = ((p + g - p * g) * v).sum() + 1e-6
            return 1 - inter / union

        def place(Vloc, Rm, txy_or_t, s):
            V = (Vloc * s) @ Rm.T
            if dof == "planar":
                zlow = -torch.topk(-V[:, 2], min(32, len(V))).values.mean()
                return V + torch.cat([txy_or_t[:2], (sz - zlow).reshape(1)])
            return V + txy_or_t

        if builder is None:
            V0, F0, UV0, FUV0, tex0, m0 = load_mesh(project / ob["mesh"])
            prox = trimesh.Trimesh(V0, F0, process=False).convex_hull if ob.get("proxy") == "hull" else None
            Vp = torch.tensor(prox.vertices if prox is not None else V0, dtype=torch.float32, device=dev)
            Fp = torch.tensor(prox.faces if prox is not None else F0, dtype=torch.int64, device=dev)
            Vfull = torch.tensor(V0, dtype=torch.float32, device=dev)
            vo = ob.get("vertex_offsets") or {}
            use_off = bool(vo.get("enabled")) and prox is None
            off = torch.zeros_like(Vp, requires_grad=use_off)
            w = torch.zeros(3, device=dev, requires_grad=dof != "fixed")
            dt = torch.zeros(3, device=dev, requires_grad=dof != "fixed")
            log_s = torch.tensor([float(np.log(sc_spec.get("init", 1.0)))], dtype=torch.float32, device=dev,
                                 requires_grad=bool(sc_spec.get("fit")))
            R0 = torch.tensor(T0[:3, :3], dtype=torch.float32, device=dev)
            t0 = torch.tensor(T0[:3, 3], dtype=torch.float32, device=dev)
            groups = []
            if dof != "fixed":
                groups += [{"params": [w], "lr": lr}, {"params": [dt], "lr": lr * 0.05}]
            if sc_spec.get("fit"):
                groups.append({"params": [log_s], "lr": lr * 0.5})
            if use_off:
                groups.append({"params": [off], "lr": lr * 0.02})
                li, lj, deg = _laplacian(np.asarray(Fp.cpu()), len(Vp), dev)
            ez = torch.tensor([0.0, 0.0, 1.0], device=dev)
            lo, hi = np.log(sc_spec.get("bounds", [0.5, 2.0]))

            def pose(Vl):
                Rm = _so3(w * ez if dof in ("planar", "yaw") else w) @ R0
                s = torch.exp(log_s.clamp(lo, hi))
                return Rm, place(Vl, Rm, t0 + dt, s), s

            opt = torch.optim.Adam(groups) if groups else None
            for it in range(iters if groups else 0):
                opt.zero_grad()
                Vl = Vp + (off.clamp(-vo.get("max_m", 0.03), vo.get("max_m", 0.03)) if use_off else 0)
                Rm, Vw, s = pose(Vl)
                L = sum(iou_loss(Vw, Fp, f) for f in frs) / len(frs)
                if dof != "planar":
                    zmin = -torch.topk(-Vw[:, 2], min(32, len(Vw))).values.mean()
                    L = L + 10.0 * (torch.relu(sz - zmin) / 0.005) ** 2
                if use_off:
                    lap = off - torch.zeros_like(off).index_add_(0, li, off[lj]) / deg.clamp(min=1)[:, None]
                    L = L + vo.get("w_laplacian", 50.0) * (lap ** 2).sum(1).mean() * 1e4     # (1 cm)^-2
                L.backward()
                opt.step()
                hist.append(float(L))
            with torch.no_grad():
                Rm, Vw_p, s = pose(Vp + (off if use_off else 0))
                # final pose of the mesh frame: T_world = [R | t] with scale baked into the exported mesh
                if use_off:                                   # offsets live on the mesh vertices (no proxy)
                    Vfull_l = (Vfull + off) * s
                else:
                    Vfull_l = Vfull * s
                Vw_full = place(Vfull, Rm, t0 + dt, s)
                t_final = (Vw_full - Vfull_l @ Rm.T)[0]
                T = np.eye(4)
                T[:3, :3] = Rm.cpu().numpy()
                T[:3, 3] = t_final.cpu().numpy()
            mesh_out = m0.copy()
            mesh_out.vertices = Vfull_l.cpu().numpy()
            Vw_eval, F_eval = Vw_full, torch.tensor(F0, dtype=torch.int64, device=dev)
            res = {"mode": "gradient", "T_world": T.tolist(), "scale": float(s), "vertex_offsets": use_off,
                   "loss_first": hist[0] if hist else None, "loss_last": hist[-1] if hist else None}
        else:
            build = _load_builder(project, builder)
            names = list(builder["params"])
            init = np.array([builder["params"][k][0] for k in names], float)
            blo = np.array([builder["params"][k][1] for k in names], float)
            bhi = np.array([builder["params"][k][2] for k in names], float)
            pose_names = {"planar": ["yaw", "dx", "dy"], "yaw": ["yaw"], "6dof": ["rx", "ry", "rz", "dx", "dy", "dz"],
                          "fixed": []}[dof]
            x0 = np.r_[(init - blo) / np.maximum(bhi - blo, 1e-9), np.zeros(len(pose_names))]
            step = {"yaw": 0.3, "rx": 0.1, "ry": 0.1, "rz": 0.3, "dx": 0.03, "dy": 0.03, "dz": 0.03}
            fill = ob.get("fill_render", True)
            w_edges = float(ob.get("w_edges", 0.0))
            cache = {}

            def make(x):
                p = {k: float(blo[i] + np.clip(x[i], 0, 1) * (bhi[i] - blo[i])) for i, k in enumerate(names)}
                p.update(builder.get("fixed", {}))
                m = build(p)
                pv = dict(zip(pose_names, [x[len(names) + i] * step[n] for i, n in enumerate(pose_names)]))
                from scipy.spatial.transform import Rotation
                if dof in ("planar", "yaw"):
                    Rz = Rotation.from_euler("z", pv.get("yaw", 0.0)).as_matrix()
                else:
                    Rz = Rotation.from_rotvec([pv.get("rx", 0), pv.get("ry", 0), pv.get("rz", 0)]).as_matrix()
                Rm = T0[:3, :3] @ Rz
                V = np.asarray(m.vertices) @ Rm.T
                t = T0[:3, 3] + [pv.get("dx", 0), pv.get("dy", 0), pv.get("dz", 0)]
                if dof == "planar":
                    t[2] = sz - V[:, 2].min()
                return p, m, Rm, t

            def objective(x):
                key = tuple(np.round(x, 5))
                if key in cache:
                    return cache[key]
                try:
                    p, m, Rm, t = make(x)
                except Exception:  # invalid parameter combination for the builder
                    cache[key] = 2.0
                    return 2.0
                Vw = torch.tensor(np.asarray(m.vertices) @ Rm.T + t, dtype=torch.float32, device=dev)
                Fm = torch.tensor(np.asarray(m.faces), dtype=torch.int64, device=dev)
                with torch.no_grad():
                    L = float(sum(iou_loss(Vw, Fm, f, fill=fill) for f in frs) / len(frs))
                if w_edges > 0:
                    ed = []
                    for f in frs[:3]:
                        Tcw = cam_T(f)
                        Vc = torch.tensor((np.asarray(m.vertices) @ Rm.T + t) @ Tcw[:3, :3].T + Tcw[:3, 3],
                                          dtype=torch.float32, device=dev)
                        obj, depth, nrm, tri = R.render_full([Vc], [Fm], K0, W0, H0)
                        region = cv2.dilate(tgt[f].astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
                        rgb_f = read_rgb(files[f])
                        d, er_, en_ = edge_alignment(rgb_f, tri, depth, nrm, region, ign[f])
                        ef = edge_fscore(rgb_f, er_, en_, depth) if d is not None else None
                        ed.append(1.0 - (ef["f"] if ef else 0.0))
                    L += w_edges * (np.mean(ed) if ed else 1.0)
                cache[key] = L
                hist.append(L)
                return L
            r = minimize(objective, x0, method=builder.get("method", "Powell"),
                         options={"maxfev": max_evals, "xtol": 1e-3, "ftol": 1e-4})
            p, m, Rm, t = make(r.x)
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = Rm, t
            mesh_out = m
            Vw_eval = torch.tensor(np.asarray(m.vertices) @ Rm.T + t, dtype=torch.float32, device=dev)
            F_eval = torch.tensor(np.asarray(m.faces), dtype=torch.int64, device=dev)
            res = {"mode": "builder", "T_world": T.tolist(), "params": p, "evaluations": len(hist),
                   "loss_first": hist[0] if hist else None, "loss_last": float(r.fun), "optimizer": str(r.message)}

        # evaluation at full resolution, every target frame
        per, tiles, caps = [], [], []
        for f in frs:
            Vc = [to_cam(Vw_eval, f)] + [to_cam(V, f) for V, _, _ in fixed]
            Fs = [F_eval] + [Fx for _, Fx, _ in fixed]
            obj, depth, nrm, tri = R.render_full(Vc, Fs, K0, W0, H0)
            obj_a, _, _, _ = R.render_full(Vc[:1], Fs[:1], K0, W0, H0)
            movable = [1 + j for j, fx in enumerate(fixed) if fx[2]]
            pr = visible_filled(obj, depth, 0, obj_a, hull_front_depth(R, Vc[0], K0, W0, H0), movable)
            g, ig = tgt[f], ign[f]
            iou = float((pr & g & ~ig).sum() / max(((pr | g) & ~ig).sum(), 1))
            bm, b95 = boundary_stats(pr, g, ig)
            region = cv2.dilate(g.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            ed, er, en = edge_alignment(read_rgb(files[f]), np.where(obj == 0, tri, -1), depth, nrm, region, ig)
            ef = edge_fscore(read_rgb(files[f]), er, en, depth) if ed is not None else None
            per.append({"frame": int(f), "iou": round(iou, 4), "boundary_mean_px": bm, "boundary_p95_px": b95,
                        "edge_chamfer_px": ed, "edge_f": ef["f"] if ef else None})
            if len(tiles) < 6:
                im = read_rgb(files[f])
                im[ig] = (im[ig] * 0.4).astype(np.uint8)
                for msk, col in ((g, (255, 0, 255)), (pr, (0, 255, 0))):
                    cn, _ = cv2.findContours(msk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(im, cn, -1, col, 2)
                ys, xs = np.nonzero(g | pr)
                if len(xs):
                    pad = 40
                    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, H0)
                    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, W0)
                    crop = im[y0:y1, x0:x1]
                    ec = np.zeros_like(crop)
                    ec[er[y0:y1, x0:x1]] = (255, 0, 255)
                    ec[en[y0:y1, x0:x1]] = (0, 255, 0)
                    tiles += [crop, ec]
                    caps += [f"f{f} IoU {iou:.3f}", f"edges real(m)/render(g) {ed if ed is None else round(ed, 2)}"]
        if tiles:
            write_rgb(od / "overlay.jpg", contact_sheet(tiles, cols=4, width=360, labels=caps))
        res["per_frame"] = per
        res["iou_mean"] = float(np.mean([p_["iou"] for p_ in per])) if per else None
        res["boundary_p95_px_max"] = float(np.max([p_["boundary_p95_px"] or 0 for p_ in per])) if per else None
        eds = [p_["edge_chamfer_px"] for p_ in per if p_["edge_chamfer_px"] is not None]
        res["edge_chamfer_px_mean"] = float(np.mean(eds)) if eds else None
        efs = [p_["edge_f"] for p_ in per if p_["edge_f"] is not None]
        res["edge_f_mean"] = float(np.mean(efs)) if efs else None
        res["loss_history"] = hist[:: max(1, len(hist) // 200)]
        mesh_out.export(str(od / "fitted.glb"))
        from ..meshkit import save_obj
        save_obj(mesh_out, od / "fitted.obj")
        write_json(res, od / "result.json")
        results[name] = {k: v for k, v in res.items() if k not in ("per_frame", "loss_history")}
        if sequential:
            fixed.append((Vw_eval.detach(), F_eval, True))
    return {"objects": results, "outputs": [str(out_root.relative_to(project))]}
