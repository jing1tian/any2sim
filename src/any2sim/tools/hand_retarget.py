"""Retarget human hand keypoints (MANO-21 or any keypoint set) to a robot hand: palm pose + finger joints per frame.

1. Palm calibration (whole clip): a similarity M (scale s, rotation, translation; constant relative to the human
   wrist frame when wrist poses are given, else per frame) that maps the rigid human palm keypoints (wrist, finger
   bases) onto the robot's corresponding palm points. Reports the scale and the residual.
2. Fingers (per frame, warm-started, limits respected): finger joints whose keypoint links reach the mapped human
   keypoints (position IK per finger chain), with a smoothness pull to the previous frame.
3. If robot joint values are supplied (e.g. an external retargeter), the same keypoint residual is reported for them.
Everything stays in the frame of the input data (e.g. a demo/base frame); placing it in the scene is the caller's job.

Keypoints are named semantically: wrist, <finger>_<k> with finger in thumb/index/middle/ring/pinky and k = 0 (base:
thumb CMC, finger MCP) .. 3 (tip). The data layout maps names to indices: "openpose21" (0 wrist, 1-4 thumb, 5-8
index, 9-12 middle, 13-16 ring, 17-20 pinky), "mano_chains21" (0 wrist, 1-4 index, 5-8 middle, 9-12 pinky, 13-16
ring, 17-20 thumb), an explicit {name: index} dict, or "auto" (five contiguous 4-point chains; thumb = the chain
whose base is nearest the wrist, then index/middle/ring/pinky by base distance from the thumb base).
The robot side comes from the bundle (robot.yaml retarget.<group>) or the `mapping` param:
  {palm_link: <link>,
   rigid: {wrist | <finger>_0: <link> | {link, offset}},          # palm points
   chains: {<finger>: {joints: [...], keypoints: {<finger>_<k>: <link> | {link, offset}}}}}

params:
  bundle: "assets:<name>" | dir | "scene:<robot id>"
  group: hand group name (e.g. right_hand)
  data: path (npz | npy + .meta.json | csv)            the per-frame table
  keypoints: column spec of (T, K*3) keypoints;  layout: "auto" | "openpose21" | "mano_chains21" | {name: index}
  wrist: null | {position: col, rotation: col, rotation_format: "auto" | fmt}   human wrist poses in the same
          frame ("auto": the format in which the keypoints are most rigid in the wrist frame)
  supplied_q: null | {column: col, joints: [...]}       robot joints from another retargeter (evaluated, and
                                                        returned as an alternative)
  rows: null | [a, b]
  smooth: 0.05      scale_mode: "fit" | "fixed:<s>"
  out: project-relative npz (default work/retarget/<group>.npz)
Output npz: palm_T (T,4,4) robot palm pose in the data frame, q (T,J) finger joints, joint_names, scale, M_palm
(T,4,4 similarity human->robot palm frame), residual_mm (T,K); supplied_q / supplied_residual_mm when given.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common import write_json

ENV = "base"
GPU = False


def _pt(spec):
    return (spec, (0, 0, 0)) if isinstance(spec, str) else (spec["link"], tuple(spec.get("offset", (0, 0, 0))))


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
LAYOUTS = {"openpose21": ["thumb", "index", "middle", "ring", "pinky"],
           "mano_chains21": ["index", "middle", "pinky", "ring", "thumb"]}


def layout_names(layout, KP) -> dict:
    """{semantic name: keypoint index}."""
    if isinstance(layout, dict):
        return {k: int(v) for k, v in layout.items()}
    if layout in LAYOUTS:
        order = LAYOUTS[layout]
    else:                                                   # auto: contiguous chains 1-4, 5-8, ... after the wrist
        k = KP[len(KP) // 2]
        bases = [1 + 4 * c for c in range(5)]
        d0 = [np.linalg.norm(k[b] - k[0]) for b in bases]
        th = int(np.argmin(d0))
        rest = sorted([c for c in range(5) if c != th], key=lambda c: np.linalg.norm(k[bases[c]] - k[bases[th]]))
        order = [None] * 5
        order[th] = "thumb"
        for name, c in zip(("index", "middle", "ring", "pinky"), rest):
            order[c] = name
    out = {"wrist": 0}
    for c, f in enumerate(order):
        for j in range(4):
            out[f"{f}_{j}"] = 1 + 4 * c + j
    return out


def _umeyama(src, dst, with_scale=True):
    """Similarity (s, R, t) minimising |s R src + t - dst|."""
    ms, md = src.mean(0), dst.mean(0)
    A, B = src - ms, dst - md
    U_, S, Vt = np.linalg.svd(B.T @ A)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U_ @ Vt))
    R = U_ @ D @ Vt
    s = (S * np.diag(D)).sum() / (A ** 2).sum() if with_scale else 1.0
    return s, R, md - s * R @ ms


def run(project: Path, bundle: str, group: str, data: str, keypoints, layout="auto", wrist: dict | None = None,
        supplied_q: dict | None = None, rows: list | None = None, smooth: float = 0.05, scale_mode: str = "fit",
        mapping: dict | None = None, out: str | None = None) -> dict:
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation as Rot
    from .. import robokit as rk
    from ..robot import datalog as DL
    if bundle.startswith("scene:"):
        from ..scene_ir import load, robot_bundle
        b = robot_bundle(project / "scene", load(project / "scene"), bundle.split(":", 1)[1])
    else:
        b = rk._bundle(bundle if not Path(bundle).is_absolute() and bundle.startswith("assets:") else
                       (bundle if Path(bundle).is_absolute() else str(project / bundle)))
    kin = rk.Kin(b)
    mp = mapping or (b.profile.get("retarget") or {}).get(group)
    if not mp:
        raise ValueError(f"no retarget mapping for {group} (bundle retarget.{group} or `mapping`)")
    tab = DL.load_table(Path(data) if Path(data).is_absolute() else project / data)
    sl = slice(*(rows or [0, len(next(iter(tab.values())))]))
    KP = DL.column(tab, keypoints)[sl].reshape(-1, len(DL.column(tab, keypoints)[0]) // 3, 3)
    T = len(KP)
    names = layout_names(layout, KP)
    Wt = None
    fmt_used = None
    if wrist:
        rot = DL.column(tab, wrist["rotation"])[sl]
        fmts = DL.candidate_formats(rot.shape[1]) if wrist.get("rotation_format", "auto") == "auto" else [wrist["rotation_format"]]
        best = None
        for fmt in fmts:                     # the format whose wrist frame is most rigidly attached to the palm
            try:
                W_ = DL.poses(DL.column(tab, wrist["position"])[sl], rot, fmt)
            except Exception:                                          # noqa: BLE001
                continue
            rels = []
            for t in range(0, len(W_), max(1, len(W_) // 60)):
                kp = DL.column(tab, keypoints)[sl][t].reshape(-1, 3)
                h = (np.linalg.inv(W_[t]) @ np.c_[kp, np.ones(len(kp))].T)[:3].T
                rels.append(h)
            spread = float(np.median(np.std(np.asarray(rels), axis=0)))  # keypoints in the wrist frame: rigid parts stay put
            if best is None or spread < best[0]:
                best = (spread, fmt, W_)
        _, fmt_used, Wt = best
    palm = mp["palm_link"]
    rig = {names[k]: _pt(v) for k, v in mp["rigid"].items()}
    kin.set(kin.q0)
    Pp = kin.pose(palm)
    Pinv = np.linalg.inv(Pp)
    robot_rigid = np.array([(Pinv @ np.r_[kin.point(l, o), 1])[:3] for l, o in rig.values()])
    idx = list(rig)
    # 1. palm similarity
    Ms = np.zeros((T, 4, 4))
    scales = []
    for t in range(T):
        s, R, tt = _umeyama(KP[t, idx], robot_rigid, scale_mode == "fit")
        if scale_mode.startswith("fixed:"):
            s = float(scale_mode.split(":")[1])
            _, R, _ = _umeyama(KP[t, idx], robot_rigid, False)
            tt = robot_rigid.mean(0) - s * R @ KP[t, idx].mean(0)
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = s * R, tt
        Ms[t] = M
        scales.append(s)
    s_all = float(np.median(scales))
    if Wt is not None:
        # constant human-wrist -> robot-palm transform (rigid), fitted over the clip with one scale
        rel = []
        for t in range(T):
            Mt = Ms[t].copy()
            Mt[:3, :3] /= scales[t]
            rel.append(Mt @ Wt[t])                             # palm <- wrist (rotation part), per frame
        R_avg = Rot.from_matrix(np.array([r[:3, :3] for r in rel])).mean().as_matrix()

        def resid(x):
            Rr = Rot.from_rotvec(x[:3]).as_matrix() @ R_avg
            out = []
            for t in range(0, T, max(1, T // 120)):
                h = (np.linalg.inv(Wt[t]) @ np.c_[KP[t, idx], np.ones(len(idx))].T)[:3].T   # wrist frame
                pr = x[6] * (h @ Rr.T) + x[3:6]
                out.append((pr - robot_rigid).ravel())
            return np.concatenate(out) * 1000
        x0 = np.r_[np.zeros(3), np.zeros(3), s_all]
        h0 = (np.linalg.inv(Wt[0]) @ np.c_[KP[0, idx], np.ones(len(idx))].T)[:3].T
        x0[3:6] = robot_rigid.mean(0) - s_all * (R_avg @ h0.mean(0))
        sol = least_squares(resid, x0, loss="soft_l1", f_scale=5.0)
        Rr = Rot.from_rotvec(sol.x[:3]).as_matrix() @ R_avg
        s_all = float(sol.x[6]) if scale_mode == "fit" else s_all
        Mw = np.eye(4)
        Mw[:3, :3], Mw[:3, 3] = s_all * Rr, sol.x[3:6]
        for t in range(T):
            Ms[t] = Mw @ np.linalg.inv(Wt[t])
    # palm pose in the data frame: the robot palm origin/axes mapped back through the (rigid part of) M
    palm_T = np.zeros((T, 4, 4))
    for t in range(T):
        s = np.cbrt(np.linalg.det(Ms[t][:3, :3]))
        R = Ms[t][:3, :3] / s
        P = np.eye(4)
        P[:3, :3] = R.T
        c_r = robot_rigid.mean(0)                              # scale about the palm centre
        c_h = (np.linalg.inv(Ms[t]) @ np.r_[c_r, 1])[:3]
        P[:3, 3] = c_h - R.T @ c_r
        palm_T[t] = P
    # 2. fingers
    chains = mp["chains"]
    jn = [j for c in chains.values() for j in c["joints"]]
    Q = np.zeros((T, len(jn)))
    kp_link = dict(rig)
    for c in chains.values():
        kp_link.update({names[k]: _pt(v) for k, v in c["keypoints"].items()})
    allk = list(kp_link)
    res = np.zeros((T, len(allk)))
    q = dict(kin.q0)

    def keypoint_err(qd, t):
        kin.set(qd)
        Pinv_ = np.linalg.inv(kin.pose(palm))
        e = []
        for k in allk:
            l, o = kp_link[k]
            pr = (Pinv_ @ np.r_[kin.point(l, o), 1])[:3]
            ph = (Ms[t] @ np.r_[KP[t, k], 1])[:3]
            e.append(np.linalg.norm(pr - ph) * 1000)
        return e

    for t in range(T):
        kin.set(q)
        Pw = kin.pose(palm)
        for cname, c in chains.items():
            tg = {}
            for k, spec in c["keypoints"].items():
                l, o = _pt(spec)
                ph = (Ms[t] @ np.r_[KP[t, names[k]], 1])[:3]         # robot palm frame
                tg[k] = {"link": l, "p": (Pw @ np.r_[ph, 1])[:3], "offset": o}
            q, _ = kin.ik(tg, q0=q, joints=c["joints"], q_rest=dict(q), rest_w=smooth if t else 0.0, iters=60)
        Q[t] = [q[j] for j in jn]
        res[t] = keypoint_err(q, t)
    outp = project / (out or f"work/retarget/{group}.npz")
    outp.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"palm_T": palm_T, "q": Q, "joint_names": np.asarray(jn, dtype="U64"), "scale": s_all, "M_palm": Ms,
              "residual_mm": res, "keypoint_ids": np.asarray(allk),
              "keypoint_names": np.asarray([next(n for n, i in names.items() if i == k) for k in allk], dtype="U16")}
    summary = {"frames": T, "scale": s_all, "wrist_rotation_format": fmt_used, "residual_mm": {"median": float(np.median(res)),
                                                           "p95": float(np.percentile(res, 95))},
               "layout": {f: names[f"{f}_0"] for f in FINGERS},
               "per_keypoint_median_mm": {next(n for n, i in names.items() if i == k): v
                                          for k, v in zip(allk, np.median(res, 0).round(2).tolist())}}
    if supplied_q:
        Sq = DL.column(tab, supplied_q)[sl]
        sres = np.zeros_like(res)
        for t in range(T):
            qd = dict(kin.q0)
            qd.update({j: float(v) for j, v in zip(supplied_q["joints"], Sq[t]) if j})
            sres[t] = keypoint_err(qd, t)
        arrays.update(supplied_q=Sq, supplied_residual_mm=sres)
        summary["supplied_residual_mm"] = {"median": float(np.median(sres)), "p95": float(np.percentile(sres, 95))}
        lim = kin.limits([j for j in supplied_q["joints"] if j])
        summary["supplied_outside_limits_frac"] = float(((Sq < lim[:, 0] - 1e-3) | (Sq > lim[:, 1] + 1e-3)).mean())
    np.savez_compressed(outp, **arrays)
    write_json(summary, outp.with_suffix(".json"))
    return {**summary, "outputs": [str(outp.relative_to(project))]}
