"""Check a robot description against logged robot data: joint ranges/velocities and forward kinematics vs logged
end-effector poses (which link and frame the log refers to, and how well the kinematics reproduce it).

For every end-effector log and candidate link L, fit constant transforms A (log reference frame <- robot base) and
B (link L -> logged EE frame) so that T_log(t) ~= A @ FK_L(q(t)) @ B, trying every rotation format that matches the
column width. Small residuals (mm, deg) mean the kinematics (link lengths, axes, signs, joint order) agree with the
real robot; the best link/B tell where the controller's EE frame is. A position-only fit is reported as well.

params:
  bundle: "robot"
  log: path (absolute or project-relative)         npz | npy (+ .meta.json) | csv | json  (datalog.load_table)
  joints: {column spec (str | {column, slice}): [joint names]}   e.g. {"joint_positions": {...}}
          given as a list of {"column", "slice", "joints"}
  ee: [{name, position: col spec, rotation: col spec | null, rotation_format: "auto" | fmt,
        links: [candidate links] | null (default: every group's tip link and its neighbours)}]
  rows: null | [a, b]        row range used
  rate_hz: null              for velocity checks (else a "timestamp" column in seconds/ns is used if present)
Writes <bundle>/checks/log_check.json; returns per joint range/velocity violations and per EE the ranked fits.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common import write_json

ENV = "base"
GPU = False


def _fit(Tl, Tf, with_rot=True):
    """A, B with Tl ~= A Tf B; returns (A, B, pos_res (T,), rot_res_deg (T,))."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation as Rot

    def unpack(x):
        A, B = np.eye(4), np.eye(4)
        A[:3, :3], A[:3, 3] = Rot.from_rotvec(x[0:3]).as_matrix(), x[3:6]
        B[:3, :3], B[:3, 3] = Rot.from_rotvec(x[6:9]).as_matrix(), x[9:12]
        return A, B

    def resid(x):
        A, B = unpack(x)
        P = A[None] @ Tf @ B[None]
        r = [(P[:, :3, 3] - Tl[:, :3, 3]).ravel() * 1000.0]            # mm
        if with_rot:
            dR = np.einsum("tij,tik->tjk", P[:, :3, :3], Tl[:, :3, :3])
            r.append(Rot.from_matrix(dR).as_rotvec().ravel() * 57.3 * 0.5)   # deg, weighted 2 deg ~ 1 mm
        return np.concatenate(r)

    best = None
    # initialisations: identity, and A from the first sample with B = I
    inits = [np.zeros(12)]
    A0 = Tl[0] @ np.linalg.inv(Tf[0])
    inits.append(np.r_[Rot.from_matrix(A0[:3, :3]).as_rotvec(), A0[:3, 3], np.zeros(6)])
    for x0 in inits:
        s = least_squares(resid, x0, loss="soft_l1", f_scale=5.0, max_nfev=400)
        if best is None or s.cost < best.cost:
            best = s
    A, B = unpack(best.x)
    P = A[None] @ Tf @ B[None]
    pr = np.linalg.norm(P[:, :3, 3] - Tl[:, :3, 3], axis=1) * 1000
    dR = np.einsum("tij,tik->tjk", P[:, :3, :3], Tl[:, :3, :3])
    rr = np.degrees(np.linalg.norm(Rot.from_matrix(dR).as_rotvec(), axis=1))
    return A, B, pr, rr


def run(project: Path, log: str, joints: list, ee: list | None = None, bundle: str = "robot",
        rows: list | None = None, rate_hz: float | None = None) -> dict:
    from ..robot.bundle import Bundle
    from ..robot import datalog as DL
    from ..robot import urdf as U
    broot = Path(bundle) if Path(bundle).is_absolute() else project / bundle
    b = Bundle(broot)
    tab = DL.load_table(Path(log) if Path(log).is_absolute() else project / log)
    n = len(next(iter(tab.values())))
    sl = slice(*(rows or [0, n]))
    q = {}
    for spec in joints:
        a = DL.column(tab, spec)[sl]
        for k, jn in enumerate(spec["joints"]):
            if jn is not None:
                q[jn] = a[:, k]
    T = len(next(iter(q.values())))
    if rate_hz:
        dt = np.full(T - 1, 1.0 / rate_hz)
    elif "timestamp" in tab:
        ts = tab["timestamp"][sl, 0]
        dt = np.diff(ts) * (1e-9 if np.median(np.abs(np.diff(ts))) > 1e3 else 1.0)
        if (dt <= 0).mean() > 0.05:                     # e.g. ns timestamps stored as float32: unusable
            dt = None
    else:
        dt = None
    lim = dict(zip(b.joints(), b.limits()))
    jrep = {}
    for jn, v in q.items():
        lo, hi = lim[jn]
        r = {"min": float(v.min()), "max": float(v.max()), "limit": [float(lo), float(hi)],
             "frac_outside": float(((v < lo - 1e-3) | (v > hi + 1e-3)).mean()),
             "max_violation": float(max(lo - v.min(), v.max() - hi, 0.0))}
        if dt is not None:
            vel = np.abs(np.diff(v)) / np.maximum(dt, 1e-6)
            vl = (b.model["joints"][jn].get("limit") or {}).get("velocity")
            r.update(max_vel=float(np.percentile(vel, 99.5)), vel_limit=vl)
        jrep[jn] = r
    q0 = b.init_q()
    Tfk = {}

    def fk_series(link):
        if link not in Tfk:
            out = np.zeros((T, 4, 4))
            for t in range(T):
                qt = dict(q0)
                qt.update({j: float(v[t]) for j, v in q.items()})
                out[t] = U.fk(b.model, qt)[link]
            Tfk[link] = out
        return Tfk[link]

    eerep = []
    for e in ee or []:
        pos = DL.column(tab, e["position"])[sl]
        rot = DL.column(tab, e["rotation"])[sl] if e.get("rotation") else None
        cands = e.get("links")
        if not cands:
            cands = []
            for G in b.profile["groups"].values():
                tl = G.get("tip_link")
                if tl:
                    cands.append(tl)
        sub = np.linspace(0, T - 1, min(T, 300)).astype(int)
        fits = []
        fmts = (DL.candidate_formats(rot.shape[1]) if e.get("rotation_format", "auto") == "auto" else [e["rotation_format"]]) \
            if rot is not None else [None]
        for L in cands:
            Tf = fk_series(L)[sub]
            for fmt in fmts:
                Tl = np.tile(np.eye(4), (len(sub), 1, 1))
                Tl[:, :3, 3] = pos[sub]
                if fmt:
                    try:
                        Tl[:, :3, :3] = DL.to_matrix(rot[sub], fmt)
                    except Exception:                                        # noqa: BLE001
                        continue
                A, B, pr, rr = _fit(Tl, Tf, with_rot=fmt is not None)
                fits.append({"link": L, "rotation_format": fmt,
                             "pos_mm": {"median": float(np.median(pr)), "p95": float(np.percentile(pr, 95))},
                             "rot_deg": ({"median": float(np.median(rr)), "p95": float(np.percentile(rr, 95))} if fmt else None),
                             "A_logref_base": A.round(6).tolist(), "B_link_ee": B.round(6).tolist(),
                             "motion_range_mm": float(np.ptp(pos[sub], axis=0).max() * 1000)})
        fits.sort(key=lambda f: f["pos_mm"]["p95"] + (f["rot_deg"]["p95"] if f["rot_deg"] else 0))
        eerep.append({"name": e.get("name", str(e["position"])), "best": fits[0] if fits else None, "fits": fits[:8]})
    res = {"rows": T, "velocity_basis": "rate_hz" if rate_hz else ("timestamp" if dt is not None else "none"), "joints": jrep,
           "joints_outside_limits": {k: v["frac_outside"] for k, v in jrep.items() if v["frac_outside"] > 0},
           "ee": eerep}
    write_json(res, broot / "checks" / "log_check.json")
    return {"rows": T, "joints_outside_limits": res["joints_outside_limits"],
            "ee": [{"name": x["name"], "best": {k: x["best"][k] for k in ("link", "rotation_format", "pos_mm", "rot_deg")}
                    if x["best"] else None} for x in eerep],
            "outputs": [str((broot / "checks" / "log_check.json").relative_to(project)) if broot.is_relative_to(project) else ""]}
