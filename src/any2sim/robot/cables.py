"""Visual cables and wires of a RobotBundle (robot.yaml `cables`): leads between links that follow the joints.

Cables are appearance, not physics (like the bundle's visuals): no collision, mass or force. Their shape at a joint
configuration is a smooth curve through anchor points fixed in links; an optional length adds gravity sag
(slack along "down"), so a lead spanning moving joints bends and droops plausibly.

robot.yaml:
cables:
  <name>:
    radius_m: 0.0035
    material: {base_color: [r, g, b] (linear), roughness: 0.5, specular: 0.3}
    shape: spline                                   # default
    points: [{link: <link>, p: [x, y, z]}, ...]     # >= 2, each fixed in a link (base_link for the robot base)
    tangents: {start: [dx, dy, dz], end: [dx, dy, dz]}   # optional, in the start / end point's link frame,
                                                    # pointing along the cable (out of start, into end)
    length_m: null | <float>                        # longer than the chord -> sag along "down"
    slack: null | {link: <link>, dir: [x, y, z]}    # slack direction fixed in a link (e.g. a service loop bowing
                                                    # outwards) instead of gravity
    provenance: <text>
  <name>:
    shape: loop                                     # a service loop / coil fixed in one link
    link: <link>, center: [x, y, z], axis: [ax, ay, az], radius: 0.03, turns: 2, pitch: 0.008, phase: 0.0
    radius_m: ..., material: ..., provenance: ...

    from any2sim.robot import cables as CB
    P = CB.centerline(spec, link_T, down)          # link_T: {link: 4x4} (e.g. Bundle.fk(q, T_world_base))
    V, F = CB.tube(P, spec["radius_m"])
"""
from __future__ import annotations

import numpy as np


def _apply(T, p):
    T = np.asarray(T, float)
    return T[:3, :3] @ np.asarray(p, float) + T[:3, 3]


def _catmull_rom(P, t_start=None, t_end=None, n=48):
    """Centripetal Catmull-Rom through P (k, 3), end tangents optional (unit, scaled by the end segment length)."""
    P = np.asarray(P, float)
    k = len(P)
    if k == 2 and t_start is None and t_end is None:
        u = np.linspace(0, 1, n)[:, None]
        return (1 - u) * P[0] + u * P[1]
    seg_len = np.linalg.norm(np.diff(P, axis=0), axis=1) + 1e-9
    tan = np.zeros_like(P)
    for i in range(k):
        if i == 0:
            tan[i] = (np.asarray(t_start, float) * seg_len[0]) if t_start is not None else (P[1] - P[0])
        elif i == k - 1:
            tan[i] = (np.asarray(t_end, float) * seg_len[-1]) if t_end is not None else (P[-1] - P[-2])
        else:
            tan[i] = 0.5 * (P[i + 1] - P[i - 1])
    # samples per segment proportional to its length
    w = seg_len / seg_len.sum()
    counts = np.maximum(2, np.round(w * (n - 1)).astype(int))
    out = []
    for i in range(k - 1):
        t = np.linspace(0, 1, counts[i] + 1)[:-1, None]
        h00, h10, h01, h11 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t, -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
        out.append(h00 * P[i] + h10 * tan[i] + h01 * P[i + 1] + h11 * tan[i + 1])
    out.append(P[-1:])
    return np.vstack(out)


def arc_length(P) -> float:
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def _resample(P, n):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    u = np.linspace(0, s[-1], n)
    return np.stack([np.interp(u, s, P[:, k]) for k in range(3)], 1)


def centerline(spec: dict, link_T: dict, down=(0, 0, -1), n: int = 48) -> np.ndarray:
    """(n, 3) centreline in the frame of link_T (world when link_T are world poses)."""
    if spec.get("shape", "spline") == "loop":
        T = link_T[spec["link"]]
        a = np.asarray(spec["axis"], float)
        a /= np.linalg.norm(a)
        e1 = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(a, e1)
        th = float(spec.get("phase", 0.0)) + np.linspace(0, 2 * np.pi * float(spec.get("turns", 1.0)), n)
        r = float(spec["radius"])
        h = float(spec.get("pitch", 0.0)) * (th - th[0]) / (2 * np.pi)
        h -= h.mean()
        P = np.asarray(spec["center"], float) + r * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2) + h[:, None] * a
        return P @ np.asarray(T)[:3, :3].T + np.asarray(T)[:3, 3]
    pts = [_apply(link_T[p["link"]], p["p"]) for p in spec["points"]]
    tg = spec.get("tangents") or {}
    ts = te = None
    if tg.get("start") is not None:
        ts = np.asarray(link_T[spec["points"][0]["link"]])[:3, :3] @ np.asarray(tg["start"], float)
        ts /= np.linalg.norm(ts) + 1e-12
    if tg.get("end") is not None:
        te = np.asarray(link_T[spec["points"][-1]["link"]])[:3, :3] @ np.asarray(tg["end"], float)
        te /= np.linalg.norm(te) + 1e-12
    base = _catmull_rom(pts, ts, te, n=4 * n)
    L = spec.get("length_m")
    if L:
        d = np.asarray(down, float)
        if spec.get("slack"):
            d = np.asarray(link_T[spec["slack"]["link"]])[:3, :3] @ np.asarray(spec["slack"]["dir"], float)
        d /= np.linalg.norm(d)
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(base, axis=0), axis=1))]
        u = s / max(s[-1], 1e-9)
        bump = np.sin(np.pi * u) ** 2                       # zero value and slope at both anchors

        def curve(a):
            return base + (a * bump)[:, None] * d
        if arc_length(base) < float(L):
            lo, hi = 0.0, 0.5
            while arc_length(curve(hi)) < float(L) and hi < 5:
                hi *= 2
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                if arc_length(curve(mid)) < float(L):
                    lo = mid
                else:
                    hi = mid
            base = curve(0.5 * (lo + hi))
    return _resample(base, n)


def tube(P, radius: float, sides: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Triangle mesh of a tube around a polyline (parallel-transport frames, open ends)."""
    P = np.asarray(P, float)
    t = np.gradient(P, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    n0 = np.cross(t[0], [0, 0, 1.0] if abs(t[0][2]) < 0.9 else [1.0, 0, 0])
    n0 /= np.linalg.norm(n0)
    N = [n0]
    for i in range(1, len(P)):
        v = N[-1] - np.dot(N[-1], t[i]) * t[i]
        N.append(v / (np.linalg.norm(v) + 1e-12))
    N = np.asarray(N)
    B = np.cross(t, N)
    a = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    V = (P[:, None] + radius * (np.cos(a)[None, :, None] * N[:, None] + np.sin(a)[None, :, None] * B[:, None])).reshape(-1, 3)
    F = []
    for i in range(len(P) - 1):
        for j in range(sides):
            a0, a1 = i * sides + j, i * sides + (j + 1) % sides
            b0, b1 = a0 + sides, a1 + sides
            F += [[a0, b0, a1], [a1, b0, b1]]
    return V, np.asarray(F, int)


def all_centerlines(profile: dict, link_T: dict, down=(0, 0, -1), n: int = 48) -> dict:
    """{cable name: (n, 3)} for every cable of a bundle profile (link_T world or base poses; robots stand upright, so
    gravity is -z in both)."""
    return {name: centerline(spec, link_T, down, n) for name, spec in (profile.get("cables") or {}).items()}
