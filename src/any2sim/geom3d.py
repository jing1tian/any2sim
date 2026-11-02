"""Geometry helpers (numpy only): transforms, projection, planes, rotations, interpolation.

Conventions (shared by every tool and backend):
  * right-handed, metres; world = table frame, +z up (table normal), origin on the table plane
  * T_a_b maps points from frame b into frame a (4x4)
  * cameras use the OpenCV optical frame (x right, y down, z forward); backends convert (Blender: x right,
    y up, -z forward) through cv_to_gl()
  * quaternions are wxyz
"""
from __future__ import annotations

import numpy as np


def hom(R=None, t=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = np.asarray(t).reshape(3)
    return T


def apply(T, p) -> np.ndarray:
    p = np.asarray(p, float)
    return p @ T[:3, :3].T + T[:3, 3]


def inv(T) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return hom(R.T, -R.T @ t)


def project(K, T_cam_world, p_world) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates (N,2) and depths (N,) of world points in a camera (OpenCV)."""
    pc = apply(T_cam_world, p_world)
    z = pc[:, 2]
    uv = (pc[:, :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)[:, None]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    return uv, z


def pixel_rays(K, uv) -> np.ndarray:
    """Unit-z rays (N,3) in the camera frame through pixels uv (N,2)."""
    uv = np.asarray(uv, float)
    return np.stack([(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(uv))], -1)


def ray_plane(origins, dirs, n, d) -> np.ndarray:
    """Intersections of rays o + t*v with plane n.x + d = 0 (t may be negative: caller checks)."""
    t = -(origins @ n + d) / (dirs @ n)
    return origins + t[:, None] * dirs


def fit_plane_ransac(P, thresh=0.005, iters=2000, seed=0) -> tuple[np.ndarray, float, np.ndarray]:
    """Plane n.x + d = 0 (|n| = 1) through 3-D points; returns n, d, inlier mask (refit by SVD on inliers)."""
    rng = np.random.default_rng(seed)
    best = None
    N = len(P)
    for _ in range(iters):
        a, b, c = P[rng.choice(N, 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        d = -n @ a
        inl = np.abs(P @ n + d) < thresh
        if best is None or inl.sum() > best[2].sum():
            best = (n, d, inl)
    n, d, inl = best
    Q = P[inl]
    c = Q.mean(0)
    n = np.linalg.eigh(np.cov((Q - c).T))[1][:, 0]       # smallest-eigenvalue direction (3x3, no N x N SVD)
    d = -n @ c
    inl = np.abs(P @ n + d) < thresh
    return n, float(d), inl


def rot_angle_deg(R) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def mat_to_quat(R) -> np.ndarray:
    """wxyz"""
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def quat_to_mat(q) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    w, x, y, z = q
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def interpolate_poses(key_ids, key_T, all_ids) -> np.ndarray:
    """Slerp rotations / lerp translations of 4x4 poses at key_ids onto all_ids (clamped at the ends)."""
    from scipy.spatial.transform import Rotation, Slerp
    key_ids = np.asarray(key_ids, float)
    key_T = np.asarray(key_T)
    q = np.clip(np.asarray(all_ids, float), key_ids[0], key_ids[-1])
    R = Slerp(key_ids, Rotation.from_matrix(key_T[:, :3, :3]))(q).as_matrix()
    t = np.stack([np.interp(q, key_ids, key_T[:, k, 3]) for k in range(3)], -1)
    out = np.tile(np.eye(4), (len(q), 1, 1))
    out[:, :3, :3] = R
    out[:, :3, 3] = t
    return out


CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def cv_to_gl(T_world_cvcam) -> np.ndarray:
    """Camera pose with an OpenCV optical frame -> the same camera with an OpenGL/Blender frame."""
    return T_world_cvcam @ CV_TO_GL
