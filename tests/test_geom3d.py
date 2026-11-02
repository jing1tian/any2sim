import numpy as np

from any2sim.geom3d import (apply, cv_to_gl, fit_plane_ransac, hom, interpolate_poses, inv, mat_to_quat, pixel_rays,
                            project, quat_to_mat, ray_plane)
from any2sim.models.raster import clip_matrix


def _rand_T(rng):
    from scipy.spatial.transform import Rotation
    return hom(Rotation.random(random_state=rng.integers(1e9)).as_matrix(), rng.normal(size=3))


def test_inverse_and_apply_roundtrip():
    rng = np.random.default_rng(0)
    T = _rand_T(rng)
    p = rng.normal(size=(10, 3))
    assert np.allclose(apply(inv(T), apply(T, p)), p)


def test_project_and_back_project():
    K = np.array([[900.0, 0, 640], [0, 900, 360], [0, 0, 1]])
    T_cam_world = hom(None, [0, 0, 2.0])
    p = np.array([[0.1, -0.2, 0.0], [0.0, 0.0, 0.5]])
    uv, z = project(K, T_cam_world, p)
    rays = pixel_rays(K, uv)
    assert np.allclose(rays * z[:, None], apply(T_cam_world, p))


def test_plane_fit_and_ray_intersection():
    rng = np.random.default_rng(1)
    n_true = np.array([0.1, -0.2, 1.0])
    n_true /= np.linalg.norm(n_true)
    xy = rng.uniform(-1, 1, (500, 2))
    pts = np.c_[xy, -(n_true[0] * xy[:, 0] + n_true[1] * xy[:, 1] + 0.3) / n_true[2]]
    pts += rng.normal(0, 1e-4, pts.shape)
    pts = np.r_[pts, rng.uniform(-1, 1, (50, 3))]                 # outliers
    n, d, inl = fit_plane_ransac(pts, thresh=0.002)
    assert abs(abs(n @ n_true) - 1) < 1e-4 and inl.sum() >= 500
    hit = ray_plane(np.zeros((1, 3)), np.array([[0, 0, -1.0]]), n, d)
    assert abs(hit[0] @ n + d) < 1e-9


def test_quaternion_wxyz_roundtrip():
    rng = np.random.default_rng(2)
    R = _rand_T(rng)[:3, :3]
    q = mat_to_quat(R)
    assert np.allclose(quat_to_mat(q), R)
    assert np.allclose(mat_to_quat(np.eye(3)), [1, 0, 0, 0])


def test_interpolation_hits_keys_and_midpoint():
    T0, T1 = hom(None, [0, 0, 0]), hom(None, [1, 0, 0])
    out = interpolate_poses([0, 10], [T0, T1], [0, 5, 10, 12])
    assert np.allclose(out[1][:3, 3], [0.5, 0, 0]) and np.allclose(out[3], T1)


def test_cv_to_gl_camera_looks_along_minus_z():
    T = hom(None, [0, 0, 1.0])
    Tgl = cv_to_gl(T)
    fwd_cv = T[:3, :3] @ [0, 0, 1]
    fwd_gl = Tgl[:3, :3] @ [0, 0, -1]
    assert np.allclose(fwd_cv, fwd_gl)


def test_clip_matrix_matches_pinhole_projection():
    K = np.array([[800.0, 0, 600], [0, 810, 330], [0, 0, 1]])
    W, H = 1280, 720
    P = clip_matrix(K, W, H)
    pc = np.array([[0.1, -0.05, 1.2, 1.0]])
    c = pc @ P.T
    ndc = c[:, :2] / c[:, 3:]
    uv = (ndc + 1) / 2 * [W, H]
    u = K[0, 0] * pc[0, 0] / pc[0, 2] + K[0, 2]
    v = K[1, 1] * pc[0, 1] / pc[0, 2] + K[1, 2]
    assert np.allclose(uv[0], [u, v], atol=1e-3)
