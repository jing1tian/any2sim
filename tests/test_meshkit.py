import numpy as np
import pytest

from any2sim import meshkit as mk


def test_save_obj_writes_welded_smooth_normals(tmp_path):
    m = mk.uv_unwrap(mk.cylinder(0.05, 0.1, sections=24))      # UV seams split vertices
    p = mk.save_obj(m, tmp_path / "c.obj")
    txt = open(p).read().splitlines()
    vn = np.array([[float(x) for x in l.split()[1:]] for l in txt if l.startswith("vn ")])
    v = np.array([[float(x) for x in l.split()[1:]] for l in txt if l.startswith("v ")])
    assert len(vn) == len(v) and np.allclose(np.linalg.norm(vn, axis=1), 1, atol=1e-3)
    # vertices that share a position (split by the UV seam) get the same normal
    key = np.unique(np.round(v, 6), axis=0, return_inverse=True)[1].reshape(-1)
    for k in np.unique(key):
        assert np.allclose(vn[key == k], vn[key == k][0], atol=1e-4)
    assert any(l.startswith("f ") and l.count("/") == 6 for l in txt)


def test_save_obj_crease_and_none(tmp_path):
    b = mk.box([0.1, 0.1, 0.1])
    txt = open(mk.save_obj(b, tmp_path / "b.obj", normals="crease", crease_deg=45)).read()
    vn = np.array([[float(x) for x in l.split()[1:]] for l in txt.splitlines() if l.startswith("vn ")])
    axes = np.abs(vn).max(1)
    assert np.allclose(axes, 1, atol=1e-4)                     # box corners keep face normals (hard edges)
    assert "vn " not in open(mk.save_obj(b, tmp_path / "n.obj", normals="none")).read()


def test_synthesize_texture_rejects_0_255_mean(tmp_path):
    from any2sim.project import Project
    from any2sim.tools import synthesize_texture as st
    import cv2
    proj = Project.create(tmp_path / "p", {})
    cv2.imwrite(str(proj.root / "ex.png"), (np.random.default_rng(0).random((120, 120, 3)) * 255).astype(np.uint8))
    with pytest.raises(ValueError, match="0-1"):
        st.run(proj.root, exemplar="ex.png", size=[64, 64], patch=24, overlap=6, match_mean=[120, 130, 140])
    r = st.run(proj.root, exemplar="ex.png", size=[64, 64], patch=24, overlap=6, match_mean=[0.5, 0.5, 0.5])
    assert abs(np.mean(r["mean_srgb"]) - 0.5) < 0.02
