"""Mesh-building helpers for agent-written geometry code (CPU; trimesh + manifold3d + xatlas).

Typical use in a project script (projects/<id>/work/code/<object>.py):

    from any2sim import meshkit as mk
    def build(p):                                   # p: dict of parameters (metres)
        body = mk.lathe([(0, 0), (p["r_bottom"], 0), (p["r_top"], p["h"])], wall=p["wall"])
        slots = mk.radial_array(mk.box([p["slot_w"], 0.02, p["slot_h"]]), n=p["n"], radius=p["r_mid"], z=p["z0"])
        return mk.normalize(mk.difference(body, slots))

Conventions: object frame z up, origin at the bottom centre (normalize()), metres. A `build(params) -> Trimesh`
function in a module can be fitted to the video with the `fit_render` tool (builder mode).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


def _manifold(m: trimesh.Trimesh):
    import manifold3d as m3
    return m3.Manifold(m3.Mesh(vert_properties=np.asarray(m.vertices, np.float32),
                                tri_verts=np.asarray(m.faces, np.uint32)))


def _trimesh(man) -> trimesh.Trimesh:
    mesh = man.to_mesh()
    return trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=True)


def union(*meshes) -> trimesh.Trimesh:
    out = _manifold(meshes[0])
    for m in meshes[1:]:
        out = out + _manifold(m)
    return _trimesh(out)


def difference(a, *cutters) -> trimesh.Trimesh:
    out = _manifold(a)
    for c in cutters:
        out = out - _manifold(c)
    return _trimesh(out)


def intersection(a, b) -> trimesh.Trimesh:
    return _trimesh(_manifold(a) ^ _manifold(b))


def box(extents, center=(0, 0, 0)) -> trimesh.Trimesh:
    return trimesh.creation.box(extents=extents, transform=trimesh.transformations.translation_matrix(center))


def cylinder(radius, height, sections=64, center=(0, 0, 0)) -> trimesh.Trimesh:
    return trimesh.creation.cylinder(radius=radius, height=height, sections=sections,
                                     transform=trimesh.transformations.translation_matrix(center))


def lathe(profile, sections=96, wall: float | None = None) -> trimesh.Trimesh:
    """Surface of revolution about z from a profile [(r, z), ...] listed bottom -> top along the outer surface.
    wall=None: solid of revolution (profile closed to the axis). wall=t: open-top vessel with wall thickness t
    (inner surface offset inward/upward by t, closed bottom of thickness t)."""
    prof = np.asarray(profile, float)
    if wall is None:
        pts = np.r_[[[0.0, prof[0, 1]]], prof, [[0.0, prof[-1, 1]]]]
    else:
        inner = []
        for k in range(len(prof) - 1, -1, -1):
            r, z = prof[k]
            inner.append([max(r - wall, 1e-4), max(z, prof[0, 1] + wall)])
        pts = np.r_[[[0.0, prof[0, 1]]], prof, inner, [[0.0, prof[0, 1] + wall]]]
    # revolve a closed polygon (r, z) into a watertight mesh
    n = len(pts)
    ang = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    V = np.array([[r * np.cos(a), r * np.sin(a), z] for a in ang for r, z in pts])
    F = []
    for i in range(sections):
        j = (i + 1) % sections
        for k in range(n):
            a, b = i * n + k, i * n + (k + 1) % n
            c, d = j * n + (k + 1) % n, j * n + k
            F += [[a, b, c], [a, c, d]]
    m = trimesh.Trimesh(V, np.asarray(F), process=True)
    m.merge_vertices()
    trimesh.repair.fix_normals(m)
    return m


def radial_array(m: trimesh.Trimesh, n: int, radius: float, z: float = 0.0, phase: float = 0.0,
                 tangential: bool = True) -> trimesh.Trimesh:
    """n copies of m placed around the z axis at `radius`, height z; tangential=True rotates each copy to face out."""
    parts = []
    for k in range(n):
        a = phase + 2 * np.pi * k / n
        T = trimesh.transformations.rotation_matrix(a if tangential else 0.0, [0, 0, 1])
        T[:3, 3] = [radius * np.cos(a), radius * np.sin(a), z]
        parts.append(m.copy().apply_transform(T))
    return trimesh.util.concatenate(parts)


def grid_array(m: trimesh.Trimesh, nx: int, ny: int, pitch_x: float, pitch_y: float, z: float = 0.0) -> trimesh.Trimesh:
    parts = []
    for i in range(nx):
        for j in range(ny):
            parts.append(m.copy().apply_translation([(i - (nx - 1) / 2) * pitch_x, (j - (ny - 1) / 2) * pitch_y, z]))
    return trimesh.util.concatenate(parts)


def transform(m: trimesh.Trimesh, T) -> trimesh.Trimesh:
    return m.copy().apply_transform(np.asarray(T, float))


def normalize(m: trimesh.Trimesh) -> trimesh.Trimesh:
    """Origin at the bottom centre of the bounding box (object frame convention)."""
    lo, hi = m.bounds
    return m.copy().apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])


def uv_unwrap(m: trimesh.Trimesh) -> trimesh.Trimesh:
    """Atlas UVs (xatlas); vertices are split along seams."""
    import xatlas
    vmap, faces, uvs = xatlas.parametrize(np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.uint32))
    out = trimesh.Trimesh(np.asarray(m.vertices)[vmap], faces, process=False)
    out.visual = trimesh.visual.TextureVisuals(uv=uvs)
    return out


def uv_planar_metres(m: trimesh.Trimesh, axes=(0, 1)) -> trimesh.Trimesh:
    """UVs = vertex coordinates (metres) on two axes: for tiled materials (Scene IR material.tile_m)."""
    out = m.copy()
    out.visual = trimesh.visual.TextureVisuals(uv=np.asarray(m.vertices)[:, list(axes)])
    return out


def smooth_normals(m: trimesh.Trimesh, crease_deg: float | None = None) -> np.ndarray:
    """Area-weighted vertex normals, welded across vertices that share a position (UV seams), so seams do not
    shade. crease_deg: faces meeting at a sharper angle keep separate normals (None: smooth everywhere)."""
    V = np.asarray(m.vertices)
    F = np.asarray(m.faces)
    key = np.unique(np.round(V, 7), axis=0, return_inverse=True)[1].reshape(-1)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])        # length = 2 * area
    acc = np.zeros((key.max() + 1, 3))
    for k in range(3):
        np.add.at(acc, key[F[:, k]], fn)
    n = acc[key]
    if crease_deg is not None:
        # per corner: average only the faces around the vertex within crease_deg of this corner's face
        fu = fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12)
        cos_lim = np.cos(np.radians(crease_deg))
        corners_v = key[F].reshape(-1)                     # welded vertex of each corner
        corners_f = np.repeat(np.arange(len(F)), 3)
        order = np.argsort(corners_v, kind="stable")
        out = np.zeros((len(corners_v), 3))
        bounds = np.flatnonzero(np.diff(corners_v[order])) + 1
        for grp in np.split(order, bounds):
            faces = corners_f[grp]
            near = (fu[faces] @ fu[faces].T) >= cos_lim       # which incident faces are smooth-connected
            out[grp] = near.astype(float) @ fn[faces]
        out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
        return out.reshape(-1, 3, 3)                        # (F, 3, 3) per-corner normals
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def save_obj(m: trimesh.Trimesh, path: str | Path, texture: str | None = None, normals: str = "smooth",
             crease_deg: float = 60.0) -> str:
    """OBJ (+MTL) with UVs when present and vertex normals; `texture` is written as map_Kd (file name relative to
    the OBJ). normals: "smooth" (welded across UV seams), "crease" (smooth below crease_deg, hard edges above),
    "none" (renderers then shade flat)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    uv = getattr(m.visual, "uv", None)
    has_uv = uv is not None and len(uv) == len(m.vertices)
    F = np.asarray(m.faces)
    lines = [f"mtllib {path.stem}.mtl", "usemtl surface"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in m.vertices]
    if has_uv:
        lines += [f"vt {u:.6f} {v:.6f}" for u, v in uv]
    if normals == "none":
        if has_uv:
            lines += [f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}" for a, b, c in F]
        else:
            lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in F]
    else:
        if normals == "crease":
            cn = smooth_normals(m, crease_deg).reshape(-1, 3)              # one normal per face corner
            nidx = np.arange(len(cn)).reshape(-1, 3)
        else:
            cn = smooth_normals(m)                                        # one normal per vertex
            nidx = F
        lines += [f"vn {x:.5f} {y:.5f} {z:.5f}" for x, y, z in cn]
        t = (lambda i: f"{i + 1}") if has_uv else (lambda i: "")
        lines += ["f " + " ".join(f"{F[k, j] + 1}/{t(F[k, j])}/{nidx[k, j] + 1}" for j in range(3)) for k in range(len(F))]
    path.write_text("\n".join(lines) + "\n")
    path.with_suffix(".mtl").write_text("newmtl surface\nKd 1 1 1\n" + (f"map_Kd {texture}\n" if texture else ""))
    return str(path)


def load(path: str | Path) -> trimesh.Trimesh:
    return trimesh.load(path, force="mesh", process=False)


def stats(m: trimesh.Trimesh) -> dict:
    lo, hi = m.bounds
    return {"vertices": len(m.vertices), "faces": len(m.faces), "extent_m": (hi - lo).tolist(),
            "watertight": bool(m.is_watertight), "volume_m3": float(m.volume) if m.is_volume else None}
