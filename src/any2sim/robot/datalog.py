"""Tabular robot/hand logs: loading (npz, npy + .meta.json sidecar, csv, json) and rotation conversions.

load_table(path) -> {column: (T, d) float array}. Column specs used by tools:
  "name"                 whole column
  {"column": "name", "slice": [a, b]}   sub-range of a column
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_table(path: str | Path) -> dict:
    p = Path(path)
    if p.suffix == ".npz":
        d = np.load(p)
        return {k: np.asarray(d[k], float).reshape(len(d[k]), -1) for k in d.files}
    if p.suffix == ".npy":
        meta = p.with_name(p.name + ".meta.json")
        a = np.load(p)
        if meta.exists():
            m = json.loads(meta.read_text())
            return {c["raw_key"]: np.asarray(a[:, c["start"]:c["end"]], float) for c in m["columns"]}
        return {"data": np.asarray(a, float).reshape(len(a), -1)}
    if p.suffix == ".csv":
        import csv
        rows = list(csv.DictReader(p.open()))
        return {k: np.array([[float(r[k])] for r in rows]) for k in rows[0]}
    if p.suffix == ".json":
        d = json.loads(p.read_text())
        if isinstance(d, dict):
            return {k: np.asarray(v, float).reshape(len(v), -1) for k, v in d.items() if isinstance(v, list)}
        keys = [k for k, v in d[0].items() if isinstance(v, (int, float, list))]
        return {k: np.asarray([r[k] for r in d], float).reshape(len(d), -1) for k in keys}
    raise ValueError(f"unsupported log format {p.suffix}")


def column(table: dict, spec) -> np.ndarray:
    if isinstance(spec, str):
        return table[spec]
    a = table[spec["column"]]
    if spec.get("slice"):
        a = a[:, spec["slice"][0]:spec["slice"][1]]
    return a


def _rot6d(a, rows: bool):
    a = np.asarray(a, float)
    if rows:
        x, y = a[:, 0:3], a[:, 3:6]
    else:
        x, y = a[:, 0::2], a[:, 1::2]
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    y = y - (x * y).sum(1, keepdims=True) * x
    y = y / np.linalg.norm(y, axis=1, keepdims=True)
    z = np.cross(x, y)
    if rows:
        return np.stack([x, y, z], 1)              # x, y, z are the matrix rows
    return np.stack([x, y, z], 2)                  # columns


ROT_FORMATS = ("rotvec", "euler_xyz", "euler_zyx", "euler_XYZ", "euler_ZYX", "quat_wxyz", "quat_xyzw",
               "rot6d_cols", "rot6d_rows", "rot6d_colsinterleaved", "matrix")


def to_matrix(a, fmt: str) -> np.ndarray:
    """(T, 3, 3) rotation matrices from a (T, d) array. rot6d_cols: first two columns of R stored as
    [c0, c1] (6 values); rot6d_rows: first two rows; rot6d_colsinterleaved: [r00, r01, r10, r11, r20, r21]."""
    from scipy.spatial.transform import Rotation as Rot
    a = np.asarray(a, float)
    if fmt == "rotvec":
        return Rot.from_rotvec(a).as_matrix()
    if fmt.startswith("euler_"):
        return Rot.from_euler(fmt.split("_")[1], a).as_matrix()
    if fmt == "quat_wxyz":
        return Rot.from_quat(a[:, [1, 2, 3, 0]]).as_matrix()
    if fmt == "quat_xyzw":
        return Rot.from_quat(a).as_matrix()
    if fmt == "rot6d_cols":
        x, y = a[:, 0:3], a[:, 3:6]
        return _rot6d(np.concatenate([x, y], 1), rows=True).transpose(0, 2, 1)
    if fmt == "rot6d_rows":
        return _rot6d(a, rows=True)
    if fmt == "rot6d_colsinterleaved":
        return _rot6d(a, rows=False)
    if fmt == "matrix":
        return a.reshape(-1, 3, 3)
    raise ValueError(fmt)


def candidate_formats(dim: int) -> list[str]:
    return {3: ["rotvec", "euler_xyz", "euler_zyx", "euler_XYZ", "euler_ZYX"], 4: ["quat_wxyz", "quat_xyzw"],
            6: ["rot6d_cols", "rot6d_rows", "rot6d_colsinterleaved"], 9: ["matrix"]}[dim]


def poses(pos, rot, fmt) -> np.ndarray:
    T = np.tile(np.eye(4), (len(pos), 1, 1))
    T[:, :3, :3] = to_matrix(rot, fmt)
    T[:, :3, 3] = pos
    return T


def joint_series(table: dict, joints: list) -> dict:
    """{joint: (T,) values} from specs [{column, slice?, joints: [names | null]}] (null skips a value)."""
    out = {}
    for spec in joints:
        a = column(table, spec)
        for k, jn in enumerate(spec["joints"]):
            if jn is not None:
                out[jn] = a[:, k]
    return out
