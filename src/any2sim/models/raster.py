"""Differentiable rasterisation with nvdiffrast for OpenCV pinhole cameras. Runs in the `sam3d` env."""
from __future__ import annotations

import numpy as np


def clip_matrix(K, W, H, near=0.02, far=20.0):
    """4x4 matrix taking OpenCV camera points (x right, y down, z forward) to nvdiffrast clip space, with image
    row 0 at the top (nvdiffrast's row 0 is ndc y = -1)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    a = (far + near) / (far - near)
    b = -2 * far * near / (far - near)
    return np.array([[2 * fx / W, 0, 2 * cx / W - 1, 0],
                     [0, 2 * fy / H, 2 * cy / H - 1, 0],
                     [0, 0, a, b],
                     [0, 0, 1, 0]], np.float32)


class Rasterizer:
    def __init__(self):
        import nvdiffrast.torch as dr
        self.dr = dr
        self.ctx = dr.RasterizeCudaContext()

    def _concat(self, V_list, F_list):
        import torch
        dev = V_list[0].device
        Vs, Fs, tri_obj, off = [], [], [], 0
        for k, (V, F) in enumerate(zip(V_list, F_list)):
            Vs.append(V)
            Fs.append(F + off)
            tri_obj.append(torch.full((len(F),), k, dtype=torch.int64, device=dev))
            off += len(V)
        return torch.cat(Vs), torch.cat(Fs).int().contiguous(), torch.cat(tri_obj)

    def render_ids(self, V_cam_list, F_list, K, W, H):
        """Several meshes (camera-frame vertices, torch (N,3)) with depth testing. Returns per-object antialiased
        visible silhouettes (M,H,W) and the camera-z depth map (H,W, 0 = empty)."""
        import torch
        dr = self.dr
        dev = V_cam_list[0].device
        P = torch.from_numpy(clip_matrix(K, W, H)).to(dev)
        V, F, tri_obj = self._concat(V_cam_list, F_list)
        pos = (torch.cat([V, torch.ones_like(V[:, :1])], 1) @ P.T)[None].contiguous()
        rast, _ = dr.rasterize(self.ctx, pos, F, resolution=[H, W])
        tid = rast[0, ..., 3].long()
        hit = tid > 0
        obj = torch.full((H, W), -1, dtype=torch.int64, device=dev)
        obj[hit] = tri_obj[tid[hit] - 1]
        M = len(V_cam_list)
        onehot = torch.stack([(obj == k).float() for k in range(M)], -1)[None]
        sil = dr.antialias(onehot.contiguous(), rast, pos, F)[0].permute(2, 0, 1)
        zc, _ = dr.interpolate(V[None, :, 2:3].contiguous(), rast, F)
        return sil, zc[0, ..., 0] * hit.float()

    def render_full(self, V_cam_list, F_list, K, W, H):
        """Non-differentiable: object id map (H,W; -1 empty), depth (H,W), camera-frame normals (H,W,3),
        triangle id (H,W; -1 empty)."""
        import torch
        dr = self.dr
        dev = V_cam_list[0].device
        P = torch.from_numpy(clip_matrix(K, W, H)).to(dev)
        V, F, tri_obj = self._concat(V_cam_list, F_list)
        pos = (torch.cat([V, torch.ones_like(V[:, :1])], 1) @ P.T)[None].contiguous()
        with torch.no_grad():
            rast, _ = dr.rasterize(self.ctx, pos, F, resolution=[H, W])
            tid = rast[0, ..., 3].long()
            hit = tid > 0
            obj = torch.full((H, W), -1, dtype=torch.int64, device=dev)
            obj[hit] = tri_obj[tid[hit] - 1]
            zc, _ = dr.interpolate(V[None, :, 2:3].contiguous(), rast, F)
            Fl = F.long()
            fn = torch.cross(V[Fl[:, 1]] - V[Fl[:, 0]], V[Fl[:, 2]] - V[Fl[:, 0]], dim=1)
            fn = torch.nn.functional.normalize(fn, dim=1)
            nrm = torch.zeros((H, W, 3), device=dev)
            nrm[hit] = fn[tid[hit] - 1]
            tri = torch.full((H, W), -1, dtype=torch.int64, device=dev)
            tri[hit] = tid[hit] - 1
        return obj.cpu().numpy(), (zc[0, ..., 0] * hit.float()).cpu().numpy(), nrm.cpu().numpy(), tri.cpu().numpy()

    def rasterize_uv(self, V, F, UV, FUV, W, H, attrs):
        """Rasterise a mesh in texture space: returns texel coverage (H,W) and interpolated per-vertex attributes
        (list of (H,W,C)), texture row 0 = v = 1 (image file convention). V/attrs indexed by F, UV by FUV."""
        import torch
        dr = self.dr
        dev = V.device
        uv = torch.as_tensor(UV, dtype=torch.float32, device=dev)
        pos = torch.cat([uv * 2 - 1, torch.zeros_like(uv[:, :1]), torch.ones_like(uv[:, :1])], 1)[None].contiguous()
        fuv = torch.as_tensor(FUV, dtype=torch.int32, device=dev).contiguous()
        with torch.no_grad():
            rast, _ = dr.rasterize(self.ctx, pos, fuv, resolution=[H, W])
            cov = (rast[0, ..., 3] > 0).cpu().numpy()[::-1]
            outs = []
            for a in attrs:
                # attributes are per mesh vertex; re-index them to the UV vertices through the face lists
                av = torch.zeros((len(uv), a.shape[1]), device=dev)
                av[fuv.long().reshape(-1)] = a[torch.as_tensor(F, device=dev).long().reshape(-1)]
                x, _ = dr.interpolate(av[None].contiguous(), rast, fuv)
                outs.append(x[0].cpu().numpy()[::-1])
        return cov, outs
