"""SAM 3 (image, text prompts) and SAM 2.1 (video propagation) adapters. Runs in the `vision` env."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np


def _np(x):
    if hasattr(x, "detach"):
        x = x.detach()
        x = x.float() if "bfloat" in str(x.dtype) else x
        x = x.cpu().numpy()
    return np.asarray(x)


class Sam3:
    """SAM 3 image model loaded strictly from the cached facebook/sam3 checkpoint (no partial/random weights)."""

    def __init__(self, min_score: float = 0.3):
        import torch
        from huggingface_hub import try_to_load_from_cache
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
        ckpt = try_to_load_from_cache("facebook/sam3", "sam3.pt")
        if not isinstance(ckpt, str):
            raise FileNotFoundError("facebook/sam3 sam3.pt is not in the HF cache (models/huggingface)")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_sam3_image_model(checkpoint_path=None, load_from_HF=False, device=self.device)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        state = state.get("model", state)
        img = {k.removeprefix("detector."): v for k, v in state.items() if k.startswith("detector.")}
        expected = set(model.state_dict())
        bad = [k for k in set(img) - expected if not k.startswith("backbone.vision_backbone.sam2_convs.")]
        if bad:
            raise ValueError(f"unexpected SAM 3 weights: {sorted(bad)[:5]}")
        model.load_state_dict({k: v for k, v in img.items() if k in expected}, strict=True)
        model.eval()
        self.proc = Sam3Processor(model, device=self.device, confidence_threshold=min_score)
        self.identity = {"model": "facebook/sam3", "checkpoint": ckpt}

    def segment(self, rgb: np.ndarray, prompts: list[str]) -> dict[str, list[dict]]:
        """Instances per prompt: [{mask (H,W) bool, score, box [x0,y0,x1,y1]}], best score first."""
        import torch
        from PIL import Image
        H, W = rgb.shape[:2]
        out = {}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
            state = self.proc.set_image(Image.fromarray(rgb))
            for p in prompts:
                r = self.proc.set_text_prompt(prompt=p, state=state)
                m = r.get("masks")
                inst = []
                if m is not None and m.shape[0] > 0:
                    M = _np(m).astype(bool).reshape(m.shape[0], -1, *m.shape[-2:]).any(axis=1)
                    sc = _np(r["scores"]).reshape(-1).astype(float) if r.get("scores") is not None else np.ones(len(M))
                    for i in np.argsort(-sc):
                        mi = M[i]
                        if mi.shape != (H, W):
                            import cv2
                            mi = cv2.resize(mi.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
                        ys, xs = np.nonzero(mi)
                        if not len(xs):
                            continue
                        inst.append({"mask": mi, "score": float(sc[i]),
                                     "box": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]})
                out[p] = inst
        return out


def sam2_propagate(frame_files: list, seeds: list[dict], checkpoint: str,
                   model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml") -> np.ndarray:
    """Track objects through the video with SAM 2.1.

    seeds: [{"obj_id": k (1..), "frame": i, "mask": (H,W) bool}] (several seeds per object allowed, all used as
    prompts). Returns a (N, K, H, W) bool array of per-object masks, propagated forward and backward from the
    earliest seed and re-anchored at every seed frame.
    """
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    from ..common import read_rgb
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pred = build_sam2_video_predictor(model_cfg, checkpoint, device=dev)
    n = len(frame_files)
    H, W = read_rgb(frame_files[0]).shape[:2]
    K = max(s["obj_id"] for s in seeds)
    out = np.zeros((n, K, H, W), bool)
    tmp = tempfile.mkdtemp(prefix="sam2_", dir=os.environ.get("TMPDIR"))
    try:
        for i, f in enumerate(frame_files):
            os.symlink(os.path.abspath(f), os.path.join(tmp, f"{i:05d}.jpg"))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            state = pred.init_state(video_path=tmp, offload_video_to_cpu=True, offload_state_to_cpu=True)
            start = min(s["frame"] for s in seeds)
            for reverse in (False, True):
                if reverse and start == 0:
                    continue
                pred.reset_state(state)
                for s in seeds:
                    pred.add_new_mask(state, frame_idx=int(s["frame"]), obj_id=int(s["obj_id"]),
                                      mask=np.asarray(s["mask"], bool))
                for fi, ids, logits in pred.propagate_in_video(state, start_frame_idx=start, reverse=reverse):
                    for j, oid in enumerate(ids):
                        m = (logits[j, 0] > 0).float().cpu().numpy() > 0
                        if m.shape != (H, W):
                            import cv2
                            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
                        out[fi, int(oid) - 1] = m
    finally:
        for f in os.listdir(tmp):
            os.unlink(os.path.join(tmp, f))
        os.rmdir(tmp)
    return out
