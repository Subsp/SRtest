"""
HRNVS geometry prior core (training hook).

Provides a minimal PriorPackDepth and L_geom-style depth supervision:
median-scale alignment between predicted (e.g. splat-render) depth and a prior proxy
(VGGT / HR-head depth, or oracle depth in tests), optionally confidence-weighted.

Scale s is computed in no_grad using the same median rule as Phase-0 AbsRel utilities
(utils.metrics.abs_rel align_scale branch): medians over valid positive pixels only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class PriorPackDepth:
    """Depth (+ optional confidence). Normals slot reserved for extensions."""

    depth: torch.Tensor  # (... H W)
    confidence: Optional[torch.Tensor] = None
    normal_world: Optional[torch.Tensor] = None  # reserved; not used in L_geom v0


def validity_mask(depth: torch.Tensor, min_depth: float = 1e-6) -> torch.Tensor:
    """Broadcast-safe finite positive mask matching depth[..., H, W]."""
    return torch.isfinite(depth) & (depth > min_depth)


def median_scale_align_factor(
    pred: torch.Tensor,
    ref: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Scalar s with median(s * pred) ~= median(ref) on masked pixels.
    Runs without autograd on the statistic (recommended for stabilising L_geom).

    pred, ref : same shape, last two dims spatial
    mask      : broadcastable bool mask
    """
    pred_flat = torch.masked_select(pred, mask)
    ref_flat = torch.masked_select(ref, mask)
    if pred_flat.numel() == 0:
        return pred.new_tensor(1.0)
    med_p = torch.median(pred_flat)
    med_r = torch.median(ref_flat)
    return (med_r / med_p.clamp(min=1e-12)).detach()


def geom_depth_loss_l1(
    pred_depth: torch.Tensor,
    prior_pack: PriorPackDepth,
    extra_mask: Optional[torch.Tensor] = None,
    min_depth: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Confidence-weighted L1 after median alignment of pred to prior depth:

        s = median(prior valid) / median(pred valid)
        diff = |s * pred - prior|

    prior is treated as supervision target (no gradient).
    Confidence multiplies pixel losses (default uniform if missing).

    pred_depth shape must match prior depth spatially.
    Returns (scalar loss, diagnostics dict with detached scalars/tensors useful for logs).
    """
    prior_d = prior_pack.depth.to(dtype=pred_depth.dtype, device=pred_depth.device)
    valid = validity_mask(pred_depth, min_depth) & validity_mask(prior_d, min_depth)
    if extra_mask is not None:
        valid = valid & extra_mask.to(device=pred_depth.device, dtype=torch.bool)

    if prior_pack.confidence is not None:
        w = prior_pack.confidence.to(dtype=pred_depth.dtype, device=pred_depth.device).clamp(min=0.0)
    else:
        w = torch.ones_like(prior_d)

    need = tuple(pred_depth.shape)
    need_p = tuple(prior_d.shape)
    if need != need_p:
        raise ValueError(f"Spatial mismatch pred {need_p} vs prior {need}")

    with torch.no_grad():
        scale = median_scale_align_factor(pred_depth, prior_d, valid)

    aligned = pred_depth * scale
    diff = torch.abs(aligned - prior_d.detach())

    wv = w * valid.to(dtype=w.dtype)
    denom = wv.sum().clamp(min=1.0)
    loss = (diff * wv).sum() / denom

    diag: Dict[str, torch.Tensor] = {
        "scale": scale.detach(),
        "valid_ratio": valid.to(dtype=torch.float32).mean().detach(),
        "weighted_pixels": denom.detach(),
    }
    return loss, diag


def prior_pack_depth_from_numpy(
    depth_np,
    confidence_np=None,
    device: str | torch.device = "cpu",
    dtype=torch.float32,
) -> PriorPackDepth:
    """Build PriorPackDepth from NumPy arrays (H,W)."""
    d = torch.from_numpy(depth_np).to(device=device, dtype=dtype)
    if d.ndim != 2:
        raise ValueError(f"Expected (H,W) depth, got {tuple(d.shape)}")
    conf = None
    if confidence_np is not None:
        conf = torch.from_numpy(confidence_np).to(device=device, dtype=dtype)
        if tuple(conf.shape) != tuple(d.shape):
            raise ValueError("Confidence shape must match depth")
    return PriorPackDepth(depth=d, confidence=conf)
