"""Rendering utility functions used by routed training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def downsample_to_match(render_hr: torch.Tensor, target_lr: torch.Tensor, mode: str = "area") -> torch.Tensor:
    """Downsample a CHW render to the spatial size of a CHW target image."""

    if render_hr.ndim != 3 or target_lr.ndim != 3:
        raise ValueError(
            f"expected CHW tensors, got render={tuple(render_hr.shape)} target={tuple(target_lr.shape)}"
        )
    target_size = tuple(int(v) for v in target_lr.shape[-2:])
    if tuple(render_hr.shape[-2:]) == target_size:
        return render_hr
    return F.interpolate(render_hr.unsqueeze(0), size=target_size, mode=mode).squeeze(0)


def l1_downsampled(render_hr: torch.Tensor, target_lr: torch.Tensor) -> torch.Tensor:
    return (downsample_to_match(render_hr, target_lr) - target_lr).abs().mean()
