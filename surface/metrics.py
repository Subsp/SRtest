"""Lightweight geometry proxy metrics for SP-IE-SRGS v0."""

from __future__ import annotations

from typing import Dict, Mapping

import torch


def render_proxy_metrics(render_pkg: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    distortion = render_pkg.get("distortion")
    if distortion is not None:
        metrics["mean_depth_distortion"] = float(distortion.detach().mean().item())
    alpha = render_pkg.get("alpha")
    if alpha is not None:
        metrics["mean_alpha"] = float(alpha.detach().mean().item())
    normal = render_pkg.get("normal")
    if normal is not None and normal.ndim == 3:
        dx = (normal[:, :, 1:] - normal[:, :, :-1]).abs().mean()
        dy = (normal[:, 1:, :] - normal[:, :-1, :]).abs().mean()
        metrics["normal_smoothness_proxy"] = float((dx + dy).detach().item())
    depth = render_pkg.get("depth")
    if depth is not None and depth.ndim == 3:
        dx = (depth[:, :, 1:] - depth[:, :, :-1]).abs().mean()
        dy = (depth[:, 1:, :] - depth[:, :-1, :]).abs().mean()
        metrics["depth_tv_proxy"] = float((dx + dy).detach().item())
    return metrics
