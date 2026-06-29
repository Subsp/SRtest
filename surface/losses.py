"""Thin surface-loss wrapper for SP-IE-SRGS v0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from hybrid_sdfgs.blocks import SOFRegularizationBlock, SOFRegularizationConfig


def surface_ramp(iteration: int, start_iter: int, end_iter: int) -> float:
    if end_iter <= start_iter:
        return 1.0 if iteration >= start_iter else 0.0
    if iteration < start_iter:
        return 0.0
    if iteration >= end_iter:
        return 1.0
    return float(iteration - start_iter) / float(end_iter - start_iter)


@dataclass(frozen=True)
class SPV0SurfaceConfig:
    lambda_surface: float = 1.0
    lambda_distortion: float = 1000.0
    lambda_depth_normal: float = 0.05
    lambda_smoothness: float = 0.01
    ramp_start_iter: int = 1000
    ramp_end_iter: int = 5000
    distortion_from_iter: int = 0
    depth_normal_from_iter: int = 0


class SPV0SurfaceLoss:
    """Surface loss with v0-only terms and an outer ramp."""

    def __init__(self, cfg: SPV0SurfaceConfig):
        self.cfg = cfg

    def _make_block(self, scale: float) -> SOFRegularizationBlock:
        cfg = SOFRegularizationConfig(
            lambda_distortion=float(self.cfg.lambda_distortion) * scale,
            lambda_depth_normal=float(self.cfg.lambda_depth_normal) * scale,
            lambda_smoothness=float(self.cfg.lambda_smoothness) * scale,
            distortion_from_iter=int(self.cfg.distortion_from_iter),
            depth_normal_from_iter=int(self.cfg.depth_normal_from_iter),
        )
        return SOFRegularizationBlock(cfg)

    def compute(
        self,
        *,
        gaussians,
        iteration: int,
        render_ctx: Dict[str, object],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        ramp = surface_ramp(iteration, self.cfg.ramp_start_iter, self.cfg.ramp_end_iter)
        scale = float(self.cfg.lambda_surface) * ramp
        metrics = {
            "surface_ramp": ramp,
            "surface_weight": scale,
        }
        if scale <= 0.0:
            return None, metrics
        loss, loss_metrics = self._make_block(scale).compute(
            gaussians,
            iteration,
            render_ctx=render_ctx,
        )
        metrics.update({f"surface_{key}": float(value) for key, value in loss_metrics.items()})
        return loss, metrics
