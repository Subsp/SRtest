"""Gradient auditing helpers for SP-IE-SRGS routing."""

from __future__ import annotations

from typing import Dict, Mapping

import torch

from .param_groups import RoutedParamGroups


def _grad_stats(param: torch.nn.Parameter) -> Dict[str, float | int]:
    grad = param.grad
    if grad is None:
        return {
            "has_grad": 0,
            "norm": 0.0,
            "mean_abs": 0.0,
            "max_abs": 0.0,
            "numel": int(param.numel()),
        }
    grad_detached = grad.detach()
    abs_grad = grad_detached.abs()
    return {
        "has_grad": 1,
        "norm": float(torch.linalg.vector_norm(grad_detached).item()),
        "mean_abs": float(abs_grad.mean().item()) if abs_grad.numel() > 0 else 0.0,
        "max_abs": float(abs_grad.max().item()) if abs_grad.numel() > 0 else 0.0,
        "numel": int(param.numel()),
    }


def capture_param_grad_stats(named_params: Mapping[str, torch.nn.Parameter]) -> Dict[str, Dict[str, float | int]]:
    return {name: _grad_stats(param) for name, param in named_params.items()}


def capture_route_grad_stats(groups: RoutedParamGroups) -> Dict[str, Dict[str, Dict[str, float | int]]]:
    return {
        "geometry": capture_param_grad_stats(groups.geometry),
        "appearance": capture_param_grad_stats(groups.appearance),
    }


def summarize_viewspace_grad(viewspace_points, visibility_filter=None) -> Dict[str, float | int]:
    if viewspace_points is None or getattr(viewspace_points, "grad", None) is None:
        return {
            "has_grad": 0,
            "visible": 0,
            "mean_norm_xy": 0.0,
            "mean_norm_z": 0.0,
            "max_norm_xy": 0.0,
        }

    grad = viewspace_points.grad.detach()
    if visibility_filter is not None:
        grad = grad[visibility_filter]
    if grad.numel() == 0:
        return {
            "has_grad": 1,
            "visible": 0,
            "mean_norm_xy": 0.0,
            "mean_norm_z": 0.0,
            "max_norm_xy": 0.0,
        }
    norm_xy = torch.linalg.vector_norm(grad[:, :2], dim=-1)
    norm_z = torch.linalg.vector_norm(grad[:, 2:], dim=-1) if grad.shape[-1] > 2 else torch.zeros_like(norm_xy)
    return {
        "has_grad": 1,
        "visible": int(grad.shape[0]),
        "mean_norm_xy": float(norm_xy.mean().item()),
        "mean_norm_z": float(norm_z.mean().item()),
        "max_norm_xy": float(norm_xy.max().item()),
    }


def max_grad_norm(stats: Mapping[str, Mapping[str, float | int]]) -> float:
    if not stats:
        return 0.0
    return max(float(item.get("norm", 0.0)) for item in stats.values())


def assert_no_geometry_grad(
    route_stats: Mapping[str, Mapping[str, Mapping[str, float | int]]],
    *,
    eps: float = 1e-12,
) -> None:
    geometry_stats = route_stats.get("geometry", {})
    leaked = {
        name: float(stats.get("norm", 0.0))
        for name, stats in geometry_stats.items()
        if float(stats.get("norm", 0.0)) > eps
    }
    if leaked:
        raise RuntimeError(f"appearance route leaked geometry gradients: {leaked}")


def format_route_grad_stats(route_name: str, route_stats) -> str:
    lines = [f"[ROUTE-AUDIT] {route_name}"]
    for role in ("geometry", "appearance"):
        role_stats = route_stats.get(role, {})
        parts = []
        for name, stats in role_stats.items():
            parts.append(f"{name}=norm:{float(stats.get('norm', 0.0)):.3e}")
        lines.append(f"  {role}: " + " ".join(parts))
    return "\n".join(lines)
