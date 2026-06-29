"""One-optimizer routed training step for SP-IE-SRGS v0."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Optional

import torch

from .grad_audit import (
    assert_no_geometry_grad,
    capture_route_grad_stats,
    summarize_viewspace_grad,
)
from .param_groups import RoutedParamGroups, build_routed_param_groups, routed_requires_grad


ClosureResult = Mapping[str, object] | torch.Tensor


@dataclass
class RoutedStepResult:
    geo_loss: float
    app_loss: float
    metrics: Dict[str, float] = field(default_factory=dict)
    grad_audit: Dict[str, object] = field(default_factory=dict)


def _as_result_dict(result: ClosureResult, default_name: str) -> Dict[str, object]:
    if torch.is_tensor(result):
        return {"loss": result, "metrics": {default_name: float(result.detach().item())}}
    if "loss" not in result:
        raise KeyError("routed closure must return a Tensor or a mapping with a 'loss' key")
    return dict(result)


def _clone_grads(named_params: Mapping[str, torch.nn.Parameter]) -> Dict[str, Optional[torch.Tensor]]:
    grads = {}
    for name, param in named_params.items():
        grads[name] = None if param.grad is None else param.grad.detach().clone()
    return grads


def _restore_grads(
    named_params: Mapping[str, torch.nn.Parameter],
    saved_grads: Mapping[str, Optional[torch.Tensor]],
) -> None:
    for name, param in named_params.items():
        grad = saved_grads.get(name)
        param.grad = None if grad is None else grad.to(device=param.device, dtype=param.dtype).clone()


def _merge_grads(*grad_maps: Mapping[str, Optional[torch.Tensor]]) -> Dict[str, Optional[torch.Tensor]]:
    merged: Dict[str, Optional[torch.Tensor]] = {}
    for grad_map in grad_maps:
        for name, grad in grad_map.items():
            if grad is None:
                merged.setdefault(name, None)
            elif merged.get(name) is None:
                merged[name] = grad
            else:
                merged[name] = merged[name] + grad
    return merged


def _zero_optimizer(optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)


def _collect_densification_stats(gaussians, render_pkg: Mapping[str, object]) -> Dict[str, float | int]:
    viewspace_points = render_pkg.get("viewspace_points")
    visibility_filter = render_pkg.get("visibility_filter")
    stats = summarize_viewspace_grad(viewspace_points, visibility_filter)
    stats["densify_grad_source"] = "geometry"
    if viewspace_points is not None and visibility_filter is not None:
        gaussians.add_densification_stats(viewspace_points, visibility_filter)
        stats["densify_consumed"] = 1
    else:
        stats["densify_consumed"] = 0
    return stats


def routed_one_optimizer_step(
    *,
    gaussians,
    geometry_closure: Callable[[], ClosureResult],
    appearance_closure: Callable[[], ClosureResult],
    groups: Optional[RoutedParamGroups] = None,
    collect_densification: bool = True,
    fail_on_app_geometry_grad: bool = True,
) -> RoutedStepResult:
    """Run two routed backward passes and one Adam step.

    The closures perform the forward pass and return either a loss tensor or a
    dict containing at least ``loss``. The geometry closure may also return
    ``render_pkg`` so densification stats can be collected from the geometry
    route only.
    """

    if gaussians.optimizer is None:
        raise RuntimeError("Gaussian optimizer is not initialized")

    groups = groups or build_routed_param_groups(gaussians)
    optimizer = gaussians.optimizer

    _zero_optimizer(optimizer)
    with routed_requires_grad(groups, geometry=True, appearance=False):
        geo_result = _as_result_dict(geometry_closure(), "loss_geo")
        geo_loss = geo_result["loss"]
        if not torch.is_tensor(geo_loss):
            raise TypeError("geometry closure 'loss' must be a tensor")
        geo_loss.backward()
        geo_audit = capture_route_grad_stats(groups)
        densify_stats = {}
        if collect_densification:
            densify_stats = _collect_densification_stats(gaussians, geo_result.get("render_pkg", {}))
        else:
            densify_stats = {
                "densify_grad_source": "disabled",
                "densify_consumed": 0,
            }
        geo_grads = _clone_grads(groups.geometry)

    _zero_optimizer(optimizer)
    with routed_requires_grad(groups, geometry=False, appearance=True):
        app_result = _as_result_dict(appearance_closure(), "loss_app")
        app_loss = app_result["loss"]
        if not torch.is_tensor(app_loss):
            raise TypeError("appearance closure 'loss' must be a tensor")
        app_loss.backward()
        app_audit = capture_route_grad_stats(groups)
        app_viewspace_stats = summarize_viewspace_grad(
            app_result.get("render_pkg", {}).get("viewspace_points")
            if isinstance(app_result.get("render_pkg"), Mapping)
            else None,
            app_result.get("render_pkg", {}).get("visibility_filter")
            if isinstance(app_result.get("render_pkg"), Mapping)
            else None,
        )
        app_viewspace_stats["densify_consumed"] = 0
        app_viewspace_stats["densify_grad_source"] = "appearance_unused"
        if fail_on_app_geometry_grad:
            assert_no_geometry_grad(app_audit)
        app_grads = _clone_grads(groups.appearance)

    _zero_optimizer(optimizer)
    merged = _merge_grads(geo_grads, app_grads)
    _restore_grads(groups.all, merged)
    optimizer.step()
    _zero_optimizer(optimizer)

    metrics: Dict[str, float] = {}
    for result in (geo_result, app_result):
        result_metrics = result.get("metrics", {})
        if isinstance(result_metrics, Mapping):
            metrics.update({str(k): float(v) for k, v in result_metrics.items()})
    metrics["loss_geo"] = float(geo_loss.detach().item())
    metrics["loss_app"] = float(app_loss.detach().item())

    return RoutedStepResult(
        geo_loss=metrics["loss_geo"],
        app_loss=metrics["loss_app"],
        metrics=metrics,
        grad_audit={
            "geometry_route": geo_audit,
            "appearance_route": app_audit,
            "densification": densify_stats,
            "appearance_viewspace_unused": app_viewspace_stats,
        },
    )
