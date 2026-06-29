"""Source-aware parameter routing helpers for SP-IE-SRGS v0."""

from .param_groups import (
    APPEARANCE_PARAM_NAMES,
    GEOMETRY_PARAM_NAMES,
    RoutedParamGroups,
    build_routed_param_groups,
    routed_requires_grad,
)
from .train_step import RoutedStepResult, routed_one_optimizer_step

__all__ = [
    "APPEARANCE_PARAM_NAMES",
    "GEOMETRY_PARAM_NAMES",
    "RoutedParamGroups",
    "RoutedStepResult",
    "build_routed_param_groups",
    "routed_one_optimizer_step",
    "routed_requires_grad",
]
