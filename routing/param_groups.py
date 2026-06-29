"""Parameter grouping utilities for source-aware routing.

SP-IE-SRGS v0 keeps the original Gaussian schema and routes losses to the
existing parameter tensors:

* geometry: xyz, scaling, rotation, opacity
* appearance: SH dc/rest features
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Mapping, MutableMapping, Tuple

import torch


GEOMETRY_PARAM_NAMES = ("xyz", "scaling", "rotation", "opacity")
APPEARANCE_PARAM_NAMES = ("f_dc", "f_rest")

_GAUSSIAN_ATTRS = {
    "xyz": "_xyz",
    "scaling": "_scaling",
    "rotation": "_rotation",
    "opacity": "_opacity",
    "f_dc": "_features_dc",
    "f_rest": "_features_rest",
}


@dataclass(frozen=True)
class RoutedParamGroups:
    geometry: Mapping[str, torch.nn.Parameter]
    appearance: Mapping[str, torch.nn.Parameter]

    @property
    def all(self) -> Dict[str, torch.nn.Parameter]:
        params = dict(self.geometry)
        params.update(self.appearance)
        return params


def build_routed_param_groups(gaussians) -> RoutedParamGroups:
    """Build v0 geometry/appearance groups from a GaussianModel instance."""

    missing = [attr for attr in _GAUSSIAN_ATTRS.values() if not hasattr(gaussians, attr)]
    if missing:
        raise AttributeError(f"Gaussian model is missing routed params: {missing}")

    params = {name: getattr(gaussians, attr) for name, attr in _GAUSSIAN_ATTRS.items()}
    return RoutedParamGroups(
        geometry={name: params[name] for name in GEOMETRY_PARAM_NAMES},
        appearance={name: params[name] for name in APPEARANCE_PARAM_NAMES},
    )


def iter_named_params(
    named_params: Mapping[str, torch.nn.Parameter] | Iterable[Tuple[str, torch.nn.Parameter]]
) -> Iterator[Tuple[str, torch.nn.Parameter]]:
    if isinstance(named_params, Mapping):
        yield from named_params.items()
    else:
        yield from named_params


def set_requires_grad(
    named_params: Mapping[str, torch.nn.Parameter] | Iterable[Tuple[str, torch.nn.Parameter]],
    enabled: bool,
) -> None:
    for _, param in iter_named_params(named_params):
        param.requires_grad_(enabled)


def snapshot_requires_grad(
    named_params: Mapping[str, torch.nn.Parameter] | Iterable[Tuple[str, torch.nn.Parameter]]
) -> Dict[str, bool]:
    return {name: bool(param.requires_grad) for name, param in iter_named_params(named_params)}


def restore_requires_grad(
    named_params: Mapping[str, torch.nn.Parameter],
    snapshot: Mapping[str, bool],
) -> None:
    for name, param in named_params.items():
        if name in snapshot:
            param.requires_grad_(bool(snapshot[name]))


@contextmanager
def routed_requires_grad(
    groups: RoutedParamGroups,
    *,
    geometry: bool,
    appearance: bool,
):
    """Temporarily enable gradients for one routing phase.

    This context must wrap the forward pass, not only the backward pass.
    """

    all_params: MutableMapping[str, torch.nn.Parameter] = dict(groups.all)
    state = snapshot_requires_grad(all_params)
    try:
        set_requires_grad(groups.geometry, geometry)
        set_requires_grad(groups.appearance, appearance)
        yield
    finally:
        restore_requires_grad(all_params, state)
