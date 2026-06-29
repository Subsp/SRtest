"""Surface helpers for SP-IE-SRGS v0."""

from .losses import SPV0SurfaceConfig, SPV0SurfaceLoss, surface_ramp
from .render_utils import downsample_to_match

__all__ = [
    "SPV0SurfaceConfig",
    "SPV0SurfaceLoss",
    "downsample_to_match",
    "surface_ramp",
]
