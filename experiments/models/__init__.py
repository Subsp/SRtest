"""Stage-1 neural modules (HR Head, etc.)."""

from .hr_head import HRGeometricPriorHead
from .hr_head_hd_vggt_style import HDVGGTStyleGeomHead

__all__ = ["HRGeometricPriorHead", "HDVGGTStyleGeomHead"]
