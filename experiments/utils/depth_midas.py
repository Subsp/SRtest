"""
MiDaS monocular depth estimator wrapper.

On first call, downloads MiDaS_small via torch.hub (~300 MB).
Returns relative depth maps, scale-normalized so median = 1.0.
"""

import torch
import torch.nn.functional as F
import numpy as np

_midas_model     = None
_midas_transform = None


def _load_midas(device: str):
    global _midas_model, _midas_transform
    if _midas_model is not None:
        return
    print("[MiDaS] Loading MiDaS_small via torch.hub …")
    _midas_model = torch.hub.load(
        "intel-isl/MiDaS", "MiDaS_small", trust_repo=True
    )
    _midas_model.eval().to(device)
    transforms = torch.hub.load(
        "intel-isl/MiDaS", "transforms", trust_repo=True
    )
    _midas_transform = transforms.small_transform
    print("[MiDaS] Ready.")


@torch.no_grad()
def depth_from_image(
    image_tensor: torch.Tensor,   # (3, H, W) float [0,1]
    device: str,
) -> torch.Tensor:
    """
    Estimate monocular depth for a single image.

    Returns: (H, W) float32, scale-normalised so median = 1.0
    """
    _load_midas(device)

    # MiDaS transform expects a numpy HWC uint8 image
    img_np = (image_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    inp = _midas_transform(img_np).to(device)     # (1, 3, h, w)

    pred = _midas_model(inp)                       # (1, h', w') disparity

    H, W = image_tensor.shape[1:]
    pred_up = F.interpolate(
        pred.unsqueeze(1), size=(H, W),
        mode="bilinear", align_corners=False,
    ).squeeze()                                    # (H, W)

    # MiDaS outputs disparity (higher = closer). Convert to pseudo-depth.
    disp = pred_up.float()
    disp = disp.clamp(min=1e-4)
    depth = 1.0 / disp

    # Scale-normalise: median → 1.0  (removes unknown metric scale)
    med = depth.median()
    if med > 0:
        depth = depth / med

    return depth   # (H, W), metric-agnostic, relative depth


def depth_batch(
    image_tensors: list,    # list of (3, H, W) float [0,1]
    device: str,
) -> list:
    """Estimate depth for a list of image tensors."""
    return [depth_from_image(t, device) for t in image_tensors]
