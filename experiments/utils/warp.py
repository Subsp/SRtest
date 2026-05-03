"""
Differentiable backward image warping via camera reprojection.

Given:
  src_image  – the SR image at view j  (to be sampled from)
  depth_tgt  – dense depth at target view i
  K_src, R_src, t_src – camera parameters of view j
  K_tgt, R_tgt, t_tgt – camera parameters of view i

We compute, for every pixel in view i, its 3-D world position (using depth_tgt
and view-i camera), then re-project that point into view j, and bilinearly
sample src_image.  This gives warp(I_sr^{v_j}) → view i.

All tensors use OpenCV convention: P_cam = R @ P_world + t
"""

import torch
import torch.nn.functional as F


def backward_warp(
    src_image:  torch.Tensor,   # (3, H, W)  float [0,1]
    depth_tgt:  torch.Tensor,   # (H, W)     metric depth of TARGET view
    K_src:      torch.Tensor,   # (3, 3)
    R_src:      torch.Tensor,   # (3, 3)  world→camera of source (j)
    t_src:      torch.Tensor,   # (3,)
    K_tgt:      torch.Tensor,   # (3, 3)
    R_tgt:      torch.Tensor,   # (3, 3)  world→camera of target (i)
    t_tgt:      torch.Tensor,   # (3,)
    depth_scale_lr_to_sr: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Backward-warp src_image into the target view using depth_tgt.

    depth_tgt must be at the same spatial resolution as src_image (H×W).
    If depth_tgt is at LR (200×200) and src_image is at SR (800×800), pass
    depth_scale_lr_to_sr = 1 (depth values are metric, not resolution-dependent).
    But you must upsample depth_tgt to SR resolution externally before calling.

    Returns:
        warped  (3, H, W) – warped src_image
        valid   (1, H, W) – binary mask: 1 where projection falls inside src bounds
    """
    _, H, W = src_image.shape
    device  = src_image.device
    dtype   = src_image.dtype

    # ── pixel grid of target view ─────────────────────────────────────────────
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )                                            # (H, W)

    # Homogeneous pixel coords in target view: K_tgt^{-1} @ [u,v,1]^T
    K_tgt_inv = torch.linalg.inv(K_tgt)
    ones = torch.ones_like(xs)
    pix_h = torch.stack([xs, ys, ones], dim=0)  # (3, H, W)
    ray   = torch.einsum("ij,jhw->ihw", K_tgt_inv, pix_h)  # (3, H, W) camera-frame directions

    d = depth_tgt.unsqueeze(0)                   # (1, H, W)
    P_cam_tgt = ray * d                          # (3, H, W) camera-frame 3-D points

    # ── target camera-frame → world ───────────────────────────────────────────
    # P_world = R_tgt^T @ (P_cam_tgt - t_tgt)
    R_tgt_T = R_tgt.T
    P_flat   = P_cam_tgt.view(3, -1) - t_tgt.unsqueeze(1)  # (3, H*W)
    P_world  = R_tgt_T @ P_flat                              # (3, H*W)

    # ── world → source camera-frame ───────────────────────────────────────────
    # P_cam_src = R_src @ P_world + t_src
    P_cam_src = R_src @ P_world + t_src.unsqueeze(1)        # (3, H*W)

    # ── project into source image ─────────────────────────────────────────────
    depth_src = P_cam_src[2:3, :]                            # (1, H*W)
    valid_depth = depth_src > 0                              # positive-depth mask

    # normalise
    proj = K_src @ (P_cam_src / depth_src.clamp(min=1e-6))  # (3, H*W)
    u_src = proj[0]   # pixel x in source
    v_src = proj[1]   # pixel y in source

    # ── convert to grid_sample coords: [-1, 1] ───────────────────────────────
    u_norm = (u_src / (W - 1)) * 2.0 - 1.0
    v_norm = (v_src / (H - 1)) * 2.0 - 1.0
    grid   = torch.stack([u_norm, v_norm], dim=1)            # (H*W, 2)
    grid   = grid.view(1, H, W, 2)

    # ── sample source image ───────────────────────────────────────────────────
    warped = F.grid_sample(
        src_image.unsqueeze(0),   # (1, 3, H, W)
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0)                   # (3, H, W)

    # valid: positive depth AND within image bounds
    in_bounds = (
        (u_src >= 0) & (u_src <= W - 1) &
        (v_src >= 0) & (v_src <= H - 1)
    )                              # (H*W,)
    valid = (valid_depth.squeeze(0) & in_bounds).view(1, H, W).float()

    return warped, valid


def upsample_depth(
    depth_lr: torch.Tensor,
    scale: int = None,
    target_hw: tuple = None,
) -> torch.Tensor:
    """
    Bilinearly upsample a depth map.

    depth_lr   : (H, W)
    scale      : integer scale factor (used when target is square or uniform)
    target_hw  : (H_out, W_out) explicit target size (takes priority over scale)

    Returns: (H_out, W_out)
    """
    d = depth_lr.unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)
    if target_hw is not None:
        d_up = F.interpolate(d, size=target_hw, mode="bilinear", align_corners=False)
    elif scale is not None:
        d_up = F.interpolate(d, scale_factor=scale, mode="bilinear", align_corners=False)
    else:
        raise ValueError("Either scale or target_hw must be provided")
    return d_up.squeeze(0).squeeze(0)
