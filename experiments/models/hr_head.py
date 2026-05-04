"""
Phase 2.2 – HR Geometric Prior Head (dual output heads, shared backbone).

**This file defines ``HRGeometricPriorHead``: a canonical **CNN U-Net**
(encoder–decoder with skip connections), *not* the HD-VGGT architecture.
HD-VGGT-style LR-ViT + φ_guide / φ_feat / φ_fuse is in ``models/hr_head_hd_vggt_style.py``
(``HDVGGTStyleGeomHead``).

Consumes LR-resolution conditioning (VGGT depth + optional LR RGB +
optional StableSR prior downsampled to LR *and*, when ``use_sr_prior``,
optional **HR-resolution** StableSR fused after the decoder upsample).
Produces HR maps aligned to
oracle / MipNeRF SR resolution (default 800×800 for 200→4×).

Inference contract (per batch of views):
  depth_hr:    (B, V, 1, H_sr, W_sr), strictly positive
  normal_hr:   (B, V, 3, H_sr, W_sr), unit vectors along channel dim
  confidence:  (B, V, 1, H_sr, W_sr), values in (0, 1)

Training losses and VGGT aggregation live in Phase 2.3 — this module is
the architecture + forward pass only.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def compose_geom_lr_stack(
    depth_lr: torch.Tensor,
    rgb_lr: Optional[torch.Tensor],
    sr_prior_hr: Optional[torch.Tensor],
    *,
    use_rgb: bool,
    use_sr_prior: bool,
    depth_in_log_space: bool,
    sr_scale: int,
) -> torch.Tensor:
    """
    Shared LR conditioning (log-depth + optional RGB @ LR + SR prior down to LR).

    depth_lr:   (B, V, 1, H, W); rgb_lr/sr_prior_hr optional per flags.
    Returns (B, V, C, H, W).
    """
    if depth_lr.dim() != 5:
        raise ValueError(f"depth_lr must be 5D, got shape {tuple(depth_lr.shape)}")
    b, v, _, h, w = depth_lr.shape
    eps = 1e-6
    d = depth_lr
    if depth_in_log_space:
        d = torch.log(d.clamp_min(eps))

    parts = [d]

    if use_rgb:
        if rgb_lr is None:
            raise ValueError("use_rgb=True but rgb_lr is None")
        parts.append(rgb_lr)

    if use_sr_prior:
        if sr_prior_hr is None:
            raise ValueError("use_sr_prior=True but sr_prior_hr is None")
        if sr_prior_hr.shape[-2:] != (h * sr_scale, w * sr_scale):
            sr_prior_hr = F.interpolate(
                sr_prior_hr.flatten(0, 1),
                size=(h * sr_scale, w * sr_scale),
                mode="bilinear",
                align_corners=False,
            ).unflatten(0, (b, v))
        sr_lr = F.interpolate(
            sr_prior_hr.flatten(0, 1),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).unflatten(0, (b, v))
        parts.append(sr_lr)

    x = torch.cat(parts, dim=2)
    return x.contiguous()


def _make_norm(num_channels: int) -> nn.Module:
    """GroupNorm with num_groups dividing num_channels (required by PyTorch)."""
    if num_channels <= 0:
        raise ValueError(f"num_channels must be positive, got {num_channels}")
    g = min(32, num_channels)
    while num_channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, num_channels)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            _make_norm(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            _make_norm(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HRGeometricPriorHead(nn.Module):
    """
    U-Net backbone (LR → HR via skip + bilinear enlarge) + three heads.

    Typical inputs (stacked along channel dim at LR spatial size):
      - depth_lr:  VGGT disparity/depth resized to LR (1 ch), log-space optional
      - rgb_lr:    LR RGB in [0,1] if use_rgb
      - sr_prior_lr: StableSR RGB downsampled to LR if use_sr_prior (compose)
      - sr_prior_hr: same StableSR at HR — **additionally** encoded and fused at HR after bilinear upsample
        (not only the low-pass LR copy).
    """

    def __init__(
        self,
        in_channels: Optional[int] = None,
        *,
        use_rgb: bool = True,
        use_sr_prior: bool = True,
        depth_in_log_space: bool = True,
        # ~28M trainable params with default 96 (dual heads share backbone).
        base_channels: int = 96,
        sr_scale: int = 4,
    ) -> None:
        super().__init__()
        self.use_rgb = use_rgb
        self.use_sr_prior = use_sr_prior
        self.depth_in_log_space = depth_in_log_space
        self.sr_scale = int(sr_scale)
        self.base_channels = base_channels

        if in_channels is not None:
            ic = int(in_channels)
        else:
            ic = 1 + (3 if use_rgb else 0) + (3 if use_sr_prior else 0)

        b = base_channels
        self.inc = DoubleConv(ic, b)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(b, b * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(b * 2, b * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(b * 4, b * 8))

        self.up1 = nn.Conv2d(b * 8 + b * 4, b * 4, 1, bias=False)
        self.dec1 = DoubleConv(b * 4, b * 4)

        self.up2 = nn.Conv2d(b * 4 + b * 2, b * 2, 1, bias=False)
        self.dec2 = DoubleConv(b * 2, b * 2)

        self.up3 = nn.Conv2d(b * 2 + b, b, 1, bias=False)
        self.dec3 = DoubleConv(b, b)

        if use_sr_prior:
            self.sr_hr_encoder = DoubleConv(3, b)
            self.sr_hr_fuse = nn.Sequential(
                nn.Conv2d(b + b, b, 1, bias=False),
                _make_norm(b),
                nn.ReLU(inplace=True),
            )
        else:
            self.sr_hr_encoder = None  # type: ignore[assignment]
            self.sr_hr_fuse = None  # type: ignore[assignment]

        hr_mid = max(32, b // 2)
        self.refine = nn.Sequential(
            DoubleConv(b, hr_mid),
            nn.Conv2d(hr_mid, hr_mid, 3, padding=1, bias=False),
            _make_norm(hr_mid),
            nn.ReLU(inplace=True),
        )

        self.head_depth = nn.Conv2d(hr_mid, 1, 3, padding=1)
        self.head_normal = nn.Conv2d(hr_mid, 3, 3, padding=1)
        self.head_confidence = nn.Conv2d(hr_mid, 1, 3, padding=1)

        self._init_conv_weights()

    def _init_conv_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def expected_lr_channels(self) -> int:
        return 1 + (3 if self.use_rgb else 0) + (3 if self.use_sr_prior else 0)

    def compose_lr_input(
        self,
        depth_lr: torch.Tensor,
        rgb_lr: Optional[torch.Tensor] = None,
        sr_prior_hr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        depth_lr: (B,V,1,H,W) positive depth or disparity from VGGT
        rgb_lr:   (B,V,3,H,W) in [0,1], optional if use_rgb
        sr_prior_hr: (B,V,3,sH,sW) StableSR RGB at HR — downsampled internally
        Returns (B,V,C,H,W).
        """
        return compose_geom_lr_stack(
            depth_lr,
            rgb_lr,
            sr_prior_hr,
            use_rgb=self.use_rgb,
            use_sr_prior=self.use_sr_prior,
            depth_in_log_space=self.depth_in_log_space,
            sr_scale=self.sr_scale,
        )

    def forward_tensors(
        self,
        lr_stack: torch.Tensor,
        *,
        sr_hr: Optional[torch.Tensor] = None,
        _lr_hw: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """
        lr_stack: (B*V, C, H_lr, W_lr)
        sr_hr:    (B*V, 3, H_sr, W_sr) optional StableSR RGB at **target HR**; fused after upsample.
        _lr_hw:   reserved for future validation / profiling hooks.
        Returns dict depth_hr, normal_hr, confidence with leading (B,V,...) restored by caller.
        """
        _, _, h_lr, w_lr = lr_stack.shape
        h_sr = int(round(h_lr * self.sr_scale))
        w_sr = int(round(w_lr * self.sr_scale))

        x0 = self.inc(lr_stack)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        y = x3

        # Align decoder feature `t` to skip `ref` spatially; skip keeps its own channels.
        def _align_spatial(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
            th, tw = ref.shape[2], ref.shape[3]
            uh, uw = t.shape[2], t.shape[3]
            if (uh, uw) == (th, tw):
                return t
            if uh >= th and uw >= tw:
                return t[..., :th, :tw]
            return F.interpolate(t, size=(th, tw), mode="bilinear", align_corners=False)

        y = F.interpolate(y, scale_factor=2.0, mode="bilinear", align_corners=False)
        y = torch.cat([x2, _align_spatial(y, x2)], dim=1)
        y = self.up1(y)
        y = self.dec1(y)

        y = F.interpolate(y, scale_factor=2.0, mode="bilinear", align_corners=False)
        y = torch.cat([x1, _align_spatial(y, x1)], dim=1)
        y = self.up2(y)
        y = self.dec2(y)

        y = F.interpolate(y, scale_factor=2.0, mode="bilinear", align_corners=False)
        y = torch.cat([x0, _align_spatial(y, x0)], dim=1)
        y = self.up3(y)
        y = self.dec3(y)

        y = F.interpolate(y, size=(h_sr, w_sr), mode="bilinear", align_corners=False)
        if self.use_sr_prior and sr_hr is not None and self.sr_hr_encoder is not None:
            if sr_hr.shape[-2:] != (h_sr, w_sr):
                sr_hr = F.interpolate(sr_hr, size=(h_sr, w_sr), mode="bilinear", align_corners=False)
            sf = self.sr_hr_encoder(sr_hr)
            y = self.sr_hr_fuse(torch.cat([y, sf], dim=1))
        y = self.refine(y)

        depth = F.softplus(self.head_depth(y)) + 1e-3
        normal = self.head_normal(y)
        normal = F.normalize(normal, dim=1, eps=1e-6)
        confidence = torch.sigmoid(self.head_confidence(y))

        return {
            "depth_hr": depth,
            "normal_hr": normal,
            "confidence_hr": confidence,
        }

    def forward(
        self,
        depth_lr: torch.Tensor,
        rgb_lr: Optional[torch.Tensor] = None,
        sr_prior_hr: Optional[torch.Tensor] = None,
        lr_stack: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Either pass lr_stack (B,V,C,H,W) fully composed,
        OR pass depth_lr (+ optional tensors) for automatic composition.

        Outputs each (B,V,…,h_sr,w_sr).
        """
        if lr_stack is None:
            lr_stack = self.compose_lr_input(
                depth_lr, rgb_lr=rgb_lr, sr_prior_hr=sr_prior_hr
            )

        if lr_stack.dim() != 5:
            raise ValueError(f"lr_stack must be (B,V,C,H,W); got {tuple(lr_stack.shape)}")

        b, v, _, h_lr, w_lr = lr_stack.shape
        bv = b * v
        x_in = lr_stack.reshape(bv, lr_stack.shape[2], h_lr, w_lr)

        sr_hv: Optional[torch.Tensor] = None
        if self.use_sr_prior and sr_prior_hr is not None:
            sr_hv = sr_prior_hr.reshape(bv, 3, sr_prior_hr.shape[-2], sr_prior_hr.shape[-1])

        out = self.forward_tensors(x_in, sr_hr=sr_hv, _lr_hw=(h_lr, w_lr))
        pooled: Dict[str, torch.Tensor] = {}
        for k, t in out.items():
            _, c, hh, ww = t.shape
            pooled[k] = t.reshape(b, v, c, hh, ww)
        return pooled


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
