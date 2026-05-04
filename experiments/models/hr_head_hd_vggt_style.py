"""
HD-VGGT–aligned HR geometric head (conceptual parity with Chen et al., arXiv:2603.27222 §3.2).

**What we implement (single-view distill head in this repo):**
  * **LR branch:** ViT encoder on patch token grid of the *same LR stack*
    ``[log depth, LR RGB?, SR-at-LR?]`` — analogue of coarse/global reasoning at low resolution;
    not the pretrained multi-view VGGT trunk.
  * **HR branch:** **Guidance-conditioned upsampler** —
    coarse ViT maps are lifted to HR, fused with convolutional embeddings of HR guidance
    (StableSR RGB if provided, otherwise bilinear-upsampled LR RGB), matching Fig. 1 /
    Eq. (3–5) style decomposition (``phi_guidance``, ``phi_feat``, ``phi_fuse``).
  * **HR refiner:** convolutional surrogate of the shallow ``T_refine`` in HD-VGGT
    (transformer refinement can replace this stack if multi-GPU / window attention is wired).

Out of scope here: full ``T_coarse`` multi-view VGGT-1B, Feature Modulation §3.3, official HD-VGGT weights.

See ``HRGeometricPriorHead`` for the U-Net baseline and ``compose_geom_lr_stack`` for inputs.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hr_head import DoubleConv, _make_norm, compose_geom_lr_stack


class _PatchEmbed(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        z = self.proj(x)
        hp, wp = z.shape[-2:]
        seq = z.flatten(2).transpose(1, 2)
        return seq, (hp, wp)


class HDVGGTStyleGeomHead(nn.Module):
    """ViT LR encoder + guided HR fusion + shallow conv outputs (depth / normal / conf)."""

    def __init__(
        self,
        *,
        use_rgb: bool = True,
        use_sr_prior: bool = True,
        depth_in_log_space: bool = True,
        sr_scale: int = 4,
        patch_size: int = 8,
        vit_dim: int = 256,
        vit_depth: int = 6,
        vit_heads: int = 8,
        mlp_ratio: float = 4.0,
        guidance_width: int = 64,
        fuse_width: int = 160,
        refiner_mid: Optional[int] = None,
        max_patch_tokens: int = 16384,
    ) -> None:
        super().__init__()
        self.use_rgb = use_rgb
        self.use_sr_prior = use_sr_prior
        self.depth_in_log_space = depth_in_log_space
        self.sr_scale = int(sr_scale)
        self.patch_size = int(patch_size)
        self.vit_dim = int(vit_dim)

        ic = 1 + (3 if use_rgb else 0) + (3 if use_sr_prior else 0)
        hidden = int(vit_dim * mlp_ratio)

        self.patch_embed = _PatchEmbed(ic, vit_dim, self.patch_size)
        self.norm_pre = nn.LayerNorm(vit_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=vit_dim,
            nhead=vit_heads,
            dim_feedforward=hidden,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.vit = nn.TransformerEncoder(layer, num_layers=vit_depth)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patch_tokens, vit_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Project backbone grid to fused width before HR lift
        self.coarse_proj = nn.Conv2d(vit_dim, fuse_width, 1, bias=False)
        gw = guidance_width
        fw = fuse_width

        self.phi_guide = nn.Sequential(
            nn.Conv2d(3, gw, 5, padding=2, bias=False),
            _make_norm(gw),
            nn.ReLU(inplace=True),
            nn.Conv2d(gw, gw, 3, padding=1, bias=False),
            _make_norm(gw),
            nn.ReLU(inplace=True),
        )
        self.phi_feat = nn.Sequential(
            nn.Conv2d(fw, fw, 3, padding=1, bias=False),
            _make_norm(fw),
            nn.ReLU(inplace=True),
            nn.Conv2d(fw, fw, 3, padding=1, bias=False),
            _make_norm(fw),
            nn.ReLU(inplace=True),
        )
        self.phi_fuse = nn.Sequential(
            nn.Conv2d(fw + gw, fw, 3, padding=1, bias=False),
            _make_norm(fw),
            nn.ReLU(inplace=True),
            nn.Conv2d(fw, fw, 3, padding=1, bias=False),
            _make_norm(fw),
            nn.ReLU(inplace=True),
        )

        mid = refiner_mid if refiner_mid is not None else max(48, fw // 2)
        self.refiner = nn.Sequential(
            DoubleConv(fw, mid),
            nn.Conv2d(mid, mid, 3, padding=1, bias=False),
            _make_norm(mid),
            nn.ReLU(inplace=True),
        )
        self.head_depth = nn.Conv2d(mid, 1, 3, padding=1)
        self.head_normal = nn.Conv2d(mid, 3, 3, padding=1)
        self.head_confidence = nn.Conv2d(mid, 1, 3, padding=1)

        self._init_conv_weights()

    def expected_lr_channels(self) -> int:
        return 1 + (3 if self.use_rgb else 0) + (3 if self.use_sr_prior else 0)

    def _init_conv_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def compose_lr_input(
        self,
        depth_lr: torch.Tensor,
        rgb_lr: Optional[torch.Tensor] = None,
        sr_prior_hr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return compose_geom_lr_stack(
            depth_lr,
            rgb_lr,
            sr_prior_hr,
            use_rgb=self.use_rgb,
            use_sr_prior=self.use_sr_prior,
            depth_in_log_space=self.depth_in_log_space,
            sr_scale=self.sr_scale,
        )

    def _hr_guidance(
        self,
        rgb_lr: Optional[torch.Tensor],
        sr_prior_hr: Optional[torch.Tensor],
        h_sr: int,
        w_sr: int,
    ) -> torch.Tensor:
        if sr_prior_hr is not None:
            gh = sr_prior_hr
            if gh.shape[-2:] != (h_sr, w_sr):
                gh = F.interpolate(
                    gh.flatten(0, 1),
                    size=(h_sr, w_sr),
                    mode="bilinear",
                    align_corners=False,
                ).unflatten(0, gh.shape[:2])
            return gh.flatten(0, 1).clamp(0.0, 1.0)
        if rgb_lr is None:
            raise ValueError(
                "HD-VGGT style head needs SR prior or RGB for HR guidance branch (§3.2.2)."
            )
        g = rgb_lr.flatten(0, 1)
        gh = F.interpolate(g, size=(h_sr, w_sr), mode="bilinear", align_corners=False)
        return gh.clamp(0.0, 1.0)

    def forward_tensors(
        self,
        lr_stack: torch.Tensor,
        hr_guidance_rgb: torch.Tensor,
        *,
        _lr_hw: Tuple[int, int],
        h_sr: int,
        w_sr: int,
    ) -> Dict[str, torch.Tensor]:
        _, _, h_lr, w_lr = lr_stack.shape
        ps = self.patch_size
        pad_h = (ps - h_lr % ps) % ps
        pad_w = (ps - w_lr % ps) % ps
        x_lr = lr_stack
        if pad_h or pad_w:
            x_lr = F.pad(x_lr, (0, pad_w, 0, pad_h))

        tokens, (hp, wp) = self.patch_embed(x_lr)
        n = tokens.size(1)
        if n > self.pos_embed.size(1):
            raise RuntimeError(
                f"Patch grid {hp}x{wp}={n} exceeds max_patch_tokens={self.pos_embed.size(1)}; "
                f"raise max_patch_tokens or increase patch_size."
            )
        pos = self.pos_embed[:, :n, :].to(dtype=tokens.dtype, device=tokens.device)
        h = self.norm_pre(tokens + pos)
        h = self.vit(h)
        feat = (
            h.transpose(1, 2)
            .reshape(lr_stack.size(0), self.vit_dim, hp, wp)
            .contiguous()
        )
        if pad_h or pad_w:
            # crop back (trim padding from bottom/right)
            h_keep = min(hp, int((h_lr + pad_h) // ps))
            w_keep = min(wp, int((w_lr + pad_w) // ps))
            feat = feat[..., :h_keep, :w_keep]

        fused_lr = self.coarse_proj(feat)
        fused_lr = self.phi_feat(fused_lr)
        fused_hr = F.interpolate(
            fused_lr, size=(h_sr, w_sr), mode="bilinear", align_corners=False
        )
        g = self.phi_guide(hr_guidance_rgb)
        y = self.phi_fuse(torch.cat([fused_hr, g], dim=1))
        y = self.refiner(y)

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
        if lr_stack is None:
            lr_stack = self.compose_lr_input(depth_lr, rgb_lr=rgb_lr, sr_prior_hr=sr_prior_hr)

        if lr_stack.dim() != 5:
            raise ValueError(f"lr_stack must be (B,V,C,H,W); got {tuple(lr_stack.shape)}")

        b, v, _, h_lr, w_lr = lr_stack.shape
        h_sr = int(round(h_lr * self.sr_scale))
        w_sr = int(round(w_lr * self.sr_scale))
        bv = b * v
        x_in = lr_stack.reshape(bv, lr_stack.shape[2], h_lr, w_lr)

        gh = self._hr_guidance(rgb_lr, sr_prior_hr, h_sr, w_sr)

        out = self.forward_tensors(
            x_in, gh, _lr_hw=(h_lr, w_lr), h_sr=h_sr, w_sr=w_sr
        )
        pooled: Dict[str, torch.Tensor] = {}
        for k, t in out.items():
            _, c, hh, ww = t.shape
            pooled[k] = t.reshape(b, v, c, hh, ww)
        return pooled
