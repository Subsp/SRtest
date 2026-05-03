"""
SwinIR x4 Super-Resolution wrapper (frozen inference only).

On first use, the script:
  1. Clones the official SwinIR repo into ./third_party/SwinIR/
  2. Downloads the pretrained classical-SR ×4 weights

Model: SwinIR-M (medium), classical SR, ×4, DF2K training
Weight URL: https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/
            001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image as PILImage


# ── paths ─────────────────────────────────────────────────────────────────────
_HERE        = Path(__file__).parent.parent          # experiments/
_THIRD_PARTY = _HERE / "third_party"
_SWINIR_DIR  = _THIRD_PARTY / "SwinIR"
_WEIGHTS_DIR = _THIRD_PARTY / "weights"
_WEIGHT_NAME = "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
_WEIGHT_URL  = (
    "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
    + _WEIGHT_NAME
)


def _ensure_swinir():
    """Clone SwinIR repo and download weights if not already present."""
    _THIRD_PARTY.mkdir(parents=True, exist_ok=True)
    _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── clone repo ────────────────────────────────────────────────────────────
    if not (_SWINIR_DIR / "models" / "network_swinir.py").exists():
        print("[SwinIR] Cloning SwinIR repository …")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/JingyunLiang/SwinIR.git",
             str(_SWINIR_DIR)],
            check=True,
        )

    # ── download weights ──────────────────────────────────────────────────────
    weight_path = _WEIGHTS_DIR / _WEIGHT_NAME
    if not weight_path.exists():
        print(f"[SwinIR] Downloading pretrained weights → {weight_path} …")
        urllib.request.urlretrieve(_WEIGHT_URL, weight_path)
        print("[SwinIR] Download complete.")

    return weight_path


def _load_model(weight_path: Path, device: str) -> torch.nn.Module:
    """Instantiate SwinIR-M ×4 and load pretrained weights."""
    if str(_SWINIR_DIR) not in sys.path:
        sys.path.insert(0, str(_SWINIR_DIR))

    from models.network_swinir import SwinIR  # type: ignore

    model = SwinIR(
        upscale      = 4,
        in_chans     = 3,
        img_size     = 64,
        window_size  = 8,
        img_range    = 1.0,
        depths       = [6, 6, 6, 6, 6, 6],
        embed_dim    = 180,
        num_heads    = [6, 6, 6, 6, 6, 6],
        mlp_ratio    = 2,
        upsampler    = "pixelshuffle",
        resi_connection = "1conv",
    )

    state = torch.load(weight_path, map_location="cpu")
    # Official checkpoint may be wrapped in "params_ema" or "params"
    if "params_ema" in state:
        state = state["params_ema"]
    elif "params" in state:
        state = state["params"]
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)
    return model


class SwinIRSuperResolver:
    """Thin wrapper around SwinIR for convenient ×4 SR inference."""

    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        weight_path  = _ensure_swinir()
        self.model   = _load_model(weight_path, self.device)
        self.window  = 8   # SwinIR window size; input must be divisible

    @torch.no_grad()
    def upscale_tensor(self, lr: torch.Tensor) -> torch.Tensor:
        """
        Upscale a single LR image tensor.

        lr  : (3, H, W) float [0, 1], on any device
        out : (3, H*4, W*4) float [0, 1], on self.device
        """
        lr = lr.to(self.device)
        _, H, W = lr.shape

        # Pad to multiple of window_size
        pad_h = (self.window - H % self.window) % self.window
        pad_w = (self.window - W % self.window) % self.window
        lr_pad = F.pad(lr.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect")

        sr_pad = self.model(lr_pad)   # (1, 3, (H+pad)*4, (W+pad)*4)

        # Crop back to exact SR size
        sr = sr_pad[0, :, :H * 4, :W * 4]
        return sr.clamp(0.0, 1.0)

    @torch.no_grad()
    def upscale_numpy(self, lr_np: np.ndarray) -> np.ndarray:
        """
        lr_np  : (H, W, 3) uint8 or float [0,1]
        returns: (H*4, W*4, 3) float [0,1]
        """
        if lr_np.dtype == np.uint8:
            lr_np = lr_np.astype(np.float32) / 255.0
        t = torch.from_numpy(lr_np).permute(2, 0, 1).float()
        sr_t = self.upscale_tensor(t)
        return sr_t.cpu().permute(1, 2, 0).numpy()

    @torch.no_grad()
    def upscale_pil(self, pil_img: PILImage.Image) -> PILImage.Image:
        """Upscale a PIL Image and return a PIL Image."""
        lr_np = np.array(pil_img.convert("RGB"), dtype=np.uint8)
        sr_np = self.upscale_numpy(lr_np)
        return PILImage.fromarray((sr_np * 255).clip(0, 255).astype(np.uint8))
