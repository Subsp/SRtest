"""
Phase 0 experiment configuration.
MipNeRF360 scenes, frame counts, resolution settings.
"""

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Task 0.1 & 0.2 scenes ────────────────────────────────────────────────────
SCENES_PHASE0 = ["garden", "kitchen", "bonsai", "room", "counter"]

# Number of training frames to sample per scene for the consistency test
FRAMES_PER_SCENE = 8

# LR / SR resolution (4× scale)
LR_SIZE   = 200   # LR input for VGGT, HR-head conditioning, warp tests
SR_SCALE  = 4
SR_SIZE   = LR_SIZE * SR_SCALE  # 800

# MipNeRF360 sub-directory used as LR source (images_8 ≈ 1/8 of full res)
LR_IMAGE_SUBDIR = "images_8"
HR_IMAGE_SUBDIR = "images_2"    # used for oracle training in Task 0.2

# StableSR (or other) HR cache: sibling of images_8 under each scene
# Example: mipnerf360/kitchen/images_8/…  ↔  mipnerf360/kitchen/priors/<stem>.png
PRIORS_SUBDIR = "priors"

# ── Task 0.1 thresholds (PSNR) ───────────────────────────────────────────────
PSNR_SEVERE    = 22.0   # < 22: view-consistent SR module mandatory
PSNR_MODERATE  = 28.0   # 22–28: confidence weighting sufficient
# >= 28: negligible, simplify narrative

# ── Task 0.2 thresholds (AbsRel) ─────────────────────────────────────────────
ABSREL_OK      = 0.10   # < 0.10: VGGT directly usable
ABSREL_FINETUNE = 0.20  # 0.10–0.20: HR Head strengthening needed
# > 0.20: LoRA fine-tune on VGGT required

# ── Paths (edit to match your local dataset location) ────────────────────────
# Expected layout:
#   MIPNERF360_ROOT/
#     <scene>/
#       images/        full resolution
#       images_2/
#       images_4/
#       images_8/      ← LR source
#       priors/        ← StableSR HR cache (PRIORS_SUBDIR)
#       sparse/0/
#         cameras.bin
#         images.bin
#         points3D.bin
MIPNERF360_ROOT = "/path/to/mipnerf360"   # ← SET THIS

# Output directory for experiment results
OUTPUT_ROOT = "./results"

# VGGT: clone root must contain importable package `vggt/` (facebook/VGGT).
# On servers, the repo is often outside SRtest — set once:
#   export VGGT_ROOT=/root/autodl-tmp/vggt
# or pass  --vggt_root /root/autodl-tmp/vggt  to task02_vggt_geometry.py
VGGT_ROOT = os.environ.get("VGGT_ROOT", str(_REPO_ROOT / "vggt"))
MIP_SPLATTING_ROOT = str(_REPO_ROOT / "mip-splatting")
