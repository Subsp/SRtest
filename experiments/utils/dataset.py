"""
MipNeRF360 dataset loader for Phase 0 experiments.

Loads a fixed number of training frames from a scene, along with COLMAP
cameras and sparse depth maps.
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image as PILImage

from .colmap_reader import (
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary_with_ids,
    camera_params_to_K,
    scale_K,
    qvec2rotmat,
    build_sparse_depth,
    interpolate_depth,
)


def _find_sparse_dir(scene_root: str) -> str:
    """Return path to sparse/0 or sparse directory."""
    for candidate in ["sparse/0", "sparse"]:
        d = os.path.join(scene_root, candidate)
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(f"Cannot find COLMAP sparse dir under {scene_root}")


def load_scene_frames(
    scene_root: str,
    image_subdir: str = "images_8",
    n_frames: int = 8,
    target_lr_size: int = 200,
    seed: int = 42,
) -> List[Dict]:
    """
    Load n_frames training frames from a MipNeRF360 scene.

    Returns a list of dicts, each containing:
        name        : image filename stem
        image_lr    : np.uint8 (target_lr_size, target_lr_size, 3)
        K_lr        : 3×3 intrinsic for LR image
        K_sr        : 3×3 intrinsic for 4× SR image (4*target_lr_size)
        R           : 3×3 rotation matrix (world→camera, OpenCV)
        t           : (3,) translation vector (world→camera)
        depth_lr    : np.float32 (target_lr_size, target_lr_size) sparse-interp depth
        orig_wh     : (W, H) original image resolution
    """
    sparse_dir = _find_sparse_dir(scene_root)
    cameras   = read_cameras_binary(os.path.join(sparse_dir, "cameras.bin"))
    images    = read_images_binary(os.path.join(sparse_dir, "images.bin"))
    points3D  = read_points3D_binary_with_ids(os.path.join(sparse_dir, "points3D.bin"))

    # Build filename → colmap_image map for the LR images that exist on disk
    lr_dir = os.path.join(scene_root, image_subdir)
    if not os.path.isdir(lr_dir):
        raise FileNotFoundError(f"LR image directory not found: {lr_dir}")

    existing = {os.path.basename(p) for p in Path(lr_dir).glob("*")
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")}

    valid_entries = [
        img for img in images.values()
        if os.path.basename(img.name) in existing
    ]
    if len(valid_entries) == 0:
        raise RuntimeError(
            f"No COLMAP images match files in {lr_dir}. "
            "Check that images.bin and image_subdir refer to the same scene."
        )

    random.seed(seed)
    selected = random.sample(valid_entries, min(n_frames, len(valid_entries)))
    selected.sort(key=lambda x: x.name)   # deterministic order after sampling

    frames = []
    lr_wh = (target_lr_size, target_lr_size)
    sr_wh = (target_lr_size * 4, target_lr_size * 4)

    for colmap_img in selected:
        camera = cameras[colmap_img.camera_id]
        orig_wh = (camera.width, camera.height)
        K_orig  = camera_params_to_K(camera)

        # ── load & resize LR image ──────────────────────────────────────────
        img_path = os.path.join(lr_dir, os.path.basename(colmap_img.name))
        pil_img  = PILImage.open(img_path).convert("RGB")
        # actual size in images_8 folder (may differ from camera.width/height
        # if the COLMAP model was built on full-res images)
        actual_wh = pil_img.size   # (W, H)
        pil_lr = pil_img.resize(lr_wh, PILImage.BICUBIC)
        image_lr = np.array(pil_lr, dtype=np.uint8)

        # ── scale intrinsics ────────────────────────────────────────────────
        K_lr = scale_K(K_orig, orig_wh, lr_wh)
        K_sr = scale_K(K_orig, orig_wh, sr_wh)

        # ── camera extrinsics (world→camera, OpenCV) ────────────────────────
        R = qvec2rotmat(colmap_img.qvec)   # 3×3
        t = colmap_img.tvec                 # (3,)

        # ── sparse depth at LR resolution ───────────────────────────────────
        us, vs, ds = build_sparse_depth(
            colmap_img, camera, points3D,
            target_wh=lr_wh,
        )
        depth_lr = interpolate_depth(us, vs, ds, *lr_wh, method="linear")

        frames.append(dict(
            name     = Path(colmap_img.name).stem,
            image_lr = image_lr,
            K_lr     = K_lr,
            K_sr     = K_sr,
            R        = R,
            t        = t,
            depth_lr = depth_lr,
            orig_wh  = orig_wh,
        ))

    return frames


def frames_to_tensors(frames: List[Dict], device="cpu"):
    """
    Convert loaded frame dicts to torch tensors (float [0,1]).

    Returns list of dicts with same keys but tensor values where applicable.
    """
    import torch

    result = []
    for f in frames:
        result.append({
            "name"    : f["name"],
            "image_lr": torch.from_numpy(f["image_lr"]).float().permute(2, 0, 1).div(255.0).to(device),
            "K_lr"    : torch.from_numpy(f["K_lr"]).float().to(device),
            "K_sr"    : torch.from_numpy(f["K_sr"]).float().to(device),
            "R"       : torch.from_numpy(f["R"]).float().to(device),
            "t"       : torch.from_numpy(f["t"]).float().to(device),
            "depth_lr": torch.from_numpy(f["depth_lr"]).float().to(device),
            "orig_wh" : f["orig_wh"],
        })
    return result
