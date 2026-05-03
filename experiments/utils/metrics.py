"""
Image and depth quality metrics used in Phase 0 experiments.

  Image metrics  : psnr, ssim, edge_weighted_psnr, edge_weighted_ssim
  Depth metrics  : abs_rel, scale_invariant_l1, rmse
"""

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


# ── image metrics (numpy, float [0,1]) ───────────────────────────────────────

def psnr(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray = None) -> float:
    """
    PSNR between two float [0,1] images of shape (H,W,3) or (H,W).
    If mask is provided (bool H×W), only masked pixels are considered.
    """
    if mask is not None:
        if img1.ndim == 3:
            mask3 = mask[:, :, None]
            sq_err = ((img1 - img2) ** 2)[mask3.repeat(3, axis=2)]
        else:
            sq_err = ((img1 - img2) ** 2)[mask]
        mse = sq_err.mean()
    else:
        mse = ((img1 - img2) ** 2).mean()

    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(1.0 / mse))


def ssim(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray = None) -> float:
    """
    SSIM between two float [0,1] RGB images (H,W,3).
    If mask is provided, the SSIM map is averaged over masked pixels only.
    """
    score, ssim_map = structural_similarity(
        img1, img2,
        multichannel=True,
        channel_axis=2,
        data_range=1.0,
        full=True,
    )
    if mask is not None:
        return float(ssim_map[mask].mean())
    return float(score)


def sobel_edge_mask(img: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """
    Return a bool H×W mask of edge pixels using the Sobel operator.

    img: float [0,1], shape (H,W,3) or (H,W).
    threshold: normalised gradient magnitude threshold.
    """
    if img.ndim == 3:
        gray = img.mean(axis=2)
    else:
        gray = img

    gray_t = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0)

    # Sobel kernels
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                       dtype=torch.float32).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                       dtype=torch.float32).view(1, 1, 3, 3)

    gx = F.conv2d(gray_t, kx, padding=1)
    gy = F.conv2d(gray_t, ky, padding=1)
    mag = (gx ** 2 + gy ** 2).sqrt().squeeze().numpy()
    return mag > threshold


# ── depth metrics (numpy, metric units) ──────────────────────────────────────

def _align_scale(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray = None):
    """Median-scale alignment: returns scale s such that median(s*pred[mask]) = median(gt[mask])."""
    if mask is None:
        mask = np.ones_like(pred, dtype=bool)
    valid = mask & (gt > 0) & (pred > 0)
    if valid.sum() == 0:
        return 1.0
    s = np.median(gt[valid]) / np.median(pred[valid])
    return float(s)


def abs_rel(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray = None,
            align_scale: bool = True) -> float:
    """
    Absolute Relative Error:  mean(|pred - gt| / gt)

    align_scale: if True, apply median-scale alignment before computing.
    """
    if mask is None:
        mask = np.ones_like(pred, dtype=bool)
    valid = mask & (gt > 0) & (pred > 0)
    if valid.sum() == 0:
        return float("nan")

    p = pred[valid]
    g = gt[valid]
    if align_scale:
        s = np.median(g) / np.median(p)
        p = p * s
    return float(np.mean(np.abs(p - g) / g))


def scale_invariant_l1(pred: np.ndarray, gt: np.ndarray,
                       mask: np.ndarray = None) -> float:
    """
    Scale-Invariant L1 (log-space):
      mean(|log(pred) - log(gt) - mean(log(pred/gt))|)
    """
    if mask is None:
        mask = np.ones_like(pred, dtype=bool)
    valid = mask & (gt > 0) & (pred > 0)
    if valid.sum() == 0:
        return float("nan")

    log_diff = np.log(pred[valid]) - np.log(gt[valid])
    return float(np.mean(np.abs(log_diff - log_diff.mean())))


def rmse(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray = None,
         align_scale: bool = True) -> float:
    """Root Mean Squared Error with optional median-scale alignment."""
    if mask is None:
        mask = np.ones_like(pred, dtype=bool)
    valid = mask & (gt > 0) & (pred > 0)
    if valid.sum() == 0:
        return float("nan")

    p = pred[valid]
    g = gt[valid]
    if align_scale:
        s = np.median(g) / np.median(p)
        p = p * s
    return float(np.sqrt(np.mean((p - g) ** 2)))


# ── summary helper ────────────────────────────────────────────────────────────

def compute_all_image_metrics(
    img_ref: np.ndarray,
    img_cmp: np.ndarray,
    edge_threshold: float = 0.1,
) -> dict:
    """
    Compute PSNR/SSIM globally and on Sobel edge regions.

    img_ref, img_cmp: float [0,1], shape (H,W,3)
    """
    edge_mask = sobel_edge_mask(img_ref, threshold=edge_threshold)
    return {
        "psnr_full"    : psnr(img_ref, img_cmp),
        "ssim_full"    : ssim(img_ref, img_cmp),
        "psnr_edge"    : psnr(img_ref, img_cmp, mask=edge_mask),
        "ssim_edge"    : ssim(img_ref, img_cmp, mask=edge_mask),
        "edge_ratio"   : float(edge_mask.mean()),
    }


def compute_all_depth_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray = None,
) -> dict:
    """Compute AbsRel, Scale-Invariant L1, and RMSE."""
    return {
        "abs_rel"          : abs_rel(pred, gt, mask),
        "scale_inv_l1"     : scale_invariant_l1(pred, gt, mask),
        "rmse"             : rmse(pred, gt, mask),
    }
