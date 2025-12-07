"""
Quality Metrics for SGAPS-MAE.

Provides functions to compute image quality metrics
for reconstruction evaluation.
"""

import logging
from typing import Dict, Optional
import numpy as np
from skimage.metrics import structural_similarity


logger = logging.getLogger(__name__)


def compute_mse(
    reconstructed: np.ndarray,
    ground_truth: np.ndarray
) -> float:
    """
    Compute Mean Squared Error between two images.
    
    Args:
        reconstructed: Reconstructed image
        ground_truth: Ground truth image
        
    Returns:
        MSE value (lower is better)
    """
    if reconstructed.shape != ground_truth.shape:
        logger.warning("Shape mismatch in MSE computation")
        return float('inf')
    
    return float(np.mean((reconstructed.astype(float) - ground_truth.astype(float)) ** 2))


def compute_psnr(
    reconstructed: np.ndarray,
    ground_truth: np.ndarray,
    max_val: float = 255.0
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.
    
    Args:
        reconstructed: Reconstructed image
        ground_truth: Ground truth image
        max_val: Maximum pixel value
        
    Returns:
        PSNR value in dB (higher is better)
    """
    mse = compute_mse(reconstructed, ground_truth)
    
    if mse == 0:
        return float('inf')
    
    return float(10 * np.log10((max_val ** 2) / mse))


def compute_ssim(
    reconstructed: np.ndarray,
    ground_truth: np.ndarray,
    **kwargs
) -> float:
    """
    Compute Structural Similarity Index (SSIM) using scikit-image.
    
    Args:
        reconstructed: Reconstructed image
        ground_truth: Ground truth image
        **kwargs: Additional arguments for scikit-image's structural_similarity
        
    Returns:
        SSIM value in range [-1, 1] (higher is better)
    """
    if reconstructed.shape != ground_truth.shape:
        logger.warning("Shape mismatch in SSIM computation")
        return 0.0

    # Ensure images are at least 2D
    if reconstructed.ndim < 2:
        logger.warning("Images must be at least 2D for SSIM computation")
        return 0.0

    # The channel axis is handled automatically by scikit-image if channel_axis is set.
    # If the image is grayscale, it should be (H, W). If color, (H, W, C).
    channel_axis = -1 if reconstructed.ndim == 3 else None

    # Use a smaller window size if the image is smaller than the default window
    win_size = kwargs.get('win_size', 7)
    min_dim = min(reconstructed.shape[:2])
    if min_dim < win_size:
        win_size = min_dim - (1 - min_dim % 2) # must be odd and smaller than image
        if win_size < 3:
             logger.warning(f"Image dimension ({min_dim}) is too small for SSIM. Returning 0.")
             return 0.0
        kwargs['win_size'] = win_size

    return structural_similarity(
        ground_truth,
        reconstructed,
        data_range=255,
        channel_axis=channel_axis,
        **kwargs
    )


def compute_all_metrics(
    reconstructed: np.ndarray,
    ground_truth: np.ndarray
) -> Dict[str, float]:
    """
    Compute all quality metrics.
    
    Args:
        reconstructed: Reconstructed image
        ground_truth: Ground truth image
        
    Returns:
        Dictionary with MSE, PSNR, and SSIM
    """
    return {
        "mse": compute_mse(reconstructed, ground_truth),
        "psnr": compute_psnr(reconstructed, ground_truth),
        "ssim": compute_ssim(reconstructed, ground_truth)
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    """
    Format metrics for display.
    
    Args:
        metrics: Dictionary of metric values
        
    Returns:
        Formatted string
    """
    return (
        f"MSE: {metrics.get('mse', 0):.4f}, "
        f"PSNR: {metrics.get('psnr', 0):.2f} dB, "
        f"SSIM: {metrics.get('ssim', 0):.4f}"
    )
