"""
Quality Metrics for SGAPS-MAE.

Provides functions to compute image quality metrics
for reconstruction evaluation.
"""

import logging
from typing import Dict, Optional
import numpy as np


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
    window_size: int = 11,
    k1: float = 0.01,
    k2: float = 0.03
) -> float:
    """
    Compute Structural Similarity Index (SSIM).
    
    Implementation based on the original SSIM paper.
    For production use, consider using skimage.metrics.structural_similarity.
    
    Args:
        reconstructed: Reconstructed image
        ground_truth: Ground truth image
        window_size: Size of the sliding window
        k1, k2: Stability constants
        
    Returns:
        SSIM value in range [-1, 1] (higher is better)
    """
    if reconstructed.shape != ground_truth.shape:
        logger.warning("Shape mismatch in SSIM computation")
        return 0.0
    
    # Constants
    L = 255.0  # Dynamic range
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    
    # Convert to float
    img1 = reconstructed.astype(np.float64)
    img2 = ground_truth.astype(np.float64)
    
    # Calculate means using simple box filter
    kernel_size = window_size
    
    # Pad images
    pad = kernel_size // 2
    img1_padded = np.pad(img1, pad, mode='reflect')
    img2_padded = np.pad(img2, pad, mode='reflect')
    
    # Calculate local means
    def local_mean(img):
        result = np.zeros_like(img1)
        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                window = img[i:i+kernel_size, j:j+kernel_size]
                result[i, j] = np.mean(window)
        return result
    
    mu1 = local_mean(img1_padded)
    mu2 = local_mean(img2_padded)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Calculate local variances and covariance
    def local_var(img, mu):
        result = np.zeros_like(img1)
        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                window = img[i:i+kernel_size, j:j+kernel_size]
                result[i, j] = np.mean(window ** 2) - mu[i, j] ** 2
        return result
    
    def local_cov(img_a, img_b, mu_a, mu_b):
        result = np.zeros_like(img1)
        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                win_a = img_a[i:i+kernel_size, j:j+kernel_size]
                win_b = img_b[i:i+kernel_size, j:j+kernel_size]
                result[i, j] = np.mean(win_a * win_b) - mu_a[i, j] * mu_b[i, j]
        return result
    
    sigma1_sq = local_var(img1_padded, mu1)
    sigma2_sq = local_var(img2_padded, mu2)
    sigma12 = local_cov(img1_padded, img2_padded, mu1, mu2)
    
    # Calculate SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return float(np.mean(ssim_map))


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
