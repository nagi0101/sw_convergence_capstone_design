"""
Image Reconstructor for SGAPS-MAE Server.

Reconstructs full images from sparse pixel samples.
Phase 1: OpenCV inpainting as baseline.
Future phases will use trained MAE model.
"""

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    logging.warning("OpenCV not available, reconstruction disabled")


logger = logging.getLogger(__name__)


class OpenCVReconstructor:
    """
    Image reconstructor using OpenCV inpainting.
    
    Phase 1 implementation that uses classical inpainting
    algorithms to fill in missing pixels.
    
    This serves as a baseline for comparison with the
    MAE-based reconstruction in later phases.
    """
    
    def __init__(
        self,
        inpaint_radius: int = 3,
        method: str = "telea"
    ):
        """
        Initialize the reconstructor.
        
        Args:
            inpaint_radius: Radius of circular neighborhood for inpainting
            method: Inpainting method - "telea" or "ns" (Navier-Stokes)
        """
        self.inpaint_radius = inpaint_radius
        self.method = method
        
        if HAS_OPENCV:
            self.inpaint_flag = (
                cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
            )
        
        logger.info(f"OpenCVReconstructor initialized: radius={inpaint_radius}, method={method}")
    
    def reconstruct(
        self,
        pixels: List[Dict],
        resolution: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """
        Reconstruct image from sparse pixel samples.
        
        Args:
            pixels: List of pixel data dicts with 'u', 'v', 'value' keys
            resolution: Target resolution (width, height)
            
        Returns:
            Reconstructed grayscale image as numpy array, or None if failed
        """
        if not HAS_OPENCV:
            logger.error("OpenCV not available for reconstruction")
            return None
        
        if not pixels:
            logger.warning("No pixels provided for reconstruction")
            return None
        
        width, height = resolution
        
        # Create image with known pixels
        image = np.zeros((height, width), dtype=np.uint8)
        mask = np.ones((height, width), dtype=np.uint8) * 255  # 255 = needs inpainting
        
        # Fill in known pixels
        for pixel in pixels:
            u = pixel.get("u", 0)
            v = pixel.get("v", 0)
            value = pixel.get("value", 0)
            
            # Convert UV to pixel coordinates
            x = int(u * (width - 1))
            y = int(v * (height - 1))
            
            # Clamp to valid range
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            
            # Set pixel value and mark as known
            image[y, x] = int(value)
            mask[y, x] = 0  # 0 = known pixel
        
        # Perform inpainting
        try:
            reconstructed = cv2.inpaint(
                image, mask,
                self.inpaint_radius,
                self.inpaint_flag
            )
            
            logger.debug(f"Reconstructed image from {len(pixels)} pixels")
            return reconstructed
            
        except Exception as e:
            logger.error(f"Inpainting failed: {e}")
            return None
    
    def compute_metrics(
        self,
        reconstructed: np.ndarray,
        ground_truth: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute quality metrics between reconstructed and ground truth.
        
        Args:
            reconstructed: Reconstructed image
            ground_truth: Original ground truth image
            
        Returns:
            Dictionary with MSE, PSNR, and SSIM metrics
        """
        if reconstructed is None or ground_truth is None:
            return {"mse": float("inf"), "psnr": 0.0, "ssim": 0.0}
        
        # Ensure same shape
        if reconstructed.shape != ground_truth.shape:
            logger.warning("Shape mismatch in metric computation")
            return {"mse": float("inf"), "psnr": 0.0, "ssim": 0.0}
        
        # MSE
        mse = float(np.mean((reconstructed.astype(float) - ground_truth.astype(float)) ** 2))
        
        # PSNR
        if mse > 0:
            psnr = float(10 * np.log10((255.0 ** 2) / mse))
        else:
            psnr = float("inf")
        
        # SSIM (simplified version)
        ssim = self._compute_ssim(reconstructed, ground_truth)
        
        return {
            "mse": mse,
            "psnr": psnr,
            "ssim": ssim
        }
    
    def _compute_ssim(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        window_size: int = 11
    ) -> float:
        """
        Compute Structural Similarity Index (SSIM).
        
        Simplified implementation - for production, consider
        using skimage.metrics.structural_similarity.
        """
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        # Mean
        mu1 = cv2.GaussianBlur(img1, (window_size, window_size), 1.5)
        mu2 = cv2.GaussianBlur(img2, (window_size, window_size), 1.5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Variance and covariance
        sigma1_sq = cv2.GaussianBlur(img1 ** 2, (window_size, window_size), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 ** 2, (window_size, window_size), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (window_size, window_size), 1.5) - mu1_mu2
        
        # SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return float(np.mean(ssim_map))
