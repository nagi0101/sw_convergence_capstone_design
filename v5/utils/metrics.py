"""
Evaluation Metrics for SGAPS-MAE
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def compute_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0
) -> torch.Tensor:
    """
    Compute Peak Signal-to-Noise Ratio.
    
    Args:
        prediction: Predicted image [B, C, H, W]
        target: Target image [B, C, H, W]
        max_val: Maximum pixel value
        
    Returns:
        PSNR value in dB
    """
    mse = torch.mean((prediction - target) ** 2)
    
    if mse == 0:
        return torch.tensor(float('inf'))
    
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    
    return psnr


def compute_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    channel: Optional[int] = None
) -> torch.Tensor:
    """
    Compute Structural Similarity Index.
    
    Args:
        prediction: Predicted image [B, C, H, W]
        target: Target image [B, C, H, W]
        window_size: Window size for local statistics
        channel: Number of channels (inferred if None)
        
    Returns:
        SSIM value
    """
    if channel is None:
        channel = prediction.shape[1]
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu_x = F.avg_pool2d(prediction, window_size, stride=1, 
                       padding=window_size // 2)
    mu_y = F.avg_pool2d(target, window_size, stride=1, 
                       padding=window_size // 2)
    
    sigma_x = F.avg_pool2d(prediction ** 2, window_size, stride=1,
                           padding=window_size // 2) - mu_x ** 2
    sigma_y = F.avg_pool2d(target ** 2, window_size, stride=1,
                           padding=window_size // 2) - mu_y ** 2
    sigma_xy = F.avg_pool2d(prediction * target, window_size, stride=1,
                            padding=window_size // 2) - mu_x * mu_y
    
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    
    return ssim_map.mean()


def compute_lpips(
    prediction: torch.Tensor,
    target: torch.Tensor,
    net: Optional[torch.nn.Module] = None
) -> torch.Tensor:
    """
    Compute Learned Perceptual Image Patch Similarity.
    
    Simplified version using basic feature extraction.
    For full LPIPS, use lpips package.
    
    Args:
        prediction: Predicted image [B, C, H, W]
        target: Target image [B, C, H, W]
        net: Optional feature network
        
    Returns:
        LPIPS value (lower is better)
    """
    if net is None:
        # Simple feature extraction
        features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(inplace=True),
        ).to(prediction.device)
        
        pred_features = features(prediction)
        target_features = features(target)
    else:
        pred_features = net(prediction)
        target_features = net(target)
    
    # L2 distance in feature space
    lpips = torch.mean((pred_features - target_features) ** 2)
    
    return lpips


class MetricsLogger:
    """Logger for tracking metrics over training."""
    
    def __init__(self):
        self.metrics = {}
    
    def update(self, name: str, value: float, n: int = 1) -> None:
        """Update metric."""
        if name not in self.metrics:
            self.metrics[name] = {'sum': 0.0, 'count': 0}
        
        self.metrics[name]['sum'] += value * n
        self.metrics[name]['count'] += n
    
    def get(self, name: str) -> float:
        """Get average metric value."""
        if name not in self.metrics or self.metrics[name]['count'] == 0:
            return 0.0
        return self.metrics[name]['sum'] / self.metrics[name]['count']
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics = {}
    
    def get_all(self) -> dict:
        """Get all averaged metrics."""
        return {name: self.get(name) for name in self.metrics}
