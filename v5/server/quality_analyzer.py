"""
Quality Analyzer for SGAPS-MAE Server
Analyzes reconstruction quality without ground truth.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalSmoothnessChecker(nn.Module):
    """Check local smoothness and detect artifacts."""
    
    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = kernel_size
        
        # Sobel kernels for edge detection
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
        
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Check local smoothness of reconstruction.
        
        Args:
            image: RGB image [B, 3, H, W]
            
        Returns:
            Smoothness map [B, 1, H, W] (higher = smoother)
        """
        B, C, H, W = image.shape
        
        # Convert to grayscale
        gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        
        # Compute gradients
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        
        # Gradient magnitude
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        
        # Local variance as artifact indicator
        local_mean = F.avg_pool2d(grad_magnitude, self.kernel_size, stride=1, 
                                   padding=self.kernel_size // 2)
        local_var = F.avg_pool2d(grad_magnitude ** 2, self.kernel_size, stride=1,
                                  padding=self.kernel_size // 2) - local_mean ** 2
        
        # High variance indicates potential artifacts
        smoothness = 1.0 / (1.0 + local_var)
        
        return smoothness


class StructuralIntegrityChecker(nn.Module):
    """Verify structural integrity of edges and textures."""
    
    def __init__(self, feature_dim: int = 64):
        super().__init__()
        
        # Edge detector
        self.edge_detector = nn.Sequential(
            nn.Conv2d(3, feature_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, 1, 1),
            nn.Sigmoid()
        )
        
        # Texture analyzer
        self.texture_analyzer = nn.Sequential(
            nn.Conv2d(3, feature_dim, 3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Check structural integrity.
        
        Args:
            image: RGB image [B, 3, H, W]
            
        Returns:
            Dictionary with edge and texture maps
        """
        edge_map = self.edge_detector(image)
        texture_map = self.texture_analyzer(image)
        
        # Combined integrity score
        integrity = (edge_map + texture_map) / 2
        
        return {
            'edge_map': edge_map,
            'texture_map': texture_map,
            'integrity': integrity
        }


class LearnedQualityPredictor(nn.Module):
    """Neural network for quality prediction."""
    
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, feature_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Quality regressor
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Spatial quality map
        self.spatial_predictor = nn.Sequential(
            nn.Conv2d(feature_dim, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict quality score and map.
        
        Args:
            image: RGB image [B, 3, H, W]
            
        Returns:
            Dictionary with global score and spatial map
        """
        features = self.features(image)
        
        global_score = self.regressor(features)
        spatial_map = self.spatial_predictor(features)
        
        # Upsample spatial map to original size
        spatial_map = F.interpolate(spatial_map, size=image.shape[-2:], mode='bilinear', align_corners=False)
        
        return {
            'global_score': global_score,
            'spatial_map': spatial_map
        }


class QualityAnalyzer(nn.Module):
    """
    Complete Quality Analyzer for SGAPS-MAE.
    Evaluates reconstruction quality without ground truth.
    """
    
    def __init__(
        self,
        feature_dim: int = 128,
        kernel_size: int = 5
    ):
        """
        Args:
            feature_dim: Feature dimension for neural components
            kernel_size: Kernel size for smoothness checking
        """
        super().__init__()
        
        self.smoothness_checker = LocalSmoothnessChecker(kernel_size)
        self.integrity_checker = StructuralIntegrityChecker(feature_dim // 2)
        self.quality_predictor = LearnedQualityPredictor(feature_dim)
    
    def forward(
        self,
        reconstruction: torch.Tensor,
        return_maps: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze reconstruction quality.
        
        Args:
            reconstruction: Reconstructed image [B, 3, H, W]
            return_maps: Whether to return detailed spatial maps
            
        Returns:
            Dictionary with quality metrics
        """
        # Check local smoothness
        smoothness = self.smoothness_checker(reconstruction)
        
        # Check structural integrity
        integrity = self.integrity_checker(reconstruction)
        
        # Learned quality prediction
        quality = self.quality_predictor(reconstruction)
        
        # Combine confidence
        confidence_map = (
            smoothness * 0.3 +
            integrity['integrity'] * 0.3 +
            quality['spatial_map'] * 0.4
        )
        
        # Overall quality score
        overall_quality = quality['global_score']
        
        result = {
            'overall_quality': overall_quality,
            'confidence_map': confidence_map,
        }
        
        if return_maps:
            result['smoothness_map'] = smoothness
            result['edge_map'] = integrity['edge_map']
            result['texture_map'] = integrity['texture_map']
            result['integrity_map'] = integrity['integrity']
            result['predicted_quality_map'] = quality['spatial_map']
        
        return result
    
    def identify_problem_regions(
        self,
        reconstruction: torch.Tensor,
        threshold: float = 0.3
    ) -> torch.Tensor:
        """
        Identify regions with potential quality issues.
        
        Args:
            reconstruction: Reconstructed image [B, 3, H, W]
            threshold: Quality threshold for problem detection
            
        Returns:
            Problem region mask [B, 1, H, W]
        """
        analysis = self.forward(reconstruction, return_maps=True)
        
        # Low confidence regions are problematic
        problem_mask = (analysis['confidence_map'] < threshold).float()
        
        return problem_mask
    
    def get_sampling_priorities(
        self,
        reconstruction: torch.Tensor
    ) -> torch.Tensor:
        """
        Get sampling priorities based on quality analysis.
        
        Args:
            reconstruction: Reconstructed image [B, 3, H, W]
            
        Returns:
            Priority map [B, 1, H, W] (higher = needs more samples)
        """
        analysis = self.forward(reconstruction, return_maps=True)
        
        # Inverse of confidence is sampling priority
        priority = 1.0 - analysis['confidence_map']
        
        return priority
