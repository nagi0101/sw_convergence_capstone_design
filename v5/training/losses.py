"""
Loss Functions for SGAPS-MAE Training
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features.
    Simplified version that doesn't require pretrained VGG.
    """
    
    def __init__(self):
        super().__init__()
        
        # Simple feature extractor (can be replaced with VGG)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )
    
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute perceptual loss.
        
        Args:
            prediction: Predicted image [B, 3, H, W]
            target: Target image [B, 3, H, W]
            
        Returns:
            Perceptual loss scalar
        """
        pred_features = self.features(prediction)
        target_features = self.features(target)
        
        return F.l1_loss(pred_features, target_features)


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency loss for video sequences.
    """
    
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
        self.prev_prediction: Optional[torch.Tensor] = None
    
    def forward(
        self,
        prediction: torch.Tensor,
        optical_flow: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute temporal consistency loss.
        
        Args:
            prediction: Current prediction [B, 3, H, W]
            optical_flow: Optional flow for warping [B, 2, H, W]
            
        Returns:
            Temporal consistency loss
        """
        if self.prev_prediction is None:
            self.prev_prediction = prediction.detach()
            return torch.tensor(0.0, device=prediction.device)
        
        # Simple L2 temporal smoothness
        if optical_flow is None:
            loss = F.mse_loss(prediction, self.prev_prediction)
        else:
            # Warp previous prediction
            warped = self._warp(self.prev_prediction, optical_flow)
            loss = F.mse_loss(prediction, warped)
        
        self.prev_prediction = prediction.detach()
        
        return self.weight * loss
    
    def _warp(
        self,
        image: torch.Tensor,
        flow: torch.Tensor
    ) -> torch.Tensor:
        """Warp image using optical flow."""
        B, C, H, W = image.shape
        
        # Create grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=image.device),
            torch.arange(W, device=image.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).float()
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
        
        # Add flow
        flow_permuted = flow.permute(0, 2, 3, 1)
        grid = grid + flow_permuted
        
        # Normalize grid to [-1, 1]
        grid[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0
        
        # Warp
        warped = F.grid_sample(image, grid, mode='bilinear', 
                               padding_mode='border', align_corners=True)
        
        return warped
    
    def reset(self) -> None:
        """Reset temporal state."""
        self.prev_prediction = None


class StructuralLoss(nn.Module):
    """
    Structural similarity loss (SSIM-based).
    """
    
    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size
    
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute structural loss (1 - SSIM).
        
        Args:
            prediction: Predicted image [B, 3, H, W]
            target: Target image [B, 3, H, W]
            
        Returns:
            Structural loss scalar
        """
        ssim = self._ssim(prediction, target)
        return 1.0 - ssim
    
    def _ssim(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """Compute SSIM."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_x = F.avg_pool2d(x, self.window_size, stride=1, 
                           padding=self.window_size // 2)
        mu_y = F.avg_pool2d(y, self.window_size, stride=1, 
                           padding=self.window_size // 2)
        
        sigma_x = F.avg_pool2d(x * x, self.window_size, stride=1,
                               padding=self.window_size // 2) - mu_x ** 2
        sigma_y = F.avg_pool2d(y * y, self.window_size, stride=1,
                               padding=self.window_size // 2) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, self.window_size, stride=1,
                                padding=self.window_size // 2) - mu_x * mu_y
        
        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
        
        return ssim_map.mean()


class SGAPSMAELoss(nn.Module):
    """
    Combined loss function for SGAPS-MAE training.
    
    L_total = λ_sampled × L_sampled + λ_perceptual × L_perceptual + 
              λ_structural × L_structural + λ_temporal × L_temporal
    """
    
    def __init__(
        self,
        sampled_weight: float = 0.3,
        perceptual_weight: float = 0.4,
        structural_weight: float = 0.2,
        temporal_weight: float = 0.1
    ):
        """
        Args:
            sampled_weight: Weight for sampled pixel loss
            perceptual_weight: Weight for perceptual loss
            structural_weight: Weight for structural loss
            temporal_weight: Weight for temporal consistency
        """
        super().__init__()
        
        self.sampled_weight = sampled_weight
        self.perceptual_weight = perceptual_weight
        self.structural_weight = structural_weight
        self.temporal_weight = temporal_weight
        
        self.perceptual_loss = PerceptualLoss()
        self.temporal_loss = TemporalConsistencyLoss(weight=1.0)
        self.structural_loss = StructuralLoss()
    
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        sampled_positions: torch.Tensor,
        importance_map: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            prediction: Predicted image [B, 3, H, W]
            target: Ground truth [B, 3, H, W]
            sampled_positions: Sampled pixel positions [N, 2]
            importance_map: Importance weights [B, 1, H, W]
            
        Returns:
            Dictionary with individual losses and total
        """
        B, C, H, W = prediction.shape
        
        # 1. Sampled pixel loss (weighted by inverse importance)
        sampled_loss = torch.tensor(0.0, device=prediction.device)
        
        for pos in sampled_positions:
            u, v = int(pos[0].item()), int(pos[1].item())
            if 0 <= u < H and 0 <= v < W:
                pixel_pred = prediction[:, :, u, v]
                pixel_target = target[:, :, u, v]
                
                if importance_map is not None:
                    weight = 1.0 / (importance_map[:, :, u, v].mean() + 0.1)
                else:
                    weight = 1.0
                
                sampled_loss += weight * F.mse_loss(pixel_pred, pixel_target)
        
        sampled_loss = sampled_loss / max(len(sampled_positions), 1)
        
        # 2. Perceptual loss
        perceptual_loss = self.perceptual_loss(prediction, target)
        
        # 3. Structural loss
        structural_loss = self.structural_loss(prediction, target)
        
        # 4. Temporal consistency
        temporal_loss = self.temporal_loss(prediction)
        
        # Total loss
        total_loss = (
            self.sampled_weight * sampled_loss +
            self.perceptual_weight * perceptual_loss +
            self.structural_weight * structural_loss +
            self.temporal_weight * temporal_loss
        )
        
        return {
            'total': total_loss,
            'sampled': sampled_loss,
            'perceptual': perceptual_loss,
            'structural': structural_loss,
            'temporal': temporal_loss
        }
    
    def reset_temporal(self) -> None:
        """Reset temporal loss state."""
        self.temporal_loss.reset()
