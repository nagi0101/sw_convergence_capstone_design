"""
SGAPS-MAE: Server-Guided Adaptive Pixel Sampling Masked Autoencoder
Main model implementation for game session replay.
"""

from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparse_encoder import SparsePixelEncoder, SparsePixelDecoder
from .temporal_memory import TemporalMemoryBank


class QualityEstimator(nn.Module):
    """
    Estimates reconstruction quality and uncertainty using Monte Carlo Dropout.
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 256,
        num_mc_samples: int = 10,
        dropout_rate: float = 0.1
    ):
        """
        Args:
            feature_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_mc_samples: Number of Monte Carlo samples for uncertainty
            dropout_rate: Dropout rate for uncertainty estimation
        """
        super().__init__()
        
        self.num_mc_samples = num_mc_samples
        self.dropout_rate = dropout_rate
        
        # Quality prediction network
        self.quality_net = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid()
        )
        
        # Entropy estimator
        self.entropy_head = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim // 2, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, 1)
        )
    
    def forward(
        self,
        features: torch.Tensor,
        return_uncertainty: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate quality and uncertainty.
        
        Args:
            features: Dense feature map [B, C, H, W]
            return_uncertainty: Whether to compute MC Dropout uncertainty
            
        Returns:
            Dictionary with quality_map, uncertainty_map
        """
        # Quality prediction
        quality_map = self.quality_net(features)
        
        result = {'quality_map': quality_map}
        
        if return_uncertainty:
            # Monte Carlo Dropout for uncertainty estimation
            self.train()  # Enable dropout
            
            predictions = []
            for _ in range(self.num_mc_samples):
                pred = self.quality_net(features)
                predictions.append(pred)
            
            predictions = torch.stack(predictions, dim=0)
            uncertainty = predictions.var(dim=0)
            
            # Entropy term
            mean_pred = predictions.mean(dim=0)
            entropy = -mean_pred * torch.log(mean_pred + 1e-8)
            
            result['uncertainty_map'] = uncertainty + 0.5 * entropy
            result['mean_prediction'] = mean_pred
        
        return result


class ImportanceAnalyzer(nn.Module):
    """
    Analyzes pixel importance for adaptive sampling.
    Combines uncertainty, attention patterns, and temporal consistency.
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 256,
        uncertainty_weight: float = 0.4,
        attention_weight: float = 0.3,
        temporal_weight: float = 0.3
    ):
        """
        Args:
            feature_dim: Input feature dimension
            hidden_dim: Hidden dimension
            uncertainty_weight: Weight for uncertainty-based importance
            attention_weight: Weight for attention-based importance
            temporal_weight: Weight for temporal consistency importance
        """
        super().__init__()
        
        self.uncertainty_weight = uncertainty_weight
        self.attention_weight = attention_weight
        self.temporal_weight = temporal_weight
        
        # Attention analysis
        self.attention_analyzer = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid()
        )
        
        # Temporal consistency checker
        self.temporal_checker = nn.Sequential(
            nn.Conv2d(feature_dim * 2, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid()
        )
        
        # Combined importance prediction
        self.importance_combiner = nn.Sequential(
            nn.Conv2d(3, hidden_dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # Previous frame storage
        self.prev_features: Optional[torch.Tensor] = None
    
    def forward(
        self,
        features: torch.Tensor,
        uncertainty_map: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute pixel importance map.
        
        Args:
            features: Dense feature map [B, C, H, W]
            uncertainty_map: Optional uncertainty map [B, 1, H, W]
            
        Returns:
            Importance map [B, 1, H, W]
        """
        # Attention-based importance
        attention_importance = self.attention_analyzer(features)
        
        # Temporal importance
        if self.prev_features is not None:
            combined = torch.cat([features, self.prev_features], dim=1)
            temporal_importance = self.temporal_checker(combined)
        else:
            temporal_importance = torch.ones_like(attention_importance) * 0.5
        
        # Update previous features
        self.prev_features = features.detach()
        
        # Uncertainty importance
        if uncertainty_map is None:
            uncertainty_importance = torch.ones_like(attention_importance) * 0.5
        else:
            uncertainty_importance = uncertainty_map
        
        # Combine all importance factors
        combined_importance = torch.cat([
            uncertainty_importance * self.uncertainty_weight,
            attention_importance * self.attention_weight,
            temporal_importance * self.temporal_weight
        ], dim=1)
        
        importance_map = self.importance_combiner(combined_importance)
        
        return importance_map
    
    def reset(self) -> None:
        """Reset temporal state."""
        self.prev_features = None


class CoordinateGenerator(nn.Module):
    """
    Generates sampling coordinates based on importance map.
    Implements hierarchical budget allocation.
    """
    
    def __init__(
        self,
        budget: int = 500,
        critical_ratio: float = 0.1,
        important_ratio: float = 0.2,
        moderate_ratio: float = 0.3
    ):
        """
        Args:
            budget: Total number of pixels to sample
            critical_ratio: Ratio for critical pixels
            important_ratio: Ratio for important pixels
            moderate_ratio: Ratio for moderate pixels
        """
        super().__init__()
        
        self.budget = budget
        self.tiers = {
            'critical': critical_ratio,
            'important': important_ratio,
            'moderate': moderate_ratio,
            'optional': 1.0 - critical_ratio - important_ratio - moderate_ratio
        }
    
    def forward(
        self,
        importance_map: torch.Tensor,
        static_mask: Optional[torch.Tensor] = None,
        budget: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate sampling coordinates.
        
        Args:
            importance_map: Pixel importance [B, 1, H, W]
            static_mask: Optional mask to exclude static regions [B, 1, H, W]
            budget: Override default budget
            
        Returns:
            Coordinates [N, 2] (u, v)
        """
        budget = budget or self.budget
        
        # Apply static mask to reduce sampling in static regions
        if static_mask is not None:
            # Invert static mask: sample more in dynamic regions
            dynamic_weight = 1.0 - static_mask * 0.8
            importance_map = importance_map * dynamic_weight
        
        B, _, H, W = importance_map.shape
        
        # Flatten and get top-k indices
        flat_importance = importance_map.view(B, -1)
        
        # Select top pixels by importance
        _, top_indices = flat_importance.topk(budget, dim=-1)
        
        # Convert to coordinates
        coords_list = []
        for b in range(B):
            indices = top_indices[b]
            u = indices // W
            v = indices % W
            coords = torch.stack([u, v], dim=-1)
            coords_list.append(coords)
        
        # Return first batch for now (single-image processing)
        return coords_list[0].float()
    
    def hierarchical_allocation(
        self,
        importance_map: torch.Tensor,
        budget: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Allocate budget hierarchically by importance tier.
        
        Args:
            importance_map: Pixel importance [B, 1, H, W]
            budget: Total budget
            
        Returns:
            Dictionary mapping tier to coordinates
        """
        budget = budget or self.budget
        B, _, H, W = importance_map.shape
        
        flat_importance = importance_map.view(B, -1)
        sorted_vals, sorted_indices = flat_importance.sort(dim=-1, descending=True)
        
        allocation = {}
        start_idx = 0
        
        for tier, ratio in self.tiers.items():
            if tier == 'optional':
                continue  # Skip optional tier
                
            tier_budget = int(budget * ratio)
            end_idx = start_idx + tier_budget
            
            tier_indices = sorted_indices[:, start_idx:end_idx]
            
            coords_list = []
            for b in range(B):
                indices = tier_indices[b]
                u = indices // W
                v = indices % W
                coords = torch.stack([u, v], dim=-1)
                coords_list.append(coords)
            
            allocation[tier] = coords_list[0].float()
            start_idx = end_idx
        
        return allocation


class SGAPS_MAE(nn.Module):
    """
    Server-Guided Adaptive Pixel Sampling Masked Autoencoder.
    
    Complete model for game session replay that:
    1. Encodes sparse pixel observations
    2. Reconstructs full frames using information diffusion
    3. Estimates quality and importance for next-frame sampling
    4. Generates adaptive sampling coordinates
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        pixel_embed_dim: int = 384,
        pos_embed_dim: int = 384,
        hidden_dim: int = 768,
        output_dim: int = 768,
        num_gat_layers: int = 6,
        num_heads: int = 8,
        knn_k: int = 8,
        diffusion_steps: int = 10,
        sampling_budget: int = 500,
        dropout: float = 0.1
    ):
        """
        Args:
            image_size: Target image resolution (H, W)
            pixel_embed_dim: Pixel value embedding dimension
            pos_embed_dim: Position embedding dimension
            hidden_dim: Hidden dimension
            output_dim: Feature output dimension
            num_gat_layers: Number of Graph Attention layers
            num_heads: Number of attention heads
            knn_k: K for KNN graph
            diffusion_steps: Number of diffusion iterations
            sampling_budget: Default sampling budget
            dropout: Dropout rate
        """
        super().__init__()
        
        self.image_size = image_size
        self.sampling_budget = sampling_budget
        
        # Sparse Pixel Encoder
        self.encoder = SparsePixelEncoder(
            pixel_embed_dim=pixel_embed_dim,
            pos_embed_dim=pos_embed_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_gat_layers=num_gat_layers,
            num_heads=num_heads,
            knn_k=knn_k,
            dropout=dropout,
            max_resolution=max(image_size),
            diffusion_steps=diffusion_steps
        )
        
        # Decoder
        self.decoder = SparsePixelDecoder(
            feature_dim=output_dim,
            hidden_dim=hidden_dim // 2,
            num_layers=4,
            num_heads=num_heads,
            output_channels=3
        )
        
        # Quality estimation
        self.quality_estimator = QualityEstimator(
            feature_dim=output_dim,
            num_mc_samples=10,
            dropout_rate=dropout
        )
        
        # Importance analysis
        self.importance_analyzer = ImportanceAnalyzer(
            feature_dim=output_dim
        )
        
        # Coordinate generation
        self.coordinate_generator = CoordinateGenerator(
            budget=sampling_budget
        )
        
        # Temporal memory
        self.memory = TemporalMemoryBank(
            resolution=image_size,
            feature_dim=3
        )
    
    def encode(
        self,
        pixel_values: torch.Tensor,
        pixel_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode sparse pixels to dense feature map.
        
        Args:
            pixel_values: RGB values [N, 3]
            pixel_positions: Pixel positions [N, 2]
            
        Returns:
            Dense feature map [1, C, H, W]
        """
        return self.encoder(pixel_values, pixel_positions, self.image_size)
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode features to RGB image.
        
        Args:
            features: Dense feature map [B, C, H, W]
            
        Returns:
            RGB image [B, 3, H, W]
        """
        return self.decoder(features)
    
    def reconstruct(
        self,
        pixel_values: torch.Tensor,
        pixel_positions: torch.Tensor,
        use_memory: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Full reconstruction from sparse pixels.
        
        Args:
            pixel_values: RGB values [N, 3]
            pixel_positions: Pixel positions [N, 2]
            use_memory: Whether to use temporal memory
            
        Returns:
            Dictionary with reconstruction, features, quality info
        """
        device = pixel_values.device
        
        # Update memory
        if use_memory:
            static_mask, static_values = self.memory(pixel_positions, pixel_values)
        else:
            static_mask = torch.zeros(1, 1, *self.image_size, device=device)
            static_values = torch.zeros(1, 3, *self.image_size, device=device)
        
        # Encode sparse pixels
        features = self.encode(pixel_values, pixel_positions)
        
        # Decode to RGB
        reconstruction = self.decode(features)
        
        # Blend with static memory
        if use_memory and static_mask.sum() > 0:
            reconstruction = reconstruction * (1 - static_mask) + static_values * static_mask
        
        # Quality estimation
        quality_info = self.quality_estimator(features, return_uncertainty=True)
        
        return {
            'reconstruction': reconstruction,
            'features': features,
            'quality_map': quality_info['quality_map'],
            'uncertainty_map': quality_info['uncertainty_map'],
            'static_mask': static_mask
        }
    
    def get_next_coordinates(
        self,
        features: torch.Tensor,
        uncertainty_map: Optional[torch.Tensor] = None,
        static_mask: Optional[torch.Tensor] = None,
        budget: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate coordinates for next frame sampling.
        
        Args:
            features: Current feature map [B, C, H, W]
            uncertainty_map: Current uncertainty [B, 1, H, W]
            static_mask: Static region mask [B, 1, H, W]
            budget: Override default budget
            
        Returns:
            Sampling coordinates [N, 2]
        """
        # Compute importance
        importance_map = self.importance_analyzer(features, uncertainty_map)
        
        # Generate coordinates
        coords = self.coordinate_generator(
            importance_map,
            static_mask=static_mask,
            budget=budget
        )
        
        return coords
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_positions: torch.Tensor,
        return_next_coords: bool = True,
        use_memory: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass.
        
        Args:
            pixel_values: RGB values [N, 3]
            pixel_positions: Pixel positions [N, 2]
            return_next_coords: Whether to compute next sampling coordinates
            use_memory: Whether to use temporal memory
            
        Returns:
            Dictionary with all outputs
        """
        # Reconstruct
        result = self.reconstruct(
            pixel_values,
            pixel_positions,
            use_memory=use_memory
        )
        
        # Generate next coordinates
        if return_next_coords:
            next_coords = self.get_next_coordinates(
                result['features'],
                result['uncertainty_map'],
                result['static_mask']
            )
            result['next_coordinates'] = next_coords
        
        return result
    
    def reset_memory(self) -> None:
        """Reset temporal memory."""
        self.memory.reset()
        self.importance_analyzer.reset()


class SGAPSMAELoss(nn.Module):
    """
    Combined loss function for SGAPS-MAE training.
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
            temporal_weight: Weight for temporal smoothness
        """
        super().__init__()
        
        self.sampled_weight = sampled_weight
        self.perceptual_weight = perceptual_weight
        self.structural_weight = structural_weight
        self.temporal_weight = temporal_weight
        
        self.mse = nn.MSELoss(reduction='none')
        self.l1 = nn.L1Loss(reduction='none')
        
        # Previous prediction for temporal loss
        self.prev_pred: Optional[torch.Tensor] = None
    
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
                
                # Inverse importance weighting
                if importance_map is not None:
                    weight = 1.0 / (importance_map[:, :, u, v] + 0.1)
                else:
                    weight = 1.0
                
                sampled_loss += weight.mean() * F.mse_loss(pixel_pred, pixel_target)
        
        sampled_loss = sampled_loss / max(len(sampled_positions), 1)
        
        # 2. Perceptual loss (simplified as L1)
        perceptual_loss = self.l1(prediction, target).mean()
        
        # 3. Structural loss (SSIM-like)
        structural_loss = 1.0 - self._ssim(prediction, target)
        
        # 4. Temporal smoothness loss
        if self.prev_pred is not None:
            temporal_loss = F.mse_loss(prediction, self.prev_pred)
        else:
            temporal_loss = torch.tensor(0.0, device=prediction.device)
        
        self.prev_pred = prediction.detach()
        
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
    
    def _ssim(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        window_size: int = 11
    ) -> torch.Tensor:
        """Simplified SSIM computation."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_x = F.avg_pool2d(x, window_size, stride=1, padding=window_size // 2)
        mu_y = F.avg_pool2d(y, window_size, stride=1, padding=window_size // 2)
        
        sigma_x = F.avg_pool2d(x * x, window_size, stride=1, padding=window_size // 2) - mu_x ** 2
        sigma_y = F.avg_pool2d(y * y, window_size, stride=1, padding=window_size // 2) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=window_size // 2) - mu_x * mu_y
        
        ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
        
        return ssim.mean()
    
    def reset(self) -> None:
        """Reset temporal state."""
        self.prev_pred = None
