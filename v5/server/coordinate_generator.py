"""
Coordinate Generator for SGAPS-MAE Server
Generates adaptive sampling coordinates based on importance analysis.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImportanceMapGenerator(nn.Module):
    """Generate importance map from multiple sources."""
    
    def __init__(self, feature_dim: int = 64):
        super().__init__()
        
        # Importance fusion network
        self.fusion = nn.Sequential(
            nn.Conv2d(4, feature_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        uncertainty_map: torch.Tensor,
        attention_map: torch.Tensor,
        temporal_error: torch.Tensor,
        quality_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse multiple importance sources.
        
        Args:
            uncertainty_map: Model uncertainty [B, 1, H, W]
            attention_map: Information gaps [B, 1, H, W]
            temporal_error: Temporal inconsistency [B, 1, H, W]
            quality_map: Quality-based importance [B, 1, H, W]
            
        Returns:
            Combined importance map [B, 1, H, W]
        """
        combined = torch.cat([
            uncertainty_map,
            attention_map,
            temporal_error,
            quality_map
        ], dim=1)
        
        importance = self.fusion(combined)
        
        return importance


class HierarchicalBudgetAllocator:
    """
    Allocate sampling budget hierarchically by importance tier.
    """
    
    def __init__(
        self,
        critical_ratio: float = 0.10,
        important_ratio: float = 0.20,
        moderate_ratio: float = 0.30,
        optional_ratio: float = 0.40
    ):
        """
        Args:
            critical_ratio: Budget for critical pixels (UI, objectives)
            important_ratio: Budget for important pixels (characters, objects)
            moderate_ratio: Budget for moderate pixels (environment details)
            optional_ratio: Budget for optional pixels (background)
        """
        self.tiers = {
            'critical': critical_ratio,
            'important': important_ratio,
            'moderate': moderate_ratio,
            'optional': optional_ratio
        }
    
    def allocate(
        self,
        importance_map: torch.Tensor,
        total_budget: int,
        exclude_optional: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Allocate budget by importance tier.
        
        Args:
            importance_map: Pixel importance [B, 1, H, W]
            total_budget: Total number of pixels to sample
            exclude_optional: Whether to skip optional tier
            
        Returns:
            Dictionary mapping tier to coordinates [N, 2]
        """
        B, _, H, W = importance_map.shape
        flat = importance_map.view(B, -1)
        
        # Sort by importance
        sorted_vals, sorted_indices = flat.sort(dim=-1, descending=True)
        
        allocation = {}
        start_idx = 0
        
        for tier, ratio in self.tiers.items():
            if exclude_optional and tier == 'optional':
                continue
            
            tier_budget = int(total_budget * ratio)
            end_idx = start_idx + tier_budget
            
            tier_indices = sorted_indices[:, start_idx:end_idx]
            
            # Convert to coordinates
            u = tier_indices // W
            v = tier_indices % W
            coords = torch.stack([u, v], dim=-1).squeeze(0)
            
            allocation[tier] = coords.float()
            start_idx = end_idx
        
        return allocation
    
    def flatten_allocation(
        self,
        allocation: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Flatten hierarchical allocation to single coordinate tensor.
        
        Args:
            allocation: Tier to coordinates mapping
            
        Returns:
            All coordinates [N, 2]
        """
        all_coords = []
        for tier in ['critical', 'important', 'moderate']:
            if tier in allocation:
                all_coords.append(allocation[tier])
        
        if all_coords:
            return torch.cat(all_coords, dim=0)
        return torch.empty(0, 2)


class PatternRecognizer(nn.Module):
    """
    Recognize patterns in coordinate distribution for efficient encoding.
    """
    
    def __init__(self):
        super().__init__()
    
    def identify_pattern(
        self,
        coordinates: torch.Tensor
    ) -> Dict[str, any]:
        """
        Identify pattern in coordinates.
        
        Args:
            coordinates: Pixel coordinates [N, 2]
            
        Returns:
            Pattern information dictionary
        """
        if len(coordinates) == 0:
            return {'type': 'empty'}
        
        # Check for grid pattern
        u_vals = coordinates[:, 0].unique()
        v_vals = coordinates[:, 1].unique()
        
        if len(u_vals) * len(v_vals) == len(coordinates):
            # Potential grid pattern
            u_diffs = u_vals[1:] - u_vals[:-1]
            v_diffs = v_vals[1:] - v_vals[:-1]
            
            if u_diffs.std() < 0.1 and v_diffs.std() < 0.1:
                return {
                    'type': 'grid',
                    'start': (u_vals.min().item(), v_vals.min().item()),
                    'step': (u_diffs.mean().item(), v_diffs.mean().item()),
                    'size': (len(u_vals), len(v_vals))
                }
        
        # Check for clustering
        from torch.cluster import KMeans if hasattr(torch, 'cluster') else None
        
        # Simple clustering check: compute mean distance to centroid
        centroid = coordinates.mean(dim=0)
        distances = (coordinates - centroid).norm(dim=1)
        mean_distance = distances.mean().item()
        std_distance = distances.std().item()
        
        if std_distance / (mean_distance + 1e-8) < 0.5:
            return {
                'type': 'cluster',
                'center': centroid.tolist(),
                'radius': mean_distance
            }
        
        # Default: raw coordinates
        return {
            'type': 'raw',
            'count': len(coordinates)
        }


class ServerCoordinateGenerator(nn.Module):
    """
    Server-side coordinate generator for adaptive pixel sampling.
    """
    
    def __init__(
        self,
        default_budget: int = 500,
        feature_dim: int = 64
    ):
        """
        Args:
            default_budget: Default number of pixels to sample
            feature_dim: Feature dimension for neural components
        """
        super().__init__()
        
        self.default_budget = default_budget
        
        self.importance_generator = ImportanceMapGenerator(feature_dim)
        self.budget_allocator = HierarchicalBudgetAllocator()
        self.pattern_recognizer = PatternRecognizer()
    
    def generate_importance_map(
        self,
        uncertainty_map: torch.Tensor,
        attention_map: Optional[torch.Tensor] = None,
        temporal_error: Optional[torch.Tensor] = None,
        quality_map: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate combined importance map.
        
        Args:
            uncertainty_map: Model uncertainty [B, 1, H, W]
            attention_map: Information gaps [B, 1, H, W]
            temporal_error: Temporal inconsistency [B, 1, H, W]
            quality_map: Quality-based importance [B, 1, H, W]
            
        Returns:
            Combined importance map [B, 1, H, W]
        """
        B, _, H, W = uncertainty_map.shape
        device = uncertainty_map.device
        
        # Default maps if not provided
        if attention_map is None:
            attention_map = torch.ones(B, 1, H, W, device=device) * 0.5
        if temporal_error is None:
            temporal_error = torch.ones(B, 1, H, W, device=device) * 0.5
        if quality_map is None:
            quality_map = 1.0 - uncertainty_map  # Inverse of uncertainty
        
        importance = self.importance_generator(
            uncertainty_map,
            attention_map,
            temporal_error,
            quality_map
        )
        
        return importance
    
    def select_top_pixels(
        self,
        importance_map: torch.Tensor,
        budget: Optional[int] = None,
        static_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Select top-N important pixels.
        
        Args:
            importance_map: Pixel importance [B, 1, H, W]
            budget: Number of pixels to select
            static_mask: Optional mask to reduce static region sampling [B, 1, H, W]
            
        Returns:
            Selected coordinates [N, 2]
        """
        budget = budget or self.default_budget
        
        # Apply static mask weighting
        if static_mask is not None:
            dynamic_boost = 1.0 - static_mask * 0.7
            importance_map = importance_map * dynamic_boost
        
        B, _, H, W = importance_map.shape
        flat = importance_map.view(B, -1)
        
        # Top-k selection
        _, top_indices = flat.topk(budget, dim=-1)
        
        # Convert to coordinates
        u = top_indices // W
        v = top_indices % W
        coords = torch.stack([u, v], dim=-1).squeeze(0)
        
        return coords.float()
    
    def forward(
        self,
        uncertainty_map: torch.Tensor,
        attention_map: Optional[torch.Tensor] = None,
        temporal_error: Optional[torch.Tensor] = None,
        quality_map: Optional[torch.Tensor] = None,
        static_mask: Optional[torch.Tensor] = None,
        budget: Optional[int] = None,
        hierarchical: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Generate sampling coordinates.
        
        Args:
            uncertainty_map: Model uncertainty [B, 1, H, W]
            attention_map: Information gaps [B, 1, H, W]
            temporal_error: Temporal inconsistency [B, 1, H, W]
            quality_map: Quality-based importance [B, 1, H, W]
            static_mask: Static region mask [B, 1, H, W]
            budget: Total sampling budget
            hierarchical: Whether to use hierarchical allocation
            
        Returns:
            Dictionary with coordinates and metadata
        """
        budget = budget or self.default_budget
        
        # Generate importance map
        importance_map = self.generate_importance_map(
            uncertainty_map,
            attention_map,
            temporal_error,
            quality_map
        )
        
        if hierarchical:
            # Hierarchical allocation
            allocation = self.budget_allocator.allocate(
                importance_map,
                budget,
                exclude_optional=True
            )
            coordinates = self.budget_allocator.flatten_allocation(allocation)
            
            return {
                'coordinates': coordinates,
                'importance_map': importance_map,
                'allocation': allocation
            }
        else:
            # Simple top-k selection
            coordinates = self.select_top_pixels(
                importance_map,
                budget,
                static_mask
            )
            
            # Identify pattern for compression
            pattern = self.pattern_recognizer.identify_pattern(coordinates)
            
            return {
                'coordinates': coordinates,
                'importance_map': importance_map,
                'pattern': pattern
            }
