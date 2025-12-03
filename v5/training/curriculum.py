"""
Curriculum Learning for SGAPS-MAE
Progressive training strategy with increasing difficulty.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import torch


class SamplingStrategy(Enum):
    """Sampling strategy types."""
    UNIFORM = "uniform"
    EDGE_FOCUSED = "edge_focused"
    HARD_NEGATIVE = "hard_negative"
    EXTREME_SPARSE = "extreme_sparse"


@dataclass
class SamplingPhase:
    """Configuration for a training phase."""
    
    strategy: SamplingStrategy
    sample_rate: float
    focus: str
    epochs_start: int
    epochs_end: int
    
    def is_active(self, epoch: int) -> bool:
        """Check if this phase is active for given epoch."""
        return self.epochs_start <= epoch < self.epochs_end


class CurriculumLearning:
    """
    Curriculum learning scheduler for SGAPS-MAE.
    
    Progressively increases task difficulty:
    1. Phase 1: High sample rate, uniform sampling
    2. Phase 2: Medium sample rate, edge-focused sampling
    3. Phase 3: Low sample rate, hard negative mining
    4. Phase 4: Extreme sparse, critical-only sampling
    """
    
    def __init__(
        self,
        total_epochs: int = 500,
        phase1_epochs: int = 100,
        phase2_epochs: int = 200,
        phase3_epochs: int = 200,
        initial_sample_rate: float = 0.10,
        final_sample_rate: float = 0.005
    ):
        """
        Args:
            total_epochs: Total training epochs
            phase1_epochs: Epochs for phase 1
            phase2_epochs: Epochs for phase 2
            phase3_epochs: Epochs for phase 3
            initial_sample_rate: Starting sample rate
            final_sample_rate: Final sample rate
        """
        self.total_epochs = total_epochs
        
        # Define phases
        self.phases = [
            SamplingPhase(
                strategy=SamplingStrategy.UNIFORM,
                sample_rate=initial_sample_rate,
                focus="global_structure",
                epochs_start=0,
                epochs_end=phase1_epochs
            ),
            SamplingPhase(
                strategy=SamplingStrategy.EDGE_FOCUSED,
                sample_rate=0.05,
                focus="boundaries",
                epochs_start=phase1_epochs,
                epochs_end=phase1_epochs + phase2_epochs
            ),
            SamplingPhase(
                strategy=SamplingStrategy.HARD_NEGATIVE,
                sample_rate=0.02,
                focus="high_error_regions",
                epochs_start=phase1_epochs + phase2_epochs,
                epochs_end=phase1_epochs + phase2_epochs + phase3_epochs
            ),
            SamplingPhase(
                strategy=SamplingStrategy.EXTREME_SPARSE,
                sample_rate=final_sample_rate,
                focus="critical_only",
                epochs_start=phase1_epochs + phase2_epochs + phase3_epochs,
                epochs_end=total_epochs
            ),
        ]
        
        self.current_epoch = 0
    
    def get_phase(self, epoch: Optional[int] = None) -> SamplingPhase:
        """
        Get the active phase for given epoch.
        
        Args:
            epoch: Epoch number (uses current if None)
            
        Returns:
            Active sampling phase
        """
        epoch = epoch if epoch is not None else self.current_epoch
        
        for phase in self.phases:
            if phase.is_active(epoch):
                return phase
        
        # Default to last phase
        return self.phases[-1]
    
    def get_config(self, epoch: Optional[int] = None) -> Dict:
        """
        Get sampling configuration for epoch.
        
        Args:
            epoch: Epoch number
            
        Returns:
            Configuration dictionary
        """
        phase = self.get_phase(epoch)
        
        return {
            'strategy': phase.strategy.value,
            'sample_rate': phase.sample_rate,
            'focus': phase.focus
        }
    
    def step(self) -> None:
        """Advance to next epoch."""
        self.current_epoch += 1
    
    def reset(self) -> None:
        """Reset to beginning."""
        self.current_epoch = 0


class AdaptiveSampler:
    """
    Adaptive pixel sampler based on curriculum phase.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16
    ):
        """
        Args:
            image_size: Image resolution
            patch_size: Patch size for sampling patterns
        """
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
    
    def sample_uniform(
        self,
        batch_size: int,
        sample_rate: float,
        device: torch.device
    ) -> torch.Tensor:
        """
        Generate uniform random samples.
        
        Args:
            batch_size: Number of images in batch
            sample_rate: Fraction of pixels to sample
            device: Target device
            
        Returns:
            Sample positions [B, N, 2]
        """
        num_samples = int(self.image_size * self.image_size * sample_rate)
        
        positions = []
        for _ in range(batch_size):
            indices = torch.randperm(self.image_size * self.image_size, device=device)[:num_samples]
            u = indices // self.image_size
            v = indices % self.image_size
            pos = torch.stack([u, v], dim=-1).float()
            positions.append(pos)
        
        return torch.stack(positions, dim=0)
    
    def sample_edge_focused(
        self,
        images: torch.Tensor,
        sample_rate: float
    ) -> torch.Tensor:
        """
        Sample with focus on edge regions.
        
        Args:
            images: Input images [B, 3, H, W]
            sample_rate: Fraction of pixels to sample
            
        Returns:
            Sample positions [B, N, 2]
        """
        import torch.nn.functional as F
        
        device = images.device
        B, C, H, W = images.shape
        num_samples = int(H * W * sample_rate)
        
        # Compute edge magnitude using Sobel
        gray = 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=images.dtype, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=images.dtype, device=device).view(1, 1, 3, 3)
        
        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        edge_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        
        # Sample weighted by edge magnitude
        positions = []
        for b in range(B):
            weights = edge_magnitude[b].flatten()
            weights = weights + 0.1  # Add small baseline
            weights = weights / weights.sum()
            
            indices = torch.multinomial(weights, num_samples, replacement=False)
            u = indices // W
            v = indices % W
            pos = torch.stack([u, v], dim=-1).float()
            positions.append(pos)
        
        return torch.stack(positions, dim=0)
    
    def sample_hard_negative(
        self,
        images: torch.Tensor,
        reconstructions: torch.Tensor,
        sample_rate: float
    ) -> torch.Tensor:
        """
        Sample focusing on high-error regions.
        
        Args:
            images: Original images [B, 3, H, W]
            reconstructions: Reconstructed images [B, 3, H, W]
            sample_rate: Fraction of pixels to sample
            
        Returns:
            Sample positions [B, N, 2]
        """
        device = images.device
        B, C, H, W = images.shape
        num_samples = int(H * W * sample_rate)
        
        # Compute error map
        error = (images - reconstructions).abs().mean(dim=1, keepdim=True)
        
        # Sample weighted by error
        positions = []
        for b in range(B):
            weights = error[b].flatten()
            weights = weights + 0.1
            weights = weights / weights.sum()
            
            indices = torch.multinomial(weights, num_samples, replacement=False)
            u = indices // W
            v = indices % W
            pos = torch.stack([u, v], dim=-1).float()
            positions.append(pos)
        
        return torch.stack(positions, dim=0)
    
    def sample(
        self,
        images: torch.Tensor,
        phase: SamplingPhase,
        reconstructions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample based on current curriculum phase.
        
        Args:
            images: Input images [B, 3, H, W]
            phase: Current sampling phase
            reconstructions: Previous reconstructions (for hard negative)
            
        Returns:
            Sample positions [B, N, 2]
        """
        if phase.strategy == SamplingStrategy.UNIFORM:
            return self.sample_uniform(images.shape[0], phase.sample_rate, images.device)
        
        elif phase.strategy == SamplingStrategy.EDGE_FOCUSED:
            return self.sample_edge_focused(images, phase.sample_rate)
        
        elif phase.strategy == SamplingStrategy.HARD_NEGATIVE:
            if reconstructions is not None:
                return self.sample_hard_negative(images, reconstructions, phase.sample_rate)
            return self.sample_uniform(images.shape[0], phase.sample_rate, images.device)
        
        elif phase.strategy == SamplingStrategy.EXTREME_SPARSE:
            # Use edge-focused for extreme sparse (focus on critical info)
            return self.sample_edge_focused(images, phase.sample_rate)
        
        # Default fallback
        return self.sample_uniform(images.shape[0], phase.sample_rate, images.device)
