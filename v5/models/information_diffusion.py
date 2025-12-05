"""
Information Diffusion Module for SGAPS-MAE
Implements anisotropic diffusion to spread sparse pixel information
to reconstruct the full image.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnisotropicDiffusionKernel(nn.Module):
    """
    Learnable anisotropic diffusion kernel that preserves edges.
    Based on Perona-Malik diffusion equation.
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        """
        Args:
            feature_dim: Dimension of feature vectors
            hidden_dim: Hidden dimension for conductance network
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Gradient computation kernels (Sobel-like)
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
        
        # Laplacian kernel for diffusion
        self.register_buffer('laplacian', torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3))
        
        # Learnable conductance function
        self.conductance_net = nn.Sequential(
            nn.Conv2d(feature_dim * 2, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, feature_dim, 1),
            nn.Sigmoid()
        )
        
        # Diffusion rate parameter
        self.diffusion_rate = nn.Parameter(torch.tensor(0.1))
    
    def compute_gradients(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute spatial gradients using Sobel filters.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Gradients in x and y directions
        """
        B, C, H, W = x.shape
        
        # Process each channel
        grad_x_list = []
        grad_y_list = []
        
        for c in range(C):
            grad_x = F.conv2d(x[:, c:c+1], self.sobel_x, padding=1)
            grad_y = F.conv2d(x[:, c:c+1], self.sobel_y, padding=1)
            grad_x_list.append(grad_x)
            grad_y_list.append(grad_y)
        
        grad_x = torch.cat(grad_x_list, dim=1)
        grad_y = torch.cat(grad_y_list, dim=1)
        
        return grad_x, grad_y
    
    def compute_conductance(
        self,
        x: torch.Tensor,
        grad_x: torch.Tensor,
        grad_y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute edge-aware conductance map.
        
        Args:
            x: Input features [B, C, H, W]
            grad_x: X gradients [B, C, H, W]
            grad_y: Y gradients [B, C, H, W]
            
        Returns:
            Conductance map [B, C, H, W]
        """
        # Gradient magnitude
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        
        # Learnable conductance
        combined = torch.cat([grad_magnitude, x], dim=1)
        conductance = self.conductance_net(combined)
        
        return conductance
    
    def diffusion_step(
        self,
        x: torch.Tensor,
        conductance: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Single diffusion step with edge preservation.
        
        Args:
            x: Input tensor [B, C, H, W]
            conductance: Conductance map [B, C, H, W]
            mask: Optional mask for known pixels [B, 1, H, W]
            
        Returns:
            Diffused tensor [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Compute Laplacian for each channel
        laplacian_list = []
        for c in range(C):
            lap = F.conv2d(x[:, c:c+1], self.laplacian, padding=1)
            laplacian_list.append(lap)
        laplacian = torch.cat(laplacian_list, dim=1)
        
        # Apply conductance-weighted diffusion
        diffusion_rate = torch.sigmoid(self.diffusion_rate)
        delta = diffusion_rate * conductance * laplacian
        
        # Update
        x_new = x + delta
        
        # Preserve known pixels if mask provided
        if mask is not None:
            x_new = x_new * (1 - mask) + x * mask
        
        return x_new


class NonlinearTransform(nn.Module):
    """
    Learnable nonlinear transformation for diffusion refinement.
    """
    
    def __init__(self, feature_dim: int, num_steps: int = 10):
        """
        Args:
            feature_dim: Feature dimension
            num_steps: Number of diffusion steps
        """
        super().__init__()
        
        self.num_steps = num_steps
        
        # Step-specific transformations
        self.transforms = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
                nn.GroupNorm(8, feature_dim),
                nn.GELU(),
                nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            )
            for _ in range(num_steps)
        ])
        
        # Residual weights
        self.residual_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.1)
            for _ in range(num_steps)
        ])
    
    def forward(self, x: torch.Tensor, step: int) -> torch.Tensor:
        """
        Apply nonlinear transformation for given diffusion step.
        
        Args:
            x: Input tensor [B, C, H, W]
            step: Current diffusion step
            
        Returns:
            Transformed tensor [B, C, H, W]
        """
        residual = self.transforms[step](x)
        weight = torch.sigmoid(self.residual_weights[step])
        return x + weight * residual


class InformationDiffusion(nn.Module):
    """
    Information Diffusion Module for spreading sparse pixel information.
    
    Implements physics-inspired anisotropic diffusion with learned components
    to reconstruct full feature maps from sparse pixel observations.
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        num_steps: int = 10,
        hidden_dim: int = 64,
        use_attention: bool = True
    ):
        """
        Args:
            feature_dim: Dimension of feature vectors
            num_steps: Number of diffusion iterations
            hidden_dim: Hidden dimension for internal networks
            use_attention: Whether to use attention-based refinement
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_steps = num_steps
        self.use_attention = use_attention
        
        # Diffusion kernel
        self.diffusion_kernel = AnisotropicDiffusionKernel(feature_dim, hidden_dim)
        
        # Nonlinear transforms
        self.nonlinear = NonlinearTransform(feature_dim, num_steps)
        
        # Attention-based refinement
        if use_attention:
            self.attention_refine = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(feature_dim)
        
        # Final projection
        self.final_proj = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.GroupNorm(8, feature_dim),
            nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, 1)
        )
    
    def initialize_grid(
        self,
        sparse_features: torch.Tensor,
        positions: torch.Tensor,
        target_shape: Tuple[int, int] = (224, 224)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize dense grid with sparse features.
        
        Args:
            sparse_features: Sparse pixel features [N, C]
            positions: Pixel positions [N, 2] (u, v)
            target_shape: Target spatial resolution (H, W)
            
        Returns:
            grid: Initialized feature grid [1, C, H, W]
            mask: Binary mask of known pixels [1, 1, H, W]
        """
        device = sparse_features.device
        dtype = sparse_features.dtype
        H, W = target_shape
        C = self.feature_dim
        
        # Initialize empty grid
        grid = torch.zeros(1, C, H, W, device=device, dtype=dtype)
        mask = torch.zeros(1, 1, H, W, device=device, dtype=dtype)
        
        # Place sparse features
        for i, (pos, feat) in enumerate(zip(positions, sparse_features)):
            u, v = int(pos[0].item()), int(pos[1].item())
            if 0 <= u < H and 0 <= v < W:
                grid[0, :, u, v] = feat
                mask[0, 0, u, v] = 1.0
        
        return grid, mask
    
    def forward(
        self,
        sparse_features: torch.Tensor,
        positions: torch.Tensor,
        target_shape: Tuple[int, int] = (224, 224)
    ) -> torch.Tensor:
        """
        Diffuse sparse pixel features to dense feature map.
        
        Args:
            sparse_features: Sparse pixel features [N, C]
            positions: Pixel positions [N, 2] (u, v)
            target_shape: Target spatial resolution (H, W)
            
        Returns:
            Dense feature map [1, C, H, W]
        """
        # Initialize grid
        grid, mask = self.initialize_grid(sparse_features, positions, target_shape)
        
        # Store original values for preservation
        original_grid = grid.clone()
        
        # Iterative diffusion
        for step in range(self.num_steps):
            # Compute gradients
            grad_x, grad_y = self.diffusion_kernel.compute_gradients(grid)
            
            # Compute edge-aware conductance
            conductance = self.diffusion_kernel.compute_conductance(grid, grad_x, grad_y)
            
            # Diffusion step
            grid = self.diffusion_kernel.diffusion_step(grid, conductance, mask)
            
            # Nonlinear refinement
            grid = self.nonlinear(grid, step)
            
            # Preserve known pixels
            grid = grid * (1 - mask) + original_grid * mask
        
        # Attention-based refinement
        if self.use_attention:
            grid = self._attention_refinement(grid, mask)
        
        # Final projection
        grid = self.final_proj(grid)
        
        return grid
    
    def _attention_refinement(
        self,
        grid: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Refine diffused features using self-attention.
        
        Args:
            grid: Feature grid [B, C, H, W]
            mask: Known pixel mask [B, 1, H, W]
            
        Returns:
            Refined grid [B, C, H, W]
        """
        B, C, H, W = grid.shape
        
        # Reshape for attention: [B, H*W, C]
        x = grid.flatten(2).transpose(1, 2)
        
        # Self-attention
        attn_out, _ = self.attention_refine(x, x, x)
        
        # Residual connection
        x = self.attention_norm(x + attn_out)
        
        # Reshape back
        grid = x.transpose(1, 2).view(B, C, H, W)
        
        return grid


class BatchInformationDiffusion(InformationDiffusion):
    """
    Batched version of Information Diffusion for efficient processing.
    """
    
    def forward_batch(
        self,
        batch_sparse_features: list,
        batch_positions: list,
        target_shape: Tuple[int, int] = (224, 224)
    ) -> torch.Tensor:
        """
        Process a batch of sparse observations.
        
        Args:
            batch_sparse_features: List of sparse features [N_i, C]
            batch_positions: List of positions [N_i, 2]
            target_shape: Target spatial resolution
            
        Returns:
            Batch of dense feature maps [B, C, H, W]
        """
        results = []
        
        for features, positions in zip(batch_sparse_features, batch_positions):
            result = self.forward(features, positions, target_shape)
            results.append(result)
        
        return torch.cat(results, dim=0)
