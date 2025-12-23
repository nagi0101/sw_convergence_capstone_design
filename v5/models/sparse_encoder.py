"""
Sparse Pixel Encoder for SGAPS-MAE
Encodes sparse pixel observations using Graph Neural Networks.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_attention import GraphAttentionLayer, build_knn_graph
from .information_diffusion import InformationDiffusion


class ContinuousPositionalEncoding(nn.Module):
    """
    Continuous positional encoding for arbitrary pixel positions.
    Uses sinusoidal encodings with learnable frequency parameters.
    """
    
    def __init__(
        self,
        embed_dim: int = 384,
        max_resolution: int = 256,
        num_frequencies: int = 64
    ):
        """
        Args:
            embed_dim: Output embedding dimension
            max_resolution: Maximum image resolution
            num_frequencies: Number of frequency bands
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_resolution = max_resolution
        self.num_frequencies = num_frequencies
        
        # Learnable frequency parameters
        self.freq_bands = nn.Parameter(
            torch.linspace(0, math.log2(max_resolution), num_frequencies)
        )
        
        # Project to embedding dimension
        # 2D position * num_frequencies * 2 (sin, cos)
        input_dim = 2 * num_frequencies * 2
        self.projection = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Encode continuous pixel positions.
        
        Args:
            positions: Pixel positions [N, 2] (u, v coordinates)
            
        Returns:
            Positional embeddings [N, embed_dim]
        """
        # Normalize positions to [0, 1]
        positions_norm = positions.float() / self.max_resolution
        
        # Compute frequency encodings
        freqs = 2 ** self.freq_bands  # [num_freq]
        
        # [N, 2] x [num_freq] -> [N, 2, num_freq]
        freq_inputs = positions_norm.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)
        
        # Sin and cos encodings
        sin_enc = torch.sin(2 * math.pi * freq_inputs)
        cos_enc = torch.cos(2 * math.pi * freq_inputs)
        
        # Concatenate: [N, 2, num_freq, 2] -> [N, 2 * num_freq * 2]
        pos_encoding = torch.cat([sin_enc, cos_enc], dim=-1)
        pos_encoding = pos_encoding.flatten(1)
        
        # Project to embedding dimension
        embedding = self.projection(pos_encoding)
        
        return embedding


class PixelValueEncoder(nn.Module):
    """
    Encoder for RGB pixel values with optional auxiliary features.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 384,
        use_layernorm: bool = True
    ):
        """
        Args:
            in_channels: Number of input channels (3 for RGB)
            embed_dim: Output embedding dimension
            use_layernorm: Whether to apply layer normalization
        """
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )
        
        if use_layernorm:
            self.norm = nn.LayerNorm(embed_dim)
        else:
            self.norm = nn.Identity()
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB pixel values.
        
        Args:
            pixel_values: RGB values [N, 3] normalized to [0, 1]
            
        Returns:
            Pixel embeddings [N, embed_dim]
        """
        x = self.encoder(pixel_values)
        x = self.norm(x)
        return x


class SparsePixelEncoder(nn.Module):
    """
    Sparse Pixel Encoder using Graph Attention Networks.
    
    Takes sparse pixel observations (positions + RGB values) and produces
    a dense feature map through graph-based processing and information diffusion.
    """
    
    def __init__(
        self,
        pixel_embed_dim: int = 384,
        pos_embed_dim: int = 384,
        hidden_dim: int = 768,
        output_dim: int = 768,
        num_gat_layers: int = 6,
        num_heads: int = 8,
        knn_k: int = 8,
        dropout: float = 0.1,
        max_resolution: int = 256,
        diffusion_steps: int = 10
    ):
        """
        Args:
            pixel_embed_dim: Dimension for pixel value embedding
            pos_embed_dim: Dimension for position embedding
            hidden_dim: Hidden dimension (pixel_embed_dim + pos_embed_dim)
            output_dim: Output feature dimension
            num_gat_layers: Number of Graph Attention layers
            num_heads: Number of attention heads
            knn_k: Number of nearest neighbors for graph construction
            dropout: Dropout rate
            max_resolution: Maximum image resolution
            diffusion_steps: Number of diffusion iterations
        """
        super().__init__()
        
        self.knn_k = knn_k
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Pixel value encoder
        self.pixel_encoder = PixelValueEncoder(
            in_channels=3,
            embed_dim=pixel_embed_dim
        )
        
        # Position encoder
        self.position_encoder = ContinuousPositionalEncoding(
            embed_dim=pos_embed_dim,
            max_resolution=max_resolution
        )
        
        # Feature combination
        combined_dim = pixel_embed_dim + pos_embed_dim
        self.feature_proj = nn.Linear(combined_dim, hidden_dim)
        
        # Graph Attention layers
        self.graph_layers = nn.ModuleList([
            GraphAttentionLayer(
                in_features=hidden_dim,
                out_features=hidden_dim,
                heads=num_heads,
                dropout=dropout,
                concat=True
            )
            for _ in range(num_gat_layers)
        ])
        
        # Layer norms between GAT layers
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_gat_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Information diffusion module
        self.diffusion = InformationDiffusion(
            feature_dim=output_dim,
            num_steps=diffusion_steps,
            use_attention=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def encode_sparse(
        self,
        pixel_values: torch.Tensor,
        pixel_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode sparse pixel observations into feature vectors.
        
        Args:
            pixel_values: RGB values [N, 3]
            pixel_positions: Pixel positions [N, 2] (u, v)
            
        Returns:
            Encoded features [N, output_dim]
        """
        # Encode pixel values and positions
        pixel_feat = self.pixel_encoder(pixel_values)  # [N, pixel_embed_dim]
        pos_feat = self.position_encoder(pixel_positions)  # [N, pos_embed_dim]
        
        # Combine features
        combined = torch.cat([pixel_feat, pos_feat], dim=-1)  # [N, combined_dim]
        features = self.feature_proj(combined)  # [N, hidden_dim]
        
        # Build KNN graph
        edge_index = build_knn_graph(pixel_positions, k=self.knn_k)
        
        # Graph Attention layers with residual connections
        for i, (gat_layer, norm) in enumerate(zip(self.graph_layers, self.layer_norms)):
            residual = features
            features = gat_layer(features, edge_index)
            features = norm(features + residual)  # Residual connection
            if i < len(self.graph_layers) - 1:
                features = F.elu(features)
                features = self.dropout(features)
        
        # Output projection
        features = self.output_proj(features)  # [N, output_dim]
        
        return features
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_positions: torch.Tensor,
        target_shape: Tuple[int, int] = (224, 224)
    ) -> torch.Tensor:
        """
        Full forward pass: encode sparse pixels and diffuse to dense map.
        
        Args:
            pixel_values: RGB values [N, 3]
            pixel_positions: Pixel positions [N, 2] (u, v)
            target_shape: Target spatial resolution (H, W)
            
        Returns:
            Dense feature map [1, output_dim, H, W]
        """
        # Encode sparse pixels
        sparse_features = self.encode_sparse(pixel_values, pixel_positions)
        
        # Diffuse to dense map
        dense_map = self.diffusion(sparse_features, pixel_positions, target_shape)
        
        return dense_map


class SparsePixelDecoder(nn.Module):
    """
    Lightweight decoder to reconstruct RGB image from dense feature map.
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 384,
        num_layers: int = 4,
        num_heads: int = 8,
        output_channels: int = 3
    ):
        """
        Args:
            feature_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            output_channels: Output channels (3 for RGB)
        """
        super().__init__()
        
        # Project to hidden dimension
        self.input_proj = nn.Conv2d(feature_dim, hidden_dim, 1)
        
        # Transformer layers (simplified as conv blocks for 2D)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim * 4, 1),
                nn.GELU(),
                nn.Conv2d(hidden_dim * 4, hidden_dim, 1),
                nn.GroupNorm(8, hidden_dim)
            ))
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, output_channels, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode features to RGB image.
        
        Args:
            features: Dense feature map [B, C, H, W]
            
        Returns:
            RGB image [B, 3, H, W]
        """
        x = self.input_proj(features)
        
        for layer in self.layers:
            x = x + layer(x)  # Residual connection
        
        output = self.output_proj(x)
        
        return output
