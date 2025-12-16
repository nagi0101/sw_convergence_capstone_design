"""
Positional Encoding for continuous, non-grid coordinates.
"""

import torch
import torch.nn as nn
import math

class ContinuousPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for continuous (u, v) coordinates.
    This allows the model to understand the precise location of each sparse pixel,
    unlike traditional positional encodings that work on a fixed grid.
    """

    def __init__(self, embed_dim: int, max_freq: int = 10, num_bands: int = 64):
        """
        Args:
            embed_dim: The dimensionality of the embedding. Must be divisible by 4.
            max_freq: The maximum frequency for the sinusoidal functions.
            num_bands: The number of frequency bands. embed_dim should be 4 * num_bands.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_freq = max_freq
        self.num_bands = embed_dim // 4

        # Create frequency bands
        freqs = torch.logspace(
            0,
            math.log10(max_freq),
            self.num_bands,
        )
        self.register_buffer('freqs', freqs)

    def forward(self, embeddings: torch.Tensor, uv_coords: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input embeddings.
        
        Args:
            embeddings: Input embeddings. Shape: [B, N, embed_dim]
            uv_coords: UV coordinates for each pixel. Shape: [B, N, 2] (u, v in [0, 1])

        Returns:
            Embeddings with added positional encoding. Shape: [B, N, embed_dim]
        """
        B, N, _ = uv_coords.shape
        if self.embed_dim % 4 != 0:
            raise ValueError(
                f"Embedding dimension {self.embed_dim} must be divisible by 4."
            )

        # Separate u and v coordinates
        u = uv_coords[:, :, 0:1]  # Shape: [B, N, 1]
        v = uv_coords[:, :, 1:2]  # Shape: [B, N, 1]

        # Create sin/cos encodings for each frequency band
        u_enc = []
        v_enc = []
        for freq in self.freqs:
            u_enc.append(torch.sin(2 * math.pi * freq * u))
            u_enc.append(torch.cos(2 * math.pi * freq * u))
            v_enc.append(torch.sin(2 * math.pi * freq * v))
            v_enc.append(torch.cos(2 * math.pi * freq * v))
        
        # Concatenate all encodings
        # The result will have embed_dim channels
        pos_encoding = torch.cat(u_enc + v_enc, dim=-1) # Shape: [B, N, embed_dim]

        # Add positional encoding to the original embeddings
        return embeddings + pos_encoding


class GaussianFourierFeatures(nn.Module):
    """
    Gaussian Fourier Features (Random Fourier Features) for continuous coordinates.
    
    This encoding overcomes the spectral bias of neural networks, enabling them
    to learn high-frequency functions. The frequencies are sampled from a Gaussian
    distribution and can optionally be made learnable.
    
    Reference:
        Tancik et al., "Fourier Features Let Networks Learn High Frequency 
        Functions in Low Dimensional Domains" (NeurIPS 2020)
    """

    def __init__(
        self, 
        embed_dim: int, 
        sigma: float = 10.0, 
        learnable: bool = True
    ):
        """
        Args:
            embed_dim: The dimensionality of the embedding. Must be divisible by 4.
            sigma: Bandwidth for Gaussian sampling of frequencies. Higher values
                   capture higher frequency details but may cause instability.
            learnable: If True, frequency matrix B becomes a learnable parameter.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.sigma = sigma
        self.learnable = learnable
        
        if embed_dim % 4 != 0:
            raise ValueError(
                f"Embedding dimension {embed_dim} must be divisible by 4."
            )
        
        # Number of frequency components per coordinate axis (u and v)
        # We need: 2 coords * 2 (sin/cos) * num_freqs = embed_dim
        # So num_freqs = embed_dim // 4
        num_freqs = embed_dim // 4
        
        # Initialize frequency matrix from Gaussian distribution
        # B has shape [2, num_freqs] for 2D coordinates (u, v)
        B_init = torch.randn(2, num_freqs) * sigma
        
        if learnable:
            self.B = nn.Parameter(B_init)
        else:
            self.register_buffer('B', B_init)
    
    def forward(self, embeddings: torch.Tensor, uv_coords: torch.Tensor) -> torch.Tensor:
        """
        Adds Gaussian Fourier positional encoding to the input embeddings.
        
        Args:
            embeddings: Input embeddings. Shape: [B, N, embed_dim]
            uv_coords: UV coordinates for each pixel. Shape: [B, N, 2] (u, v in [0, 1])

        Returns:
            Embeddings with added positional encoding. Shape: [B, N, embed_dim]
        """
        # Project coordinates through frequency matrix
        # uv_coords: [B, N, 2], self.B: [2, num_freqs]
        # Result: [B, N, num_freqs]
        proj = 2 * math.pi * torch.matmul(uv_coords, self.B)
        
        # Create sin/cos features and concatenate
        # Each gives [B, N, num_freqs], concatenated to [B, N, num_freqs * 2]
        sin_features = torch.sin(proj)
        cos_features = torch.cos(proj)
        
        # Repeat for both axes to match embed_dim
        # Final shape: [B, N, embed_dim] where embed_dim = 4 * num_freqs
        pos_encoding = torch.cat([
            sin_features, 
            cos_features, 
            sin_features,  # Effectively doubled for compatibility
            cos_features
        ], dim=-1)
        
        # Add positional encoding to the original embeddings
        return embeddings + pos_encoding


def create_positional_encoding(
    embed_dim: int,
    pe_type: str = "sinusoidal",
    max_freq: int = 10,
    sigma: float = 10.0,
    learnable: bool = True
) -> nn.Module:
    """
    Factory function to create positional encoding based on configuration.
    
    Args:
        embed_dim: The dimensionality of the embedding.
        pe_type: Type of positional encoding. Options: "sinusoidal", "gaussian_fourier"
        max_freq: Maximum frequency for sinusoidal encoding.
        sigma: Bandwidth for Gaussian Fourier Features.
        learnable: Whether frequencies are learnable (for Gaussian Fourier).
    
    Returns:
        A positional encoding module.
    """
    if pe_type == "sinusoidal":
        return ContinuousPositionalEncoding(
            embed_dim=embed_dim,
            max_freq=max_freq
        )
    elif pe_type == "gaussian_fourier":
        return GaussianFourierFeatures(
            embed_dim=embed_dim,
            sigma=sigma,
            learnable=learnable
        )
    else:
        raise ValueError(f"Unknown positional encoding type: {pe_type}")

