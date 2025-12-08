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

