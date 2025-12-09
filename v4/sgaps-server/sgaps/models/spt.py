"""
Main model file for the Sparse Pixel Transformer (SPT).
"""
import torch
import torch.nn as nn
from omegaconf import DictConfig

from .positional_encoding import ContinuousPositionalEncoding

class StateVectorEncoder(nn.Module):
    """
    Encodes the game state vector into a dense embedding.
    It handles variable-length state vectors by using a mask derived from
    a sentinel value, ensuring that only valid state information is used.
    """
    def __init__(self, max_state_dim: int, embed_dim: int, sentinel_value: float = -999.0):
        super().__init__()
        self.max_state_dim = max_state_dim
        self.sentinel_value = sentinel_value
        self.linear = nn.Linear(max_state_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, state_vector: torch.Tensor, state_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_vector: The state vector. Shape: [B, max_state_dim]
            state_mask: Mask indicating valid states (1 for valid, 0 for sentinel).
                        Shape: [B, max_state_dim]
        Returns:
            The state embedding. Shape: [B, 1, embed_dim]
        """
        masked_state = state_vector * state_mask
        embeds = self.linear(masked_state)
        embeds = self.norm(embeds)
        return embeds.unsqueeze(1)


class CrossAttentionDecoderLayer(nn.Module):
    """
    A custom Transformer Decoder layer that performs only Cross-Attention,
    completely skipping the expensive self-attention step. This is crucial for
    decoding large query grids (e.g., full image resolutions) without causing
    CUDA OutOfMemory errors.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, batch_first=True):
        super().__init__()
        if not batch_first:
            raise ValueError("This custom layer only supports batch_first=True")
        
        # Cross-Attention components
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Feed-forward components
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.functional.relu
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt, memory, memory_key_padding_mask=None, return_attn_weights=False):
        """
        Forward pass for cross-attention decoder layer.

        Args:
            tgt: Query tensor [B, num_queries, embed_dim]
            memory: Key/Value tensor [B, num_keys, embed_dim]
            memory_key_padding_mask: Padding mask for memory
            return_attn_weights: If True, return attention weights

        Returns:
            tgt: Output tensor [B, num_queries, embed_dim]
            attn_weights (optional): [B, num_heads, num_queries, num_keys]
        """
        # Cross-Attention block (query: tgt, key/value: memory)
        attn_output, attn_weights = self.cross_attn(
            query=tgt,
            key=memory,
            value=memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
            average_attn_weights=False  # Keep all heads for analysis
        )
        tgt = self.norm1(tgt + self.dropout1(attn_output))

        # Feed-forward block
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm2(tgt + self.dropout2(ff_output))

        if return_attn_weights:
            return tgt, attn_weights  # [B, num_queries, embed_dim], [B, num_heads, num_queries, num_keys]
        else:
            return tgt

class SparsePixelTransformer(nn.Module):
    """
    Reconstructs a full image frame from a sparse set of pixels and a game state vector.
    This architecture is designed to be efficient by only performing self-attention
    on the small set of sparse pixels.
    """
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        # Extract hyperparameters from config
        self.embed_dim = config.model.architecture.embed_dim
        self.num_heads = config.model.architecture.num_heads
        self.num_encoder_layers = config.model.architecture.num_encoder_layers
        self.num_decoder_layers = config.model.architecture.num_decoder_layers
        self.max_state_dim = config.model.input_constraints.max_state_dim
        self.sentinel_value = config.model.sentinel_value

        # 1. Pixel Embedding: (u, v, value) -> embed_dim
        self.pixel_embed = nn.Linear(3, self.embed_dim)

        # 2. State Vector Encoder
        self.state_encoder = StateVectorEncoder(
            max_state_dim=self.max_state_dim,
            embed_dim=self.embed_dim,
            sentinel_value=self.sentinel_value
        )

        # 3. Positional Encoding
        self.pos_encoder = ContinuousPositionalEncoding(
            embed_dim=self.embed_dim,
            max_freq=config.model.positional_encoding.max_freq
        )

        # 4. Sparse Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=config.model.architecture.feedforward_dim,
            dropout=config.model.architecture.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_encoder_layers
        )

        # 5. State-Pixel Cross-Attention
        self.state_pixel_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=config.model.architecture.dropout,
            batch_first=True
        )
        
        # 6. Cross-Attention Decoder (Memory Efficient)
        decoder_layers = [
            CrossAttentionDecoderLayer(
                d_model=self.embed_dim,
                nhead=self.num_heads,
                dim_feedforward=config.model.architecture.feedforward_dim,
                dropout=config.model.architecture.dropout,
                batch_first=True
            ) for _ in range(self.num_decoder_layers)
        ]
        self.decoder = nn.ModuleList(decoder_layers)

        # 7. CNN Refinement Head
        refinement_channels = [self.embed_dim] + config.model.refinement_head.channels + [1]
        refinement_layers = []
        for i in range(len(refinement_channels) - 2):
            refinement_layers.append(nn.Conv2d(refinement_channels[i], refinement_channels[i+1], kernel_size=3, padding=1))
            refinement_layers.append(nn.ReLU(inplace=True))
        refinement_layers.append(nn.Conv2d(refinement_channels[-2], refinement_channels[-1], kernel_size=3, padding=1))
        refinement_layers.append(nn.Sigmoid())
        self.refine_head = nn.Sequential(*refinement_layers)

        self.register_buffer('query_grid', None)
        
    def _generate_query_grid(self, batch_size, height, width, device):
        """
        Generate a grid of UV coordinates for the entire frame.

        Args:
            batch_size: Batch size
            height: Frame height
            width: Frame width
            device: torch device

        Returns:
            Grid of shape [B, H*W, 2] with UV coordinates in [0, 1] range
        """
        # Check if cached grid can be reused
        if self.query_grid is None or self.query_grid.shape[0] != height * width:
            # Create normalized coordinates [0, 1]
            y_coords = torch.linspace(0, 1, height, device=device)
            x_coords = torch.linspace(0, 1, width, device=device)

            # Create meshgrid
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

            # Stack and flatten: [H, W, 2] -> [H*W, 2]
            grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)

            # Cache it
            self.query_grid = grid

        # Repeat for batch: [H*W, 2] -> [B, H*W, 2]
        return self.query_grid.unsqueeze(0).repeat(batch_size, 1, 1)

    def forward(self, sparse_pixels, state_vector, state_mask, resolution, return_attention=False):
        """
        Main forward pass for the SPT model.

        Args:
            sparse_pixels: [B, N, 3] - Sparse pixels (u, v, value)
            state_vector: [B, max_state_dim] - State vector with sentinels
            state_mask: [B, max_state_dim] - Mask for valid states
            resolution: tuple (H, W) - Output resolution
            return_attention: bool - Whether to return attention weights

        Returns:
            output: [B, 1, H, W] - Reconstructed frame
            attn_weights (optional): Attention weights from decoder
        """
        B, N, _ = sparse_pixels.shape
        H, W = resolution
        device = sparse_pixels.device

        # 1. Pixel embedding: [B, N, 3] -> [B, N, embed_dim]
        pixel_embeds = self.pixel_embed(sparse_pixels)

        # 2. Positional encoding on pixel coordinates
        uv_coords = sparse_pixels[:, :, :2]  # [B, N, 2]
        pixel_embeds = self.pos_encoder(pixel_embeds, uv_coords)

        # 3. State vector encoding: [B, max_state_dim] -> [B, 1, embed_dim]
        state_embeds = self.state_encoder(state_vector, state_mask)

        # 4. Sparse transformer encoder: [B, N, embed_dim] -> [B, N, embed_dim]
        encoded = self.encoder(pixel_embeds)

        # 5. State-pixel cross-attention
        state_conditioned, _ = self.state_pixel_attention(
            query=state_embeds,  # [B, 1, embed_dim]
            key=encoded,         # [B, N, embed_dim]
            value=encoded        # [B, N, embed_dim]
        )  # Output: [B, 1, embed_dim]

        # 6. Broadcast state information to all pixels
        encoded = encoded + state_conditioned.expand(-1, N, -1)

        # 7. Generate query grid for full frame: [B, H*W, 2]
        query_positions = self._generate_query_grid(B, H, W, device)

        # 8. Create query embeddings with only positional information
        query_embeds = torch.zeros(B, H*W, self.embed_dim, device=device)
        query_embeds = self.pos_encoder(query_embeds, query_positions)

        # 9. Decoder: cross-attention between queries and encoded sparse pixels
        decoded = query_embeds
        all_attn_weights = []

        for layer in self.decoder:
            if return_attention:
                decoded, layer_attn = layer(
                    tgt=decoded,
                    memory=encoded,
                    return_attn_weights=True
                )
                all_attn_weights.append(layer_attn)  # [B, num_heads, H*W, N]
            else:
                decoded = layer(tgt=decoded, memory=encoded)
        # Output: [B, H*W, embed_dim]

        # 10. Reshape to 2D spatial format
        decoded_2d = decoded.view(B, H, W, self.embed_dim)
        decoded_2d = decoded_2d.permute(0, 3, 1, 2)  # [B, embed_dim, H, W]

        # 11. CNN refinement head
        output = self.refine_head(decoded_2d)  # [B, 1, H, W]

        # 12. Return with or without attention weights
        if return_attention:
            # Aggregate attention weights: average across layers and heads
            # Stack: [num_layers, B, num_heads, H*W, N]
            stacked_attn = torch.stack(all_attn_weights, dim=0)

            # Average across layers: [B, num_heads, H*W, N]
            avg_attn_layers = stacked_attn.mean(dim=0)

            # Average across heads: [B, H*W, N]
            final_attn = avg_attn_layers.mean(dim=1)

            return output, final_attn
        else:
            return output

    @staticmethod
    def load_from_checkpoint(checkpoint_path: str, config: DictConfig):
        """Loads a model from a saved checkpoint file."""
        model = SparsePixelTransformer(config)
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Successfully loaded model from {checkpoint_path}")
        except FileNotFoundError:
            print(f"Warning: Checkpoint file not found at {checkpoint_path}. Initializing with random weights.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Initializing with random weights.")
        return model
