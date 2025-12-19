"""
Main model file for the Sparse Pixel Transformer (SPT).
"""
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchvision.models.resnet import BasicBlock


from .positional_encoding import create_positional_encoding

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
        
        # Skip connection settings
        self.skip_enabled = config.model.architecture.skip.enabled
        self.skip_weight = config.model.architecture.skip.weight

        # 1. Pixel Embedding: (u, v, value) -> embed_dim
        self.pixel_embed = nn.Linear(3, self.embed_dim)

        # 2. State Vector Encoder
        self.state_encoder = StateVectorEncoder(
            max_state_dim=self.max_state_dim,
            embed_dim=self.embed_dim,
            sentinel_value=self.sentinel_value
        )

        # 3. Positional Encoding (configurable via Hydra)
        pe_config = config.model.positional_encoding
        self.pos_encoder = create_positional_encoding(
            embed_dim=self.embed_dim,
            pe_type=pe_config.type,
            max_freq=pe_config.get('max_freq', 10),
            sigma=pe_config.get('sigma', 10.0),
            learnable=pe_config.get('learnable', True)
        )

        # 4. Sparse Transformer Encoder (Pre-LN enabled)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=config.model.architecture.feedforward_dim,
            dropout=config.model.architecture.dropout,
            batch_first=True,
            norm_first=True  # Optimization: Pre-LN for stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_encoder_layers
        )

        # 5. Skip Connection Projection (Option A: Encoder-Decoder Skip)
        if self.skip_enabled:
            self.skip_proj = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.Dropout(config.model.architecture.skip.dropout),
                nn.LayerNorm(self.embed_dim)
            )

        # 6. Decoder (Standard PyTorch Transformer Decoder, Pre-LN)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=config.model.architecture.feedforward_dim,
            dropout=config.model.architecture.dropout,
            batch_first=True,
            norm_first=True  # Optimization: Pre-LN for stability
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.num_decoder_layers
        )

        # 7. ResNet-based Refinement Head (Hybrid Architecture, Grayscale Output)
        # Replaces simple Conv2d stack with BasicBlock for better gradient flow
        self.refine_head = nn.Sequential(
            BasicBlock(self.embed_dim, self.embed_dim),
            BasicBlock(self.embed_dim, self.embed_dim),
            nn.Conv2d(self.embed_dim, 1, kernel_size=3, padding=1), # Output 1 channel (Grayscale)
            # nn.Sigmoid() # Optional: clamp to 0-1. Can be omitted if loss handles logits, but Report suggests using it.
                           # We will omit it for now to avoid saturation if using L2 on logits, 
                           # but typically L2 expects same range. 
                           # Re-enabling Sigmoid based on Report recommendation to ensure 0-1 output.
            nn.Sigmoid()
        )

        self.register_buffer('query_grid', None)
        
        # Initialize weights
        self.apply(self._init_weights)

        # Prevent Sigmoid saturation at the start:
        # Initialize the final Conv2d (before Sigmoid) to small values/zeros.
        # This ensures the model starts outputting neutral gray (0.5) instead of binary 0/1.
        final_conv = self.refine_head[2] # Index 2 is the Conv2d
        if isinstance(final_conv, nn.Conv2d):
            nn.init.constant_(final_conv.weight, 0)
            nn.init.constant_(final_conv.bias, 0)

    def _init_weights(self, m):
        """
        ViT/BERT-style weight initialization.
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        
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

    def forward(self, sparse_pixels, state_vector, state_mask, resolution, num_pixels=None, return_attention=False):
        """
        Main forward pass for the SPT model.

        Args:
            sparse_pixels: [B, N, 3] - Sparse pixels (u, v, value)
            state_vector: [B, max_state_dim] - State vector with sentinels
            state_mask: [B, max_state_dim] - Mask for valid states
            resolution: tuple (H, W) - Output resolution
            num_pixels: [B] - Number of valid pixels per sample
            return_attention: bool - Whether to return attention weights

        Returns:
            output: [B, 1, H, W] - Reconstructed frame
            attn_weights (optional): Attention weights from decoder
        """
        B, N, _ = sparse_pixels.shape
        H, W = resolution
        device = sparse_pixels.device

        # Create padding mask if num_pixels is provided
        # pixel_padding_mask: [B, N], True where padding
        pixel_padding_mask = None
        if num_pixels is not None:
            indices = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
            pixel_padding_mask = indices >= num_pixels.unsqueeze(1)

        # 1. Pixel embedding: [B, N, embed_dim]
        pixel_embeds = self.pixel_embed(sparse_pixels)

        # Apply mask to embeddings to zero out padded pixels
        if pixel_padding_mask is not None:
            pixel_embeds = pixel_embeds.masked_fill(pixel_padding_mask.unsqueeze(-1), 0.0)

        # 2. Positional encoding on pixel coordinates
        uv_coords = sparse_pixels[:, :, :2]
        pixel_embeds = self.pos_encoder(pixel_embeds, uv_coords)

        # Re-apply mask after positional encoding
        if pixel_padding_mask is not None:
            pixel_embeds = pixel_embeds.masked_fill(pixel_padding_mask.unsqueeze(-1), 0.0)

        # 3. State vector encoding: [B, 1, embed_dim]
        state_embeds = self.state_encoder(state_vector, state_mask)

        # 4. Integrate State as a Special Token
        # Prepend state token to pixel embeddings: [B, N+1, embed_dim]
        encoder_input = torch.cat([state_embeds, pixel_embeds], dim=1)

        # Adjust padding mask for the new token
        # The state token (index 0) is always valid (False in padding mask)
        if pixel_padding_mask is not None:
            # Prepend a column of False (valid) to the mask
            state_token_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
            encoder_padding_mask = torch.cat([state_token_mask, pixel_padding_mask], dim=1) # [B, N+1]
        else:
            encoder_padding_mask = None

        # 5. Sparse Transformer Encoder
        # Output: [B, N+1, embed_dim]
        encoded_full = self.encoder(encoder_input, src_key_padding_mask=encoder_padding_mask)

        # Separate state token and pixel tokens ??
        # For Decoder memory, we can use the full sequence (State + Pixels) as Memory
        # This allows the decoder to attend to the Global State as well.
        memory = encoded_full
        memory_key_padding_mask = encoder_padding_mask

        # Extract pixel part for Skip Connection (if needed)
        # encoded_pixels = encoded_full[:, 1:, :] # [B, N, D]

        # 6. Skip Connection Calculation
        if self.skip_enabled:
            # We want to use the Global Context from the encoder.
            # The State Token itself (encoded_full[:, 0, :]) is a good candidate for global context.
            # Or we can average the pixel tokens as before.
            # Let's use the average of *pixel* tokens for consistency with previous logic,
            # but using the State Token might be cleaner.
            # Given the previous logic was "Global Context", let's try using the State Token 
            # as the Representative Global Context if it learns well. 
            # However, to be safe and robust, let's Stick to the Average Pooling of Pixels,
            # as the State Token might focus only on State info.
            
            encoded_pixels = encoded_full[:, 1:, :]
            
            if pixel_padding_mask is not None:
                valid_mask = (~pixel_padding_mask).float().unsqueeze(-1)
                encoded_masked = encoded_pixels * valid_mask
                
                encoded_sum = encoded_masked.sum(dim=1, keepdim=True)
                valid_count = valid_mask.sum(dim=1, keepdim=True)
                valid_count = torch.clamp(valid_count, min=1.0)
                encoded_avg = encoded_sum / valid_count
            else:
                encoded_avg = encoded_pixels.mean(dim=1, keepdim=True)
            
            encoded_avg_proj = self.skip_proj(encoded_avg)

        # 7. Generate query grid for full frame: [B, H*W, 2]
        query_positions = self._generate_query_grid(B, H, W, device)

        # 8. Create query embeddings with only positional information
        query_embeds = torch.zeros(B, H*W, self.embed_dim, device=device)
        query_embeds = self.pos_encoder(query_embeds, query_positions)

        # 9. Decoder: Standard Transformer Decoder
        # tgt: [B, H*W, D]
        # memory: [B, N+1, D]
        decoded = self.decoder(
            tgt=query_embeds,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask
            # tgt_mask=None (non-causal, all positions attend to all)
        )

        # Skip connection injection
        if self.skip_enabled:
             decoded = decoded + self.skip_weight * encoded_avg_proj.expand(-1, decoded.shape[1], -1)

        # 10. Reshape to 2D spatial format
        decoded_2d = decoded.view(B, H, W, self.embed_dim)
        decoded_2d = decoded_2d.permute(0, 3, 1, 2)  # [B, embed_dim, H, W]

        # 11. CNN refinement head (ResNet + Sigmoid)
        output = self.refine_head(decoded_2d)  # [B, 1, H, W]

        # 12. Return
        if return_attention:
            # Native TransformerDecoder doesn't easily return weights from forward()
            # We would need to use hooks or access layers. 
            # For now, return None for weights or implement a custom hook if strictly needed.
            # Considering the complexity, we will return None for weights in this standard implementation.
            return output, None
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
