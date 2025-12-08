"""
Image Reconstructor for SGAPS-MAE Server.
This module is responsible for reconstructing full images from sparse pixel data
using a trained Sparse Pixel Transformer (SPT) model.
"""

import logging
from typing import Dict, Tuple, Any
import numpy as np
import torch
from omegaconf import DictConfig
from pathlib import Path

# Import the model - assuming it's in the models directory
from sgaps.models.spt import SparsePixelTransformer

logger = logging.getLogger(__name__)

class FrameReconstructor:
    """
    Manages loading SPT models and performing inference to reconstruct frames.
    """
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded_models: Dict[str, torch.nn.Module] = {}
        # Removed: self.default_key - no default fallback needed

        logger.info(f"FrameReconstructor initialized. Device: {self.device}")
        # Removed: Pre-loading default model - models load on-demand only

    def get_model(self, checkpoint_key: str) -> torch.nn.Module:
        """
        Retrieves a model based on its checkpoint key.
        If the model is not already loaded, it loads it from disk.
        If the checkpoint does not exist, initializes with random weights.
        """
        if checkpoint_key not in self.loaded_models:
            model_path = Path(self.config.paths.checkpoint_dir) / checkpoint_key / "best.pth"

            if model_path.exists():
                # Load from checkpoint
                logger.info(f"Loading checkpoint from {model_path}")
                model = SparsePixelTransformer.load_from_checkpoint(str(model_path), self.config)
            else:
                # Initialize with random weights
                logger.warning(f"Checkpoint not found at {model_path}. Initializing model with random weights.")
                model = SparsePixelTransformer(self.config)

            model.to(self.device)
            model.eval()

            # Explicitly disable gradients on all parameters for inference-only mode
            for param in model.parameters():
                param.requires_grad = False

            # Cache the model
            self.loaded_models[checkpoint_key] = model

        return self.loaded_models[checkpoint_key]

    @torch.no_grad()
    async def reconstruct(
        self, 
        sparse_pixels: np.ndarray, 
        state_vector: np.ndarray, 
        resolution: Tuple[int, int], 
        checkpoint_key: str = "default"
    ) -> Tuple[np.ndarray, Any]:
        """
        Performs frame reconstruction using the appropriate SPT model.

        Args:
            sparse_pixels: NumPy array of sparse pixels. Shape: [N, 3] (u, v, value)
            state_vector: NumPy array of the game state. Shape: [max_state_dim]
            resolution: The (height, width) of the target frame.
            checkpoint_key: The key to identify which trained model to use.

        Returns:
            A tuple containing:
            - reconstructed_frame: The reconstructed frame as a NumPy array. Shape: [H, W]
            - attention_weights: The attention weights from the decoder for importance analysis.
        """
        model = self.get_model(checkpoint_key)

        # 1. Convert NumPy arrays to PyTorch Tensors
        pixels_tensor = torch.from_numpy(sparse_pixels).float().unsqueeze(0).to(self.device)
        state_tensor = torch.from_numpy(state_vector).float().unsqueeze(0).to(self.device)
        
        # 2. Create the state mask (1 for valid data, 0 for sentinel)
        sentinel = self.config.model.sentinel_value
        state_mask = (state_tensor != sentinel).float()

        # 3. Perform inference
        try:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=self.config.training.use_amp):
                recons_frame, attn_weights = model(
                    pixels_tensor,
                    state_tensor,
                    state_mask,
                    resolution,
                    return_attention=True
                )
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            # Return a black frame on failure
            h, w = resolution
            return np.zeros((h, w), dtype=np.uint8), None

        # 4. Convert output tensor back to a NumPy array for visualization/storage
        # Output is [1, 1, H, W], needs to be [H, W] and scaled to [0, 255]
        # Use .detach() to ensure no gradient tracking (even with @torch.no_grad())
        recons_frame_np = recons_frame.detach().cpu().squeeze().numpy()
        recons_frame_np = (recons_frame_np * 255).astype(np.uint8)

        return recons_frame_np, attn_weights