"""
Loss functions for SGAPS model training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SampledPixelL2Loss(nn.Module):
    """
    Calculates L2 (MSE) loss only on the pixels that were actually sampled.
    This forces the model to learn to reconstruct the given "hints" (the sparse pixels)
    perfectly, encouraging it to generalize to the unsampled areas.
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config

    def forward(self, pred: torch.Tensor, target: torch.Tensor, sampled_coords: torch.Tensor, pixel_padding_mask: torch.Tensor = None) -> dict:
        """
        Calculates the loss.

        Args:
            pred: The reconstructed (predicted) frame from the model.
                  Shape: [B, 1, H, W]
            target: The ground truth frame.
                    Shape: [B, 1, H, W]
            sampled_coords: The (u, v) coordinates of the pixels that were sampled.
                            Shape: [B, N, 2] (u, v in [0, 1] range)
            pixel_padding_mask: Boolean mask where True indicates padding (to be ignored).
                                Shape: [B, N]

        Returns:
            A dictionary containing the calculated loss values.
            e.g., {"total": loss_tensor, "l2_sampled": float_value}
        """
        B, _, H, W = pred.shape
        
        # 1. Convert continuous UV coordinates [0, 1] to discrete pixel indices [0, H-1] or [0, W-1].
        u = sampled_coords[:, :, 0]  # Shape: [B, N]
        v = sampled_coords[:, :, 1]  # Shape: [B, N]

        # Note: We use round() instead of floor() or ceil() to get the nearest pixel index.
        x_indices = (u * (W - 1)).round().long().clamp(0, W - 1)  # Shape: [B, N]
        y_indices = (v * (H - 1)).round().long().clamp(0, H - 1)  # Shape: [B, N]

        # 2. To use torch.gather on the flattened image, convert 2D indices (x, y) to 1D indices.
        # The image is flattened in row-major order, so index = y * width + x.
        flat_indices = y_indices * W + x_indices  # Shape: [B, N]

        # 3. Flatten the spatial dimensions (H, W) of the predicted and target images.
        pred_flat = pred.view(B, H * W)    # Shape: [B, H*W]
        target_flat = target.view(B, H * W)  # Shape: [B, H*W]

        # 4. Use `torch.gather` to efficiently select the pixel values at the sampled locations.
        # `gather` needs the indices to have the same number of dimensions as the source tensor.
        # So we expand `flat_indices` to match `pred_flat` and `target_flat`.
        pred_sampled = torch.gather(pred_flat, dim=1, index=flat_indices)    # Shape: [B, N]
        target_sampled = torch.gather(target_flat, dim=1, index=flat_indices)  # Shape: [B, N]

        # 5. Calculate loss, ignoring padded pixels if mask is provided
        diff = pred_sampled - target_sampled
        squared_diff = diff ** 2  # [B, N]

        if pixel_padding_mask is not None:
            # pixel_padding_mask: True=Padding (Ignore), False=Valid
            valid_mask = (~pixel_padding_mask).float() # [B, N]
            
            # Apply mask
            squared_diff = squared_diff * valid_mask
            
            # Per-sample sum of errors: [B]
            sample_error_sum = squared_diff.sum(dim=1)
            
            # Per-sample count of valid pixels: [B]
            sample_valid_count = valid_mask.sum(dim=1)
            
            # Avoid division by zero (if a sample has 0 valid pixels, loss is 0)
            sample_valid_count = torch.clamp(sample_valid_count, min=1.0)
            
            # Per-sample MSE: [B]
            sample_mse = sample_error_sum / sample_valid_count
            
            # Batch Loss: Mean of sample MSEs
            l2_loss = sample_mse.mean()
        else:
            # Fallback if no mask (assuming all valid)
            l2_loss = squared_diff.mean()

        return {
            "total": l2_loss,
            "l2_sampled": l2_loss.item(),
        }
