"""
Attention Entropy-based Importance Map Calculator.

This module calculates pixel importance from cross-attention entropy,
enabling adaptive sampling strategies that focus on uncertain regions.
"""
import torch
import numpy as np
from typing import Tuple, List, Dict
from omegaconf import DictConfig


class AttentionEntropyImportanceCalculator:
    """
    Calculates pixel importance from cross-attention entropy.

    The core insight: High entropy indicates the model is uncertain about
    which sparse pixels to use for reconstruction, suggesting that region
    needs more sampling. Low entropy means the model is confident (focused
    on specific pixels), so the region is well-represented.

    Formula: H(X) = -Σ p(x) * log(p(x))
    where p(x) is the attention weight distribution over sparse input pixels.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the importance calculator.

        Args:
            config: Hydra config with sampling.importance settings
        """
        self.epsilon = config.sampling.importance.epsilon  # 1e-9
        self.normalize = config.sampling.importance.normalize  # True

    def calculate(
        self,
        attention_weights: torch.Tensor,  # [B, H*W, N]
        resolution: Tuple[int, int]       # (H, W)
    ) -> np.ndarray:
        """
        Calculate importance map from attention weights.

        Args:
            attention_weights: Cross-attention weights from decoder
                Shape: [batch_size, num_queries, num_keys]
                where num_queries = H*W (output pixels)
                      num_keys = N (sparse input pixels)
                Attention weights are assumed to be already softmaxed
                (sum to ~1.0 along dim=-1)
            resolution: Target resolution (H, W)

        Returns:
            importance_map: [B, H, W] normalized to [0, 1]
                Higher values indicate regions needing more sampling

        Edge cases handled:
            - All zero attention → Returns uniform importance (all 0.5)
            - NaN/Inf values → Clipped and warning logged
            - Extreme entropy range → Tanh squashing applied
        """
        # Validate input shape
        B, HW, N = attention_weights.shape
        H, W = resolution
        assert HW == H * W, f"Mismatch: attention has {HW} queries but resolution is {H}x{W} = {H*W}"

        # Handle edge case: all zero attention
        if torch.all(attention_weights == 0):
            print("Warning: All attention weights are zero. Returning uniform importance.")
            return np.full((B, H, W), 0.5, dtype=np.float32)

        # Check for NaN/Inf and clip
        if torch.any(torch.isnan(attention_weights)) or torch.any(torch.isinf(attention_weights)):
            print("Warning: NaN or Inf detected in attention weights. Clipping values.")
            attention_weights = torch.nan_to_num(
                attention_weights,
                nan=self.epsilon,
                posinf=1.0,
                neginf=self.epsilon
            )

        # Calculate Shannon entropy: H(X) = -Σ p(x) * log(p(x))
        # Clamp attention weights to avoid log(0)
        attn_clamped = torch.clamp(attention_weights, min=self.epsilon, max=1.0)

        # Compute entropy per query (output pixel)
        entropy = -(attn_clamped * torch.log(attn_clamped)).sum(dim=-1)
        # Shape: [B, H*W]

        # Reshape to spatial dimensions
        importance_map = entropy.view(B, H, W)  # [B, H, W]

        # Normalize to [0, 1] per batch item
        if self.normalize:
            for b in range(B):
                map_min = importance_map[b].min()
                map_max = importance_map[b].max()

                # Handle edge case: constant importance map
                if map_max - map_min < self.epsilon:
                    importance_map[b] = torch.full_like(importance_map[b], 0.5)
                else:
                    importance_map[b] = (importance_map[b] - map_min) / (map_max - map_min)

        # Apply tanh squashing if entropy range is extreme (>10)
        entropy_range = importance_map.max() - importance_map.min()
        if entropy_range > 10:
            print(f"Warning: Extreme entropy range ({entropy_range:.2f}). Applying tanh squashing.")
            importance_map = torch.tanh(importance_map)

        return importance_map.cpu().numpy()

    def get_top_k_regions(
        self,
        importance_map: np.ndarray,  # [H, W]
        k: int = 10
    ) -> List[Dict[str, float]]:
        """
        Get top-k most important regions for debugging and analysis.

        Args:
            importance_map: 2D importance map [H, W]
            k: Number of top regions to return

        Returns:
            List of dicts with 'u', 'v' (normalized coords) and 'importance' score
        """
        H, W = importance_map.shape
        flat_map = importance_map.flatten()

        # Get indices of top-k values
        top_indices = np.argsort(flat_map)[-k:][::-1]

        regions = []
        for idx in top_indices:
            v_idx = idx // W
            u_idx = idx % W

            # Convert to normalized UV coordinates [0, 1]
            u = (u_idx + 0.5) / W  # Center of pixel
            v = (v_idx + 0.5) / H

            score = float(flat_map[idx])

            regions.append({
                'u': float(u),
                'v': float(v),
                'importance': score,
                'pixel_index': int(idx)
            })

        return regions

    def compute_statistics(
        self,
        importance_map: np.ndarray  # [H, W]
    ) -> Dict[str, float]:
        """
        Compute statistical metrics for the importance map.

        Args:
            importance_map: 2D importance map [H, W]

        Returns:
            Dictionary with mean, std, max, min, entropy metrics
        """
        flat_map = importance_map.flatten()

        # Normalize to probability distribution for entropy calculation
        prob = flat_map / (flat_map.sum() + self.epsilon)
        prob = prob[prob > 0]  # Remove zeros for log

        # Compute entropy of importance distribution
        importance_entropy = float(-(prob * np.log(prob + self.epsilon)).sum())

        return {
            'mean': float(np.mean(flat_map)),
            'std': float(np.std(flat_map)),
            'max': float(np.max(flat_map)),
            'min': float(np.min(flat_map)),
            'median': float(np.median(flat_map)),
            'importance_entropy': importance_entropy,  # Entropy of importance distribution
            'dynamic_range': float(np.max(flat_map) - np.min(flat_map))
        }
