"""
Debug visualization using matplotlib subplots.
Standard ML visualization approach for WandB logging.

Creates a 2x4 grid dashboard with:
- Row 1: Original, Reconstructed, Difference, Sampled Pixels
- Row 2: Importance Map, Loss Map, Attention, State Vector
"""

import numpy as np
import torch

# CRITICAL: Set Agg backend BEFORE importing pyplot for thread-safe rendering
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from io import BytesIO
from typing import Dict, Optional, Any
import logging
import threading

logger = logging.getLogger(__name__)

# Lock for matplotlib operations (matplotlib is not thread-safe)
_matplotlib_lock = threading.Lock()


class DebugVisualizer:
    """Creates composite visualization dashboards using matplotlib subplots."""

    def __init__(self, config):
        self.config = config
        logger.info("DebugVisualizer initialized with matplotlib Agg backend.")

    def _apply_style(self):
        """Apply matplotlib style within the lock - called before each figure creation."""
        plt.style.use('dark_background')
        plt.rcParams.update({
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'figure.facecolor': '#1a1a1a',
            'axes.facecolor': '#2a2a2a',
            'axes.edgecolor': '#444444',
            'figure.autolayout': False,
        })

    def create_composite_dashboard(
        self,
        original_frame: np.ndarray,
        reconstructed_frame: np.ndarray,
        sampled_pixels: np.ndarray,
        state_vector: np.ndarray,
        importance_map: np.ndarray,
        attention_weights: Optional[torch.Tensor] = None,
        metadata: Optional[Dict] = None,
        best_metrics: Optional[Dict[str, float]] = None
    ) -> Image.Image:
        """
        Create 2x4 grid dashboard using matplotlib subplots.
        Thread-safe via _matplotlib_lock.
        
        Layout:
        [Original] [Reconstructed] [Difference] [Sampled Pixels]
        [Importance] [Loss Map] [Metrics] [State Vector]
        """
        # Use lock to ensure thread safety for matplotlib operations
        with _matplotlib_lock:
            try:
                # Apply style before each call (thread-safe)
                self._apply_style()
                
                # Get figure settings from config with defaults
                viz_cfg = getattr(self.config.debug, 'visualization', {})
                fig_cfg = getattr(viz_cfg, 'figure', {})
                figsize = getattr(fig_cfg, 'figsize', [16, 8])
                dpi = getattr(fig_cfg, 'dpi', 100)
                cmap = getattr(viz_cfg, 'colormap', 'viridis')

                # Ensure figsize is a tuple/list
                if hasattr(figsize, '__iter__'):
                    figsize = tuple(figsize)
                else:
                    figsize = (16, 8)

                # Create figure with subplots
                fig, axes = plt.subplots(2, 4, figsize=figsize, dpi=dpi)
                
                # Build suptitle with metrics
                meta = metadata or {}
                frame_id = meta.get('frame_id', 0)
                psnr = meta.get('psnr', 0)
                ssim = meta.get('ssim', 0)
                mse = meta.get('mse', 0)
                
                fig.suptitle(
                    f"Frame {frame_id}  |  PSNR: {psnr:.2f} dB  |  SSIM: {ssim:.4f}  |  MSE: {mse:.2f}",
                    fontsize=16, fontweight='bold', color='white', y=0.98
                )

                # Normalize frames to same size
                original = self._normalize_frame(original_frame)
                reconstructed = self._normalize_frame(reconstructed_frame)

                # Row 1: Frame comparisons
                self._plot_frame(axes[0, 0], original, "Original (sRGB)")
                self._plot_frame(axes[0, 1], reconstructed, "Reconstructed (sRGB)")
                
                # Difference
                self._plot_difference(axes[0, 2], original, reconstructed, cmap)
                
                # Sampled Pixels
                self._plot_sampled_pixels(axes[0, 3], original, sampled_pixels)

                # Row 2: Analysis heatmaps
                self._plot_heatmap(axes[1, 0], importance_map, "Importance Map", cmap)
                self._plot_loss_map(axes[1, 1], original, reconstructed, cmap)
                # self._plot_attention(axes[1, 2], attention_weights, cmap) # Removed
                self._plot_metrics_summary(axes[1, 2], meta, best_metrics) # New Text Panel
                self._plot_state_vector(axes[1, 3], state_vector)

                # Adjust layout with room for suptitle
                plt.tight_layout(rect=[0, 0, 1, 0.95])

                # Convert to PIL Image
                buf = BytesIO()
                fig.savefig(buf, format='png', facecolor=fig.get_facecolor(), 
                            edgecolor='none', bbox_inches='tight', pad_inches=0.1)
                buf.seek(0)
                img = Image.open(buf).convert('RGB')
                plt.close(fig)

                logger.info(f"Created dashboard image: {img.size}")
                return img

            except Exception as e:
                logger.error(f"Error creating dashboard: {e}", exc_info=True)
                plt.close('all')
                return self._create_error_image(str(e))
                
    def _plot_metrics_summary(self, ax, metrics: Dict[str, Any], best_metrics: Optional[Dict[str, float]] = None):
        """Render a text-based metrics summary panel."""
        ax.set_facecolor('#2a2a2a')
        ax.axis('off')
        
        # Title
        ax.set_title("Metrics Summary", fontweight='bold', pad=8, color='white')
        
        # Helper to format
        def fmt(val): return f"{val:.4f}"
        
        # Define display list with keys, labels, and optimization direction
        # direction: 'max' (higher is better) or 'min' (lower is better)
        restoration_items = [
            ("psnr", "PSNR", "max", "dB"),
            ("ssim", "SSIM", "max", ""),
            ("mse",  "MSE",  "min", ""),
            ("mae",  "MAE",  "min", ""),
            ("peak_error", "Peak", "min", "")
        ]

        y_pos = 0.90
        line_height = 0.08
        
        ax.text(0.05, y_pos, "Restoration Metrics:", transform=ax.transAxes, 
                fontsize=11, fontweight='bold', color='#CCCCCC')
        y_pos -= line_height

        best_metrics = best_metrics or {}

        for key, label, direction, suffix in restoration_items:
            current_val = metrics.get(key, 0.0)
            best_val = best_metrics.get(key, None)
            
            is_best = False
            if best_val is not None:
                # Check for equality with small epsilon
                if abs(current_val - best_val) < 1e-6:
                    is_best = True
            
            # Format text
            # Add arrow to label to indicate direction
            arrow = "↑" if direction == "max" else "↓"
            label_fmt = f"{label} {arrow}"
            
            # e.g., "  PSNR ↑ : 30.22 dB"
            current_str = f"{current_val:.2f}" if key == 'psnr' else fmt(current_val)
            text_line = f"  {label_fmt:<8}: {current_str} {suffix}"
            
            # Add Best info
            if best_val is not None:
                best_str = f"{best_val:.2f}" if key == 'psnr' else fmt(best_val)
                text_line += f" (Best: {best_str})"
            
            # Determine Color
            color = '#FFD700' if is_best else 'white' # Gold if best
            if is_best:
                 text_line += " *" # Add star for reinforcement
            
            ax.text(0.05, y_pos, text_line, transform=ax.transAxes, 
                    fontsize=10, family='monospace', color=color)
            y_pos -= line_height

        # Spacer
        y_pos -= 0.02

        # Attention Metrics
        ax.text(0.05, y_pos, "Attention Stats:", transform=ax.transAxes, 
                fontsize=11, fontweight='bold', color='#CCCCCC')
        y_pos -= line_height
        
        # Max Attention
        max_attn = metrics.get('max_attention', 0.0)
        ax.text(0.05, y_pos, f"  Max Attn:   {fmt(max_attn)}", transform=ax.transAxes, 
                fontsize=10, family='monospace', color='white')
        y_pos -= line_height

        # Attention Entropy
        attn_entropy = metrics.get('attention_entropy', 0.0)
        ax.text(0.05, y_pos, f"  Attn Entr:  {fmt(attn_entropy)}", transform=ax.transAxes, 
                fontsize=10, family='monospace', color='white')


    def _normalize_frame(self, frame: np.ndarray, target_shape=None) -> np.ndarray:
        """Normalize frame to [0, 255] uint8 and optionally resize."""
        # Resize if needed
        if target_shape is not None and frame.shape[:2] != target_shape[:2]:
            pil_img = Image.fromarray(frame.astype(np.uint8))
            pil_img = pil_img.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
            frame = np.array(pil_img)
        
        # Normalize to uint8
        if frame.dtype != np.uint8:
            if frame.max() > 1.0:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            else:
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        return frame

    def _plot_frame(self, ax, frame: np.ndarray, title: str):
        """Plot a grayscale frame."""
        ax.imshow(frame, cmap='gray', vmin=0, vmax=255)
        ax.set_title(title, fontweight='bold', pad=8)
        ax.axis('off')

    def _plot_difference(self, ax, original: np.ndarray, reconstructed: np.ndarray, cmap: str):
        """Plot difference map with colorbar."""
        diff = np.abs(original.astype(np.float32) - reconstructed.astype(np.float32))
        max_diff = diff.max() if diff.max() > 0 else 1.0
        
        im = ax.imshow(diff, cmap=cmap, vmin=0, vmax=max_diff)
        ax.set_title(f"Difference (Max: {max_diff:.1f})", fontweight='bold', pad=8)
        ax.axis('off')
        self._add_colorbar(ax, im)

    def _plot_sampled_pixels(self, ax, frame: np.ndarray, sampled_pixels: np.ndarray):
        """Plot sampled pixel locations as scatter on frame."""
        ax.imshow(frame, cmap='gray', vmin=0, vmax=255)
        
        if len(sampled_pixels) > 0:
            H, W = frame.shape[:2]
            us = sampled_pixels[:, 0] * W
            vs = sampled_pixels[:, 1] * H
            
            # Use smaller markers for many points, larger for few
            marker_size = max(1, min(10, 500 / len(sampled_pixels)))
            ax.scatter(us, vs, c='#FF5252', s=marker_size, alpha=0.8, marker='.', linewidths=0)
        
        ax.set_title(f"Sampled Pixels ({len(sampled_pixels)})", fontweight='bold', pad=8)
        ax.axis('off')

    def _plot_heatmap(self, ax, heatmap: Optional[np.ndarray], title: str, cmap: str):
        """Plot a heatmap with colorbar."""
        if heatmap is None:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center',
                    transform=ax.transAxes, fontsize=14, color='gray')
            ax.set_facecolor('#2a2a2a')
            ax.set_title(title, fontweight='bold', pad=8)
            ax.axis('off')
            return

        # Handle potential NaN/Inf
        heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Normalize to [0, 1]
        if heatmap.size > 0:
            hmin, hmax = heatmap.min(), heatmap.max()
            if hmax - hmin > 1e-8:
                heatmap_norm = (heatmap - hmin) / (hmax - hmin)
            else:
                heatmap_norm = np.zeros_like(heatmap)
        else:
            heatmap_norm = heatmap
        
        im = ax.imshow(heatmap_norm, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f"{title} (μ={heatmap.mean():.3f})", fontweight='bold', pad=8)
        ax.axis('off')
        self._add_colorbar(ax, im)

    def _plot_loss_map(self, ax, original: np.ndarray, reconstructed: np.ndarray, cmap: str):
        """Plot pixel-wise MSE loss map."""
        loss = (original.astype(np.float32) - reconstructed.astype(np.float32)) ** 2
        mse = loss.mean()
        
        max_loss = loss.max() if loss.max() > 0 else 1.0
        loss_norm = loss / max_loss
        
        im = ax.imshow(loss_norm, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f"Loss Map (MSE: {mse:.2f})", fontweight='bold', pad=8)
        ax.axis('off')
        self._add_colorbar(ax, im)

    def _plot_attention(self, ax, attention_weights: Optional[Any], cmap: str):
        """Plot attention weights visualization (supports Tensor or Numpy)."""
        if attention_weights is None:
            ax.text(0.5, 0.5, "No Attention Data", ha='center', va='center',
                    transform=ax.transAxes, fontsize=14, color='gray')
            ax.set_facecolor('#2a2a2a')
            ax.set_title("Attention", fontweight='bold', pad=8)
            ax.axis('off')
            return

        try:
            # Convert to numpy if tensor
            if isinstance(attention_weights, torch.Tensor):
                attn_data = attention_weights.detach().cpu().numpy()
            elif isinstance(attention_weights, (list, tuple)):
                # Handle list of heads/layers - assume we want the last one
                last_item = attention_weights[-1]
                if isinstance(last_item, torch.Tensor):
                    attn_data = last_item.detach().cpu().numpy()
                else:
                    attn_data = np.array(last_item)
            else:
                attn_data = np.array(attention_weights)

            # Expected shape: [batch, num_heads, seq_len, seq_len] or similar
            # Or [1, tokens, tokens]
            # Squeeze batch dim if present
            if attn_data.ndim == 4:
                attn_data = attn_data[0] # [heads, S, S]
            
            # Average across heads/layers usually
            # Here implementation relies on mean(dim=-1). Let's see original code intent.
            # Original: attention_weights[0].mean(dim=-1)
            # Implies input was [batch, S, S] or similar?
            # If shape is [heads, S, S], mean(axis=0) gives [S, S].
            
            # Let's target getting a 1D importance score per token for visualization, 
            # Or a 2D attention map? 
            # The original code: `attn_mean = attention_weights[0].mean(dim=-1)`
            # Then `H = W = int(num_elements ** 0.5)`
            # This implies it's visualizing SELF-ATTENTION significance per token, arranged spatially.
            
            if attn_data.ndim == 3: # [heads, S, S]
                 attn_map_2d = np.mean(attn_data, axis=0) # [S, S]
                 # But we want significance?
                 # If original code did .mean(dim=-1), it reduces [S, S] -> [S] (average attention received/given?)
                 
                 # Let's stick to what worked before, just translated to Numpy
                 # If input was [1, S, S], then [0] is [S, S]. mean(dim=-1) is [S].
                 # If input is now numpy [1, S, S]
                 attn_frame = attn_data
                 if attn_frame.ndim >= 2:
                      # Reduce to 1D score vector
                      attn_scores = np.mean(attn_frame, axis=-1)
                      # If multiple heads/batch, take mean?
                      if attn_scores.ndim > 1:
                           attn_scores = np.mean(attn_scores, axis=0) # Flatten
                      
                      attn_mean = attn_scores
                 else:
                      attn_mean = attn_frame
            else:
                 # Fallback
                 attn_mean = attn_data.flatten()

            num_elements = attn_mean.shape[0]
            H = W = int(num_elements ** 0.5)
            
            if H * W == num_elements:
                attn_map = attn_mean.reshape(H, W)
                attn_min, attn_max = attn_map.min(), attn_map.max()
                if attn_max - attn_min > 1e-8:
                    attn_norm = (attn_map - attn_min) / (attn_max - attn_min)
                else:
                    attn_norm = np.zeros_like(attn_map)
                
                im = ax.imshow(attn_norm, cmap=cmap, vmin=0, vmax=1)
                self._add_colorbar(ax, im)
                ax.set_title("Attention Average", fontweight='bold', pad=8)
            else:
                # Can't reshape to square, just plot regular plot or error
                # ax.plot(attn_mean) # Too messy
                ax.text(0.5, 0.5, f"Non-square: {num_elements}", ha='center', va='center',
                        transform=ax.transAxes, fontsize=12, color='orange')
                ax.set_title("Attention (Error)", fontweight='bold', pad=8)
        except Exception as e:
            logger.warning(f"Error plotting attention: {e}")
            ax.text(0.5, 0.5, "Error", ha='center', va='center',
                    transform=ax.transAxes, fontsize=14, color='red')
            ax.set_title("Attention", fontweight='bold', pad=8)
        
        ax.axis('off')

    def _plot_state_vector(self, ax, state_vector: np.ndarray):
        """Plot state vector as bar chart."""
        sentinel = getattr(self.config.model, 'sentinel_value', -999.0)
        valid_mask = state_vector != sentinel
        valid_values = state_vector[valid_mask]
        valid_indices = np.where(valid_mask)[0]

        if len(valid_values) == 0:
            ax.text(0.5, 0.5, "No Valid State", ha='center', va='center',
                    transform=ax.transAxes, fontsize=14, color='gray')
            ax.set_facecolor('#2a2a2a')
            ax.set_title("State Vector", fontweight='bold', pad=8)
            ax.axis('off')
            return

        # Normalize values for visualization
        max_abs = np.abs(valid_values).max()
        if max_abs > 0:
            normalized = valid_values / max_abs
        else:
            normalized = valid_values

        # Create bar chart with colors based on sign
        colors = ['#4CAF50' if v >= 0 else '#F44336' for v in normalized]
        ax.bar(valid_indices, normalized, color=colors, width=0.8, edgecolor='none')
        
        # Styling
        ax.axhline(y=0, color='white', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.set_xlim(-1, len(state_vector))
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlabel('Dimension', fontsize=10, color='white')
        ax.set_ylabel('Normalized', fontsize=10, color='white')
        ax.set_title(f"State Vector ({len(valid_values)}/{len(state_vector)} valid)", 
                     fontweight='bold', pad=8)
        ax.tick_params(colors='white', labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#666666')
        ax.spines['left'].set_color('#666666')

    def _add_colorbar(self, ax, im):
        """Add a colorbar to the right of an axes."""
        try:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=8, colors='white')
        except Exception as e:
            logger.warning(f"Could not add colorbar: {e}")

    def _create_error_image(self, error_msg: str) -> Image.Image:
        """Create an error placeholder image."""
        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#2a2a2a')
        ax.text(0.5, 0.5, f"Visualization Error:\n{error_msg[:200]}", 
                ha='center', va='center', fontsize=12, color='#FF5252',
                transform=ax.transAxes, wrap=True)
        ax.axis('off')
        
        buf = BytesIO()
        fig.savefig(buf, format='png', facecolor=fig.get_facecolor())
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        plt.close(fig)
        return img
