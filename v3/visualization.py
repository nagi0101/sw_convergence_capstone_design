"""
Reconstruction Visualization and Metrics for VideoMAE
Provides tools for visualizing reconstruction quality and computing metrics
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


class ReconstructionVisualizer:
    """Handles reconstruction visualization and metric computation"""
    
    def __init__(self, num_samples: int = 5):
        """
        Args:
            num_samples: Number of fixed samples to track across epochs
        """
        self.num_samples = num_samples
        self.fixed_samples = None
        self.fixed_masks = None
        
    def set_fixed_samples(self, pixel_values: torch.Tensor, bool_masked_pos: torch.Tensor) -> None:
        """Store fixed samples for consistent tracking across epochs"""
        self.fixed_samples = pixel_values[:self.num_samples].detach().cpu()
        self.fixed_masks = bool_masked_pos[:self.num_samples].detach().cpu()
        
    def denormalize_frame(self, frame: torch.Tensor) -> np.ndarray:
        """Convert normalized frame tensor to numpy image [0, 1]"""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        frame_cpu = frame.detach().cpu() * std + mean
        frame_cpu = torch.clamp(frame_cpu, 0.0, 1.0)
        return frame_cpu.permute(1, 2, 0).numpy()
    
    def apply_mask_to_frame(self, frame: torch.Tensor, mask_indices: torch.Tensor, 
                           patch_size: int = 16, tubelet_size: int = 2) -> torch.Tensor:
        """Apply mask visualization to frame for display"""
        # This is a simplified version - just gray out masked regions
        masked_frame = frame.clone()
        # For visualization, we'll just return the frame as is
        # The actual masking happens in the model
        return masked_frame
    
    def compute_metrics(self, original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, float]:
        """
        Compute reconstruction quality metrics
        
        Args:
            original: Original frames [B, T, C, H, W]
            reconstructed: Reconstructed frames [B, T, C, H, W]
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'mse': 0.0,
            'mae': 0.0,
            'psnr': 0.0,
            'ssim': 0.0
        }
        
        batch_size = original.shape[0]
        
        # Denormalize for metric calculation
        original_np = []
        reconstructed_np = []
        
        for b in range(batch_size):
            for t in range(original.shape[1]):
                orig_frame = self.denormalize_frame(original[b, t])
                recon_frame = self.denormalize_frame(reconstructed[b, t])
                original_np.append(orig_frame)
                reconstructed_np.append(recon_frame)
        
        # Calculate metrics
        mse_values = []
        mae_values = []
        psnr_values = []
        ssim_values = []
        
        for orig, recon in zip(original_np, reconstructed_np):
            # MSE
            mse = np.mean((orig - recon) ** 2)
            mse_values.append(mse)
            
            # MAE
            mae = np.mean(np.abs(orig - recon))
            mae_values.append(mae)
            
            # PSNR
            psnr_val = psnr(orig, recon, data_range=1.0)
            psnr_values.append(psnr_val)
            
            # SSIM
            ssim_val = ssim(orig, recon, channel_axis=2, data_range=1.0)
            ssim_values.append(ssim_val)
        
        metrics['mse'] = float(np.mean(mse_values))
        metrics['mae'] = float(np.mean(mae_values))
        metrics['psnr'] = float(np.mean(psnr_values))
        metrics['ssim'] = float(np.mean(ssim_values))
        metrics['psnr_std'] = float(np.std(psnr_values))
        metrics['ssim_std'] = float(np.std(ssim_values))
        metrics['best_psnr'] = float(np.max(psnr_values))
        metrics['worst_psnr'] = float(np.min(psnr_values))
        metrics['median_psnr'] = float(np.median(psnr_values))
        
        return metrics
    
    def create_comparison_grid(self, 
                              original: torch.Tensor,
                              reconstructed: torch.Tensor,
                              sample_idx: int = 0,
                              frame_indices: Optional[list] = None) -> np.ndarray:
        """
        Create 4-column comparison grid: Original | Masked | Reconstructed | Difference
        
        Args:
            original: Original video tensor [T, C, H, W]
            reconstructed: Reconstructed video tensor [T, C, H, W]
            sample_idx: Sample index for labeling
            frame_indices: Which frames to show (default: [0, T//4, T//2, 3*T//4, T-1])
        
        Returns:
            Grid image as numpy array
        """
        num_frames = original.shape[0]
        
        if frame_indices is None:
            # Show 5 evenly spaced frames
            frame_indices = [0, num_frames//4, num_frames//2, 3*num_frames//4, num_frames-1]
        
        num_display = len(frame_indices)
        
        # Create figure
        fig, axes = plt.subplots(num_display, 4, figsize=(16, 4*num_display))
        if num_display == 1:
            axes = axes.reshape(1, -1)
        
        for row, frame_idx in enumerate(frame_indices):
            # Original
            orig_frame = self.denormalize_frame(original[frame_idx])
            axes[row, 0].imshow(orig_frame)
            axes[row, 0].set_title(f'Original (Frame {frame_idx})')
            axes[row, 0].axis('off')
            
            # Masked (for now, just show original with label)
            # In practice, you'd visualize the masked regions
            axes[row, 1].imshow(orig_frame * 0.5)  # Darken to indicate masking
            axes[row, 1].set_title('Masked Input')
            axes[row, 1].axis('off')
            
            # Reconstructed
            recon_frame = self.denormalize_frame(reconstructed[frame_idx])
            axes[row, 2].imshow(recon_frame)
            axes[row, 2].set_title('Reconstructed')
            axes[row, 2].axis('off')
            
            # Difference (error map)
            diff = np.abs(orig_frame - recon_frame)
            im = axes[row, 3].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
            axes[row, 3].set_title('Absolute Error')
            axes[row, 3].axis('off')
            plt.colorbar(im, ax=axes[row, 3], fraction=0.046)
        
        plt.suptitle(f'Sample {sample_idx} Reconstruction Comparison', fontsize=16)
        plt.tight_layout()
        
        # Convert to numpy array
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return img
    
    def create_temporal_view(self, 
                            original: torch.Tensor, 
                            reconstructed: torch.Tensor,
                            sample_idx: int = 0) -> np.ndarray:
        """
        Create temporal sequence view showing all frames
        
        Args:
            original: Original video [T, C, H, W]
            reconstructed: Reconstructed video [T, C, H, W]
            sample_idx: Sample index
            
        Returns:
            Temporal view image
        """
        num_frames = original.shape[0]
        
        # Show all frames in 2 rows (original and reconstructed)
        fig, axes = plt.subplots(2, num_frames, figsize=(2*num_frames, 4))
        
        for t in range(num_frames):
            # Original
            orig_frame = self.denormalize_frame(original[t])
            axes[0, t].imshow(orig_frame)
            axes[0, t].set_title(f'F{t}')
            axes[0, t].axis('off')
            
            # Reconstructed
            recon_frame = self.denormalize_frame(reconstructed[t])
            axes[1, t].imshow(recon_frame)
            axes[1, t].axis('off')
        
        axes[0, 0].set_ylabel('Original', fontsize=12)
        axes[1, 0].set_ylabel('Reconstructed', fontsize=12)
        
        plt.suptitle(f'Sample {sample_idx} Temporal Sequence', fontsize=14)
        plt.tight_layout()
        
        # Convert to numpy
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return img
    
    def create_error_heatmap(self, 
                            original: torch.Tensor, 
                            reconstructed: torch.Tensor,
                            sample_idx: int = 0,
                            frame_idx: int = None) -> np.ndarray:
        """
        Create error heatmap for a specific frame
        
        Args:
            original: Original video [T, C, H, W]
            reconstructed: Reconstructed video [T, C, H, W]
            sample_idx: Sample index
            frame_idx: Frame to visualize (default: middle frame)
            
        Returns:
            Heatmap image
        """
        if frame_idx is None:
            frame_idx = original.shape[0] // 2
        
        orig_frame = self.denormalize_frame(original[frame_idx])
        recon_frame = self.denormalize_frame(reconstructed[frame_idx])
        
        # Calculate per-pixel MSE
        mse_map = np.mean((orig_frame - recon_frame) ** 2, axis=2)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original
        axes[0].imshow(orig_frame)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Reconstructed
        axes[1].imshow(recon_frame)
        axes[1].set_title('Reconstructed')
        axes[1].axis('off')
        
        # Error heatmap
        im = axes[2].imshow(mse_map, cmap='hot', interpolation='nearest')
        axes[2].set_title(f'MSE Heatmap (Frame {frame_idx})')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046)
        
        plt.suptitle(f'Sample {sample_idx} Error Analysis', fontsize=14)
        plt.tight_layout()
        
        # Convert to numpy
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return img
