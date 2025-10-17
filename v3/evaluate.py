"""
Evaluation script for VideoMAE baseline
Computes PSNR, SSIM, MSE and other metrics
"""

import os
import time
import yaml
from typing import Dict

import torch
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from transformers import VideoMAEForPreTraining, VideoMAEImageProcessor, VideoMAEConfig
from dataset import create_dataloaders
from utils import set_seed, AverageMeter


class VideoMAEEvaluator:
    """Evaluator for VideoMAE baseline model"""

    def __init__(self, config: DictConfig, checkpoint_path: str) -> None:
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print(f"Run directory: {os.getcwd()}")

        self.visualization_dir = os.path.abspath(self.config.evaluation.visualization_dir)
        os.makedirs(self.visualization_dir, exist_ok=True)

        # Load model
        self._load_model(checkpoint_path)

        # Setup data
        self._setup_data()

    def _load_model(self, checkpoint_path: str) -> None:
        """Load trained model from checkpoint"""
        model_config = VideoMAEConfig(
            image_size=self.config.data.image_size,
            patch_size=16,
            num_channels=3,
            num_frames=self.config.data.num_frames,
            tubelet_size=2,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            decoder_num_hidden_layers=4,
            decoder_hidden_size=384,
            decoder_num_attention_heads=6,
            decoder_intermediate_size=1536,
            mask_ratio=self.config.model.mask_ratio,
            norm_pix_loss=self.config.model.norm_pix_loss
        )

        self.model = VideoMAEForPreTraining(model_config)

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Loaded checkpoint from {checkpoint_path}")

        self.model = self.model.to(self.device)
        self.model.eval()

        # Image processor
        self.processor = VideoMAEImageProcessor(
            do_resize=True,
            size={"height": self.config.data.image_size,
                  "width": self.config.data.image_size},
            do_normalize=True
        )

    def _setup_data(self) -> None:
        """Setup test data loader"""
        data_root = to_absolute_path(self.config.data.data_root)
        _, _, self.test_loader = create_dataloaders(
            data_root=data_root,
            batch_size=self.config.evaluation.batch_size,
            num_workers=self.config.data.num_workers,
            num_frames=self.config.data.num_frames,
            image_size=self.config.data.image_size
        )

    def _generate_mask(self, batch_size: int, num_patches: int) -> torch.Tensor:
        """Generate tube masking for VideoMAE"""
        mask_ratio = self.config.model.mask_ratio
        num_masked = int(num_patches * mask_ratio)

        # Create mask for each sample in batch
        bool_masked_pos = torch.zeros((batch_size, num_patches), dtype=torch.bool)

        for i in range(batch_size):
            # Randomly select patches to mask
            mask_indices = torch.randperm(num_patches)[:num_masked]
            bool_masked_pos[i, mask_indices] = True

        return bool_masked_pos

    def reconstruct_video(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Reconstruct video from masked inputs"""
        with torch.no_grad():
            # Generate mask
            batch_size = pixel_values.shape[0]
            # Calculate number of patches dynamically
            # VideoMAE expects input shape: (batch, num_frames, channels, height, width)
            num_frames = pixel_values.shape[1]
            height = pixel_values.shape[3]
            width = pixel_values.shape[4]

            # Number of patches = (num_frames // tubelet_size) * (image_size // patch_size)^2
            tubelet_size = 2
            patch_size = 16
            seq_length = (num_frames // tubelet_size) * ((height // patch_size) * (width // patch_size))
            num_patches = seq_length
            bool_masked_pos = self._generate_mask(batch_size, num_patches)
            bool_masked_pos = bool_masked_pos.to(self.device)

            # Forward pass with mask
            outputs = self.model(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos)

            # Get reconstruction
            reconstruction = outputs.reconstruction
            if reconstruction is None:
                # If model doesn't provide reconstruction, compute it manually
                logits = outputs.logits
                reconstruction = self.model.unpatchify(logits)

        return reconstruction

    def compute_metrics(self, original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, float]:
        """Compute evaluation metrics"""
        metrics: Dict[str, float] = {}

        # Convert to numpy arrays
        orig_np = original.cpu().numpy()
        recon_np = reconstructed.cpu().numpy()

        # Denormalize if needed (assuming standard ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3, 1, 1)

        orig_np = orig_np * std + mean
        recon_np = recon_np * std + mean

        # Clip to [0, 1]
        orig_np = np.clip(orig_np, 0, 1)
        recon_np = np.clip(recon_np, 0, 1)

        # Compute metrics for each sample in batch
        batch_size = orig_np.shape[0]
        psnr_scores = []
        ssim_scores = []
        mse_scores = []

        for i in range(batch_size):
            # Average across frames
            for j in range(orig_np.shape[1]):  # num_frames
                # Convert to HWC format for metrics
                orig_frame = np.transpose(orig_np[i, j], (1, 2, 0))
                recon_frame = np.transpose(recon_np[i, j], (1, 2, 0))

                # PSNR
                psnr_score = psnr(orig_frame, recon_frame, data_range=1.0)
                psnr_scores.append(psnr_score)

                # SSIM
                ssim_score = ssim(orig_frame, recon_frame,
                                 data_range=1.0, channel_axis=2)
                ssim_scores.append(ssim_score)

                # MSE
                mse_score = np.mean((orig_frame - recon_frame) ** 2)
                mse_scores.append(mse_score)

        metrics['psnr'] = float(np.mean(psnr_scores))
        metrics['ssim'] = float(np.mean(ssim_scores))
        metrics['mse'] = float(np.mean(mse_scores))

        return metrics

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run full evaluation"""
        print("Starting evaluation...")

        all_metrics = {
            'psnr': AverageMeter(),
            'ssim': AverageMeter(),
            'mse': AverageMeter(),
            'inference_time': AverageMeter()
        }

        pbar = tqdm(self.test_loader, desc="Evaluating")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            pixel_values = batch['pixel_values'].to(self.device)

            # Measure inference time
            start_time = time.time()
            reconstruction = self.reconstruct_video(pixel_values)
            inference_time = time.time() - start_time

            # Compute metrics
            metrics = self.compute_metrics(pixel_values, reconstruction)

            # Update averages
            batch_size = pixel_values.size(0)
            all_metrics['psnr'].update(metrics['psnr'], batch_size)
            all_metrics['ssim'].update(metrics['ssim'], batch_size)
            all_metrics['mse'].update(metrics['mse'], batch_size)
            all_metrics['inference_time'].update(inference_time / batch_size, batch_size)

            # Update progress bar
            pbar.set_postfix({
                'PSNR': f"{all_metrics['psnr'].avg:.2f}",
                'SSIM': f"{all_metrics['ssim'].avg:.4f}",
                'MSE': f"{all_metrics['mse'].avg:.4f}"
            })

            # Save sample visualizations
            if batch_idx < self.config.evaluation.num_visualizations:
                self.save_visualization(
                    pixel_values[0],
                    reconstruction[0],
                    batch_idx
                )

        # Compute FPS
        fps = 1.0 / all_metrics['inference_time'].avg

        # Print final results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"PSNR: {all_metrics['psnr'].avg:.2f} dB")
        print(f"SSIM: {all_metrics['ssim'].avg:.4f}")
        print(f"MSE: {all_metrics['mse'].avg:.6f}")
        print(f"Inference FPS: {fps:.2f}")

        # Measure memory usage
        memory_mb = 0.0
        if torch.cuda.is_available():
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"GPU Memory: {memory_mb:.2f} MB")

        # Save results
        results = {
            'psnr': float(all_metrics['psnr'].avg),
            'ssim': float(all_metrics['ssim'].avg),
            'mse': float(all_metrics['mse'].avg),
            'fps': float(fps),
            'memory_mb': float(memory_mb)
        }

        return results

    def save_visualization(self, original: torch.Tensor, reconstruction: torch.Tensor, idx: int) -> None:
        """Save visualization of original vs reconstructed frames"""
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        # Select middle frame
        mid_frame = original.shape[0] // 2

        orig_frame = original[mid_frame] * std + mean
        recon_frame = reconstruction[mid_frame] * std + mean

        # Clip to [0, 1]
        orig_frame = torch.clamp(orig_frame, 0, 1)
        recon_frame = torch.clamp(recon_frame, 0, 1)

        # Convert to numpy
        orig_np = orig_frame.cpu().numpy().transpose(1, 2, 0)
        recon_np = recon_frame.cpu().numpy().transpose(1, 2, 0)

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(orig_np)
        axes[0].set_title('Original')
        axes[0].axis('off')

        axes[1].imshow(recon_np)
        axes[1].set_title('Reconstructed')
        axes[1].axis('off')

        # Save figure
        save_path = os.path.join(self.visualization_dir, f'sample_{idx}.png')
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if not cfg.evaluation.checkpoint_path:
        raise ValueError("evaluation.checkpoint_path must be provided (e.g., evaluation.checkpoint_path=path/to/best_model.pth)")

    checkpoint_path = to_absolute_path(str(cfg.evaluation.checkpoint_path))
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    results_path = to_absolute_path(str(cfg.evaluation.results_file))
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # Set seed
    set_seed(cfg.training.seed)

    # Create evaluator
    evaluator = VideoMAEEvaluator(cfg, checkpoint_path)

    # Evaluate
    results = evaluator.evaluate()

    # Save results
    with open(results_path, 'w') as f:
        yaml.dump(results, f)

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()  # type: ignore[misc]