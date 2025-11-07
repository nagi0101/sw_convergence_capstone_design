"""
Evaluation script for VideoMAE baseline
Computes PSNR, SSIM, MSE and other metrics
"""

import os
import time
import yaml
from typing import Dict, Optional

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

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
        self.artifacts_dir = os.path.abspath(self.config.output.artifacts_dir)
        os.makedirs(self.artifacts_dir, exist_ok=True)

        # Load model
        self._load_model(checkpoint_path)

        # Setup data
        self._setup_data()

        # Optional W&B logging
        self._setup_wandb()

    def _load_model(self, checkpoint_path: str) -> None:
        """Load trained model from checkpoint"""

        # Create model using same logic as train.py
        # Load pretrained model or create new
        if self.config.model.pretrained:
            print(f"Loading pretrained model: {self.config.model.pretrained}")
            # Load the pretrained model with its original config
            self.model = VideoMAEForPreTraining.from_pretrained(
                self.config.model.pretrained,
                ignore_mismatched_sizes=True
            )
            # Update mask ratio if needed
            self.model.config.mask_ratio = self.config.model.mask_ratio
            self.model.config.norm_pix_loss = self.config.model.norm_pix_loss
        else:
            print("Creating new model from scratch")
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

        # Load checkpoint weights
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']

                # Handle torch.compile() wrapped models
                # Check if keys have _orig_mod. prefix
                if state_dict and '_orig_mod.' in list(state_dict.keys())[0]:
                    print("Detected torch.compile() checkpoint, removing _orig_mod. prefix...")
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        new_key = k.replace('_orig_mod.', '')
                        new_state_dict[new_key] = v
                    state_dict = new_state_dict

                self.model.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Loaded checkpoint from {checkpoint_path}")

        self.model = self.model.to(self.device)
        self.model.eval()

        # Ensure position embeddings are on the correct device (fixes CUDA graphs warning)
        for name, module in self.model.named_modules():
            if hasattr(module, 'position_embeddings'):
                if hasattr(module.position_embeddings, 'data'):
                    module.position_embeddings.data = module.position_embeddings.data.to(self.device)
                elif isinstance(module.position_embeddings, torch.nn.Parameter):
                    module.position_embeddings.data = module.position_embeddings.data.to(self.device)

        # Also check for position_ids which might be registered as buffers
        for name, buffer in self.model.named_buffers():
            if 'position' in name.lower():
                buffer.data = buffer.data.to(self.device)

        # Optional: Compile model for faster inference
        if hasattr(self.config.evaluation, 'use_compile') and self.config.evaluation.use_compile:
            print("Compiling model with torch.compile()...")
            self.model = torch.compile(self.model, mode='reduce-overhead')
            print("Model compiled successfully")

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

    def _setup_wandb(self) -> None:
        """Initialise optional W&B logging."""

        self.use_wandb = False
        self._wandb: Optional[object] = None
        self.wandb_run: Optional[object] = None
        self._wandb_visualizations = 0
        self._wandb_max_visualizations = 0
        self._wandb_log_visuals = False
        self._wandb_log_results = False

        if not hasattr(self.config.logging, "wandb"):
            return

        wandb_cfg = self.config.logging.wandb
        if not bool(wandb_cfg.enable):
            return

        try:
            import wandb  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Weights & Biases logging requested for evaluation but the `wandb` package is missing. "
                "Install it with `pip install wandb` or disable logging with logging.wandb.enable=false."
            ) from exc

        wandb_cfg_dict = OmegaConf.to_container(wandb_cfg, resolve=True)
        if not isinstance(wandb_cfg_dict, dict):
            raise TypeError("logging.wandb configuration must resolve to a dictionary")

        tags_value = wandb_cfg_dict.get("tags")
        tags = list(tags_value) if tags_value else []
        tags.append("evaluation")

        run_name = wandb_cfg_dict.get("run_name")
        eval_run_name = f"{run_name}-eval" if run_name else None

        init_kwargs = {
            "mode": wandb_cfg_dict.get("mode", "online"),
            "config": OmegaConf.to_container(self.config, resolve=True),
            "job_type": "evaluation",
            "tags": tags,
        }

        mapping = {
            "project": "project",
            "entity": "entity",
            "group": "group",
            "notes": "notes",
        }

        for source_key, target_key in mapping.items():
            value = wandb_cfg_dict.get(source_key)
            if value:
                init_kwargs[target_key] = value

        if eval_run_name:
            init_kwargs["name"] = eval_run_name

        self.wandb_run = wandb.init(**init_kwargs)
        self._wandb = wandb
        self.use_wandb = True

        self._wandb_visualizations = 0
        self._wandb_max_visualizations = int(wandb_cfg_dict.get("max_visualizations", 0) or 0)
        self._wandb_log_visuals = bool(wandb_cfg_dict.get("log_visualizations", False))
        self._wandb_log_results = bool(wandb_cfg_dict.get("log_results", True))

    def _log_wandb_batch_metrics(
        self,
        metrics: Dict[str, float],
        inference_time: float,
        batch_idx: int
    ) -> None:
        if not self.use_wandb or self._wandb is None:
            return

        payload = {
            'eval/batch_psnr': metrics['psnr'],
            'eval/batch_ssim': metrics['ssim'],
            'eval/batch_mse': metrics['mse'],
            'eval/batch_inference_time': inference_time,
        }
        wandb_module = self._wandb
        if wandb_module is None:
            return
        wandb_module.log(payload, step=batch_idx)

    def _log_wandb_visualization(
        self,
        original_np: np.ndarray,
        reconstruction_np: np.ndarray,
        idx: int,
        global_step: int
    ) -> None:
        if (
            not self.use_wandb
            or self._wandb is None
            or not self._wandb_log_visuals
        ):
            return

        if self._wandb_max_visualizations and self._wandb_visualizations >= self._wandb_max_visualizations:
            return

        wandb_module = self._wandb
        if wandb_module is None:
            return

        images = [
            wandb_module.Image(original_np, caption=f"original_{idx}"),
            wandb_module.Image(reconstruction_np, caption=f"reconstruction_{idx}"),
        ]
        wandb_module.log({'eval/visualizations': images}, step=global_step)
        self._wandb_visualizations += 1

    def _log_wandb_videos(
        self,
        video_samples: list,
        fps: int,
        step: int
    ) -> None:
        """Log video samples to wandb"""
        if not self.use_wandb or self._wandb is None:
            return

        wandb_module = self._wandb
        if wandb_module is None:
            return

        videos = []
        for sample in video_samples:
            original = sample['original']
            reconstruction = sample['reconstruction']
            idx = sample['idx']

            # Denormalization parameters (shape for [T, C, H, W] tensors)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

            # Denormalize [T, C, H, W] -> [T, C, H, W]
            # Use detach().cpu() to ensure gradient-free CPU tensors
            original_denorm = (original.detach().cpu() * std + mean).clamp(0, 1)
            recon_denorm = (reconstruction.detach().cpu() * std + mean).clamp(0, 1)

            # Convert to [T, H, W, C] for wandb.Video
            original_np = original_denorm.permute(0, 2, 3, 1).numpy()
            recon_np = recon_denorm.permute(0, 2, 3, 1).numpy()

            # Convert to uint8 [T, H, W, C]
            original_uint8 = (original_np * 255).astype(np.uint8)
            recon_uint8 = (recon_np * 255).astype(np.uint8)

            # Stack original and reconstruction side by side [T, H, W*2, C]
            combined = np.concatenate([original_uint8, recon_uint8], axis=2)

            # Create wandb.Video
            video = wandb_module.Video(combined, fps=fps, format="mp4", caption=f"Sample {idx}: Original (left) vs Reconstruction (right)")
            videos.append(video)

        # Log all videos
        wandb_module.log({'eval/reconstruction_videos': videos}, step=step)

    def _log_wandb_final_metrics(self, metrics: Dict[str, float], step: int) -> None:
        if not self.use_wandb or self._wandb is None:
            return

        payload = {f"eval/{key}": value for key, value in metrics.items()}
        wandb_module = self._wandb
        if wandb_module is None:
            return
        wandb_module.log(payload, step=step)

        if self.wandb_run is not None:
            summary = getattr(self.wandb_run, 'summary', None)
            if summary is not None:
                for key, value in payload.items():
                    summary[key] = value

    def log_results_artifact(self, results_path: str) -> None:
        if (
            not self.use_wandb
            or self._wandb is None
            or self.wandb_run is None
            or not self._wandb_log_results
            or not os.path.exists(results_path)
        ):
            return

        wandb_module = self._wandb
        if wandb_module is None:
            return

        artifact_name = f"evaluation-results-{getattr(self.wandb_run, 'id', 'latest')}"
        artifact = wandb_module.Artifact(artifact_name, type="evaluation-results")
        artifact.add_file(results_path)
        if hasattr(self.wandb_run, 'log_artifact'):
            self.wandb_run.log_artifact(artifact)

    def finish_wandb(self) -> None:
        if self.use_wandb and self._wandb is not None:
            self._wandb.finish()

    def _generate_mask(self, batch_size: int, num_patches: int) -> torch.Tensor:
        """Generate tube masking for VideoMAE"""
        mask_ratio = self.config.model.mask_ratio
        num_masked = int(num_patches * mask_ratio)

        # Create mask on target device directly (avoid CPU->GPU copy)
        bool_masked_pos = torch.zeros((batch_size, num_patches), dtype=torch.bool, device=self.device)

        for i in range(batch_size):
            # Randomly select patches to mask (on target device)
            mask_indices = torch.randperm(num_patches, device=self.device)[:num_masked]
            bool_masked_pos[i, mask_indices] = True

        return bool_masked_pos

    def patchify(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Patchify video into patches
        Args:
            pixel_values: [B, T, C, H, W]
        Returns:
            patches: [B, num_patches, patch_dim]
        """
        batch_size, num_frames, channels, height, width = pixel_values.shape
        patch_size = 16
        tubelet_size = 2

        # Number of patches in each dimension
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size
        num_patches_t = num_frames // tubelet_size

        # Reshape to patches
        # [B, T, C, H, W] -> [B, T//tubelet_size, tubelet_size, C, H//patch_size, patch_size, W//patch_size, patch_size]
        patches = pixel_values.reshape(
            batch_size,
            num_patches_t, tubelet_size,
            channels,
            num_patches_h, patch_size,
            num_patches_w, patch_size
        )

        # Permute to group patches
        # -> [B, num_patches_t, num_patches_h, num_patches_w, tubelet_size, patch_size, patch_size, C]
        patches = patches.permute(0, 1, 4, 6, 2, 5, 7, 3)

        # Flatten patches
        # -> [B, num_patches_t * num_patches_h * num_patches_w, tubelet_size * patch_size * patch_size * C]
        num_patches = num_patches_t * num_patches_h * num_patches_w
        patch_dim = tubelet_size * patch_size * patch_size * channels
        patches = patches.reshape(batch_size, num_patches, patch_dim)

        return patches

    def unpatchify(self, patches: torch.Tensor, original_shape: tuple) -> torch.Tensor:
        """
        Unpatchify patches back to video
        Args:
            patches: [B, num_patches, patch_dim]
            original_shape: (B, T, C, H, W)
        Returns:
            video: [B, T, C, H, W]
        """
        batch_size, num_frames, channels, height, width = original_shape
        patch_size = 16
        tubelet_size = 2

        num_patches_h = height // patch_size
        num_patches_w = width // patch_size
        num_patches_t = num_frames // tubelet_size

        # Reshape patches
        # [B, num_patches, patch_dim] -> [B, num_patches_t, num_patches_h, num_patches_w, tubelet_size, patch_size, patch_size, C]
        patches = patches.reshape(
            batch_size,
            num_patches_t, num_patches_h, num_patches_w,
            tubelet_size, patch_size, patch_size, channels
        )

        # Permute back
        # -> [B, num_patches_t, tubelet_size, C, num_patches_h, patch_size, num_patches_w, patch_size]
        patches = patches.permute(0, 1, 4, 7, 2, 5, 3, 6)

        # Reshape to original video shape
        video = patches.reshape(batch_size, num_frames, channels, height, width)

        return video

    def reconstruct_video(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Reconstruct video from masked inputs

        VideoMAE reconstruction process:
        1. Patchify original video
        2. Mask some patches
        3. Model reconstructs masked patches
        4. Combine unmasked (original) + masked (reconstructed)
        5. Unpatchify to full video
        """
        batch_size = pixel_values.shape[0]
        num_frames = pixel_values.shape[1]
        height = pixel_values.shape[3]
        width = pixel_values.shape[4]

        # Calculate number of patches
        tubelet_size = 2
        patch_size = 16
        seq_length = (num_frames // tubelet_size) * ((height // patch_size) * (width // patch_size))
        num_patches = seq_length

        # Generate mask (already on self.device)
        bool_masked_pos = self._generate_mask(batch_size, num_patches)

        # Patchify original video
        original_patches = self.patchify(pixel_values)

        # Forward pass with mask
        outputs = self.model(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos)

        # Get logits (reconstruction of masked patches)
        logits = outputs.logits  # [B, num_masked, patch_dim]

        # Denormalize logits if model uses norm_pix_loss
        if self.config.model.norm_pix_loss:
            # Denormalize: logits are normalized per patch
            # Calculate mean and std from original patches
            mean = original_patches.mean(dim=-1, keepdim=True)  # [B, num_patches, 1]
            var = original_patches.var(dim=-1, keepdim=True)    # [B, num_patches, 1]

            # Select mean and var for masked positions only
            # logits is [B, num_masked, patch_dim]
            # We need mean/var for masked positions only
            mean_masked_list = []
            var_masked_list = []
            for i in range(batch_size):
                mask = bool_masked_pos[i]
                mean_masked_list.append(mean[i, mask, :])
                var_masked_list.append(var[i, mask, :])

            mean_masked = torch.stack(mean_masked_list)  # [B, num_masked, 1]
            var_masked = torch.stack(var_masked_list)    # [B, num_masked, 1]

            # Denormalize
            logits = logits * (var_masked + 1.e-6)**.5 + mean_masked

        # Create full reconstruction by combining original and reconstructed patches
        reconstructed_patches = original_patches.clone()

        # Replace masked positions with reconstructed patches
        for i in range(batch_size):
            mask = bool_masked_pos[i]
            reconstructed_patches[i, mask] = logits[i]

        # Unpatchify to get full video
        original_shape = pixel_values.shape
        reconstruction = self.unpatchify(reconstructed_patches, original_shape)

        return reconstruction

    def compute_metrics(self, original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, float]:
        """Compute evaluation metrics using GPU-based operations"""
        metrics: Dict[str, float] = {}

        # Denormalize on GPU (assuming standard ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406], device=original.device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=original.device).view(1, 1, 3, 1, 1)

        orig_normalized = original * std + mean
        recon_normalized = reconstructed * std + mean

        # Clip to [0, 1]
        orig_normalized = torch.clamp(orig_normalized, 0, 1)
        recon_normalized = torch.clamp(recon_normalized, 0, 1)

        # Compute MSE (vectorized across all dimensions)
        mse = torch.mean((orig_normalized - recon_normalized) ** 2).item()

        # Compute PSNR from MSE
        psnr_value = 10 * torch.log10(1.0 / (torch.tensor(mse) + 1e-10)).item()

        # Simplified SSIM approximation on GPU (batch computation)
        # Using structural similarity based on mean, variance, and covariance
        # Reshape to (B*T, C, H, W) for easier computation
        B, T, C, H, W = orig_normalized.shape
        orig_flat = orig_normalized.reshape(B * T, C, H, W)
        recon_flat = recon_normalized.reshape(B * T, C, H, W)

        # Constants for SSIM
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Compute means
        mu1 = torch.mean(orig_flat, dim=[2, 3], keepdim=True)
        mu2 = torch.mean(recon_flat, dim=[2, 3], keepdim=True)

        # Compute variances and covariance
        sigma1_sq = torch.mean((orig_flat - mu1) ** 2, dim=[2, 3], keepdim=True)
        sigma2_sq = torch.mean((recon_flat - mu2) ** 2, dim=[2, 3], keepdim=True)
        sigma12 = torch.mean((orig_flat - mu1) * (recon_flat - mu2), dim=[2, 3], keepdim=True)

        # SSIM formula
        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

        # Average SSIM across all frames and channels
        ssim_value = torch.mean(ssim_map).item()

        metrics['psnr'] = float(psnr_value)
        metrics['ssim'] = float(ssim_value)
        metrics['mse'] = float(mse)

        return metrics

    @torch.inference_mode()
    def evaluate(self) -> Dict[str, float]:
        """Run full evaluation with inference optimizations"""
        print("Starting evaluation...")

        # Check if AMP should be used
        use_amp = hasattr(self.config.evaluation, 'use_amp') and self.config.evaluation.use_amp
        if use_amp:
            print("Using Automatic Mixed Precision (AMP) for inference")

        all_metrics = {
            'psnr': AverageMeter(),
            'ssim': AverageMeter(),
            'mse': AverageMeter(),
            'inference_time': AverageMeter()
        }

        # Episode-level metrics tracking
        episode_metrics = {}
        log_episode_metrics = (
            hasattr(self.config.logging.wandb, 'log_episode_metrics') and
            self.config.logging.wandb.log_episode_metrics and
            self.use_wandb
        )

        # Video logging
        log_videos = (
            hasattr(self.config.logging.wandb, 'log_video') and
            self.config.logging.wandb.log_video and
            self.use_wandb
        )
        video_fps = self.config.logging.wandb.get('video_fps', 10) if log_videos else 10
        video_samples = []

        # Queue for async visualization (collect during loop, save after)
        visualization_queue = []

        pbar = tqdm(self.test_loader, desc="Evaluating")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            pixel_values = batch['pixel_values'].to(self.device)

            # Get episode IDs if available
            episode_ids = batch.get('episode_id', None)

            # Measure inference time
            start_time = time.time()

            # Use AMP if enabled
            if use_amp and torch.cuda.is_available():
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    reconstruction = self.reconstruct_video(pixel_values)
            else:
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

            # Track episode-level metrics if enabled
            if log_episode_metrics and episode_ids is not None:
                for i, episode_id in enumerate(episode_ids):
                    if episode_id not in episode_metrics:
                        episode_metrics[episode_id] = {
                            'psnr': [],
                            'ssim': [],
                            'mse': []
                        }
                    # Note: metrics are averaged over batch, so we use the same value for all samples
                    episode_metrics[episode_id]['psnr'].append(metrics['psnr'])
                    episode_metrics[episode_id]['ssim'].append(metrics['ssim'])
                    episode_metrics[episode_id]['mse'].append(metrics['mse'])

            self._log_wandb_batch_metrics(metrics, inference_time, batch_idx)

            # Update progress bar
            pbar.set_postfix({
                'PSNR': f"{all_metrics['psnr'].avg:.2f}",
                'SSIM': f"{all_metrics['ssim'].avg:.4f}",
                'MSE': f"{all_metrics['mse'].avg:.4f}"
            })

            # Collect data for visualizations (save after loop to avoid I/O blocking)
            if batch_idx < self.config.evaluation.num_visualizations:
                visualization_queue.append({
                    'original': pixel_values[0].cpu(),
                    'reconstruction': reconstruction[0].cpu(),
                    'idx': batch_idx
                })

                # Collect video samples for wandb logging
                if log_videos and len(video_samples) < self._wandb_max_visualizations:
                    # Store original and reconstruction for video creation
                    video_samples.append({
                        'original': pixel_values[0].cpu(),
                        'reconstruction': reconstruction[0].cpu(),
                        'idx': batch_idx
                    })

        # Save all visualizations (async - after evaluation loop to avoid I/O blocking)
        if visualization_queue:
            print(f"\nSaving {len(visualization_queue)} visualizations...")
            for viz_data in visualization_queue:
                self.save_visualization(
                    viz_data['original'],
                    viz_data['reconstruction'],
                    viz_data['idx'],
                    viz_data['idx']
                )

        # Calculate final step for wandb logging
        final_step = len(self.test_loader) if self.test_loader is not None else 0

        # Log episode-level metrics if enabled
        if log_episode_metrics and episode_metrics:
            print("\n" + "="*50)
            print("EPISODE-LEVEL METRICS")
            print("="*50)
            for episode_id, metrics_list in sorted(episode_metrics.items()):
                avg_psnr = np.mean(metrics_list['psnr'])
                avg_ssim = np.mean(metrics_list['ssim'])
                avg_mse = np.mean(metrics_list['mse'])

                print(f"Episode {episode_id}:")
                print(f"  PSNR: {avg_psnr:.2f} dB")
                print(f"  SSIM: {avg_ssim:.4f}")
                print(f"  MSE: {avg_mse:.6f}")

                # Log to wandb if available
                if self.use_wandb and self._wandb is not None:
                    self._wandb.log({
                        f'eval/episode_{episode_id}/psnr': avg_psnr,
                        f'eval/episode_{episode_id}/ssim': avg_ssim,
                        f'eval/episode_{episode_id}/mse': avg_mse,
                    }, step=final_step)

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

        # Log video samples to wandb if enabled
        if log_videos and video_samples and self.use_wandb and self._wandb is not None:
            print(f"\nLogging {len(video_samples)} video samples to wandb...")
            self._log_wandb_videos(video_samples, video_fps, final_step)

        self._log_wandb_final_metrics(results, final_step)

        return results

    def save_visualization(
        self,
        original: torch.Tensor,
        reconstruction: torch.Tensor,
        idx: int,
        global_step: int
    ) -> None:
        """Save enhanced visualization with difference heatmap"""
        # Denormalize (use input tensor's device for compatibility)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(original.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(original.device)

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

        # Calculate absolute difference
        diff_np = np.abs(orig_np - recon_np)

        # Calculate MSE for this frame
        mse = np.mean((orig_np - recon_np) ** 2)
        psnr_val = 10 * np.log10(1.0 / (mse + 1e-10))

        # Create enhanced figure with 4 panels
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        # Original
        axes[0, 0].imshow(orig_np)
        axes[0, 0].set_title('Original', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')

        # Reconstructed
        axes[0, 1].imshow(recon_np)
        axes[0, 1].set_title('Reconstructed', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')

        # Difference heatmap (grayscale)
        diff_gray = np.mean(diff_np, axis=2)
        im = axes[1, 0].imshow(diff_gray, cmap='hot', vmin=0, vmax=0.5)
        axes[1, 0].set_title('Difference Heatmap', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

        # Difference overlay (amplified for visibility)
        diff_amplified = np.clip(diff_np * 5, 0, 1)
        axes[1, 1].imshow(diff_amplified)
        axes[1, 1].set_title('Difference (5x amplified)', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')

        # Add overall metrics as text
        fig.suptitle(f'Sample {idx} - PSNR: {psnr_val:.2f} dB, MSE: {mse:.6f}',
                     fontsize=16, fontweight='bold')

        # Save figure
        save_path = os.path.join(self.visualization_dir, f'sample_{idx}_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Also save individual images for easier inspection
        plt.figure(figsize=(6, 6))
        plt.imshow(orig_np)
        plt.title(f'Original - Sample {idx}')
        plt.axis('off')
        plt.savefig(os.path.join(self.visualization_dir, f'sample_{idx}_original.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(6, 6))
        plt.imshow(recon_np)
        plt.title(f'Reconstructed - Sample {idx}')
        plt.axis('off')
        plt.savefig(os.path.join(self.visualization_dir, f'sample_{idx}_reconstructed.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(6, 6))
        plt.imshow(diff_gray, cmap='hot', vmin=0, vmax=0.5)
        plt.title(f'Difference Heatmap - Sample {idx}')
        plt.axis('off')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.savefig(os.path.join(self.visualization_dir, f'sample_{idx}_difference.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        self._log_wandb_visualization(orig_np, recon_np, idx, global_step)


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

    evaluator.log_results_artifact(results_path)
    evaluator.finish_wandb()

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()  # type: ignore[misc]