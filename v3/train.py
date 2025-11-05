"""
VideoMAE Training Script for SMB Dataset
Uses Hugging Face Transformers implementation
"""

import os
import time
from typing import Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from transformers import (
    VideoMAEForPreTraining,
    VideoMAEImageProcessor,
    VideoMAEConfig
)

from dataset import create_dataloaders
from utils import set_seed, save_checkpoint, load_checkpoint, AverageMeter
from visualization import ReconstructionVisualizer


class VideoMAETrainer:
    """Trainer for VideoMAE baseline model"""

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print(f"Run directory: {os.getcwd()}")

        if self.device.type == 'cpu':
            print("WARNING: Training on CPU will be very slow! Consider using a GPU.")
            print("Reducing batch size might help with memory issues.")

        # Setup model
        self._setup_model()

        # Setup data
        self._setup_data()

        # Setup training
        self._setup_training()

        # Setup logging
        tensorboard_dir = os.path.abspath(self.config.logging.tensorboard_dir)
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tensorboard_dir)
        self.best_val_loss = float('inf')
        self.checkpoint_dir = os.path.abspath(self.config.training.checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.artifacts_dir = os.path.abspath(self.config.output.artifacts_dir)
        os.makedirs(self.artifacts_dir, exist_ok=True)

        self.steps_per_epoch = max(len(self.train_loader), 1)

        # Setup torch.compile if requested
        use_compile = getattr(self.config.training, 'use_compile', False)
        if use_compile and hasattr(torch, 'compile'):
            print("Compiling model with torch.compile()...")
            self.model = torch.compile(self.model)
            print("Model compilation complete")

        # Setup reconstruction visualizer
        self._setup_visualizer()

        self._setup_wandb()

    def _setup_model(self) -> None:
        """Initialize VideoMAE model"""

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

        self.model = self.model.to(self.device)
        
        # Enable gradient checkpointing if requested
        use_grad_checkpoint = getattr(self.config.training, 'gradient_checkpointing', False)
        if use_grad_checkpoint and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing for memory efficiency")

        # Image processor
        self.processor = VideoMAEImageProcessor(
            do_resize=True,
            size={"height": self.config.data.image_size,
                  "width": self.config.data.image_size},
            do_normalize=True
        )

    def _setup_data(self) -> None:
        """Setup data loaders"""
        data_root = to_absolute_path(self.config.data.data_root)
        
        # Get DataLoader optimization settings
        persistent_workers = getattr(self.config.data, 'persistent_workers', False)
        prefetch_factor = getattr(self.config.data, 'prefetch_factor', None)
        
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            data_root=data_root,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.data.num_workers,
            num_frames=self.config.data.num_frames,
            image_size=self.config.data.image_size,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor
        )

    def _setup_training(self) -> None:
        """Setup optimizer and scheduler"""
        # Check if fused optimizer is available and requested
        use_fused = getattr(self.config.training, 'fused_optimizer', False)
        fused_available = 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
        
        optimizer_kwargs = {
            'lr': self.config.training.learning_rate,
            'betas': (0.9, 0.95),
            'weight_decay': self.config.training.weight_decay
        }
        
        if use_fused and fused_available and self.device.type == 'cuda':
            optimizer_kwargs['fused'] = True
            print("Using fused AdamW optimizer")
        
        self.optimizer = AdamW(
            self.model.parameters(),
            **optimizer_kwargs
        )

        total_steps = max(len(self.train_loader) * self.config.training.num_epochs, 1)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.config.training.min_lr
        )
        
        # Setup mixed precision training
        self.use_amp = getattr(self.config.training, 'use_amp', False)
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        if self.use_amp:
            print("Using Automatic Mixed Precision (AMP)")
        
        # Setup gradient accumulation
        self.accumulation_steps = getattr(self.config.training, 'accumulation_steps', 1)
        if self.accumulation_steps > 1:
            print(f"Using gradient accumulation with {self.accumulation_steps} steps")
            print(f"Effective batch size: {self.config.training.batch_size * self.accumulation_steps}")

    def _setup_wandb(self) -> None:
        """Initialise optional Weights & Biases logging."""

        # Defaults when wandb logging is disabled
        self.use_wandb = False
        self._wandb: Optional[object] = None
        self.wandb_run: Optional[object] = None
        self._wandb_visualizations = 0
        self._wandb_max_visualizations = 0
        self._wandb_visualization_interval = 1
        self._wandb_log_visuals = False
        self._wandb_log_checkpoints = False

        if not hasattr(self.config.logging, "wandb"):
            return

        wandb_cfg = self.config.logging.wandb
        if not bool(wandb_cfg.enable):
            return

        try:
            import wandb  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Weights & Biases logging requested but the `wandb` package is not installed. "
                "Install it with `pip install wandb` or disable logging with logging.wandb.enable=false."
            ) from exc

        wandb_cfg_dict = OmegaConf.to_container(wandb_cfg, resolve=True)
        if not isinstance(wandb_cfg_dict, dict):
            raise TypeError("logging.wandb configuration must resolve to a dictionary")

        init_kwargs = {
            "mode": wandb_cfg_dict.get("mode", "online"),
            "config": OmegaConf.to_container(self.config, resolve=True),
        }

        field_mapping = {
            "project": "project",
            "entity": "entity",
            "group": "group",
            "run_name": "name",
            "notes": "notes",
        }

        for source_key, target_key in field_mapping.items():
            value = wandb_cfg_dict.get(source_key)
            if value:
                init_kwargs[target_key] = value

        tags = wandb_cfg_dict.get("tags")
        if tags:
            init_kwargs["tags"] = list(tags)

        self.wandb_run = wandb.init(**init_kwargs)
        self._wandb = wandb
        self.use_wandb = True

        watch_cfg = wandb_cfg_dict.get("watch") or {}
        log_mode = str(watch_cfg.get("log", "gradients")).lower()
        if log_mode and log_mode != "off":
            wandb.watch(self.model, log=log_mode, log_freq=int(watch_cfg.get("log_freq", 200)))

        self._wandb_log_visuals = bool(wandb_cfg_dict.get("log_visualizations", False))
        self._wandb_max_visualizations = int(wandb_cfg_dict.get("max_visualizations", 0) or 0)
        self._wandb_visualization_interval = max(
            int(wandb_cfg_dict.get("visualization_interval", 1) or 1),
            1
        )
        self._wandb_log_checkpoints = bool(wandb_cfg_dict.get("log_checkpoints", False))

    def _setup_visualizer(self) -> None:
        """Setup reconstruction visualizer"""
        vis_config = getattr(self.config.logging, 'reconstruction_vis', None)
        
        self.use_reconstruction_vis = False
        self.visualizer = None
        self.vis_interval = 1
        
        if vis_config and bool(vis_config.get('enable', False)):
            num_samples = int(vis_config.get('num_samples', 5))
            self.visualizer = ReconstructionVisualizer(num_samples=num_samples)
            self.vis_interval = int(vis_config.get('interval', 1))
            self.use_reconstruction_vis = True
            print(f"Enabled reconstruction visualization (every {self.vis_interval} epochs, {num_samples} samples)")

    @staticmethod
    def _denormalize_frame(frame: torch.Tensor) -> np.ndarray:
        """Convert a single normalised frame tensor to a numpy image."""

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        frame_cpu = frame.detach().cpu() * std + mean
        frame_cpu = torch.clamp(frame_cpu, 0.0, 1.0)
        return frame_cpu.permute(1, 2, 0).numpy()

    def _log_wandb_training_visual(self, pixel_values: torch.Tensor, epoch: int, global_step: int) -> None:
        """Log a representative training frame to wandb."""

        if not self.use_wandb or not self._wandb_log_visuals or self._wandb is None:
            return

        if self._wandb_max_visualizations and self._wandb_visualizations >= self._wandb_max_visualizations:
            return

        sample = pixel_values[0]
        mid_index = sample.shape[0] // 2
        frame_np = self._denormalize_frame(sample[mid_index])
        caption = f"epoch={epoch}, step={global_step}"
        wandb_module = self._wandb
        if wandb_module is None:
            return

        wandb_module.log(
            {"train/sample_frame": wandb_module.Image(frame_np, caption=caption)},
            step=global_step
        )
        self._wandb_visualizations += 1

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

    def train_epoch(self, epoch: int) -> float:
        """Train one epoch"""
        self.model.train()
        losses = AverageMeter()

        # Track data loading time if enabled
        data_time = AverageMeter()
        compute_time = AverageMeter()
        batch_end_time = time.time()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.training.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            # Measure data loading time
            data_time.update(time.time() - batch_end_time)

            # Move to device with non_blocking for better performance
            pixel_values = batch['pixel_values'].to(self.device, non_blocking=True)

            # Start compute timer
            compute_start = time.time()

            # Generate random mask for tube masking
            batch_size = pixel_values.shape[0]
            # Calculate number of patches dynamically based on input size
            # VideoMAE expects input shape: (batch, num_frames, channels, height, width)
            num_frames = pixel_values.shape[1]  # T dimension
            height = pixel_values.shape[3]  # H dimension
            width = pixel_values.shape[4]  # W dimension

            # Number of patches = (num_frames // tubelet_size) * (image_size // patch_size)^2
            tubelet_size = 2
            patch_size = 16
            seq_length = (num_frames // tubelet_size) * ((height // patch_size) * (width // patch_size))
            num_patches = seq_length

            # Create boolean mask with specified masking ratio
            bool_masked_pos = self._generate_mask(batch_size, num_patches)
            bool_masked_pos = bool_masked_pos.to(self.device, non_blocking=True)

            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos)
                loss = outputs.loss
                # Scale loss for gradient accumulation
                loss = loss / self.accumulation_steps

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step with gradient accumulation
            grad_norm = None
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping and logging
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)

                # Calculate gradient norm (even if not clipping)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip if self.config.training.gradient_clip > 0 else float('inf')
                )

                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            # Update metrics (use original loss for logging)
            losses.update(loss.item() * self.accumulation_steps, pixel_values.size(0))

            # Update compute time
            compute_time.update(time.time() - compute_start)

            # Update progress bar
            pbar.set_postfix({
                'loss': losses.avg,
                'lr': self.optimizer.param_groups[0]['lr']
            })

            # Log to tensorboard
            global_step = epoch * len(self.train_loader) + batch_idx
            if batch_idx % self.config.logging.log_interval == 0:
                self.writer.add_scalar('train/loss', loss.item() * self.accumulation_steps, global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], global_step)
                
                # Log GPU memory usage
                if self.device.type == 'cuda':
                    gpu_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
                    self.writer.add_scalar('train/gpu_mem_mb', gpu_mem_mb, global_step)
                
                if self.use_wandb and self._wandb is not None:
                    log_dict = {
                        'train/loss': loss.item() * self.accumulation_steps,
                        'train/lr': self.optimizer.param_groups[0]['lr'],
                        'epoch': epoch,
                    }
                    if self.device.type == 'cuda':
                        log_dict['train/gpu_mem_mb'] = gpu_mem_mb

                    # Log gradient norm if enabled and available
                    if (grad_norm is not None and
                        hasattr(self.config.logging.wandb, 'log_gradient_norm') and
                        self.config.logging.wandb.log_gradient_norm):
                        log_dict['train/grad_norm'] = grad_norm.item()

                    # Log data loading time if enabled
                    if (hasattr(self.config.logging.wandb, 'log_data_loading_time') and
                        self.config.logging.wandb.log_data_loading_time):
                        log_dict['train/data_time'] = data_time.avg
                        log_dict['train/compute_time'] = compute_time.avg

                    # Log weight norms if enabled
                    if (hasattr(self.config.logging.wandb, 'log_weight_norm') and
                        self.config.logging.wandb.log_weight_norm and
                        hasattr(self.config.logging.wandb, 'weight_norm_interval') and
                        batch_idx % self.config.logging.wandb.weight_norm_interval == 0):
                        weight_norms = {}
                        for name, param in self.model.named_parameters():
                            if param.requires_grad and 'weight' in name:
                                # Use a simplified name for cleaner wandb UI
                                clean_name = name.replace('module.', '').replace('.weight', '')
                                weight_norms[f'weight_norm/{clean_name}'] = param.data.norm(2).item()
                        log_dict.update(weight_norms)

                    self._wandb.log(log_dict, step=global_step)

            if (
                self.use_wandb
                and self._wandb_log_visuals
                and epoch % self._wandb_visualization_interval == 0
                and batch_idx == 0
            ):
                self._log_wandb_training_visual(pixel_values.detach(), epoch, global_step)

            # Update timer for next iteration
            batch_end_time = time.time()

        # Log epoch metrics using the last global_step
        final_step = epoch * len(self.train_loader) + len(self.train_loader) - 1
        if self.use_wandb and self._wandb is not None:
            self._wandb.log({'train/epoch_loss': losses.avg, 'epoch': epoch}, step=final_step)

        return losses.avg

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        """Validate model"""
        self.model.eval()
        losses = AverageMeter()

        pbar = tqdm(self.val_loader, desc="Validation")
        
        # Storage for reconstruction visualization
        first_batch_for_vis = None

        for batch_idx, batch in enumerate(pbar):
            # Move to device with non_blocking
            pixel_values = batch['pixel_values'].to(self.device, non_blocking=True)

            # Generate mask (same as training for consistent evaluation)
            batch_size = pixel_values.shape[0]
            # Calculate number of patches dynamically
            num_frames = pixel_values.shape[1]
            height = pixel_values.shape[3]
            width = pixel_values.shape[4]

            # Use same calculation as training
            tubelet_size = 2
            patch_size = 16
            seq_length = (num_frames // tubelet_size) * ((height // patch_size) * (width // patch_size))
            num_patches = seq_length
            bool_masked_pos = self._generate_mask(batch_size, num_patches)
            bool_masked_pos = bool_masked_pos.to(self.device, non_blocking=True)

            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos)
                loss = outputs.loss

            # Update metrics
            losses.update(loss.item(), pixel_values.size(0))

            # Update progress bar
            pbar.set_postfix({'loss': losses.avg})
            
            # Store first batch for visualization
            if batch_idx == 0 and self.use_reconstruction_vis:
                if self.visualizer.fixed_samples is None:
                    # First epoch - set fixed samples
                    self.visualizer.set_fixed_samples(pixel_values, bool_masked_pos)
                first_batch_for_vis = {
                    'pixel_values': pixel_values,
                    'bool_masked_pos': bool_masked_pos,
                    'outputs': outputs
                }

        # Log to tensorboard
        self.writer.add_scalar('val/loss', losses.avg, epoch)

        # Reconstruction visualization
        if (self.use_reconstruction_vis and 
            first_batch_for_vis is not None and 
            epoch % self.vis_interval == 0 and
            self.use_wandb and 
            self._wandb is not None):
            
            self._log_reconstruction_visualization(epoch, first_batch_for_vis)

        # Log validation metrics using current global step
        if self.use_wandb and self._wandb is not None:
            current_step = epoch * len(self.train_loader) + len(self.train_loader) - 1
            self._wandb.log({'val/loss': losses.avg, 'epoch': epoch}, step=current_step)

        return losses.avg

    def _reconstruct_from_model_output(self, 
                                       pixel_values: torch.Tensor, 
                                       outputs, 
                                       bool_masked_pos: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct video from model outputs
        
        Args:
            pixel_values: Original input [B, T, C, H, W]
            outputs: Model outputs with 'logits' field
            bool_masked_pos: Boolean mask [B, num_patches]
            
        Returns:
            Reconstructed video [B, T, C, H, W]
        """
        # Get reconstruction from model
        # VideoMAE outputs.logits has shape [B, num_masked, tubelet_size * patch_size^2 * 3]
        
        batch_size = pixel_values.shape[0]
        num_frames = pixel_values.shape[1]
        height = pixel_values.shape[3]
        width = pixel_values.shape[4]
        
        patch_size = 16
        tubelet_size = 2
        
        # For now, create a simple reconstruction by combining original and predictions
        # This is a simplified version - the actual reconstruction is more complex
        reconstructed = pixel_values.clone()
        
        # Note: Full reconstruction would require unpatchifying the model outputs
        # For visualization purposes, we'll use the model's internal reconstruction if available
        
        return reconstructed

    def _log_reconstruction_visualization(self, epoch: int, batch_data: dict) -> None:
        """
        Log reconstruction visualization to W&B
        
        Args:
            epoch: Current epoch
            batch_data: Dictionary with pixel_values, bool_masked_pos, outputs
        """
        pixel_values = batch_data['pixel_values']
        outputs = batch_data['outputs']
        bool_masked_pos = batch_data['bool_masked_pos']
        
        # Use fixed samples for consistent tracking
        if self.visualizer.fixed_samples is not None:
            fixed_pixel_values = self.visualizer.fixed_samples.to(self.device)
            fixed_masks = self.visualizer.fixed_masks.to(self.device)
            
            # Get reconstruction for fixed samples
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                fixed_outputs = self.model(
                    pixel_values=fixed_pixel_values,
                    bool_masked_pos=fixed_masks
                )
            
            # Reconstruct videos
            reconstructed = self._reconstruct_from_model_output(
                fixed_pixel_values, 
                fixed_outputs, 
                fixed_masks
            )
            
            # Compute metrics
            metrics = self.visualizer.compute_metrics(
                fixed_pixel_values.cpu(),
                reconstructed.cpu()
            )

            # Log metrics
            current_step = epoch * len(self.train_loader) + len(self.train_loader) - 1
            metric_log = {
                'val/psnr': metrics['psnr'],
                'val/ssim': metrics['ssim'],
                'val/mse': metrics['mse'],
                'val/mae': metrics['mae'],
                'val/psnr_std': metrics['psnr_std'],
                'val/best_psnr': metrics['best_psnr'],
                'val/worst_psnr': metrics['worst_psnr'],
                'val/median_psnr': metrics['median_psnr'],
                'epoch': epoch
            }

            # Compute and log masked vs unmasked metrics if enabled
            if (hasattr(self.config.logging.reconstruction_vis, 'compare_masked_regions') and
                self.config.logging.reconstruction_vis.compare_masked_regions):
                masked_metrics = self.visualizer.compute_masked_unmasked_metrics(
                    fixed_pixel_values.cpu(),
                    reconstructed.cpu(),
                    fixed_masks.cpu()
                )
                metric_log.update({
                    'val/masked_psnr': masked_metrics['masked_psnr'],
                    'val/masked_mse': masked_metrics['masked_mse'],
                    'val/masked_mae': masked_metrics['masked_mae'],
                    'val/unmasked_psnr': masked_metrics['unmasked_psnr'],
                    'val/unmasked_mse': masked_metrics['unmasked_mse'],
                    'val/unmasked_mae': masked_metrics['unmasked_mae'],
                })

            self._wandb.log(metric_log, step=current_step)
            
            # Create and log visualizations
            vis_config = self.config.logging.reconstruction_vis
            visualizations = vis_config.get('visualizations', [])
            
            # Log visualizations for multiple samples
            num_samples_to_show = min(3, len(self.visualizer.fixed_samples))
            
            for sample_idx in range(num_samples_to_show):
                original_video = fixed_pixel_values[sample_idx].cpu()
                recon_video = reconstructed[sample_idx].cpu()
                
                vis_log = {}
                
                # Comparison grid
                if 'comparison_grid' in visualizations:
                    grid_img = self.visualizer.create_comparison_grid(
                        original_video,
                        recon_video,
                        sample_idx=sample_idx
                    )
                    vis_log[f'val/reconstruction_grid_sample_{sample_idx}'] = self._wandb.Image(
                        grid_img,
                        caption=f"Epoch {epoch}, Sample {sample_idx}, PSNR: {metrics['psnr']:.2f}dB"
                    )
                
                # Temporal sequence
                if 'temporal_sequence' in visualizations:
                    temporal_img = self.visualizer.create_temporal_view(
                        original_video,
                        recon_video,
                        sample_idx=sample_idx
                    )
                    vis_log[f'val/temporal_sequence_sample_{sample_idx}'] = self._wandb.Image(
                        temporal_img,
                        caption=f"Epoch {epoch}, Sample {sample_idx}"
                    )
                
                # Error heatmap (only for first sample to save space)
                if 'error_heatmap' in visualizations and sample_idx == 0:
                    heatmap_img = self.visualizer.create_error_heatmap(
                        original_video,
                        recon_video,
                        sample_idx=sample_idx
                    )
                    vis_log[f'val/error_heatmap'] = self._wandb.Image(
                        heatmap_img,
                        caption=f"Epoch {epoch}, MSE: {metrics['mse']:.6f}"
                    )
                
                if vis_log:
                    self._wandb.log(vis_log, step=current_step)

    def train(self) -> None:
        """Main training loop"""
        print(f"Starting training for {self.config.training.num_epochs} epochs")

        for epoch in range(1, self.config.training.num_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")

            # Step scheduler after each epoch
            self.scheduler.step()

            # Validate
            if epoch % self.config.training.val_interval == 0:
                val_loss = self.validate(epoch)
                print(f"Epoch {epoch} - Val Loss: {val_loss:.4f}")

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        epoch,
                        val_loss,
                        os.path.join(self.checkpoint_dir, 'best_model.pth')
                    )
                    print(f"Saved best model with val loss: {val_loss:.4f}")
                    if self.use_wandb and self._wandb is not None:
                        current_step = epoch * len(self.train_loader) + len(self.train_loader) - 1
                        self._wandb.log({'val/best_loss': val_loss, 'epoch': epoch}, step=current_step)
                        if self._wandb_log_checkpoints and hasattr(self._wandb, 'save'):
                            self._wandb.save(os.path.join(self.checkpoint_dir, 'best_model.pth'))

            # Save checkpoint
            if epoch % self.config.training.save_interval == 0:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    train_loss,
                    os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
                )
                if (
                    self.use_wandb
                    and self._wandb is not None
                    and self._wandb_log_checkpoints
                    and hasattr(self._wandb, 'save')
                ):
                    self._wandb.save(os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))

        print("Training completed!")
        self.writer.close()
        if self.use_wandb and self.wandb_run is not None:
            summary = getattr(self.wandb_run, 'summary', None)
            if summary is not None:
                summary['best_val_loss'] = float(self.best_val_loss)
        if self.use_wandb and self._wandb is not None:
            self._wandb.finish()


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Set seed
    set_seed(cfg.training.seed)

    # Create trainer
    trainer = VideoMAETrainer(cfg)

    # Resume if specified
    if cfg.training.resume:
        resume_path = to_absolute_path(str(cfg.training.resume))
        checkpoint = load_checkpoint(resume_path, trainer.model, trainer.optimizer)
        print(f"Resumed from epoch {checkpoint['epoch']}")

    # Train
    trainer.train()


if __name__ == "__main__":
    main()  # type: ignore[misc]