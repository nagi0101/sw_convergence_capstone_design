"""
VideoMAE Training Script for SMB Dataset
Uses Hugging Face Transformers implementation
"""

import os
import argparse
import yaml
from typing import Dict, Any, Tuple, Optional, Union

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import (
    VideoMAEForPreTraining,
    VideoMAEImageProcessor,
    VideoMAEConfig
)

from dataset import create_dataloaders
from utils import set_seed, save_checkpoint, load_checkpoint, AverageMeter


class VideoMAETrainer:
    """Trainer for VideoMAE baseline model"""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

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
        self.writer = SummaryWriter(log_dir=config['logging']['tensorboard_dir'])
        self.best_val_loss = float('inf')

    def _setup_model(self) -> None:
        """Initialize VideoMAE model"""

        # Load pretrained model or create new
        if self.config['model']['pretrained']:
            print(f"Loading pretrained model: {self.config['model']['pretrained']}")
            # Load the pretrained model with its original config
            self.model = VideoMAEForPreTraining.from_pretrained(
                self.config['model']['pretrained'],
                ignore_mismatched_sizes=True
            )
            # Update mask ratio if needed
            self.model.config.mask_ratio = self.config['model']['mask_ratio']
            self.model.config.norm_pix_loss = self.config['model']['norm_pix_loss']
        else:
            print("Creating new model from scratch")
            model_config = VideoMAEConfig(
                image_size=self.config['data']['image_size'],
                patch_size=16,
                num_channels=3,
                num_frames=self.config['data']['num_frames'],
                tubelet_size=2,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                decoder_num_hidden_layers=4,
                decoder_hidden_size=384,
                decoder_num_attention_heads=6,
                decoder_intermediate_size=1536,
                mask_ratio=self.config['model']['mask_ratio'],
                norm_pix_loss=self.config['model']['norm_pix_loss']
            )
            self.model = VideoMAEForPreTraining(model_config)

        self.model = self.model.to(self.device)

        # Image processor
        self.processor = VideoMAEImageProcessor(
            do_resize=True,
            size={"height": self.config['data']['image_size'],
                  "width": self.config['data']['image_size']},
            do_normalize=True
        )

    def _setup_data(self) -> None:
        """Setup data loaders"""
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            data_root=self.config['data']['data_root'],
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            num_frames=self.config['data']['num_frames'],
            image_size=self.config['data']['image_size']
        )

    def _setup_training(self) -> None:
        """Setup optimizer and scheduler"""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            betas=(0.9, 0.95),
            weight_decay=self.config['training']['weight_decay']
        )

        total_steps = len(self.train_loader) * self.config['training']['num_epochs']
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.config['training']['min_lr']
        )

    def _generate_mask(self, batch_size: int, num_patches: int) -> torch.Tensor:
        """Generate tube masking for VideoMAE"""
        mask_ratio = self.config['model']['mask_ratio']
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

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['training']['num_epochs']}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            pixel_values = batch['pixel_values'].to(self.device)

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
            bool_masked_pos = bool_masked_pos.to(self.device)

            # Forward pass
            outputs = self.model(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos)
            loss = outputs.loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config['training']['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )

            self.optimizer.step()

            # Update metrics
            losses.update(loss.item(), pixel_values.size(0))

            # Update progress bar
            pbar.set_postfix({
                'loss': losses.avg,
                'lr': self.optimizer.param_groups[0]['lr']
            })

            # Log to tensorboard
            global_step = epoch * len(self.train_loader) + batch_idx
            if batch_idx % self.config['logging']['log_interval'] == 0:
                self.writer.add_scalar('train/loss', loss.item(), global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], global_step)

        return losses.avg

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        """Validate model"""
        self.model.eval()
        losses = AverageMeter()

        pbar = tqdm(self.val_loader, desc="Validation")

        for batch in pbar:
            # Move to device
            pixel_values = batch['pixel_values'].to(self.device)

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
            bool_masked_pos = bool_masked_pos.to(self.device)

            # Forward pass
            outputs = self.model(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos)
            loss = outputs.loss

            # Update metrics
            losses.update(loss.item(), pixel_values.size(0))

            # Update progress bar
            pbar.set_postfix({'loss': losses.avg})

        # Log to tensorboard
        self.writer.add_scalar('val/loss', losses.avg, epoch)

        return losses.avg

    def train(self) -> None:
        """Main training loop"""
        print(f"Starting training for {self.config['training']['num_epochs']} epochs")

        for epoch in range(1, self.config['training']['num_epochs'] + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")

            # Step scheduler after each epoch
            self.scheduler.step()

            # Validate
            if epoch % self.config['training']['val_interval'] == 0:
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
                        os.path.join(self.config['training']['checkpoint_dir'], 'best_model.pth')
                    )
                    print(f"Saved best model with val loss: {val_loss:.4f}")

            # Save checkpoint
            if epoch % self.config['training']['save_interval'] == 0:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    train_loss,
                    os.path.join(self.config['training']['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pth')
                )

        print("Training completed!")
        self.writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description='Train VideoMAE on SMB dataset')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set seed
    set_seed(config['training']['seed'])

    # Create trainer
    trainer = VideoMAETrainer(config)

    # Resume if specified
    if args.resume:
        checkpoint = load_checkpoint(args.resume, trainer.model, trainer.optimizer)
        print(f"Resumed from epoch {checkpoint['epoch']}")

    # Train
    trainer.train()


if __name__ == "__main__":
    main()