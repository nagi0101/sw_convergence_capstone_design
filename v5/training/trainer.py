"""
SGAPS-MAE Trainer
Main training loop for the SGAPS-MAE model.
"""

import os
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..models import SGAPS_MAE
from .curriculum import CurriculumLearning, AdaptiveSampler
from .losses import SGAPSMAELoss


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SGAPSMAETrainer:
    """
    Trainer for SGAPS-MAE model.
    """
    
    def __init__(
        self,
        model: SGAPS_MAE,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device = None
    ):
        """
        Args:
            model: SGAPS-MAE model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Target device
        """
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model
        self.model = model.to(self.device)
        
        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Loss
        self.criterion = SGAPSMAELoss(
            sampled_weight=config.get('sampled_weight', 0.3),
            perceptual_weight=config.get('perceptual_weight', 0.4),
            structural_weight=config.get('structural_weight', 0.2),
            temporal_weight=config.get('temporal_weight', 0.1)
        ).to(self.device)
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1.5e-4),
            betas=(0.9, 0.95),
            weight_decay=config.get('weight_decay', 0.05)
        )
        
        # Scheduler
        total_steps = len(train_loader) * config.get('num_epochs', 500)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=config.get('min_lr', 1e-6)
        )
        
        # Curriculum learning
        self.curriculum = CurriculumLearning(
            total_epochs=config.get('num_epochs', 500)
        )
        self.sampler = AdaptiveSampler(
            image_size=config.get('image_size', 224)
        )
        
        # Logging
        self.log_dir = config.get('log_dir', 'results/tensorboard')
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Checkpointing
        self.checkpoint_dir = config.get('checkpoint_dir', 'results/checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # State
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.criterion.reset_temporal()
        
        losses = {
            'total': AverageMeter(),
            'sampled': AverageMeter(),
            'perceptual': AverageMeter(),
            'structural': AverageMeter(),
            'temporal': AverageMeter()
        }
        
        phase = self.curriculum.get_phase(self.current_epoch)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Get images
            if isinstance(batch, dict):
                images = batch['pixel_values']
            else:
                images = batch[0]
            
            images = images.to(self.device)
            
            # Handle video format [B, T, C, H, W] -> use middle frame
            if images.dim() == 5:
                mid_idx = images.shape[1] // 2
                images = images[:, mid_idx]
            
            # Sample pixels based on curriculum
            positions = self.sampler.sample(images, phase)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Extract pixel values for sampled positions
            B = images.shape[0]
            reconstructions = []
            all_losses = []
            
            for b in range(B):
                pos = positions[b]  # [N, 2]
                
                # Extract pixel values
                pixel_values = images[b, :, pos[:, 0].long(), pos[:, 1].long()].T  # [N, 3]
                
                # Reconstruct
                result = self.model.reconstruct(
                    pixel_values,
                    pos,
                    use_memory=False
                )
                
                reconstruction = result['reconstruction']
                reconstructions.append(reconstruction)
                
                # Compute loss
                loss_dict = self.criterion(
                    reconstruction,
                    images[b:b+1],
                    pos,
                    result.get('importance_map')
                )
                all_losses.append(loss_dict)
            
            # Average losses across batch
            batch_loss = sum(l['total'] for l in all_losses) / B
            
            # Backward pass
            batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.get('gradient_clip', 1.0)
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            losses['total'].update(batch_loss.item(), B)
            for key in ['sampled', 'perceptual', 'structural', 'temporal']:
                avg_loss = sum(l[key].item() for l in all_losses) / B
                losses[key].update(avg_loss, B)
            
            # Logging
            self.global_step += 1
            
            if batch_idx % self.config.get('log_interval', 10) == 0:
                self.writer.add_scalar('train/loss', losses['total'].avg, self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
                
                pbar.set_postfix({
                    'loss': losses['total'].avg,
                    'lr': self.optimizer.param_groups[0]['lr']
                })
        
        return {key: meter.avg for key, meter in losses.items()}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        self.criterion.reset_temporal()
        
        losses = {
            'total': AverageMeter(),
            'psnr': AverageMeter(),
            'ssim': AverageMeter()
        }
        
        phase = self.curriculum.get_phase(self.current_epoch)
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            if isinstance(batch, dict):
                images = batch['pixel_values']
            else:
                images = batch[0]
            
            images = images.to(self.device)
            
            if images.dim() == 5:
                mid_idx = images.shape[1] // 2
                images = images[:, mid_idx]
            
            positions = self.sampler.sample(images, phase)
            
            B = images.shape[0]
            
            for b in range(B):
                pos = positions[b]
                pixel_values = images[b, :, pos[:, 0].long(), pos[:, 1].long()].T
                
                result = self.model.reconstruct(pixel_values, pos, use_memory=False)
                reconstruction = result['reconstruction']
                
                # Compute loss
                loss_dict = self.criterion(
                    reconstruction,
                    images[b:b+1],
                    pos
                )
                losses['total'].update(loss_dict['total'].item())
                
                # Compute PSNR
                mse = torch.mean((reconstruction - images[b:b+1]) ** 2)
                psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
                losses['psnr'].update(psnr.item())
        
        return {key: meter.avg for key, meter in losses.items()}
    
    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{self.current_epoch}.pth')
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
    
    def train(self) -> None:
        """Main training loop."""
        num_epochs = self.config.get('num_epochs', 500)
        val_interval = self.config.get('val_interval', 5)
        save_interval = self.config.get('save_interval', 10)
        
        print(f"Starting training on {self.device}")
        print(f"Total epochs: {num_epochs}")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            print(f"Epoch {epoch} - Train Loss: {train_losses['total']:.4f}")
            
            # Curriculum step
            self.curriculum.step()
            
            # Validate
            if epoch % val_interval == 0:
                val_losses = self.validate()
                print(f"Epoch {epoch} - Val Loss: {val_losses['total']:.4f}, PSNR: {val_losses['psnr']:.2f}")
                
                self.writer.add_scalar('val/loss', val_losses['total'], epoch)
                self.writer.add_scalar('val/psnr', val_losses['psnr'], epoch)
                
                # Save best
                if val_losses['total'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total']
                    self.save_checkpoint(is_best=True)
            
            # Save checkpoint
            if epoch % save_interval == 0:
                self.save_checkpoint()
        
        self.writer.close()
        print("Training complete!")
