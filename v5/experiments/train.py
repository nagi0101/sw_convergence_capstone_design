"""
SGAPS-MAE Training Script
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from models import SGAPS_MAE
from training import SGAPSMAETrainer


def create_dataloader(config):
    """Create data loaders using the SMB dataset."""
    from torch.utils.data import DataLoader
    
    # Import dataset from v3 for compatibility
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'v3'))
    from dataset import SMBVideoDataset
    
    train_dataset = SMBVideoDataset(
        data_root=config.data.data_root,
        split='train',
        num_frames=1,  # Single frame for SGAPS-MAE
        image_size=config.data.image_size
    )
    
    val_dataset = SMBVideoDataset(
        data_root=config.data.data_root,
        split='val',
        num_frames=1,
        image_size=config.data.image_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Set seed
    torch.manual_seed(cfg.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.training.seed)
    
    # Create model
    model = SGAPS_MAE(
        image_size=(cfg.model.image_size, cfg.model.image_size),
        pixel_embed_dim=cfg.model.pixel_embed_dim,
        pos_embed_dim=cfg.model.pos_embed_dim,
        hidden_dim=cfg.model.hidden_dim,
        output_dim=cfg.model.output_dim,
        num_gat_layers=cfg.model.num_gat_layers,
        num_heads=cfg.model.num_heads,
        knn_k=cfg.model.knn_k,
        diffusion_steps=cfg.model.diffusion_steps,
        sampling_budget=cfg.model.sampling_budget,
        dropout=cfg.model.dropout
    )
    
    # Create data loaders
    train_loader, val_loader = create_dataloader(cfg)
    
    # Training config
    train_config = {
        'num_epochs': cfg.training.num_epochs,
        'learning_rate': cfg.training.learning_rate,
        'min_lr': cfg.training.min_lr,
        'weight_decay': cfg.training.weight_decay,
        'gradient_clip': cfg.training.gradient_clip,
        'val_interval': cfg.training.val_interval,
        'save_interval': cfg.training.save_interval,
        'sampled_weight': cfg.loss.sampled_weight,
        'perceptual_weight': cfg.loss.perceptual_weight,
        'structural_weight': cfg.loss.structural_weight,
        'temporal_weight': cfg.loss.temporal_weight,
        'log_dir': cfg.logging.log_dir,
        'checkpoint_dir': cfg.logging.checkpoint_dir,
        'log_interval': cfg.logging.log_interval,
        'image_size': cfg.model.image_size
    }
    
    # Create trainer
    trainer = SGAPSMAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config
    )
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
