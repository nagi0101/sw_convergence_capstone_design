"""
Main training script for the SGAPS-MAE project.
This script uses Hydra for configuration management.
"""
import sys
import os

# Add the project root to the Python path
# This allows running the script from anywhere and ensures modules are found
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import hydra
from omegaconf import DictConfig

# Adjust this import path if your project structure is different
from pathlib import Path
import torch.nn.utils.rnn as rnn_utils
from sgaps.models.spt import SparsePixelTransformer
from sgaps.data.dataset import SGAPSDataset
from sgaps.training.trainer import SGAPSTrainer

import torch

def collate_fn(batch):
    """
    Custom collate function to handle variable-sized tensors in a batch.
    """
    # Separate items
    sparse_pixels = [item['sparse_pixels'] for item in batch]
    gt_frames = [item['gt_frame'] for item in batch]
    state_vectors = [item['state_vector'] for item in batch]
    state_masks = [item['state_mask'] for item in batch]
    resolutions = [item['resolution'] for item in batch]
    num_pixels = [item['num_pixels'] for item in batch]

    # Pad sparse_pixels (variable length N)
    sparse_pixels_padded = rnn_utils.pad_sequence(sparse_pixels, batch_first=True, padding_value=0)

    # Stack other tensors
    gt_frames_stacked = torch.stack(gt_frames)
    state_vectors_stacked = torch.stack(state_vectors)
    state_masks_stacked = torch.stack(state_masks)
    resolutions_stacked = torch.stack(resolutions)
    num_pixels_tensor = torch.tensor(num_pixels, dtype=torch.long)

    return {
        'sparse_pixels': sparse_pixels_padded,
        'gt_frame': gt_frames_stacked,
        'state_vector': state_vectors_stacked,
        'state_mask': state_masks_stacked,
        'resolution': resolutions_stacked,
        'num_pixels': num_pixels_tensor
    }

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training function.
    
    Args:
        cfg: The configuration object loaded by Hydra.
    """
    print("--- SGAPS-MAE Training ---")

    # 1. Immediately access checkpoint_key to fail fast if the mandatory value is not provided.
    checkpoint_key = cfg.training.checkpoint.checkpoint_key
    print(f"Starting training run for checkpoint_key: '{checkpoint_key}'")

    # 2. Initialize Weights & Biases if configured, using the key in the run name for clarity.
    if cfg.logging.type == "wandb":
        try:
            import wandb
            # Use the key in the run name for better tracking, e.g., "sgaps-v4-sgaps-mae-fps-train"
            run_name = f"{cfg.logging.wandb.name}-{checkpoint_key}-train"
            wandb.init(
                project=cfg.logging.wandb.project,
                entity=cfg.logging.wandb.entity,
                name=run_name,
                config=dict(cfg)
            )
            print("Weights & Bienses initialized successfully for training run.")
        except ImportError:
            print("WandB not installed. Skipping initialization.")
        except Exception as e:
            print(f"Failed to initialize Weights & Biases: {e}")

    # 3. Initialize Model
    print("Initializing model...")
    model = SparsePixelTransformer(cfg)
    
    # 4. Initialize Datasets and DataLoaders
    print("Initializing datasets...")
    
    # Find all HDF5 files in the configured checkpoint_path directory
    checkpoint_path = cfg.training.checkpoint.checkpoint_path
    all_h5_files = list(Path(checkpoint_path).glob("**/*.h5"))
    
    if not all_h5_files:
        raise FileNotFoundError(f"No HDF5 data found in {checkpoint_path}. Please ensure HDF5 files are present.")
    
    print(f"Found {len(all_h5_files)} HDF5 files.")

    # Create a single dataset with all files
    full_dataset = SGAPSDataset(all_h5_files, cfg)

    # Split the dataset into training and validation sets dynamically
    train_ratio = cfg.data.split_ratios[0]
    
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size # Ensure all data is used
    
    # For reproducibility of the split
    generator = torch.Generator().manual_seed(cfg.seed)
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=generator)
    
    print(f"Splitting dataset: {train_size} for training, {val_size} for validation.")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    print(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")

    # 5. Initialize Trainer
    print("Initializing trainer...")
    trainer = SGAPSTrainer(model, train_loader, val_loader, cfg, checkpoint_key=checkpoint_key)

    # 6. Start Training Loop
    print("--- Starting Training ---")

    for epoch in range(cfg.training.num_epochs):
        avg_train_loss = trainer.train_epoch(epoch)
        print(f"Epoch {epoch+1}/{cfg.training.num_epochs} | Average Training Loss: {avg_train_loss:.4f}")

        # Validation
        if (epoch + 1) % cfg.training.checkpoint.save_interval == 0:
            val_metrics = trainer.validate(epoch)
            # The metric to decide the best model is defined in the config (e.g., 'val_loss')
            score = val_metrics.get(cfg.training.checkpoint.metric, avg_train_loss)
            print(f"Epoch {epoch+1} | Validation Loss: {val_metrics.get('val_loss', 0):.4f}")

            # Trainer handles best score tracking internally
            trainer.save_checkpoint(epoch, score=score)

        trainer.scheduler.step()

    print("--- Training Finished ---")

if __name__ == "__main__":
    main()
