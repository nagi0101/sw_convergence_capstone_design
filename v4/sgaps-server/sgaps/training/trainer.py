"""
Training pipeline for the Sparse Pixel Transformer.
"""
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
import torch.nn as nn
from tqdm import tqdm
from omegaconf import DictConfig
import numpy as np
import os
from pathlib import Path

# Utility imports
from sgaps.utils.metrics import compute_all_metrics

class SGAPSTrainer:
    """
    Manages the training and validation pipeline for the SPT model.
    """
    def __init__(self, model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, 
                 val_loader: torch.utils.data.DataLoader, config: DictConfig, checkpoint_key: str):
        """
        Args:
            model: The SparsePixelTransformer model.
            train_loader: DataLoader for the training set.
            val_loader: DataLoader for the validation set.
            config: The Hydra configuration object.
            checkpoint_key: The key to use for saving checkpoints.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.checkpoint_key = checkpoint_key
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.optimizer.learning_rate,
            weight_decay=config.training.optimizer.weight_decay
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.num_epochs,
            eta_min=config.training.scheduler.eta_min
        )

        # Loss function
        # Loss function selection
        if config.loss.type == "sampled_pixel_l2":
            from sgaps.models.losses import SampledPixelL2Loss
            self.criterion = SampledPixelL2Loss()
        elif config.loss.type == "full_l2":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss type: {config.loss.type}")

        # Automatic Mixed Precision
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.config.training.use_amp)
        
        # WandB
        self.wandb = None
        try:
            import wandb
            self.wandb = wandb
            self.use_wandb = True if config.logging.type == "wandb" else False
        except ImportError:
            self.use_wandb = False

        # Track best score for checkpoint management
        self.best_score = float('inf') if config.training.checkpoint.mode == 'min' else float('-inf')

    def train_epoch(self, epoch: int):
        """
        Runs a single training epoch.
        """
        self.model.train()
        total_loss = 0
        progress = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.training.num_epochs} [Train]")

        for batch in progress:
            sparse_pixels = batch["sparse_pixels"].to(self.device)
            gt_frame = batch["gt_frame"].to(self.device)
            state_vector = batch["state_vector"].to(self.device)
            state_mask = batch["state_mask"].to(self.device)
            resolution = batch["resolution"].to(self.device)
            num_pixels = batch["num_pixels"].to(self.device)

            # Enforce FlashAttention for speed and memory efficiency
            # Fallback to mem_efficient if flash is not available, but avoid slow math kernel
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]), \
                 torch.amp.autocast('cuda', enabled=self.config.training.use_amp):
                # Assuming all resolutions in the batch are the same, use the first one.
                resolution_tuple = (resolution[0, 0].item(), resolution[0, 1].item())
                
                # Pass num_pixels to model to enable internal masking (Attn, Skip, etc.)
                pred = self.model(
                    sparse_pixels, 
                    state_vector, 
                    state_mask, 
                    resolution_tuple,
                    num_pixels=num_pixels
                )
                
                # Create pixel_padding_mask for loss function
                # Logic must match model's mask generation (True=Padding/Ignore)
                B, N, _ = sparse_pixels.shape
                indices = torch.arange(N, device=self.device).unsqueeze(0).expand(B, N)
                pixel_padding_mask = indices >= num_pixels.unsqueeze(1)
                
                if self.config.loss.type == "sampled_pixel_l2":
                    losses = self.criterion(pred, gt_frame, sparse_pixels[:, :, :2], pixel_padding_mask=pixel_padding_mask)
                else:
                     # For Full L2, we compare full prediction against full GT
                    losses = {"total": self.criterion(pred, gt_frame)}

            self.optimizer.zero_grad()
            self.scaler.scale(losses["total"]).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += losses["total"].item()
            progress.set_postfix(loss=losses["total"].item())

            if self.use_wandb and self.wandb:
                self.wandb.log({
                    "Train/loss_step": losses["total"].item(),
                    "Train/lr": self.optimizer.param_groups[0]["lr"]
                })
            
            # --- DEBUG VISUALIZATION ---
            # Save debug images every 10 batches to verify data and model output
            if progress.n % 50 == 0:
                step = progress.n
                debug_dir = Path("debug_outputs") / f"epoch_{epoch}"
                debug_dir.mkdir(parents=True, exist_ok=True)
                
                # Use the first item in the batch for visualization
                # Detach and move to CPU
                sparse_pixels_viz = sparse_pixels[0].detach().cpu().numpy() # [N, 3]
                gt_frame_viz = gt_frame[0].detach().cpu().numpy().squeeze() # [H, W] (0-1 float)
                pred_viz = pred[0].detach().cpu().numpy().squeeze() # [H, W] (sigmoid output 0-1)
                
                # Rescale GT for visualization (0-255 uint8) if it's float [0, 1]
                if gt_frame_viz.max() <= 1.0:
                    gt_frame_viz = (gt_frame_viz * 255).astype(np.uint8)
                else:
                    gt_frame_viz = gt_frame_viz.astype(np.uint8)

                # Rescale Pred
                pred_viz = np.clip(pred_viz, 0.0, 1.0) # Clamp to valid range (since Sigmoid is removed)
                pred_viz = (pred_viz * 255).astype(np.uint8)

                # Helper to create a simple grid using PIL if DebugVisualizer fails or adds too much overhead
                # But here we will try to use the existing DebugVisualizer if possible, 
                # or a simple fallback since we want to be sure it works.
                try:
                    from sgaps.utils.visualization import DebugVisualizer
                    if not hasattr(self, 'viz'):
                        self.viz = DebugVisualizer(self.config)
                    
                    # Create dashboard
                    # We pass dummy importance/state for now to reuse the class
                    img = self.viz.create_composite_dashboard(
                        original_frame=gt_frame_viz,
                        reconstructed_frame=pred_viz,
                        sampled_pixels=sparse_pixels_viz[:, :2], # UVs
                        state_vector=state_vector[0].detach().cpu().numpy(),
                        importance_map=np.zeros_like(gt_frame_viz, dtype=np.float32), # Placeholder
                        metadata={'frame_id': f"{epoch}_{step}", 'loss': losses["total"].item()}
                    )
                    img.save(debug_dir / f"step_{step}.png")
                except Exception as e:
                    # Fallback: Simple concatenation using standard libraries
                    import cv2
                    # Difference map
                    diff = np.abs(gt_frame_viz.astype(int) - pred_viz.astype(int)).astype(np.uint8)
                    # Stack: GT | Pred | Diff
                    debug_img = np.hstack([gt_frame_viz, pred_viz, diff])
                    cv2.imwrite(str(debug_dir / f"step_{step}_fallback.png"), debug_img)
                    print(f"Viz Error: {e}")

        avg_loss = total_loss / len(self.train_loader)

        if self.use_wandb and self.wandb:
            self.wandb.log({"Train/loss_epoch": avg_loss, "epoch": epoch})
        
        return avg_loss

    def validate(self, epoch: int):
        """
        Runs validation on the model, computes loss on sampled pixels, and logs them.
        """
        self.model.eval()
        total_val_loss = 0
        val_progress = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.config.training.num_epochs} [Val]")

        with torch.no_grad():
            for i, batch in enumerate(val_progress):
                sparse_pixels = batch["sparse_pixels"].to(self.device)
                gt_frame = batch["gt_frame"].to(self.device)
                state_vector = batch["state_vector"].to(self.device)
                state_mask = batch["state_mask"].to(self.device)
                resolution = batch["resolution"].to(self.device)

                with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]), \
                     torch.amp.autocast('cuda', enabled=self.config.training.use_amp):
                    # Assuming all resolutions in the batch are the same, use the first one.
                    resolution_tuple = (resolution[0, 0].item(), resolution[0, 1].item())
                    pred = self.model(sparse_pixels, state_vector, state_mask, resolution_tuple)
                    
                    # Use the same criterion as training to calculate validation loss
                    if self.config.loss.type == "sampled_pixel_l2":
                        losses = self.criterion(pred, gt_frame, sparse_pixels[:, :, :2])
                    else:
                        losses = {"total": self.criterion(pred, gt_frame)}
                
                total_val_loss += losses["total"].item()
                val_progress.set_postfix(val_loss=losses["total"].item())

                # Log first batch images to wandb
                if i == 0 and self.use_wandb and self.wandb:
                    # For visualization, create a composite GT
                    # Start with the prediction as a background
                    gt_composite = pred.clone().cpu()
                    # Get the ground truth sparse image
                    gt_sparse = gt_frame.clone().cpu()
                    # Create a mask of the sampled pixels
                    mask = gt_sparse > 0
                    # Stamp the true pixels onto the prediction background, ensuring dtype match
                    gt_composite[mask] = gt_sparse[mask].to(gt_composite.dtype)
                    
                    # Create a difference image
                    diff_image = torch.abs(gt_composite - pred.cpu())

                    # Convert to numpy for wandb
                    pred_np = (pred.cpu().numpy() * 255).astype(np.uint8)
                    gt_comp_np = (gt_composite.numpy() * 255).astype(np.uint8)
                    diff_np = (diff_image.numpy() * 255).astype(np.uint8)

                    # Log a list of images for comparison
                    images_to_log = []
                    for k in range(min(5, pred_np.shape[0])):
                        # Create a single comparison image: GT | Pred | Diff
                        comparison_img = np.hstack([
                            gt_comp_np[k].squeeze(), 
                            pred_np[k].squeeze(), 
                            diff_np[k].squeeze()
                        ])
                        caption = f"Item {k}: GT (Composite) | Prediction | Difference"
                        images_to_log.append(self.wandb.Image(comparison_img, caption=caption))
                    
                    self.wandb.log({"Val/Examples": images_to_log, "epoch": epoch})

        # Aggregate metrics
        avg_val_loss = total_val_loss / len(self.val_loader)

        if self.use_wandb and self.wandb:
            self.wandb.log({
                "Val/loss": avg_val_loss,
                "epoch": epoch
            })
            
        return {"val_loss": avg_val_loss}


    def save_checkpoint(self, epoch: int, score: float):
        """
        Saves a model checkpoint in hierarchical structure.

        Strategy:
        - Always save to {checkpoint_key}/best.pth for the best model
        - Optionally save epoch checkpoints to {checkpoint_key}/epoch_{N}.pth if keep_all_epochs=True
        """
        # Get checkpoint_key from the instance variable provided during initialization
        checkpoint_key = self.checkpoint_key

        # Create checkpoint directory structure
        checkpoint_dir = Path(self.config.paths.checkpoint_dir) / checkpoint_key
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Prepare checkpoint data, excluding dynamically generated buffers like 'query_grid'
        state_dict = self.model.state_dict()
        if 'query_grid' in state_dict:
            del state_dict['query_grid']

        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'score': score,
            'config': self.config
        }

        # Determine if this is the best model
        is_best = False
        if self.config.training.checkpoint.mode == 'min':
            is_best = score < self.best_score
        else:
            is_best = score > self.best_score

        # Save best.pth if this is the best model
        if is_best:
            self.best_score = score
            best_path = checkpoint_dir / "best.pth"
            torch.save(checkpoint_data, best_path)
            print(f"New best checkpoint saved to {best_path} (score: {score:.4f})")

        # Optionally save epoch checkpoint
        if self.config.training.checkpoint.get('keep_all_epochs', False):
            epoch_path = checkpoint_dir / f"epoch_{epoch}_score_{score:.4f}.pth"
            torch.save(checkpoint_data, epoch_path)
            print(f"Epoch checkpoint saved to {epoch_path}")
