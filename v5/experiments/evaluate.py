"""
SGAPS-MAE Evaluation Script
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
from tqdm import tqdm

from models import SGAPS_MAE
from utils.metrics import compute_psnr, compute_ssim, MetricsLogger


def evaluate(model, data_loader, device, sample_rate=0.02):
    """Evaluate model on test data."""
    model.eval()
    metrics = MetricsLogger()
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            if isinstance(batch, dict):
                images = batch['pixel_values']
            else:
                images = batch[0]
            
            images = images.to(device)
            
            # Handle video format
            if images.dim() == 5:
                images = images[:, images.shape[1] // 2]
            
            B, C, H, W = images.shape
            num_samples = int(H * W * sample_rate)
            
            for b in range(B):
                # Random sampling
                indices = torch.randperm(H * W, device=device)[:num_samples]
                u = indices // W
                v = indices % W
                positions = torch.stack([u, v], dim=-1).float()
                
                # Extract pixels
                pixel_values = images[b, :, u, v].T
                
                # Reconstruct
                result = model.reconstruct(pixel_values, positions, use_memory=False)
                reconstruction = result['reconstruction']
                
                # Compute metrics
                psnr = compute_psnr(reconstruction, images[b:b+1])
                ssim = compute_ssim(reconstruction, images[b:b+1])
                
                metrics.update('psnr', psnr.item())
                metrics.update('ssim', ssim.item())
    
    return metrics.get_all()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='../smbdataset/data-smb')
    parser.add_argument('--sample_rate', type=float, default=0.02)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = SGAPS_MAE().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create test loader
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'v3'))
    from dataset import SMBVideoDataset
    from torch.utils.data import DataLoader
    
    test_dataset = SMBVideoDataset(
        data_root=args.data_root,
        split='test',
        num_frames=1,
        image_size=224
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Evaluate
    results = evaluate(model, test_loader, device, args.sample_rate)
    
    print("\nEvaluation Results:")
    print(f"  PSNR: {results['psnr']:.2f} dB")
    print(f"  SSIM: {results['ssim']:.4f}")
    print(f"  Sample Rate: {args.sample_rate * 100:.1f}%")


if __name__ == "__main__":
    main()
