# Reconstruction Visualization

This module provides comprehensive reconstruction quality tracking and visualization for VideoMAE training.

## Features

### 1. Metrics Tracked (Every Epoch)

-   **PSNR** (Peak Signal-to-Noise Ratio): Reconstruction quality in dB
-   **SSIM** (Structural Similarity Index): Perceptual similarity (0-1)
-   **MSE** (Mean Squared Error): Pixel-wise reconstruction error
-   **MAE** (Mean Absolute Error): Average absolute difference
-   **Statistical Metrics**: Best/Worst/Median PSNR, Standard deviations

### 2. Visualizations (Every Epoch)

#### Comparison Grid

4-column comparison for multiple frames:

-   Original frames
-   Masked input (visualization)
-   Reconstructed frames
-   Absolute error maps (hot colormap)

Shows 5 evenly-spaced frames per sample.

#### Temporal Sequence View

Full temporal sequence showing:

-   All 16 original frames in one row
-   All 16 reconstructed frames below
-   Side-by-side comparison

#### Error Heatmap

Detailed error analysis:

-   Original vs Reconstructed side-by-side
-   Per-pixel MSE heatmap
-   Color-coded error intensity

### 3. W&B Dashboard Organization

```
Validation Metrics:
├── val/psnr           # Average PSNR
├── val/ssim           # Average SSIM
├── val/mse            # Average MSE
├── val/mae            # Average MAE
├── val/psnr_std       # PSNR standard deviation
├── val/best_psnr      # Best sample PSNR
├── val/worst_psnr     # Worst sample PSNR
└── val/median_psnr    # Median PSNR

Reconstruction Visualizations:
├── val/reconstruction_grid_sample_0
├── val/reconstruction_grid_sample_1
├── val/reconstruction_grid_sample_2
├── val/temporal_sequence_sample_0
├── val/temporal_sequence_sample_1
├── val/temporal_sequence_sample_2
└── val/error_heatmap
```

## Configuration

Edit `conf/logging/base.yaml`:

```yaml
reconstruction_vis:
    enable: true # Enable/disable visualization
    interval: 1 # Log every N epochs
    num_samples: 5 # Number of fixed samples to track
    metrics: # Metrics to compute
        - psnr
        - ssim
        - mse
        - mae
    visualizations: # Visualizations to generate
        - comparison_grid
        - temporal_sequence
        - error_heatmap
```

## Implementation Details

### Fixed Samples

-   First validation batch of each run
-   Tracked consistently across all epochs
-   Allows monitoring progressive improvement

### Memory Efficient

-   Only stores 5 samples (configurable)
-   Visualizations generated on-demand
-   No disk storage (W&B only)

### Quality Metrics

-   PSNR: Higher is better (typical: 20-40 dB)
-   SSIM: Higher is better (0-1 scale)
-   MSE/MAE: Lower is better

## Example Output

After training for 10 epochs:

```
Epoch 10 Metrics:
- val/psnr: 28.4 dB
- val/ssim: 0.876
- val/mse: 0.0023
- val/mae: 0.032

Best sample: 32.1 dB
Worst sample: 25.3 dB
```

## Performance Impact

-   **Computation**: ~5-10 seconds per epoch for visualization
-   **W&B Upload**: Minimal (3 images per sample)
-   **Total Overhead**: <5% of training time

## Dependencies

All dependencies already in requirements.txt:

-   scikit-image (PSNR, SSIM)
-   matplotlib (visualization)
-   numpy (metrics)
