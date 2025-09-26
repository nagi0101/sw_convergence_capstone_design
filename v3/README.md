# VideoMAE Baseline (v3)

Standard VideoMAE implementation for performance comparison with v4 HST-MAE.

## Overview

This is a baseline implementation of VideoMAE (Video Masked Autoencoders) using Hugging Face Transformers. It serves as a performance benchmark for comparing with the advanced HST-MAE model in v4.

### Key Features
- Standard VideoMAE architecture from NeurIPS 2022 paper
- 90% tube masking for self-supervised learning
- Pretrained weights from MCG-NJU/videomae-base
- Evaluation metrics: PSNR, SSIM, MSE, FPS

### Architecture Comparison

| Component | v3 (VideoMAE) | v4 (HST-MAE) |
|-----------|---------------|--------------|
| Encoder | Single-scale ViT | Hierarchical Swin |
| Attention | Global self-attention | Shifted window |
| Masking | Tube masking (90%) | Dual masking + region-aware |
| Input | Video frames only | Video + state vectors |
| Compression | None | RVQ codebook |

## Installation

```bash
# 1. Create virtual environment
cd v3
python -m venv .venv

# 2. Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Dataset Setup

Ensure the SMB dataset is available:
```
../smbdataset/
└── data-smb/
    ├── Rafael_cyj0sfke_e0_8-1_fail/
    ├── Rafael_cyj0sfke_e1_8-1_win/
    └── ...
```

## Usage

### Training

```bash
# Train from scratch
python train.py --config config.yaml

# Resume training from checkpoint
python train.py --config config.yaml --resume results/checkpoints/checkpoint_epoch_100.pth
```

### Evaluation

```bash
# Evaluate trained model
python evaluate.py --config config.yaml --checkpoint results/checkpoints/best_model.pth

# Save evaluation results
python evaluate.py --config config.yaml --checkpoint results/checkpoints/best_model.pth --output results/evaluation.yaml
```

## Configuration

Edit `config.yaml` to modify training parameters:

```yaml
model:
  pretrained: "MCG-NJU/videomae-base"  # Pretrained model
  mask_ratio: 0.9                      # 90% masking

training:
  num_epochs: 200                      # Training epochs
  batch_size: 8                        # Batch size
  learning_rate: 1.5e-4                # Learning rate
```

## Expected Performance

### Baseline (v3) vs Advanced (v4)

| Metric | VideoMAE (v3) | HST-MAE (v4) | Improvement |
|--------|---------------|--------------|-------------|
| PSNR | 33-35 dB | 38.5 dB | +3.5-5.5 dB |
| SSIM | 0.83-0.87 | 0.94 | +0.07-0.11 |
| FPS | 100-150 | 85 | -15-65 |
| Memory | 5-7 GB | 8 GB | +1-3 GB |

### Key Observations
- **Reconstruction Quality**: Standard VideoMAE shows limitations in complex scenes
- **Spatial Coherence**: Artifacts at region boundaries
- **Temporal Consistency**: Frame-to-frame flickering
- **Computational Efficiency**: Faster inference due to simpler architecture

## Project Structure

```
v3/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── config.yaml           # Training configuration
├── dataset.py            # SMB dataset loader
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── utils.py              # Utility functions
└── results/
    ├── checkpoints/      # Model checkpoints
    ├── visualizations/   # Sample reconstructions
    └── tensorboard/      # Training logs
```

## Monitoring Training

```bash
# Launch TensorBoard
tensorboard --logdir results/tensorboard
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config.yaml
- Reduce `num_frames` from 16 to 8
- Use gradient accumulation

### Slow Data Loading
- Increase `num_workers` in config.yaml
- Ensure data is on SSD
- Use smaller `image_size` (e.g., 112 instead of 224)

## Citation

If using this baseline, please cite:

```bibtex
@inproceedings{tong2022videomae,
  title={VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training},
  author={Tong, Zhan and Song, Yibing and Wang, Jue and Wang, Limin},
  booktitle={NeurIPS},
  year={2022}
}
```

## License

This implementation follows the CC-BY-NC 4.0 license of the original VideoMAE.