#!/usr/bin/bash

#SBATCH -J videomae
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -w aurora-g6
#SBATCH -p batch_ugrad
#SBATCH -t 1-0
#SBATCH -o logs/videomae_%A.out
#SBATCH --error=logs/videomae_%A.err

# Decompress dataset(../smbdataset/data-smb.7z to /local_datasets/data-smb) with yes all
# 7z x ../smbdataset/data-smb.7z -o/local_datasets/data-smb -y
wandb online
python train.py training.resume=/data/nagi0101/dev/sw_convergence_capstone_design/v3/results/hydra/20251106_121356/checkpoints/latest.pth
