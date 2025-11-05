#!/usr/bin/bash

#SBATCH -J videomae
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -t 1-0
#SBATCH -o logs/videomae_%A.out
#SBATCH --error=logs/videomae_%A.err

wandb online
python train.py training=debug
