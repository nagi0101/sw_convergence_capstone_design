"""
SMB Dataset Loader for VideoMAE Baseline
Loads Super Mario Bros gameplay episodes for video reconstruction
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


class SMBVideoDataset(Dataset):
    """Super Mario Bros Video Dataset for VideoMAE training"""

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        num_frames: int = 16,
        frame_interval: int = 1,
        image_size: int = 224,
        random_seed: int = 42,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1
    ) -> None:
        """
        Args:
            data_root: Path to smbdataset/data-smb directory
            split: 'train', 'val', or 'test'
            num_frames: Number of frames per video clip
            frame_interval: Interval between sampled frames
            image_size: Size to resize images to (square)
            random_seed: Random seed for reproducibility
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
        """
        self.data_root = Path(data_root)
        self.split = split
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.image_size = image_size

        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Load episode list
        self.episodes = self._load_episodes()

        # Split data
        random.seed(random_seed)
        random.shuffle(self.episodes)

        n_train = int(len(self.episodes) * train_ratio)
        n_val = int(len(self.episodes) * val_ratio)

        if split == 'train':
            self.episodes = self.episodes[:n_train]
        elif split == 'val':
            self.episodes = self.episodes[n_train:n_train + n_val]
        else:  # test
            self.episodes = self.episodes[n_train + n_val:]

        # Build frame index for all episodes
        self.frame_data: List[List[Path]] = []
        self._build_frame_index()

        print(f"Loaded {len(self.episodes)} episodes with {len(self.frame_data)} clips for {split} split")

    def _load_episodes(self) -> List[Path]:
        """Load all episode directories"""
        episodes = []
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root {self.data_root} does not exist")

        for episode_dir in self.data_root.iterdir():
            if episode_dir.is_dir():
                episodes.append(episode_dir)

        if len(episodes) == 0:
            raise ValueError(f"No episodes found in {self.data_root}")

        return sorted(episodes)

    def _build_frame_index(self) -> None:
        """Build index of all valid frame sequences"""
        for episode_dir in tqdm(self.episodes, desc=f"Building {self.split} index"):
            frames = sorted(list(episode_dir.glob("*.png")))

            # Create clips with sliding window
            clip_length = self.num_frames * self.frame_interval
            for i in range(0, len(frames) - clip_length + 1, self.num_frames // 2):
                # Sample frames with interval
                clip_frames = []
                for j in range(self.num_frames):
                    frame_idx = i + j * self.frame_interval
                    if frame_idx < len(frames):
                        clip_frames.append(frames[frame_idx])

                if len(clip_frames) == self.num_frames:
                    self.frame_data.append(clip_frames)

    def __len__(self) -> int:
        return len(self.frame_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with:
                - 'pixel_values': Tensor of shape (num_frames, channels, height, width)
                - 'episode_id': Episode identifier
        """
        clip_frames = self.frame_data[idx]

        # Load and transform frames
        frames = []
        for frame_path in clip_frames:
            img = Image.open(frame_path).convert('RGB')
            img_tensor = self.transform(img)
            frames.append(img_tensor)

        # Stack frames: (num_frames, channels, height, width)
        video_tensor = torch.stack(frames, dim=0)

        # Get episode ID from first frame path
        episode_id = clip_frames[0].parent.name

        return {
            'pixel_values': video_tensor,
            'episode_id': episode_id
        }


def create_dataloaders(
    data_root: str,
    batch_size: int = 8,
    num_workers: int = 4,
    num_frames: int = 16,
    image_size: int = 224
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, and test dataloaders"""

    train_dataset = SMBVideoDataset(
        data_root=data_root,
        split='train',
        num_frames=num_frames,
        image_size=image_size
    )

    val_dataset = SMBVideoDataset(
        data_root=data_root,
        split='val',
        num_frames=num_frames,
        image_size=image_size
    )

    test_dataset = SMBVideoDataset(
        data_root=data_root,
        split='test',
        num_frames=num_frames,
        image_size=image_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    import sys

    data_root = "../smbdataset/data-smb"

    print("Testing SMB dataset loader...")
    dataset = SMBVideoDataset(
        data_root=data_root,
        split='train',
        num_frames=16,
        image_size=224
    )

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample shape: {sample['pixel_values'].shape}")
        print(f"Episode ID: {sample['episode_id']}")
        print(f"Dataset size: {len(dataset)} clips")
        print("Dataset loading successful!")
    else:
        print("No data found!")