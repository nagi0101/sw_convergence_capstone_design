"""
PyTorch Dataset for SGAPS-MAE Training.

Loads collected frame data from HDF5 storage for model training.
"""

import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    import h5py
    import numpy as np
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logging.warning("PyTorch or h5py not available")


logger = logging.getLogger(__name__)


if HAS_TORCH:
    class SGAPSDataset(Dataset):
        """
        PyTorch Dataset for SGAPS frame data.
        
        Loads sparse pixel samples and state vectors from HDF5 storage.
        Designed for training MAE-based reconstruction models.
        
        Attributes:
            file_path: Path to HDF5 file
            session_ids: List of session IDs to include
            max_state_dim: Maximum state vector dimension (for padding)
            sentinel_value: Sentinel value for unused state dimensions
        """
        
        def __init__(
            self,
            file_path: str,
            session_ids: List[str] = None,
            max_state_dim: int = 64,
            sentinel_value: float = -999.0,
            resolution: Tuple[int, int] = (640, 480)
        ):
            """
            Initialize the dataset.
            
            Args:
                file_path: Path to HDF5 file
                session_ids: List of sessions to include (None = all)
                max_state_dim: Maximum state vector dimension
                sentinel_value: Sentinel value for padding
                resolution: Target resolution for samples
            """
            self.file_path = Path(file_path)
            self.max_state_dim = max_state_dim
            self.sentinel_value = sentinel_value
            self.resolution = resolution
            
            if not self.file_path.exists():
                raise FileNotFoundError(f"HDF5 file not found: {file_path}")
            
            # Build index of all frames
            self.frame_index: List[Tuple[str, int]] = []
            self._build_index(session_ids)
            
            logger.info(f"SGAPSDataset loaded: {len(self.frame_index)} frames from {file_path}")
        
        def _build_index(self, session_ids: List[str] = None):
            """Build index of all frames across sessions."""
            with h5py.File(self.file_path, 'r') as f:
                for session_id in f.keys():
                    if session_ids is not None and session_id not in session_ids:
                        continue
                    
                    if "frames" not in f[session_id]:
                        continue
                    
                    frames = f[session_id]["frames"]
                    for frame_id in frames.keys():
                        self.frame_index.append((session_id, int(frame_id)))
            
            # Sort by session and frame ID
            self.frame_index.sort(key=lambda x: (x[0], x[1]))
        
        def __len__(self) -> int:
            """Return number of frames in dataset."""
            return len(self.frame_index)
        
        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            """
            Get a single frame sample.
            
            Returns:
                Dictionary containing:
                - pixels: (N, 3) tensor of (u, v, value)
                - state_vector: (max_state_dim,) tensor
                - mask: (H, W) binary mask of sampled positions
                - sparse_image: (1, H, W) sparse grayscale image
            """
            session_id, frame_id = self.frame_index[idx]
            
            with h5py.File(self.file_path, 'r') as f:
                frame_group = f[session_id]["frames"][str(frame_id)]
                
                # Load pixels
                pixels = frame_group["pixels"][:] if "pixels" in frame_group else np.zeros((0, 3))
                
                # Load state vector
                if "state_vector" in frame_group:
                    state = frame_group["state_vector"][:]
                else:
                    state = np.array([])
            
            # Pad state vector
            state_padded = np.full(self.max_state_dim, self.sentinel_value, dtype=np.float32)
            if len(state) > 0:
                state_padded[:min(len(state), self.max_state_dim)] = state[:self.max_state_dim]
            
            # Create sparse image and mask
            width, height = self.resolution
            sparse_image = np.zeros((height, width), dtype=np.float32)
            mask = np.zeros((height, width), dtype=np.float32)
            
            for u, v, value in pixels:
                x = int(u * (width - 1))
                y = int(v * (height - 1))
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                
                sparse_image[y, x] = value / 255.0  # Normalize to [0, 1]
                mask[y, x] = 1.0
            
            return {
                "pixels": torch.from_numpy(pixels.astype(np.float32)),
                "state_vector": torch.from_numpy(state_padded),
                "mask": torch.from_numpy(mask),
                "sparse_image": torch.from_numpy(sparse_image).unsqueeze(0),
                "session_id": session_id,
                "frame_id": frame_id
            }
        
        def get_dataloader(
            self,
            batch_size: int = 32,
            shuffle: bool = True,
            num_workers: int = 4
        ) -> DataLoader:
            """
            Create a DataLoader for this dataset.
            
            Args:
                batch_size: Batch size
                shuffle: Whether to shuffle
                num_workers: Number of worker processes
                
            Returns:
                PyTorch DataLoader instance
            """
            return DataLoader(
                self,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=self._collate_fn
            )
        
        @staticmethod
        def _collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
            """Custom collate function for variable-length pixel arrays."""
            result = {
                "state_vector": torch.stack([b["state_vector"] for b in batch]),
                "mask": torch.stack([b["mask"] for b in batch]),
                "sparse_image": torch.stack([b["sparse_image"] for b in batch]),
                "session_ids": [b["session_id"] for b in batch],
                "frame_ids": [b["frame_id"] for b in batch]
            }
            
            # Pad pixels to same length
            max_pixels = max(b["pixels"].shape[0] for b in batch)
            padded_pixels = []
            for b in batch:
                p = b["pixels"]
                if p.shape[0] < max_pixels:
                    pad = torch.zeros(max_pixels - p.shape[0], 3)
                    p = torch.cat([p, pad], dim=0)
                padded_pixels.append(p)
            
            result["pixels"] = torch.stack(padded_pixels)
            
            return result

else:
    # Placeholder when PyTorch is not available
    class SGAPSDataset:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for SGAPSDataset")
