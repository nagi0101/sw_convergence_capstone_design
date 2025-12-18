"""
PyTorch Dataset for loading SGAPS data from HDF5 files.
"""
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from pathlib import Path
from typing import List, Tuple

class SGAPSDataset(Dataset):
    """
    Custom PyTorch Dataset to load frame data from HDF5 files
    created by the SGAPS data collection process.
    """
    def __init__(self, h5_paths: List[str], config, transforms=None):
        """
        Args:
            h5_paths: List of paths to HDF5 files.
            config: Hydra configuration object.
            transforms: Optional transformations to apply to the data.
        """
        self.h5_paths = [p for p in h5_paths if Path(p).exists()]
        self.config = config
        self.transforms = transforms
        self.max_state_dim = config.model.input_constraints.max_state_dim
        self.sentinel_value = config.model.sentinel_value

        self.frame_indices = []
        self._build_index()

        # Dynamic Data Cleaning (In-Memory)
        if hasattr(config, 'cleaning') and config.cleaning.enabled:
            from .cleaner import DataBalancer
            balancer = DataBalancer(config)
            
            # 1. Load all state vectors for clustering
            print("Loading state vectors for data cleaning...")
            state_vectors = []
            pixel_counts = []
            valid_indices_map = [] # To map back to self.frame_indices
            
            for i, (h5_path, session_key, frame_key) in enumerate(self.frame_indices):
                try:
                    with h5py.File(h5_path, 'r') as f:
                        frame_grp = f[session_key]['frames'][frame_key]
                        
                        raw_state = frame_grp['state_vector'][:]
                        # Pad if necessary
                        sv = np.full(self.max_state_dim, self.sentinel_value, dtype=np.float32)
                        if len(raw_state) > 0:
                            sv[:len(raw_state)] = raw_state
                        
                        # Get pixel count
                        sparse_pixels = frame_grp['pixels'][:] if 'pixels' in frame_grp else []
                        n_pixels = len(sparse_pixels)

                        state_vectors.append(sv)
                        pixel_counts.append(n_pixels)
                        valid_indices_map.append(i)
                except Exception as e:
                    # Skip broken frames during this check
                    continue
            
            if state_vectors:
                state_vectors_tensor = torch.tensor(np.array(state_vectors), dtype=torch.float32)
                pixel_counts_tensor = torch.tensor(np.array(pixel_counts), dtype=torch.float32)
                
                # 2. Get indices to keep (these are indices into the list we just created)
                # Pass pixel counts for enhanced clustering/filtering
                keep_indices_local = balancer.fit_transform(state_vectors_tensor, pixel_counts_tensor)
                
                # 3. Map back to original self.frame_indices
                new_frame_indices = [self.frame_indices[valid_indices_map[k]] for k in keep_indices_local]
                self.frame_indices = new_frame_indices
                print(f"Dataset filtered. {len(self.frame_indices)} frames remaining.")

    def _build_index(self):
        """
        Builds an index of all frames across all HDF5 files.
        The index correctly handles the session/frame structure.
        Each item in the index is a tuple (file_path, session_key, frame_key).
        """
        print("Building dataset index...")
        for h5_path in self.h5_paths:
            try:
                with h5py.File(h5_path, 'r') as f:
                    # Iterate over each session (top-level group) in the file
                    for session_key in f.keys():
                        session_group = f[session_key]
                        if 'frames' in session_group and isinstance(session_group['frames'], h5py.Group):
                            frame_keys = sorted(session_group['frames'].keys())
                            for frame_key in frame_keys:
                                self.frame_indices.append((h5_path, session_key, frame_key))
            except Exception as e:
                print(f"Warning: Could not read file {h5_path}. Skipping. Error: {e}")
        print(f"Index built. Found {len(self.frame_indices)} total frames.")

    def __len__(self) -> int:
        return len(self.frame_indices)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves a single data point from the dataset.
        """
        h5_path, session_key, frame_key = self.frame_indices[idx]

        with h5py.File(h5_path, 'r') as f:
            frame_group = f[session_key]['frames'][frame_key]
            
            # Load sparse pixel data
            sparse_pixels = frame_group['pixels'][:]  # Shape: [N, 3] (u, v, value)

            # Get target resolution from config
            target_resolution = tuple(self.config.data.resolution)
            H, W = target_resolution
            
            # --- Create Ground Truth from sparse pixels for the loss function ---
            gt_frame = np.zeros(target_resolution, dtype=np.uint8)
            u_coords = sparse_pixels[:, 0]
            v_coords = sparse_pixels[:, 1]
            values = sparse_pixels[:, 2]
            
            x_indices = (u_coords * (W - 1)).round().astype(int)
            y_indices = (v_coords * (H - 1)).round().astype(int)
            
            # Clamp indices to be within bounds
            x_indices = np.clip(x_indices, 0, W - 1)
            y_indices = np.clip(y_indices, 0, H - 1)

            gt_frame[y_indices, x_indices] = values.astype(np.uint8)
            # --- End of GT creation ---

            # Load state vector and pad to fixed size
            raw_state = frame_group['state_vector'][:] if 'state_vector' in frame_group else []
            state_vector = np.full(self.max_state_dim, self.sentinel_value, dtype=np.float32)
            if len(raw_state) > 0:
                state_vector[:len(raw_state)] = raw_state

        # Create state mask (1 for valid data, 0 for sentinel)
        state_mask = (state_vector != self.sentinel_value).astype(np.float32)

        # Apply transforms if any
        if self.transforms:
            sparse_pixels, gt_frame = self.transforms(sparse_pixels, gt_frame)
        
        # Convert to PyTorch tensors
        # Normalize gt_frame to [0, 1] to match the model's sigmoid output
        gt_tensor = (torch.from_numpy(gt_frame).float() / 255.0).unsqueeze(0).unsqueeze(0) # Shape: [1, 1, H_orig, W_orig]
        
        # Downsample GT frame to target resolution
        if gt_tensor.shape[2:] != target_resolution:
            gt_tensor = torch.nn.functional.interpolate(gt_tensor, size=target_resolution, mode='bilinear', align_corners=False)

        return {
            "sparse_pixels": torch.from_numpy(sparse_pixels).float(),
            "gt_frame": gt_tensor.squeeze(0),  # Shape: [1, H_new, W_new]
            "state_vector": torch.from_numpy(state_vector).float(),
            "state_mask": torch.from_numpy(state_mask).float(),
            "num_pixels": len(sparse_pixels),
            "resolution": torch.tensor(target_resolution, dtype=torch.long)
        }