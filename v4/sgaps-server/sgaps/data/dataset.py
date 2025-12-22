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
        
        # Masking Ratio for Few-Shot / Inpainting training
        # Default to 0.0 if not specified (no masking)
        self.mask_ratio = getattr(config.training, 'mask_ratio', 0.0)
        
        # Data Augmentation: Repeats
        # Default to 1 (no repetition)
        self.repeats = 1
        if hasattr(config.training, 'augmentation') and config.training.augmentation.enabled:
            self.repeats = config.training.augmentation.repeats

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
            
            for i, data_item in enumerate(self.frame_indices):
                h5_path, session_key, frame_key = data_item[:3]

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
                keep_indices_local, stats = balancer.fit_transform(state_vectors_tensor, pixel_counts_tensor)
                
                # 3. Map back to original self.frame_indices
                new_frame_indices = [self.frame_indices[valid_indices_map[k]] for k in keep_indices_local]
                self.frame_indices = new_frame_indices
                print(f"Data Balancing filtered. {len(self.frame_indices)} frames remaining.")

                # --- Visualization (Balancing) ---
                if hasattr(self, 'cleaning_figures') is False: self.cleaning_figures = {}
                
                try:
                    import matplotlib.pyplot as plt
                    import matplotlib
                    matplotlib.use('Agg')
                    
                    n_clusters = stats['n_clusters']
                    orig_counts = stats['original_counts']
                    kept_counts = stats['kept_counts']
                    
                    x = list(range(n_clusters))
                    y_orig = [orig_counts.get(i, 0) for i in x]
                    y_kept = [kept_counts.get(i, 0) for i in x]
                    y_removed = [y_orig[i] - y_kept[i] for i in x]
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.bar(x, y_kept, label='Kept', color='lime')
                    ax.bar(x, y_removed, bottom=y_kept, label='Removed (Balanced)', color='orange', hatch='//')
                    ax.set_title("Data Balancing: Cluster Distribution")
                    ax.set_xlabel("Cluster ID")
                    ax.set_ylabel("Frames")
                    ax.legend()
                    
                    self.cleaning_figures['balancing_distribution'] = fig
                except Exception as e:
                    print(f"Failed to plot balancing stats: {e}")

        # --- Stationarity Filter (Session Level) ---
        if hasattr(config, 'data') and hasattr(config.data, 'filtering') and config.data.filtering.get('enabled', False):
            print("\nApplying Stationarity Filter...")
            from .cleaner import StationarityFilter
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg') # Ensure non-interactive backend
            
            filter_cfg = OmegaConf.create({"cleaning": config.data.filtering})
            s_filter = StationarityFilter(filter_cfg)
            
            # --- Visualization Data Collection ---
            all_velocities_before = []
            
            # 1. Group frames by session
            sessions = {} # full_sid -> {indices, states}
            
            # Pre-load all sessions to compute velocities for visualization/filtering
            session_list_for_filter = []
            
            # Temp storage to avoid re-reading files multiple times if possible, 
            # or just accept the IO cost for the sake of robust filtering.
            # We implemented a load loop previously (Step 220), let's reuse/refine it.
            
            temp_sessions = {}
            for idx, data_item in enumerate(self.frame_indices):
                path, sid, fid = data_item[:3]
                full_sid = f"{path}::{sid}"
                if full_sid not in temp_sessions:
                    temp_sessions[full_sid] = {"states": [], "indices": []}
                temp_sessions[full_sid]["indices"].append(idx)
                
            # Load stats
            frames_by_file = {}
            for full_sid, data in temp_sessions.items():
                fspath, fsid = full_sid.split("::")
                if fspath not in frames_by_file: frames_by_file[fspath] = []
                frames_by_file[fspath].append((fsid, data))
            
            for fpath, items in frames_by_file.items():
                try:
                    with h5py.File(fpath, 'r') as f:
                        for (fsid, data) in items:
                            indices = data['indices']
                            indices.sort()
                            s_states = []
                            for global_idx in indices:
                                _, _, fkey = self.frame_indices[global_idx][:3]
                                raw = f[fsid]['frames'][fkey]['state_vector'][:]
                                sv = np.full(self.max_state_dim, self.sentinel_value, dtype=np.float32)
                                if len(raw) > 0: sv[:len(raw)] = raw
                                s_states.append(sv)
                            data['states'] = np.array(s_states)
                except: pass
                
            # Build list and collect BEFORE stats
            from .cleaner import VelocityCalculator
            
            for full_sid, data in temp_sessions.items():
                if len(data.get('states', [])) > 0:
                    session_list_for_filter.append({
                        "id": full_sid,
                        "states": data['states'],
                        "indices": data['indices']
                    })
                    # Collect velocities for viz
                    vels = VelocityCalculator.compute_velocities(data['states'])
                    all_velocities_before.extend(vels)

            # Run Filter
            keep_session_indices, stats = s_filter.filter_sessions(session_list_for_filter)
            
            # Rebuild indices
            final_frame_indices_list = []
            kept_session_ids = set()
            
            for idx in keep_session_indices:
                s_data = session_list_for_filter[idx]
                kept_session_ids.add(s_data['id'])
                for frame_idx in s_data['indices']:
                    final_frame_indices_list.append(self.frame_indices[frame_idx])
            
            self.frame_indices = final_frame_indices_list
            print(f"Stationarity Filter applied. Remaining frames: {len(self.frame_indices)}")
            
            # --- Generate Visualization ---
            self.cleaning_figures = {}
            
            # 1. Velocity Distribution
            try:
                all_velocities_after = []
                for s_data in session_list_for_filter:
                    if s_data['id'] in kept_session_ids:
                        vels = VelocityCalculator.compute_velocities(s_data['states'])
                        all_velocities_after.extend(vels)
                        
                fig, ax = plt.subplots(figsize=(10, 6))
                # Histograms in log scale
                bins = np.logspace(np.log10(1e-4), np.log10(100), 50)
                
                if all_velocities_before:
                    ax.hist(all_velocities_before, bins=bins, alpha=0.5, label='Original', color='cyan', density=True)
                if all_velocities_after:
                    ax.hist(all_velocities_after, bins=bins, alpha=0.5, label='Filtered', color='lime', density=True)
                    
                ax.set_xscale('log')
                ax.set_title(f"Velocity Distribution (Threshold: {stats.get('threshold', 0):.4f})")
                ax.set_xlabel("Velocity")
                ax.legend()
                
                self.cleaning_figures['velocity_distribution'] = fig
            except Exception as e:
                print(f"Failed to generate velocity plot: {e}")



    @property
    def pixel_counts(self):
        """
        Returns the pixel counts for the available dataset.
        Handles repeats (augmentation) by expanding the list.
        """
        # Extract counts from the 4-tuple (path, sid, fid, count)
        base_counts = [item[3] for item in self.frame_indices]
        if self.repeats > 1:
            return np.repeat(base_counts, self.repeats)
        return np.array(base_counts)

    def _build_index(self):
        """
        Builds an index of all frames across all HDF5 files.
        The index correctly handles the session/frame structure.
        Each item in the index is a tuple (file_path, session_key, frame_key, pixel_count).
        """
        print("Building dataset index...")
        # self.pixel_counts is now a property
        
        for h5_path in self.h5_paths:
            try:
                with h5py.File(h5_path, 'r') as f:
                    # Iterate over each session (top-level group) in the file
                    for session_key in f.keys():
                        session_group = f[session_key]
                        if 'frames' in session_group and isinstance(session_group['frames'], h5py.Group):
                            frame_keys = sorted(session_group['frames'].keys())
                            for frame_key in frame_keys:
                                # Fast read of pixel count (metadata only)
                                count = 0
                                try:
                                    if 'pixels' in session_group['frames'][frame_key]:
                                        count = session_group['frames'][frame_key]['pixels'].shape[0]
                                except Exception:
                                    pass

                                self.frame_indices.append((h5_path, session_key, frame_key, count))
                                
            except Exception as e:
                print(f"Warning: Could not read file {h5_path}. Skipping. Error: {e}")
        print(f"Index built. Found {len(self.frame_indices)} total frames.")

    def __len__(self) -> int:
        return len(self.frame_indices) * self.repeats

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves a single data point from the dataset.
        """
        # Handle repeats: map augmented index back to original frame index
        original_idx = idx // self.repeats
        h5_path, session_key, frame_key, _ = self.frame_indices[original_idx] # Unpack 4 items

        if not hasattr(self, 'file_handles'):
            self.file_handles = {}

        if h5_path not in self.file_handles:
            # Open file and cache it.
            # Note: We do NOT use 'with' here because we want it to persist.
            # When the worker process dies, OS closes files.
            # Swmr=True might be safer if writing happens concurrently, but r is usually fine.
            self.file_handles[h5_path] = h5py.File(h5_path, 'r', swmr=True, libver='latest')
        
        f = self.file_handles[h5_path]
        
        # Access data directly from cached file handle
        try:
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
        except Exception as e:
            # Handle potential file access errors (e.g. if file handle is stale)
            # In strict training, maybe raise? For now let's raise to see issues.
            raise e

        # Create state mask (1 for valid data, 0 for sentinel)
        state_mask = (state_vector != self.sentinel_value).astype(np.float32)

        # Apply transforms if any
        if self.transforms:
            sparse_pixels, gt_frame = self.transforms(sparse_pixels, gt_frame)
        
        # Convert to PyTorch tensors
        # Normalize gt_frame to [0, 1] to match the model's sigmoid output
        gt_tensor = (torch.from_numpy(gt_frame).float() / 255.0).unsqueeze(0).unsqueeze(0) # Shape: [1, 1, H_orig, W_orig]
        
        # Interpolate logic removed as it is redundant (gt_frame is created at target_resolution)

        # Normalize sparse_pixels values (column 2) to [0, 1]
        sparse_pixels_norm = torch.from_numpy(sparse_pixels).float()
        sparse_pixels_norm[:, 2] /= 255.0

        # --- Subsampling / Masking Logic ---
        # masked_pixels: Inputs (approx (1-mask_ratio) * N)
        # sparse_pixels: Targets (N)
        
        num_pixels = len(sparse_pixels_norm)
        if self.mask_ratio > 0.0:
            # Determine number of pixels to keep (visible)
            num_keep = int(num_pixels * (1 - self.mask_ratio))
            num_keep = max(1, num_keep) # Ensure at least 1 pixel is visible
            
            # Random shuffle indices
            perm = torch.randperm(num_pixels)
            keep_indices = perm[:num_keep]
            
            masked_pixels_norm = sparse_pixels_norm[keep_indices]
        else:
            masked_pixels_norm = sparse_pixels_norm.clone()

        return {
            "sparse_pixels": sparse_pixels_norm,   # Target: All pixels
            "masked_pixels": masked_pixels_norm,   # Input: Visible pixels subset
            "gt_frame": gt_tensor.squeeze(0),      # Shape: [1, H_new, W_new]
            "state_vector": torch.from_numpy(state_vector).float(),
            "state_mask": torch.from_numpy(state_mask).float(),
            "num_pixels": len(sparse_pixels),      # Original count (Target)
            "num_input_pixels": len(masked_pixels_norm), # Input count
            "resolution": torch.tensor(target_resolution, dtype=torch.long)
        }

    def __del__(self):
        """
        Close all cached file handles.
        """
        if hasattr(self, 'file_handles'):
            for f in self.file_handles.values():
                try:
                    f.close()
                except:
                    pass
            self.file_handles.clear()