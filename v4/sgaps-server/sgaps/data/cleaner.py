"""
Data Cleaning and Balancing Module
"""
import torch
import numpy as np
from omegaconf import DictConfig
from typing import List, Tuple

class DataBalancer:
    """
    Balances the dataset by clustering state vectors and downsampling over-represented clusters.
    """
    def __init__(self, config: DictConfig):
        self.config = config
        self.n_clusters = config.cleaning.clustering.n_clusters
        self.seed = config.cleaning.clustering.get('seed', 42)
        
        # Strategy settings
        self.strategy = config.cleaning.balancing.strategy
        self.strategy_value = config.cleaning.balancing.value

    def _kmeans(self, X: torch.Tensor, num_clusters: int, distance: str = 'euclidean', tol: float = 1e-4, max_iter: int = 100):
        """
        Simple K-Means implementation using PyTorch.
        """
        # Linear assignment of clusters as initial state
        # X: [N, D]
        N, D = X.shape
        
        # Randomly select initial centroids
        generator = torch.Generator(device=X.device)
        generator.manual_seed(self.seed)
        perm = torch.randperm(N, generator=generator)
        centroids = X[perm[:num_clusters]] # [K, D]
        
        for i in range(max_iter):
            # Compute distances: [N, K]
            # dist_sq = (x - c)^2 = x^2 + c^2 - 2xc
            x_sq = (X ** 2).sum(dim=1, keepdim=True) # [N, 1]
            c_sq = (centroids ** 2).sum(dim=1) # [K]
            params = torch.matmul(X, centroids.t()) # [N, K]
            dists = x_sq + c_sq - 2 * params
            
            # Assign to nearest centroid
            labels = dists.argmin(dim=1)
            
            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            unique_labels, counts = labels.unique(return_counts=True)
            
            # Check for convergence
            # Optimized scatter_add equivalent for mean calculation
            for k in range(num_clusters):
                mask = labels == k
                if mask.any():
                    new_centroids[k] = X[mask].mean(dim=0)
                else:
                    # Re-initialize empty cluster to a random point
                    new_centroids[k] = X[torch.randint(0, N, (1,), generator=generator)]

            shift = torch.norm(centroids - new_centroids)
            centroids = new_centroids
            
            if shift < tol:
                break
                
        # Final assignment
        x_sq = (X ** 2).sum(dim=1, keepdim=True)
        c_sq = (centroids ** 2).sum(dim=1)
        params = torch.matmul(X, centroids.t())
        dists = x_sq + c_sq - 2 * params
        labels = dists.argmin(dim=1)
        
        return labels, centroids

    def fit_transform(self, state_vectors: torch.Tensor, num_pixels: torch.Tensor = None) -> List[int]:
        """
        Clusters the data and returns the indices of frames to KEEP.
        
        Args:
            state_vectors: [N, D] tensor of state vectors
            num_pixels: [N] tensor of pixel counts (optional but recommended)
            
        Returns:
            List[int]: Indices of the original dataset to keep.
        """
        N_total = len(state_vectors)
        print(f"Starting Data Balancing on {N_total} frames...")
        
        # 1. Pre-filter based on min_pixels
        min_pixels = self.config.cleaning.get('filters', {}).get('min_pixels', 0)
        valid_mask = torch.ones(N_total, dtype=torch.bool)
        
        if num_pixels is not None and min_pixels > 0:
            print(f"Filtering frames with < {min_pixels} pixels...")
            valid_mask = num_pixels >= min_pixels
            
        # These are the indices (from original 0..N-1) that passed the pre-filter
        passed_indices = valid_mask.nonzero(as_tuple=True)[0]
        
        if len(passed_indices) == 0:
            print("Warning: No frames passed the filter! Returning empty list.")
            return []
            
        # Select data for clustering
        data_filtered = state_vectors[passed_indices]
        if num_pixels is not None:
            pixels_filtered = num_pixels[passed_indices]
        
        # Ensure data is on CPU
        device = torch.device('cpu') 
        data_filtered = data_filtered.to(device)

        # 1.5 Optimize: Remove Padding Dimensions
        # Identify columns that are NOT sentinel value (at least in some frames)
        # If a column is always sentinel, it provides no info.
        # But wait, sentinel is for *masking*.
        # If we just look at std/mean, sentinel values (-999) will skew it heavily.
        # We must mask them out OR slice to the "active" dimensions.
        # Assuming the state vector is filled from 0..K and rest is sentinel.
        # We find the max column index that has non-sentinel values.
        sentinel = self.config.model.get('sentinel_value', -999.0)
        
        # Check usually first frame to see valid dim? No, batch.
        # Check max valid index across batch
        is_valid = (data_filtered != sentinel) # [N, D]
        # Keep columns where any frame has valid data
        valid_cols = is_valid.any(dim=0) # [D]
        
        if valid_cols.any():
            data_filtered = data_filtered[:, valid_cols]
            print(f"Reduced state dimension from {state_vectors.shape[1]} to {data_filtered.shape[1]} (removed padding).")
        else:
             print("Warning: All state vectors seem to be padding (sentinel). Clustering might be random.")

        # Normalize State Vectors
        mean = data_filtered.mean(dim=0)
        std = data_filtered.std(dim=0) + 1e-6
        data_norm = (data_filtered - mean) / std
        
        # 2. Feature Augmentation (Pixel Count)
        include_pixel_feature = self.config.cleaning.clustering.get('include_pixel_count', False)
        if include_pixel_feature and num_pixels is not None:
            pixel_weight = self.config.cleaning.clustering.get('pixel_count_weight', 0.5)
            
            # Log transform: log10(x + 1)
            p_log = torch.log10(pixels_filtered.float() + 1).unsqueeze(1).to(device)
            
            # Standardize pixel feature
            p_mean = p_log.mean()
            p_std = p_log.std() + 1e-6
            p_norm = (p_log - p_mean) / p_std
            
            # Concatenate
            features = torch.cat([data_norm, p_norm * pixel_weight], dim=1)
            print("Using State Vector + Pixel Count features for clustering.")
        else:
            features = data_norm

        # 3. Run K-Means
        print(f"Running K-Means (k={self.n_clusters})...")
        labels, _ = self._kmeans(features, self.n_clusters)
        
        # Count samples per cluster
        unique_labels, counts = labels.unique(return_counts=True)
        counts_np = counts.numpy()
        
        # Determine Cap Limit
        cap_limit = float('inf')
        if self.strategy == "percentile":
            cap_limit = np.percentile(counts_np, self.strategy_value)
        elif self.strategy == "median":
            cap_limit = np.median(counts_np)
        elif self.strategy == "mean":
            cap_limit = np.mean(counts_np)
        else: # Fixed number
             cap_limit = float(self.strategy_value)
             
        cap_limit = max(int(cap_limit), 1) # Ensure at least 1
        print(f"Balancing Strategy: {self.strategy} ({self.strategy_value}) -> Cap Limit: {cap_limit}")

        indices_to_keep_local = []
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        # Iterate over clusters and select indices (relative to passed_indices)
        for k in range(self.n_clusters):
            # Get indices belonging to cluster k
            cluster_indices = (labels == k).nonzero(as_tuple=True)[0]
            num_samples = len(cluster_indices)
            
            if num_samples == 0:
                continue
                
            if num_samples > cap_limit:
                # Downsample
                perm = torch.randperm(num_samples, generator=generator)
                selected = cluster_indices[perm[:cap_limit]]
                indices_to_keep_local.extend(selected.tolist())
            else:
                # Keep all
                indices_to_keep_local.extend(cluster_indices.tolist())
                
        # 4. Map back to original indices
        # indices_to_keep_local are indices into passed_indices
        indices_to_keep_local = torch.tensor(indices_to_keep_local, dtype=torch.long)
        final_indices = passed_indices[indices_to_keep_local]
        
        # Sort indices
        final_indices_sorted, _ = final_indices.sort()
        indices_to_keep = final_indices_sorted.tolist()
        
        print(f"Balancing Complete. Reduced {N_total} -> {len(indices_to_keep)} frames "
              f"({(1 - len(indices_to_keep)/N_total)*100:.1f}% reduction). "
              f"Pre-filter kept {len(passed_indices)}.")
              
        return indices_to_keep
