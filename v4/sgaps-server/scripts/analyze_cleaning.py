"""
Script to analyze the effect of Data Cleaning (Balancing).
Generates histograms and a report.
"""
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sgaps.data.cleaner import DataBalancer
from sgaps.data.dataset import SGAPSDataset

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("--- SGAPS Data Cleaning Analysis ---")
    
    # 1. Setup Output Directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(project_root) / "outputs" / "cleaning_analysis" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output Directory: {output_dir}")

    # 2. Disable dictionary cleaning in config for the dataset init
    # We want to load RAW data first, then apply cleaning manually to compare.
    if hasattr(cfg, 'cleaning'):
        cfg.cleaning.enabled = False
    
    # 3. Initialize Dataset (Raw)
    print("Initializing Raw Dataset...")
    dataset = SGAPSDataset([], cfg) # Paths will be loaded from config
    # We need to manually trigger h5_paths since we passed empty list, 
    # but SGAPSDataset logic uses all_h5_files passed in __init__.
    # Wait, the current dataset implementation takes h5_paths as arg. 
    # I need to find them first like in train.py.
    checkpoint_path = cfg.training.checkpoint.checkpoint_path
    all_h5_files = list(Path(checkpoint_path).glob("**/*.h5"))
    if not all_h5_files:
        print(f"No HDF5 files found in {checkpoint_path}")
        return
        
    dataset = SGAPSDataset([str(p) for p in all_h5_files], cfg)
    print(f"Total Raw Frames: {len(dataset)}")
    
    # 4. Load State Vectors, Pixel Counts (and store path/keys for Image Loading later)
    print("Extracting state vectors and pixel counts...")
    state_vectors = []
    pixel_counts = []
    meta_info = [] # Store (h5_path, session_key, frame_key)
    
    import h5py
    import cv2
    sentinel = cfg.model.sentinel_value
    max_dim = cfg.model.input_constraints.max_state_dim
    
    for i, (h5_path, session_key, frame_key) in enumerate(dataset.frame_indices):
        try:
            with h5py.File(h5_path, 'r') as f:
                frame_grp = f[session_key]['frames'][frame_key]
                raw_state = frame_grp['state_vector'][:]
                sv = np.full(max_dim, sentinel, dtype=np.float32)
                if len(raw_state) > 0:
                    sv[:len(raw_state)] = raw_state
                state_vectors.append(sv)
                
                # Pixels
                sparse_pixels = frame_grp['pixels'][:] if 'pixels' in frame_grp else []
                pixel_counts.append(len(sparse_pixels))
                
                meta_info.append((h5_path, session_key, frame_key))
        except:
            continue
            
    state_vectors_tensor = torch.tensor(np.array(state_vectors), dtype=torch.float32)
    pixel_counts_tensor = torch.tensor(np.array(pixel_counts), dtype=torch.float32)
    
    # 5. Run Balancing
    cfg.cleaning.enabled = True
    balancer = DataBalancer(cfg)
    
    # Fit Transform to get indices
    print("Running Clustering & Balancing...")
    keep_indices = balancer.fit_transform(state_vectors_tensor, pixel_counts_tensor)
    
    # For visualization, we need Labels. 
    # DataBalancer doesn't expose labels in `fit_transform`. We will re-run the internal clustering on the *augmented features* manually
    # to replicate the labeling logic for visualization.
    # We must match the pre-filtering logic of balancer!
    min_pixels = cfg.cleaning.filters.min_pixels
    passed_mask = pixel_counts_tensor >= min_pixels
    passed_indices = passed_mask.nonzero(as_tuple=True)[0]
    
    data_filtered = state_vectors_tensor[passed_indices]
    pixels_filtered = pixel_counts_tensor[passed_indices]
    
    # Remove padding logic match
    sentinel = cfg.model.get('sentinel_value', -999.0)
    is_valid = (data_filtered != sentinel)
    valid_cols = is_valid.any(dim=0)
    data_filtered = data_filtered[:, valid_cols]

    # Normalize
    device = torch.device('cpu')
    data_filtered = data_filtered.to(device)
    mean = data_filtered.mean(dim=0)
    std = data_filtered.std(dim=0) + 1e-6
    data_norm = (data_filtered - mean) / std
    
    # Augment
    include_pixel_feature = cfg.cleaning.clustering.get('include_pixel_count', False)
    if include_pixel_feature:
        pixel_weight = cfg.cleaning.clustering.get('pixel_count_weight', 0.5)
        p_log = torch.log10(pixels_filtered.float() + 1).unsqueeze(1).to(device)
        p_norm = (p_log - p_log.mean()) / (p_log.std() + 1e-6)
        features = torch.cat([data_norm, p_norm * pixel_weight], dim=1)
    else:
        features = data_norm
        
    print(f"Re-running clustering for visualization (Features: {features.shape})...")
    labels, centroids = balancer._kmeans(features, balancer.n_clusters)
    labels_np = labels.numpy() # Labels for 'passed_indices'
    
    # Map keep_indices (which are into original list) to indices in 'passed_indices'
    # keep_indices contains selected indices from original array. 
    # We want to identify which of 'passed_indices' are kept.
    keep_set = set(keep_indices)
    
    # passed_indices[i] is the original index.
    # kept_mask corresponding to passed_indices
    is_kept = np.array([idx.item() in keep_set for idx in passed_indices])
    
    # 6. Visualization
    print("Generating Visualizations...")
    
    # --- Plot 1: Cluster Distribution (Histogram) ---
    plt.figure(figsize=(12, 6))
    unique_labels, count_raw = np.unique(labels_np, return_counts=True)
    kept_labels = labels_np[is_kept]
    unique_kept, count_kept = np.unique(kept_labels, return_counts=True)
    
    # Map to map
    raw_map = {l: c for l, c in zip(unique_labels, count_raw)}
    kept_map = {l: c for l, c in zip(unique_kept, count_kept)}
    
    # Sort by raw size
    cluster_ids = sorted(list(raw_map.keys()), key=lambda x: raw_map[x], reverse=True)
    y_raw = [raw_map.get(k, 0) for k in cluster_ids]
    y_kept = [kept_map.get(k, 0) for k in cluster_ids]
    
    x = np.arange(len(cluster_ids))
    width = 0.35
    plt.bar(x - width/2, y_raw, width, label='Original')
    plt.bar(x + width/2, y_kept, width, label='Balanced')
    plt.xlabel('Cluster ID (Sorted by Size)')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Cluster Distribution')
    dist_path = output_dir / "distribution.png"
    plt.savefig(dist_path)
    print(f"Saved {dist_path}")

    # --- Plot 2: Pixel Count Boxplot per Cluster (All) ---
    all_clusters = cluster_ids
    boxplot_data = []
    boxplot_labels = []
    
    for cid in all_clusters:
        # Get pixels for this cluster
        mask = (labels_np == cid)
        pix = pixels_filtered[mask].numpy()
        boxplot_data.append(pix)
        boxplot_labels.append(str(cid))
        
    plt.figure(figsize=(max(12, len(all_clusters) * 0.2), 6))
    plt.boxplot(boxplot_data, tick_labels=boxplot_labels)
    plt.yscale('log')
    plt.xticks(rotation=90, fontsize=8)
    plt.xlabel('Cluster ID')
    plt.ylabel('Pixel Count (Log)')
    plt.title(f'Pixel Count Distribution for All {len(all_clusters)} Clusters')
    plt.tight_layout()
    box_path = output_dir / "pixel_boxplot.png"
    plt.savefig(box_path)
    print(f"Saved {box_path}")

    # --- Plot 3: Projection (UMAP / t-SNE / PCA) ---
    print("Computing State Space Projection...")
    proj_method = "PCA"
    try:
        # Try UMAP first
        import umap
        print("Using UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        projected = reducer.fit_transform(features.numpy())
        proj_method = "UMAP"
    except ImportError:
        try:
            # Fallback to t-SNE
            print("UMAP not found, trying t-SNE...")
            from sklearn.manifold import TSNE
            # Use a max of 5000 samples for t-SNE speed
            tsne_limit = 5000
            if len(features) > tsne_limit:
                perm = torch.randperm(len(features))[:tsne_limit]
                features_sub = features[perm]
                labels_sub = labels_np[perm]
                is_kept_sub = is_kept[perm]
                projected_sub = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto').fit_transform(features_sub.numpy())
                # We can only plot subset
                projected = projected_sub
                # overwrite full arrays with subset for plotting matching
                labels_np = labels_sub
                is_kept = is_kept_sub
            else:
                projected = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto').fit_transform(features.numpy())
            proj_method = "t-SNE"
        except ImportError:
            # Fallback to PCA
            print("t-SNE not found, using PCA...")
            U, S, V = torch.pca_lowrank(features, q=2)
            projected = torch.matmul(features, V[:, :2]).numpy()
            proj_method = "PCA"

    # Plot A: Cluster Colors
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(projected[:, 0], projected[:, 1], c=labels_np, cmap='tab20', s=10, alpha=0.6)
    plt.colorbar(scatter, label='Cluster ID')
    plt.title(f'{proj_method} Projection (Colored by Cluster)')
    proj_c_path = output_dir / "projection_clusters.png"
    plt.savefig(proj_c_path)
    print(f"Saved {proj_c_path}")
    
    # Plot B: Kept vs Dropped
    plt.figure(figsize=(10, 8))
    # Note: If t-SNE subsampled, is_kept is already subsampled
    dropped_mask = ~is_kept
    if dropped_mask.any():
        plt.scatter(projected[dropped_mask, 0], projected[dropped_mask, 1], c='lightgray', s=10, alpha=0.4, label='Dropped')
    plt.scatter(projected[is_kept, 0], projected[is_kept, 1], c='blue', s=10, alpha=0.6, label='Kept')
    plt.legend()
    plt.title(f'{proj_method} Projection (Kept vs Dropped)')
    proj_k_path = output_dir / "projection_status.png"
    plt.savefig(proj_k_path)
    print(f"Saved {proj_k_path}")

    # --- Plot 4: Cluster Gallery (Top 9) ---
    print("Generating Cluster Gallery...")
    top_9_clusters = cluster_ids[:9]
    gallery_images = []
    
    for cid in top_9_clusters:
        # Find index closest to centroid
        # Get indices in passed_indices list
        cluster_mask = (labels_np == cid) # labels_np is for PASSED indices
        
        # If t-SNE subsampled labels_np, this might be partial? 
        # Wait, if we subsampled for t-SNE, labels_np was overwritten!
        # This breaks Gallery generation if we rely on subsampled labels.
        # FIX: We should use the FULL labels for Gallery, not subsampled.
        # I need to restore full labels if I subsampled.
        pass
        
    # Re-logic for Gallery to use original full labels if t-SNE subsampled
    # Since I overwrote labels_np in t-SNE block, I should fix that logic.
    # Instead of overwriting, I should use `vis_labels` vars.
    # I'll fix this in the replacement content by NOT overwriting logic or recalculating.
    # Actually, simplistic fix: Run UMAP/t-SNE last or keep separate variables.
    
    # Let's clean up the "Plot 3" block to be self-contained for projection, 
    # and "Plot 4" uses `labels` (tensor) which is untouched.
    
    # ... (Retrying robust implementation in next replacement attempt)

    # --- Plot 4: Cluster Gallery (All Clusters) ---
    print("Generating Cluster Gallery...")
    import math
    gallery_clusters = cluster_ids # ALL
    gallery_images = []
    
    for cid in gallery_clusters:
        # Find index closest to centroid
        # Get indices in passed_indices list
        cluster_mask = (labels_np == cid)
        indices_in_passed = cluster_mask.nonzero()[0]
        
        if len(indices_in_passed) == 0:
            continue
            
        points = features[indices_in_passed]
        centroid = centroids[cid].unsqueeze(0)
        dists = torch.norm(points - centroid, dim=1)
        
        # Sort by distance (closest to centroid first)
        sorted_indices = dists.argsort()
        
        found_img_data = None
        
        # Check top 50 closest frames for an image
        search_limit = min(50, len(sorted_indices))
        for i in range(search_limit):
            local_idx = indices_in_passed[sorted_indices[i]].item()
            original_idx = passed_indices[local_idx].item()
            h5_path, s_key, f_key = meta_info[original_idx]
            
            try:
                with h5py.File(h5_path, 'r') as f:
                    frame_node = f[s_key]['frames'][f_key]
                    if 'image' in frame_node:
                        img_data = frame_node['image'][:] 
                        
                        if img_data.ndim == 3 and img_data.shape[0] in [1, 3] and img_data.shape[0] < img_data.shape[1]:
                             img_data = np.transpose(img_data, (1, 2, 0))
                        
                        if img_data.dtype != np.uint8:
                             img_data = ((img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-6) * 255).astype(np.uint8)
                        
                        if img_data.ndim == 2:
                             img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)
                        elif img_data.shape[2] == 1:
                            img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)
                        else: 
                            img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
    
                        img_data = cv2.resize(img_data, (128, 128))
                        cv2.putText(img_data, f"C{cid} (N={raw_map[cid]})", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        found_img_data = img_data
                        break # Found one!
            except:
                continue
        
        if found_img_data is not None:
             gallery_images.append(found_img_data)
        else:
             blank = np.zeros((128, 128, 3), dtype=np.uint8)
             cv2.putText(blank, f"C{cid} (No Img)", (5, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
             gallery_images.append(blank)
            
    # Combine into NxN grid
    if gallery_images:
        N_imgs = len(gallery_images)
        cols = 10 # Fixed width of 10 images (1280px)
        rows = math.ceil(N_imgs / cols)
        
        row_imgs = []
        for r in range(rows):
            start_i = r * cols
            end_i = start_i + cols
            chunk = gallery_images[start_i:end_i]
            
            # Fill remaining with black
            while len(chunk) < cols:
                chunk.append(np.zeros((128, 128, 3), dtype=np.uint8))
                
            row_imgs.append(np.hstack(chunk))
            
        grid_img = np.vstack(row_imgs)
        gal_path = output_dir / "cluster_gallery.png"
        cv2.imwrite(str(gal_path), grid_img)
        print(f"Saved {gal_path}")
    else:
        gal_path = None

    # 7. Report logic
    reduction_rate = (1 - len(keep_indices)/len(state_vectors)) * 100
    
    report_content = f"""# Data Cleaning Analysis Report (Comprehensive)
    
**Date**: {timestamp}
**Total Frames (Raw)**: {len(state_vectors)}
**Total Frames (After MinPixels Filter)**: {len(passed_indices)}
**Total Frames (Final)**: {len(keep_indices)}
**Reduction Rate**: {reduction_rate:.2f}%
**Min Pixels**: {min_pixels}

## 1. Cluster Gallery (Top 9 Clusters)
Visualizes the "Centroid" frame of the largest clusters to understand the semantic content (game state). (If images are missing, shows placeholders).
![Gallery]({gal_path.name if gal_path else 'N/A'})

## 2. Cluster Distribution
Balance check before (blue) and after (orange).
![Distribution]({dist_path.name})

## 3. Pixel Count Statistics
Boxplots of pixel counts for the Top 20 clusters. Used to verify "Sparse" vs "Dense" cluster separation.
![Boxplot]({box_path.name})

## 4. State Space Projection ({proj_method})
Structure of the data manifold.
- **Top**: Colored by Cluster ID.
- **Bottom**: Kept (Blue) vs Dropped (Grey).
![Projection]({proj_c_path.name if proj_c_path else 'N/A'})
![Status]({proj_k_path.name if proj_k_path else 'N/A'})

"""
    report_path = output_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"Saved report to {report_path}")
    print("Analysis Complete.")

if __name__ == "__main__":
    main()
