"""
UV Coordinate Sampler for SGAPS-MAE Server.

Generates UV coordinates for pixel sampling.
Phase 1: Fixed uniform grid sampling.
Phase 3: Adaptive importance-based sampling with attention entropy.
"""

import logging
from typing import List, Tuple, Optional
import numpy as np
from omegaconf import DictConfig


logger = logging.getLogger(__name__)


class FixedUVSampler:
    """
    Fixed UV coordinate sampler using uniform grid pattern.
    
    Phase 1 implementation that generates a fixed set of UV coordinates
    distributed uniformly across the image.
    
    Attributes:
        sample_count: Number of samples to generate
        resolution: Image resolution (width, height)
        current_coords: Current set of UV coordinates
    """
    
    def __init__(
        self,
        sample_count: int = 500,
        resolution: Tuple[int, int] = (640, 480)
    ):
        """
        Initialize the sampler.
        
        Args:
            sample_count: Number of pixels to sample
            resolution: Image resolution (width, height)
        """
        self.sample_count = sample_count
        self.resolution = resolution
        self.current_coords: List[Tuple[float, float]] = []
        
        logger.info(f"FixedUVSampler initialized: {sample_count} samples at {resolution}")
    
    def generate_uniform_grid(self) -> List[Tuple[float, float]]:
        """
        Generate uniform grid UV coordinates.
        
        Creates a grid pattern that covers the image evenly,
        with coordinates normalized to [0, 1] range.
        
        Returns:
            List of (u, v) coordinate tuples
        """
        # Calculate grid dimensions
        aspect_ratio = self.resolution[0] / self.resolution[1]
        
        # Approximate grid size
        rows = int(np.sqrt(self.sample_count / aspect_ratio))
        cols = int(rows * aspect_ratio)
        
        # Adjust to match sample count as closely as possible
        while rows * cols < self.sample_count:
            if cols / rows < aspect_ratio:
                cols += 1
            else:
                rows += 1
        
        # Generate grid coordinates
        coords = []
        for i in range(rows):
            for j in range(cols):
                if len(coords) >= self.sample_count:
                    break
                
                # Normalize to [0, 1] with offset for centering
                u = (j + 0.5) / cols
                v = (i + 0.5) / rows
                coords.append((u, v))
        
        self.current_coords = coords[:self.sample_count]
        logger.debug(f"Generated {len(self.current_coords)} uniform grid coordinates")
        
        return self.current_coords
    
    def generate_random(self, seed: int = None) -> List[Tuple[float, float]]:
        """
        Generate random UV coordinates.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            List of (u, v) coordinate tuples
        """
        if seed is not None:
            np.random.seed(seed)
        
        coords = [
            (float(np.random.random()), float(np.random.random()))
            for _ in range(self.sample_count)
        ]
        
        self.current_coords = coords
        logger.debug(f"Generated {len(coords)} random coordinates")
        
        return coords
    
    def generate_stratified(self) -> List[Tuple[float, float]]:
        """
        Generate stratified random UV coordinates.
        
        Divides the image into cells and places one random sample
        in each cell for more even coverage than pure random.
        
        Returns:
            List of (u, v) coordinate tuples
        """
        # Calculate grid for stratification
        aspect_ratio = self.resolution[0] / self.resolution[1]
        rows = int(np.sqrt(self.sample_count / aspect_ratio))
        cols = int(rows * aspect_ratio)
        
        # Adjust to match sample count
        while rows * cols < self.sample_count:
            if cols / rows < aspect_ratio:
                cols += 1
            else:
                rows += 1
        
        # Generate stratified coordinates
        coords = []
        for i in range(rows):
            for j in range(cols):
                if len(coords) >= self.sample_count:
                    break
                
                # Random position within cell
                u = (j + np.random.random()) / cols
                v = (i + np.random.random()) / rows
                coords.append((float(u), float(v)))
        
        self.current_coords = coords[:self.sample_count]
        logger.debug(f"Generated {len(self.current_coords)} stratified coordinates")
        
        return self.current_coords
    
    def get_current_coordinates(self) -> List[Tuple[float, float]]:
        """
        Get the current set of UV coordinates.
        
        If no coordinates have been generated yet, generates uniform grid.
        
        Returns:
            List of (u, v) coordinate tuples
        """
        if not self.current_coords:
            return self.generate_uniform_grid()
        return self.current_coords
    
    def update_sample_count(self, new_count: int):
        """
        Update the sample count and regenerate coordinates.

        Args:
            new_count: New number of samples
        """
        self.sample_count = new_count
        self.generate_uniform_grid()
        logger.info(f"Updated sample count to {new_count}")


class AdaptiveUVSampler:
    """
    Adaptive UV sampler using importance-weighted + uniform sampling.

    Strategy:
    - importance_ratio% from high-importance regions (attention entropy)
    - uniform_ratio% uniform coverage (prevent overfitting to current model state)

    This sampler implements the closed-loop adaptive sampling for Phase 3,
    dynamically adjusting sampling locations based on reconstruction quality.
    """

    def __init__(
        self,
        config: DictConfig,
        resolution: Tuple[int, int]
    ):
        """
        Initialize the adaptive sampler.

        Args:
            config: Hydra config with sampling settings
            resolution: Image resolution (width, height)
        """
        self.sample_count = config.default_sample_count
        self.resolution = resolution
        self.importance_ratio = config.distribution.importance_ratio  # 0.6
        self.uniform_ratio = config.distribution.uniform_ratio  # 0.4

        # Validate config
        ratio_sum = self.importance_ratio + self.uniform_ratio
        if abs(ratio_sum - 1.0) > 1e-6:
            raise ValueError(
                f"importance_ratio ({self.importance_ratio}) + "
                f"uniform_ratio ({self.uniform_ratio}) must equal 1.0, got {ratio_sum}"
            )

        # Initialize with uniform grid for cold start
        self.current_coords = self._generate_uniform_fallback()
        self.importance_map: Optional[np.ndarray] = None
        self.frame_count = 0
        self.warmup_frames = config["update"]["warmup_frames"]  # 10 - Use dict access to avoid method conflict

        logger.info(
            f"AdaptiveUVSampler initialized: {self.sample_count} samples, "
            f"ratio {self.importance_ratio:.1%}/{self.uniform_ratio:.1%}, "
            f"warmup {self.warmup_frames} frames"
        )

    def _generate_uniform_fallback(self) -> List[Tuple[float, float]]:
        """
        Generate uniform grid for cold start fallback.

        Returns:
            List of (u, v) coordinate tuples
        """
        W, H = self.resolution
        grid_size = int(np.sqrt(self.sample_count))

        coords = []
        for i in range(grid_size):
            for j in range(grid_size):
                u = (j + 0.5) / grid_size
                v = (i + 0.5) / grid_size
                coords.append((float(u), float(v)))

        # Fill remaining with random
        while len(coords) < self.sample_count:
            coords.append((float(np.random.rand()), float(np.random.rand())))

        return coords[:self.sample_count]

    def update_from_importance(
        self,
        importance_map: np.ndarray  # [H, W]
    ):
        """
        Update sampling coordinates based on new importance map.

        Args:
            importance_map: Normalized importance scores [0, 1]
                Shape: [H, W] where higher values indicate regions
                needing more sampling
        """
        self.importance_map = importance_map
        self.frame_count += 1

        # Warmup period: use uniform sampling
        if self.frame_count < self.warmup_frames:
            logger.debug(
                f"Frame {self.frame_count}/{self.warmup_frames}: "
                "Using uniform sampling (warmup)"
            )
            return  # Keep uniform coords

        # Calculate sample counts
        importance_count = int(self.sample_count * self.importance_ratio)
        uniform_count = self.sample_count - importance_count

        # 1. Importance-weighted sampling (60%)
        importance_coords = self._sample_importance_based(
            importance_map,
            importance_count
        )

        # 2. Uniform sampling (40%) - avoid duplicates
        uniform_coords = self._sample_uniform_avoiding(
            importance_coords,
            uniform_count
        )

        # Combine and shuffle
        all_coords = importance_coords + uniform_coords
        np.random.shuffle(all_coords)

        self.current_coords = all_coords

        logger.debug(
            f"Frame {self.frame_count}: Updated sampling "
            f"({importance_count} importance + {uniform_count} uniform)"
        )

    def _sample_importance_based(
        self,
        importance_map: np.ndarray,
        count: int
    ) -> List[Tuple[float, float]]:
        """
        Sample pixels proportional to importance scores.

        Args:
            importance_map: 2D importance map [H, W]
            count: Number of samples to generate

        Returns:
            List of (u, v) coordinate tuples
        """
        H, W = importance_map.shape

        # Flatten importance map to probability distribution
        flat_importance = importance_map.flatten()

        # Add small epsilon to avoid zero probabilities
        flat_importance = flat_importance + 1e-9
        prob = flat_importance / flat_importance.sum()

        # Weighted sampling without replacement
        try:
            indices = np.random.choice(
                H * W,
                size=count,
                replace=False,
                p=prob
            )
        except ValueError as e:
            # If probabilities don't sum to 1 exactly, renormalize
            logger.warning(f"Probability normalization issue: {e}. Re-normalizing.")
            prob = prob / prob.sum()
            indices = np.random.choice(H * W, size=count, replace=False, p=prob)

        # Convert to UV coordinates
        coords = []
        for idx in indices:
            v_idx = idx // W
            u_idx = idx % W
            # Center of pixel
            u = (u_idx + 0.5) / W
            v = (v_idx + 0.5) / H
            coords.append((float(u), float(v)))

        return coords

    def _sample_uniform_avoiding(
        self,
        existing_coords: List[Tuple[float, float]],
        count: int
    ) -> List[Tuple[float, float]]:
        """
        Sample uniformly while avoiding existing coordinates.

        Args:
            existing_coords: List of already sampled (u, v) coordinates
            count: Number of additional samples to generate

        Returns:
            List of (u, v) coordinate tuples
        """
        W, H = self.resolution

        # Create occupancy grid (quantized to pixel indices)
        occupied = set()
        for u, v in existing_coords:
            u_idx = int(u * W)
            v_idx = int(v * H)
            # Clamp to valid range
            u_idx = min(max(u_idx, 0), W - 1)
            v_idx = min(max(v_idx, 0), H - 1)
            occupied.add((v_idx, u_idx))

        # Generate all possible indices
        all_indices = [(v, u) for v in range(H) for u in range(W)]
        available = [idx for idx in all_indices if idx not in occupied]

        # Sample
        if len(available) < count:
            # Fallback: allow duplicates (sample from all indices)
            logger.warning(
                f"Not enough unoccupied pixels ({len(available)} < {count}). "
                "Allowing some duplicates."
            )
            sampled_idx = np.random.choice(len(all_indices), size=count, replace=False)
            sampled_indices = [all_indices[i] for i in sampled_idx]
        else:
            sampled_idx = np.random.choice(len(available), size=count, replace=False)
            sampled_indices = [available[i] for i in sampled_idx]

        # Convert to UV
        coords = []
        for v_idx, u_idx in sampled_indices:
            u = (u_idx + 0.5) / W
            v = (v_idx + 0.5) / H
            coords.append((float(u), float(v)))

        return coords

    def get_current_coordinates(self) -> List[Tuple[float, float]]:
        """
        Get current sampling coordinates.

        Returns:
            List of (u, v) coordinate tuples
        """
        return self.current_coords

    def update_sample_count(self, new_count: int):
        """
        Dynamically adjust sample count (for quality-based adaptation).

        Args:
            new_count: New number of samples
        """
        old_count = self.sample_count
        self.sample_count = new_count

        # Force regeneration on next update
        if self.importance_map is not None:
            self.update_from_importance(self.importance_map)
        else:
            # If no importance map yet, regenerate uniform
            self.current_coords = self._generate_uniform_fallback()

        logger.info(f"Updated sample count: {old_count} → {new_count}")
