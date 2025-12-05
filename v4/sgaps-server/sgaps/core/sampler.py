"""
UV Coordinate Sampler for SGAPS-MAE Server.

Generates UV coordinates for pixel sampling.
Phase 1: Fixed uniform grid sampling.
Future phases will include adaptive/gradient-based sampling.
"""

import logging
from typing import List, Tuple
import numpy as np


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
