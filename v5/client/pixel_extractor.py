"""
Pixel Extractor for SGAPS-MAE Client
Simple pixel sampling from game frames.
"""

from typing import List, Tuple, Optional

import numpy as np


class PixelExtractor:
    """
    Lightweight pixel extractor for game frames.
    Simply reads pixel values at specified coordinates.
    
    Designed to be minimal for low client CPU usage.
    """
    
    def __init__(
        self,
        frame_width: int = 256,
        frame_height: int = 240
    ):
        """
        Args:
            frame_width: Game frame width
            frame_height: Game frame height
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
    
    def extract(
        self,
        frame: np.ndarray,
        coordinates: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract pixel values at specified coordinates.
        
        Args:
            frame: Game frame [H, W, 3] uint8
            coordinates: Sampling coordinates [N, 2] (u, v)
            
        Returns:
            valid_coords: Valid coordinates [M, 2]
            pixel_values: RGB values [M, 3] float32
        """
        valid_coords = []
        pixel_values = []
        
        for u, v in coordinates:
            u_int, v_int = int(u), int(v)
            
            # Bounds check
            if 0 <= u_int < frame.shape[0] and 0 <= v_int < frame.shape[1]:
                valid_coords.append([u_int, v_int])
                pixel_values.append(frame[u_int, v_int] / 255.0)
        
        return np.array(valid_coords), np.array(pixel_values, dtype=np.float32)
    
    def extract_fast(
        self,
        frame: np.ndarray,
        coordinates: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast vectorized pixel extraction.
        
        Args:
            frame: Game frame [H, W, 3] uint8
            coordinates: Sampling coordinates [N, 2] (u, v)
            
        Returns:
            valid_coords: Valid coordinates [M, 2]
            pixel_values: RGB values [M, 3] float32
        """
        coords_int = coordinates.astype(np.int32)
        
        # Bounds mask
        valid_mask = (
            (coords_int[:, 0] >= 0) & 
            (coords_int[:, 0] < frame.shape[0]) &
            (coords_int[:, 1] >= 0) & 
            (coords_int[:, 1] < frame.shape[1])
        )
        
        valid_coords = coords_int[valid_mask]
        
        # Extract pixels
        pixel_values = frame[valid_coords[:, 0], valid_coords[:, 1]] / 255.0
        
        return valid_coords, pixel_values.astype(np.float32)
    
    def scale_coordinates(
        self,
        coordinates: np.ndarray,
        target_height: int,
        target_width: int
    ) -> np.ndarray:
        """
        Scale coordinates from target resolution to frame resolution.
        
        Args:
            coordinates: Coordinates in target resolution [N, 2]
            target_height: Target height (e.g., 224)
            target_width: Target width (e.g., 224)
            
        Returns:
            Scaled coordinates [N, 2]
        """
        scale_u = self.frame_height / target_height
        scale_v = self.frame_width / target_width
        
        scaled = coordinates.copy().astype(np.float32)
        scaled[:, 0] *= scale_u
        scaled[:, 1] *= scale_v
        
        return scaled


class AdaptivePixelExtractor(PixelExtractor):
    """
    Adaptive pixel extractor with priority-based extraction.
    """
    
    def __init__(
        self,
        frame_width: int = 256,
        frame_height: int = 240,
        priority_threshold: float = 0.5
    ):
        """
        Args:
            frame_width: Game frame width
            frame_height: Game frame height
            priority_threshold: Threshold for priority extraction
        """
        super().__init__(frame_width, frame_height)
        self.priority_threshold = priority_threshold
    
    def extract_with_priority(
        self,
        frame: np.ndarray,
        coordinates: np.ndarray,
        priorities: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract pixels with priority information.
        
        Args:
            frame: Game frame [H, W, 3] uint8
            coordinates: Sampling coordinates [N, 2]
            priorities: Priority values [N] float
            
        Returns:
            valid_coords: Valid coordinates [M, 2]
            pixel_values: RGB values [M, 3]
            valid_priorities: Priorities [M]
        """
        valid_coords, pixel_values = self.extract_fast(frame, coordinates)
        
        # Get priorities for valid coordinates
        coords_int = coordinates.astype(np.int32)
        valid_mask = (
            (coords_int[:, 0] >= 0) & 
            (coords_int[:, 0] < frame.shape[0]) &
            (coords_int[:, 1] >= 0) & 
            (coords_int[:, 1] < frame.shape[1])
        )
        valid_priorities = priorities[valid_mask]
        
        return valid_coords, pixel_values, valid_priorities
