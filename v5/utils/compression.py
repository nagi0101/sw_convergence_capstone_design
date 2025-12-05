"""
Compression Utilities for SGAPS-MAE
"""

import struct
import zlib
from typing import Tuple

import numpy as np


def compress_packet(
    frame_idx: int,
    coordinates: np.ndarray,
    pixel_values: np.ndarray,
    compression_level: int = 1
) -> bytes:
    """
    Compress pixel data for network transmission.
    
    Args:
        frame_idx: Frame index (non-negative integer)
        coordinates: Pixel coordinates [N, 2] with uint16-compatible values (0-65535).
            Shape must be (N, 2) where N is the number of pixels.
        pixel_values: RGB values [N, 3] as float32 in range [0.0, 1.0].
            Shape must be (N, 3) where N matches coordinates.
        compression_level: zlib compression level (1-9, default 1 for speed)
        
    Returns:
        Compressed bytes ready for network transmission
        
    Note:
        Coordinates are packed as uint16 (2 bytes each).
        RGB values are quantized to uint8 (0-255).
    """
    num_pixels = len(coordinates)
    
    # Header
    data = struct.pack('<II', frame_idx, num_pixels)
    
    # Pack coordinates as uint16
    coords_packed = coordinates.astype(np.uint16).tobytes()
    
    # Pack RGB as uint8
    rgb_packed = (pixel_values * 255).astype(np.uint8).tobytes()
    
    # Combine and compress
    data = data + coords_packed + rgb_packed
    compressed = zlib.compress(data, level=compression_level)
    
    return compressed


def decompress_packet(data: bytes) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Decompress pixel data packet.
    
    Args:
        data: Compressed packet bytes
        
    Returns:
        frame_idx, coordinates, pixel_values
    """
    # Decompress
    decompressed = zlib.decompress(data)
    
    # Parse header
    frame_idx = struct.unpack('<I', decompressed[:4])[0]
    num_pixels = struct.unpack('<I', decompressed[4:8])[0]
    
    # Parse coordinates
    coord_size = num_pixels * 4  # 2 uint16 per coordinate
    coords_bytes = decompressed[8:8 + coord_size]
    coordinates = np.frombuffer(coords_bytes, dtype=np.uint16).reshape(-1, 2)
    
    # Parse RGB values
    rgb_bytes = decompressed[8 + coord_size:]
    pixel_values = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(-1, 3) / 255.0
    
    return frame_idx, coordinates.astype(np.float32), pixel_values.astype(np.float32)


def compress_coordinates(coordinates: np.ndarray, compression_level: int = 1) -> bytes:
    """
    Compress coordinate list.
    
    Args:
        coordinates: Coordinates [N, 2]
        compression_level: Compression level
        
    Returns:
        Compressed bytes
    """
    num_coords = len(coordinates)
    
    data = struct.pack('<I', num_coords)
    data += coordinates.astype(np.uint16).tobytes()
    
    return zlib.compress(data, level=compression_level)


def decompress_coordinates(data: bytes) -> np.ndarray:
    """
    Decompress coordinate list.
    
    Args:
        data: Compressed bytes
        
    Returns:
        Coordinates [N, 2]
    """
    decompressed = zlib.decompress(data)
    
    num_coords = struct.unpack('<I', decompressed[:4])[0]
    coords_bytes = decompressed[4:]
    
    coordinates = np.frombuffer(coords_bytes, dtype=np.uint16).reshape(-1, 2)
    
    return coordinates.astype(np.float32)
