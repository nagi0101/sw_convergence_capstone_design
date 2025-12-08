"""
Packet Compressor for SGAPS-MAE Client
Efficient compression for pixel data transmission.
"""

import struct
import zlib
from typing import Tuple, Optional

import numpy as np


class PacketCompressor:
    """
    Compress pixel data for network transmission.
    """
    
    def __init__(
        self,
        compression_level: int = 1,
        quantization_bits: int = 8
    ):
        """
        Args:
            compression_level: zlib compression level (1-9)
            quantization_bits: Bits per color channel
        """
        self.compression_level = compression_level
        self.quantization_bits = quantization_bits
    
    def compress(
        self,
        frame_idx: int,
        coordinates: np.ndarray,
        pixel_values: np.ndarray
    ) -> bytes:
        """
        Compress pixel data into network packet.
        
        Args:
            frame_idx: Frame index
            coordinates: Pixel coordinates [N, 2] (u, v)
            pixel_values: RGB values [N, 3] float32 in [0, 1]
            
        Returns:
            Compressed packet bytes
        """
        num_pixels = len(coordinates)
        
        # Build packet
        # Header: frame_idx (4 bytes) + num_pixels (4 bytes)
        data = struct.pack('<II', frame_idx, num_pixels)
        
        # Pixel data: u (2 bytes) + v (2 bytes) + RGB (3 bytes)
        for i in range(num_pixels):
            u, v = coordinates[i]
            r, g, b = pixel_values[i]
            
            # Quantize to uint8
            r_int = int(np.clip(r * 255, 0, 255))
            g_int = int(np.clip(g * 255, 0, 255))
            b_int = int(np.clip(b * 255, 0, 255))
            
            data += struct.pack('<HH', int(u), int(v))
            data += struct.pack('BBB', r_int, g_int, b_int)
        
        # Compress
        compressed = zlib.compress(data, level=self.compression_level)
        
        return compressed
    
    def compress_fast(
        self,
        frame_idx: int,
        coordinates: np.ndarray,
        pixel_values: np.ndarray
    ) -> bytes:
        """
        Fast compression using numpy operations.
        
        Args:
            frame_idx: Frame index
            coordinates: Pixel coordinates [N, 2]
            pixel_values: RGB values [N, 3] float32
            
        Returns:
            Compressed packet bytes
        """
        num_pixels = len(coordinates)
        
        # Header
        header = struct.pack('<II', frame_idx, num_pixels)
        
        # Pack coordinates as uint16
        coords_packed = coordinates.astype(np.uint16).tobytes()
        
        # Pack RGB as uint8
        rgb_packed = (pixel_values * 255).astype(np.uint8).tobytes()
        
        # Combine
        data = header + coords_packed + rgb_packed
        
        # Compress
        compressed = zlib.compress(data, level=self.compression_level)
        
        return compressed


class PacketDecompressor:
    """
    Decompress coordinate packets from server.
    """
    
    def decompress(self, data: bytes) -> np.ndarray:
        """
        Decompress coordinate packet.
        
        Args:
            data: Compressed packet bytes
            
        Returns:
            Coordinates [N, 2]
        """
        # Decompress
        decompressed = zlib.decompress(data)
        
        # Parse header: num_coords (4 bytes)
        num_coords = struct.unpack('<I', decompressed[:4])[0]
        
        # Parse coordinates
        coordinates = []
        offset = 4
        
        for _ in range(num_coords):
            u = struct.unpack('<H', decompressed[offset:offset+2])[0]
            v = struct.unpack('<H', decompressed[offset+2:offset+4])[0]
            coordinates.append([u, v])
            offset += 4
        
        return np.array(coordinates, dtype=np.float32)
    
    def decompress_fast(self, data: bytes) -> np.ndarray:
        """
        Fast decompression using numpy.
        
        Args:
            data: Compressed packet bytes
            
        Returns:
            Coordinates [N, 2]
        """
        # Decompress
        decompressed = zlib.decompress(data)
        
        # Parse header
        num_coords = struct.unpack('<I', decompressed[:4])[0]
        
        # Parse as numpy array
        coords_bytes = decompressed[4:4 + num_coords * 4]
        coordinates = np.frombuffer(coords_bytes, dtype=np.uint16).reshape(-1, 2)
        
        return coordinates.astype(np.float32)


class DifferentialCompressor(PacketCompressor):
    """
    Differential compression for temporal redundancy.
    """
    
    def __init__(
        self,
        compression_level: int = 1,
        quantization_bits: int = 8
    ):
        super().__init__(compression_level, quantization_bits)
        self.prev_coordinates: Optional[np.ndarray] = None
        self.prev_values: Optional[np.ndarray] = None
    
    def compress_differential(
        self,
        frame_idx: int,
        coordinates: np.ndarray,
        pixel_values: np.ndarray
    ) -> bytes:
        """
        Compress with differential encoding.
        
        Args:
            frame_idx: Frame index
            coordinates: Pixel coordinates [N, 2]
            pixel_values: RGB values [N, 3]
            
        Returns:
            Compressed packet bytes
        """
        if self.prev_coordinates is None:
            # First frame: full encoding
            self.prev_coordinates = coordinates.copy()
            self.prev_values = pixel_values.copy()
            return self.compress(frame_idx, coordinates, pixel_values)
        
        # Compute deltas
        coord_delta = coordinates - self.prev_coordinates
        value_delta = pixel_values - self.prev_values
        
        # Encode deltas (can be negative, use int16)
        num_pixels = len(coordinates)
        header = struct.pack('<II', frame_idx, num_pixels)
        header += b'\x01'  # Flag: differential encoding
        
        # Pack deltas
        coord_packed = coord_delta.astype(np.int16).tobytes()
        
        # Quantize value deltas to int8 (-128 to 127 range for -0.5 to 0.5)
        value_delta_quantized = np.clip(value_delta * 255, -128, 127).astype(np.int8)
        value_packed = value_delta_quantized.tobytes()
        
        data = header + coord_packed + value_packed
        compressed = zlib.compress(data, level=self.compression_level)
        
        # Update previous
        self.prev_coordinates = coordinates.copy()
        self.prev_values = pixel_values.copy()
        
        return compressed
    
    def reset(self) -> None:
        """Reset differential state."""
        self.prev_coordinates = None
        self.prev_values = None
