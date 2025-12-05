"""
Minimal Game Client for SGAPS-MAE
Lightweight client for game session recording.
"""

import time
from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np


@dataclass
class ClientConfig:
    """Configuration for minimal game client."""
    
    server_host: str = "localhost"
    server_port: int = 8888
    frame_width: int = 256
    frame_height: int = 240
    target_width: int = 224
    target_height: int = 224
    compression_level: int = 1
    buffer_size: int = 3


class MinimalGameClient:
    """
    Minimal game client for SGAPS-MAE.
    
    Designed for extremely low CPU overhead:
    - Simple pixel extraction
    - Basic compression
    - Coordinate buffering
    
    Target: <0.1% CPU usage
    """
    
    def __init__(
        self,
        config: Optional[ClientConfig] = None,
        on_coordinates_received: Optional[Callable] = None
    ):
        """
        Args:
            config: Client configuration
            on_coordinates_received: Callback for coordinate updates
        """
        from .pixel_extractor import PixelExtractor
        from .compressor import PacketCompressor, PacketDecompressor
        
        self.config = config or ClientConfig()
        self.on_coordinates_received = on_coordinates_received
        
        # Components
        self.pixel_extractor = PixelExtractor(
            frame_width=self.config.frame_width,
            frame_height=self.config.frame_height
        )
        self.compressor = PacketCompressor(
            compression_level=self.config.compression_level
        )
        self.decompressor = PacketDecompressor()
        
        # State
        self.frame_idx = 0
        self.current_coordinates: Optional[np.ndarray] = None
        self.coordinate_buffer = []
        
        # Statistics
        self.total_bytes_sent = 0
        self.total_bytes_received = 0
        self.total_frames = 0
        self.total_time = 0.0
        
        # Connection (placeholder for actual network)
        self.connected = False
    
    def connect(self) -> bool:
        """
        Connect to replay server.
        
        Returns:
            True if connected successfully
        """
        # In production, this would establish network connection
        # For now, just mark as connected
        self.connected = True
        return True
    
    def disconnect(self) -> None:
        """Disconnect from server."""
        self.connected = False
    
    def set_initial_coordinates(self, coordinates: np.ndarray) -> None:
        """
        Set initial sampling coordinates.
        
        Args:
            coordinates: Initial coordinates [N, 2]
        """
        # Scale coordinates to frame resolution
        self.current_coordinates = self.pixel_extractor.scale_coordinates(
            coordinates,
            self.config.target_height,
            self.config.target_width
        )
    
    def process_frame(
        self,
        frame: np.ndarray
    ) -> Optional[bytes]:
        """
        Process a game frame.
        
        Args:
            frame: Game frame [H, W, 3] uint8
            
        Returns:
            Compressed packet or None if no coordinates
        """
        if self.current_coordinates is None:
            return None
        
        start_time = time.time()
        
        # Extract pixels at current coordinates
        valid_coords, pixel_values = self.pixel_extractor.extract_fast(
            frame,
            self.current_coordinates
        )
        
        # Scale coordinates back to target resolution for server
        scale_u = self.config.target_height / self.config.frame_height
        scale_v = self.config.target_width / self.config.frame_width
        
        coords_for_server = valid_coords.astype(np.float32)
        coords_for_server[:, 0] *= scale_u
        coords_for_server[:, 1] *= scale_v
        
        # Compress packet
        packet = self.compressor.compress(
            self.frame_idx,
            coords_for_server,
            pixel_values
        )
        
        # Update statistics
        self.total_bytes_sent += len(packet)
        self.total_frames += 1
        self.frame_idx += 1
        self.total_time += time.time() - start_time
        
        return packet
    
    def receive_coordinates(self, data: bytes) -> np.ndarray:
        """
        Receive and process new coordinates from server.
        
        Args:
            data: Compressed coordinate packet
            
        Returns:
            Decoded coordinates [N, 2]
        """
        self.total_bytes_received += len(data)
        
        # Decompress coordinates
        coordinates = self.decompressor.decompress_fast(data)
        
        # Scale to frame resolution
        self.current_coordinates = self.pixel_extractor.scale_coordinates(
            coordinates,
            self.config.target_height,
            self.config.target_width
        )
        
        # Callback
        if self.on_coordinates_received is not None:
            self.on_coordinates_received(coordinates)
        
        return coordinates
    
    def get_statistics(self) -> dict:
        """Get client statistics."""
        avg_time_ms = (self.total_time / max(self.total_frames, 1)) * 1000
        
        return {
            'total_frames': self.total_frames,
            'total_bytes_sent': self.total_bytes_sent,
            'total_bytes_received': self.total_bytes_received,
            'avg_processing_time_ms': avg_time_ms,
            'avg_bandwidth_up_kbps': (self.total_bytes_sent * 30 / 1024) if self.total_frames > 0 else 0,
            'avg_bandwidth_down_kbps': (self.total_bytes_received * 30 / 1024) if self.total_frames > 0 else 0,
        }
    
    def reset(self) -> None:
        """Reset client state."""
        self.frame_idx = 0
        self.current_coordinates = None
        self.coordinate_buffer.clear()
        self.total_bytes_sent = 0
        self.total_bytes_received = 0
        self.total_frames = 0
        self.total_time = 0.0


class SimulatedClient(MinimalGameClient):
    """
    Simulated client for testing without network.
    """
    
    def __init__(self, config: Optional[ClientConfig] = None):
        super().__init__(config)
        self.pending_packets = []
    
    def send_packet(self, packet: bytes) -> None:
        """Store packet for simulated transmission."""
        self.pending_packets.append(packet)
    
    def get_pending_packets(self) -> list:
        """Get and clear pending packets."""
        packets = self.pending_packets.copy()
        self.pending_packets.clear()
        return packets
