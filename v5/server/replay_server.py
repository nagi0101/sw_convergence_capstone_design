"""
Replay Server for SGAPS-MAE
Main server implementation for game session replay.
"""

import asyncio
import time
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import struct

import torch
import torch.nn as nn

from ..models import SGAPS_MAE, TemporalMemoryBank
from .quality_analyzer import QualityAnalyzer
from .coordinate_generator import ServerCoordinateGenerator
from .latency_compensator import LatencyCompensator


@dataclass
class ClientSession:
    """State for a single client session."""
    
    client_id: str
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    frame_count: int = 0
    
    # Last sent coordinates
    last_coords: Optional[torch.Tensor] = None
    
    # Client-specific memory
    memory: Optional[TemporalMemoryBank] = None
    
    # Latency tracking
    latency_samples: List[float] = field(default_factory=list)
    estimated_latency: float = 33.0  # Default 33ms
    
    # Quality metrics
    last_quality_score: float = 0.0
    cumulative_quality: float = 0.0
    
    def update_latency(self, latency_ms: float) -> None:
        """Update latency estimate."""
        self.latency_samples.append(latency_ms)
        if len(self.latency_samples) > 100:
            self.latency_samples = self.latency_samples[-100:]
        
        # EMA update
        self.estimated_latency = 0.9 * self.estimated_latency + 0.1 * latency_ms
    
    def update_quality(self, quality_score: float) -> None:
        """Update quality metrics."""
        self.last_quality_score = quality_score
        self.cumulative_quality += quality_score
    
    def get_average_quality(self) -> float:
        """Get average reconstruction quality."""
        if self.frame_count == 0:
            return 0.0
        return self.cumulative_quality / self.frame_count


class ReplayServer(nn.Module):
    """
    Main server for SGAPS-MAE game session replay.
    
    Handles multiple client sessions, performs reconstruction,
    and generates adaptive sampling coordinates.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        sampling_budget: int = 500,
        max_sessions: int = 10000,
        device: str = 'cuda'
    ):
        """
        Args:
            image_size: Target image resolution
            sampling_budget: Default sampling budget
            max_sessions: Maximum concurrent sessions
            device: Processing device
        """
        super().__init__()
        
        self.image_size = image_size
        self.sampling_budget = sampling_budget
        self.max_sessions = max_sessions
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Core model
        self.model = SGAPS_MAE(
            image_size=image_size,
            sampling_budget=sampling_budget
        ).to(self.device)
        
        # Server components
        self.quality_analyzer = QualityAnalyzer().to(self.device)
        self.coordinate_generator = ServerCoordinateGenerator(
            default_budget=sampling_budget
        ).to(self.device)
        self.latency_compensator = LatencyCompensator().to(self.device)
        
        # Session management
        self.sessions: Dict[str, ClientSession] = {}
        
        # Statistics
        self.total_frames_processed = 0
        self.total_bytes_received = 0
        self.total_bytes_sent = 0
    
    def create_session(self, client_id: str) -> ClientSession:
        """
        Create a new client session.
        
        Args:
            client_id: Unique client identifier
            
        Returns:
            New client session
        """
        if len(self.sessions) >= self.max_sessions:
            # Remove oldest inactive session
            oldest_id = min(
                self.sessions.keys(),
                key=lambda x: self.sessions[x].last_active
            )
            del self.sessions[oldest_id]
        
        session = ClientSession(
            client_id=client_id,
            memory=TemporalMemoryBank(
                resolution=self.image_size,
                feature_dim=3
            )
        )
        
        self.sessions[client_id] = session
        return session
    
    def get_session(self, client_id: str) -> Optional[ClientSession]:
        """Get existing session or create new one."""
        if client_id not in self.sessions:
            return self.create_session(client_id)
        return self.sessions[client_id]
    
    def remove_session(self, client_id: str) -> None:
        """Remove a client session."""
        if client_id in self.sessions:
            del self.sessions[client_id]
    
    def decode_client_data(
        self,
        data: bytes
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Decode compressed client data.
        
        Args:
            data: Compressed packet bytes
            
        Returns:
            frame_idx, pixel_values, pixel_positions
        """
        # Decompress
        decompressed = zlib.decompress(data)
        
        # Parse header: frame_idx (4 bytes) + num_pixels (4 bytes)
        frame_idx = struct.unpack('<I', decompressed[:4])[0]
        num_pixels = struct.unpack('<I', decompressed[4:8])[0]
        
        # Parse pixel data
        offset = 8
        positions = []
        values = []
        
        for _ in range(num_pixels):
            # Position: u (2 bytes) + v (2 bytes)
            u = struct.unpack('<H', decompressed[offset:offset+2])[0]
            v = struct.unpack('<H', decompressed[offset+2:offset+4])[0]
            
            # RGB values (3 bytes)
            r = decompressed[offset+4] / 255.0
            g = decompressed[offset+5] / 255.0
            b = decompressed[offset+6] / 255.0
            
            positions.append([u, v])
            values.append([r, g, b])
            
            offset += 7
        
        pixel_positions = torch.tensor(positions, device=self.device)
        pixel_values = torch.tensor(values, device=self.device)
        
        return frame_idx, pixel_values, pixel_positions
    
    def encode_coordinates(
        self,
        coordinates: torch.Tensor
    ) -> bytes:
        """
        Encode coordinates for transmission.
        
        Args:
            coordinates: Sampling coordinates [N, 2]
            
        Returns:
            Compressed coordinate bytes
        """
        coords_cpu = coordinates.cpu().numpy().astype('uint16')
        
        # Build packet
        data = struct.pack('<I', len(coords_cpu))  # num_coords
        
        for u, v in coords_cpu:
            data += struct.pack('<HH', int(u), int(v))
        
        # Compress
        compressed = zlib.compress(data, level=1)
        
        return compressed
    
    @torch.no_grad()
    def process_client_data(
        self,
        client_id: str,
        data: bytes
    ) -> Dict[str, any]:
        """
        Process client pixel data and generate response.
        
        Args:
            client_id: Client identifier
            data: Compressed pixel data
            
        Returns:
            Dictionary with reconstruction and next coordinates
        """
        session = self.get_session(client_id)
        session.last_active = time.time()
        
        # Track bandwidth
        self.total_bytes_received += len(data)
        
        # Decode data
        frame_idx, pixel_values, pixel_positions = self.decode_client_data(data)
        
        # Get static information from memory
        if session.memory is not None:
            static_mask, static_values = session.memory.get_static_pixels(self.device)
        else:
            static_mask = torch.zeros(1, 1, *self.image_size, device=self.device)
            static_values = torch.zeros(1, 3, *self.image_size, device=self.device)
        
        # Reconstruct
        result = self.model.reconstruct(
            pixel_values,
            pixel_positions,
            use_memory=session.memory is not None
        )
        
        reconstruction = result['reconstruction']
        features = result['features']
        uncertainty_map = result['uncertainty_map']
        
        # Analyze quality
        quality_info = self.quality_analyzer(reconstruction, return_maps=True)
        quality_score = quality_info['overall_quality'].item()
        session.update_quality(quality_score)
        
        # Generate next coordinates
        coord_result = self.coordinate_generator(
            uncertainty_map=uncertainty_map,
            quality_map=1.0 - quality_info['confidence_map'],
            static_mask=static_mask,
            budget=self.sampling_budget
        )
        
        next_coords = coord_result['coordinates']
        
        # Compensate for latency
        compensated_coords = self.latency_compensator.compensate(
            next_coords,
            prediction_frames=int(session.estimated_latency / 33) + 1
        )
        
        # Store for next frame
        session.last_coords = compensated_coords
        session.frame_count += 1
        self.total_frames_processed += 1
        
        # Encode response
        coord_bytes = self.encode_coordinates(compensated_coords)
        self.total_bytes_sent += len(coord_bytes)
        
        return {
            'reconstruction': reconstruction,
            'next_coordinates': compensated_coords,
            'coordinate_bytes': coord_bytes,
            'quality_score': quality_score,
            'frame_idx': frame_idx
        }
    
    def get_initial_coordinates(
        self,
        client_id: str
    ) -> bytes:
        """
        Get initial random coordinates for new session.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Encoded coordinate bytes
        """
        session = self.get_session(client_id)
        
        H, W = self.image_size
        
        # Random initial sampling
        num_coords = self.sampling_budget
        u = torch.randint(0, H, (num_coords,), device=self.device)
        v = torch.randint(0, W, (num_coords,), device=self.device)
        coords = torch.stack([u, v], dim=1).float()
        
        session.last_coords = coords
        
        return self.encode_coordinates(coords)
    
    def get_statistics(self) -> Dict[str, any]:
        """Get server statistics."""
        return {
            'total_sessions': len(self.sessions),
            'total_frames_processed': self.total_frames_processed,
            'total_bytes_received': self.total_bytes_received,
            'total_bytes_sent': self.total_bytes_sent,
            'avg_bandwidth_up_kbps': self.total_bytes_received / 1024 if self.total_frames_processed > 0 else 0,
            'avg_bandwidth_down_kbps': self.total_bytes_sent / 1024 if self.total_frames_processed > 0 else 0,
            'device': str(self.device),
            'gpu_memory_mb': torch.cuda.memory_allocated(self.device) / 1024**2 if self.device.type == 'cuda' else 0
        }
    
    def forward(
        self,
        client_id: str,
        data: bytes
    ) -> bytes:
        """
        Process client request and return coordinate response.
        
        Args:
            client_id: Client identifier
            data: Compressed pixel data
            
        Returns:
            Compressed coordinate response
        """
        result = self.process_client_data(client_id, data)
        return result['coordinate_bytes']
