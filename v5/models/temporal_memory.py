"""
Temporal Memory Bank for SGAPS-MAE
Maintains static and dynamic pixel information across frames.
"""

from collections import deque
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn


class StaticMemory:
    """
    Long-term memory for static/UI elements.
    Uses Exponential Moving Average (EMA) for stable updates.
    """
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (224, 224),
        feature_dim: int = 3,
        ema_decay: float = 0.95,
        confidence_threshold: float = 0.9
    ):
        """
        Args:
            resolution: Image resolution (H, W)
            feature_dim: Feature dimension (3 for RGB)
            ema_decay: EMA decay factor
            confidence_threshold: Threshold for considering pixel as static
        """
        self.resolution = resolution
        self.feature_dim = feature_dim
        self.ema_decay = ema_decay
        self.confidence_threshold = confidence_threshold
        
        # Storage as dictionaries (sparse representation)
        self.values: Dict[Tuple[int, int], torch.Tensor] = {}
        self.confidence: Dict[Tuple[int, int], float] = {}
        self.last_update: Dict[Tuple[int, int], int] = {}
    
    def update(
        self,
        position: Tuple[int, int],
        value: torch.Tensor,
        frame_idx: int
    ) -> None:
        """
        Update static memory with new observation.
        
        Args:
            position: Pixel position (u, v)
            value: Pixel value tensor
            frame_idx: Current frame index
        """
        if position in self.values:
            # EMA update
            old_value = self.values[position]
            new_value = self.ema_decay * old_value + (1 - self.ema_decay) * value
            self.values[position] = new_value
            
            # Increase confidence
            self.confidence[position] = min(
                self.confidence[position] + 0.1,
                1.0
            )
        else:
            # New observation
            self.values[position] = value.clone()
            self.confidence[position] = 0.5
        
        self.last_update[position] = frame_idx
    
    def get_static_pixels(
        self,
        confidence_threshold: Optional[float] = None
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        """
        Get pixels with high confidence.
        
        Args:
            confidence_threshold: Override threshold
            
        Returns:
            Dictionary of position -> value
        """
        threshold = confidence_threshold or self.confidence_threshold
        
        return {
            pos: val
            for pos, val in self.values.items()
            if self.confidence.get(pos, 0) > threshold
        }
    
    def decay_confidence(self, decay_rate: float = 0.99) -> None:
        """Decay confidence for pixels not recently updated."""
        for pos in list(self.confidence.keys()):
            self.confidence[pos] *= decay_rate
            if self.confidence[pos] < 0.1:
                # Remove low-confidence entries
                del self.values[pos]
                del self.confidence[pos]
                if pos in self.last_update:
                    del self.last_update[pos]
    
    def get_dense_mask(self, device: torch.device) -> torch.Tensor:
        """
        Get binary mask of static pixels.
        
        Args:
            device: Target device
            
        Returns:
            Mask tensor [1, 1, H, W]
        """
        H, W = self.resolution
        mask = torch.zeros(1, 1, H, W, device=device)
        
        for (u, v), conf in self.confidence.items():
            if 0 <= u < H and 0 <= v < W and conf > self.confidence_threshold:
                mask[0, 0, u, v] = 1.0
        
        return mask
    
    def get_dense_values(self, device: torch.device) -> torch.Tensor:
        """
        Get dense tensor of static pixel values.
        
        Args:
            device: Target device
            
        Returns:
            Value tensor [1, C, H, W]
        """
        H, W = self.resolution
        values = torch.zeros(1, self.feature_dim, H, W, device=device)
        
        for (u, v), val in self.values.items():
            if 0 <= u < H and 0 <= v < W:
                if self.confidence.get((u, v), 0) > self.confidence_threshold:
                    values[0, :, u, v] = val.to(device)
        
        return values


class DynamicMemory:
    """
    Short-term memory for dynamic/moving objects.
    Uses FIFO queue for recent observations.
    """
    
    def __init__(
        self,
        max_frames: int = 100,
        max_objects: int = 1000
    ):
        """
        Args:
            max_frames: Maximum number of frames to keep
            max_objects: Maximum number of objects to track
        """
        self.max_frames = max_frames
        self.max_objects = max_objects
        
        # FIFO queue of observations
        self.observations: deque = deque(maxlen=max_frames)
        
        # Motion vectors for tracked objects
        self.motion_history: deque = deque(maxlen=max_frames)
    
    def add_observation(
        self,
        positions: torch.Tensor,
        values: torch.Tensor,
        motion_vectors: Optional[torch.Tensor] = None,
        frame_idx: int = 0
    ) -> None:
        """
        Add new observations to dynamic memory.
        
        Args:
            positions: Pixel positions [N, 2]
            values: Pixel values [N, C]
            motion_vectors: Optional motion vectors [N, 2]
            frame_idx: Current frame index
        """
        observation = {
            'positions': positions.clone(),
            'values': values.clone(),
            'frame_idx': frame_idx
        }
        
        self.observations.append(observation)
        
        if motion_vectors is not None:
            self.motion_history.append(motion_vectors.clone())
    
    def get_recent(
        self,
        num_frames: int = 10
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Get recent observations.
        
        Args:
            num_frames: Number of recent frames
            
        Returns:
            List of observation dictionaries
        """
        return list(self.observations)[-num_frames:]
    
    def estimate_motion(self) -> Optional[torch.Tensor]:
        """
        Estimate average motion from history.
        
        Returns:
            Average motion vector or None
        """
        if len(self.motion_history) < 2:
            return None
        
        recent_motion = list(self.motion_history)[-10:]
        
        # Average motion
        avg_motion = torch.stack(recent_motion).mean(dim=0)
        
        return avg_motion
    
    def clear(self) -> None:
        """Clear all dynamic memory."""
        self.observations.clear()
        self.motion_history.clear()


class TemporalMemoryBank(nn.Module):
    """
    Combined Temporal Memory Bank for SGAPS-MAE.
    
    Manages both static (UI/HUD) and dynamic (moving objects) elements
    for efficient reconstruction and sampling budget optimization.
    """
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (224, 224),
        feature_dim: int = 3,
        motion_threshold: float = 0.1,
        ema_decay: float = 0.95,
        max_dynamic_frames: int = 100
    ):
        """
        Args:
            resolution: Image resolution (H, W)
            feature_dim: Feature dimension (3 for RGB)
            motion_threshold: Threshold for static/dynamic classification
            ema_decay: EMA decay for static memory
            max_dynamic_frames: Max frames in dynamic memory
        """
        super().__init__()
        
        self.resolution = resolution
        self.feature_dim = feature_dim
        self.motion_threshold = motion_threshold
        
        # Static memory for UI/HUD elements
        self.static_memory = StaticMemory(
            resolution=resolution,
            feature_dim=feature_dim,
            ema_decay=ema_decay
        )
        
        # Dynamic memory for moving objects
        self.dynamic_memory = DynamicMemory(
            max_frames=max_dynamic_frames
        )
        
        # Frame counter
        self.frame_counter = 0
        
        # Previous frame for motion estimation
        self.prev_positions: Optional[torch.Tensor] = None
        self.prev_values: Optional[torch.Tensor] = None
    
    def update(
        self,
        positions: torch.Tensor,
        values: torch.Tensor,
        motion_scores: Optional[torch.Tensor] = None
    ) -> None:
        """
        Update memory with new pixel observations.
        
        Args:
            positions: Pixel positions [N, 2]
            values: Pixel values [N, C]
            motion_scores: Optional motion scores [N]
        """
        self.frame_counter += 1
        
        # Compute motion scores if not provided
        if motion_scores is None:
            motion_scores = self._compute_motion_scores(positions, values)
        
        # Classify and update
        for i, (pos, val, motion) in enumerate(zip(positions, values, motion_scores)):
            pos_tuple = (int(pos[0].item()), int(pos[1].item()))
            
            if motion < self.motion_threshold:
                # Static: update long-term memory
                self.static_memory.update(
                    pos_tuple,
                    val,
                    self.frame_counter
                )
            else:
                # Dynamic: will be handled in batch
                pass
        
        # Update dynamic memory with all observations
        self.dynamic_memory.add_observation(
            positions,
            values,
            motion_vectors=None,
            frame_idx=self.frame_counter
        )
        
        # Store for next frame
        self.prev_positions = positions.clone()
        self.prev_values = values.clone()
        
        # Periodic confidence decay
        if self.frame_counter % 10 == 0:
            self.static_memory.decay_confidence()
    
    def _compute_motion_scores(
        self,
        positions: torch.Tensor,
        values: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute motion scores based on value changes.
        
        Args:
            positions: Current positions [N, 2]
            values: Current values [N, C]
            
        Returns:
            Motion scores [N]
        """
        if self.prev_values is None or self.prev_positions is None:
            return torch.ones(len(positions), device=positions.device) * 0.5
        
        motion_scores = []
        
        for pos, val in zip(positions, values):
            pos_tuple = (int(pos[0].item()), int(pos[1].item()))
            
            # Check if position exists in static memory
            if pos_tuple in self.static_memory.values:
                old_val = self.static_memory.values[pos_tuple]
                diff = (val - old_val.to(val.device)).abs().mean().item()
                motion_scores.append(diff)
            else:
                motion_scores.append(0.5)  # Unknown
        
        return torch.tensor(motion_scores, device=positions.device)
    
    def get_static_pixels(
        self,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get reliable static pixels for reconstruction.
        
        Args:
            device: Target device
            
        Returns:
            mask: Static pixel mask [1, 1, H, W]
            values: Static pixel values [1, C, H, W]
        """
        mask = self.static_memory.get_dense_mask(device)
        values = self.static_memory.get_dense_values(device)
        
        return mask, values
    
    def create_dynamic_mask(
        self,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create mask indicating dynamic regions (inverse of static).
        
        Args:
            device: Target device
            
        Returns:
            Dynamic region mask [1, 1, H, W]
        """
        static_mask = self.static_memory.get_dense_mask(device)
        return 1.0 - static_mask
    
    def get_sampling_mask(
        self,
        device: torch.device,
        budget_reduction: float = 0.6
    ) -> torch.Tensor:
        """
        Get mask for adaptive sampling (focus on dynamic regions).
        
        Args:
            device: Target device
            budget_reduction: Factor to reduce sampling in static regions
            
        Returns:
            Sampling weight mask [1, 1, H, W]
        """
        static_mask = self.static_memory.get_dense_mask(device)
        
        # Higher weight for dynamic regions, lower for static
        sampling_mask = 1.0 - static_mask * (1 - budget_reduction)
        
        return sampling_mask
    
    def reset(self) -> None:
        """Reset all memory."""
        self.static_memory.values.clear()
        self.static_memory.confidence.clear()
        self.static_memory.last_update.clear()
        self.dynamic_memory.clear()
        self.frame_counter = 0
        self.prev_positions = None
        self.prev_values = None
    
    def forward(
        self,
        positions: torch.Tensor,
        values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process new observations and return static information.
        
        Args:
            positions: Pixel positions [N, 2]
            values: Pixel values [N, C]
            
        Returns:
            static_mask: Static region mask
            static_values: Static pixel values
        """
        # Update memory
        self.update(positions, values)
        
        # Return static information
        device = positions.device
        return self.get_static_pixels(device)
