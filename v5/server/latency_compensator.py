"""
Latency Compensator for SGAPS-MAE Server
Predicts future pixel positions to compensate for network latency.
"""

from collections import deque
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class OpticalFlowEstimator(nn.Module):
    """
    Lightweight optical flow estimator for motion prediction.
    """
    
    def __init__(self, feature_dim: int = 64):
        super().__init__()
        
        # Feature extractor
        self.encoder = nn.Sequential(
            nn.Conv2d(6, feature_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim * 2, feature_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Flow decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_dim * 2, feature_dim, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feature_dim, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 3, padding=1),  # 2 channels for (u, v) flow
        )
    
    def forward(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate optical flow between two frames.
        
        Args:
            frame1: First frame [B, 3, H, W]
            frame2: Second frame [B, 3, H, W]
            
        Returns:
            Optical flow [B, 2, H, W]
        """
        # Concatenate frames
        combined = torch.cat([frame1, frame2], dim=1)
        
        # Encode
        features = self.encoder(combined)
        
        # Decode to flow
        flow = self.decoder(features)
        
        return flow


class CameraMotionExtractor(nn.Module):
    """
    Extract global camera motion from frame sequence.
    """
    
    def __init__(self, feature_dim: int = 64):
        super().__init__()
        
        # Global motion network
        self.motion_net = nn.Sequential(
            nn.Conv2d(2, feature_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2)  # Global translation (dx, dy)
        )
    
    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        """
        Extract camera motion from optical flow.
        
        Args:
            flow: Optical flow [B, 2, H, W]
            
        Returns:
            Camera motion [B, 2] (dx, dy)
        """
        camera_motion = self.motion_net(flow)
        return camera_motion


class MotionPredictor(nn.Module):
    """
    Predict future pixel positions based on motion history.
    """
    
    def __init__(self, max_history: int = 10):
        super().__init__()
        
        self.max_history = max_history
        
        # Motion history
        self.flow_history: deque = deque(maxlen=max_history)
        self.camera_history: deque = deque(maxlen=max_history)
    
    def update_history(
        self,
        flow: torch.Tensor,
        camera_motion: torch.Tensor
    ) -> None:
        """
        Update motion history.
        
        Args:
            flow: Current optical flow
            camera_motion: Current camera motion
        """
        self.flow_history.append(flow.detach())
        self.camera_history.append(camera_motion.detach())
    
    def predict_flow(self, num_frames: int = 2) -> Optional[torch.Tensor]:
        """
        Predict future optical flow.
        
        Args:
            num_frames: Number of frames to predict ahead
            
        Returns:
            Predicted flow or None if insufficient history
        """
        if len(self.flow_history) < 2:
            return None
        
        # Simple linear extrapolation
        recent_flows = list(self.flow_history)[-5:]
        avg_flow = torch.stack(recent_flows).mean(dim=0)
        
        # Scale by prediction horizon
        predicted_flow = avg_flow * num_frames
        
        return predicted_flow
    
    def predict_positions(
        self,
        current_coords: torch.Tensor,
        num_frames: int = 2
    ) -> torch.Tensor:
        """
        Predict future pixel positions.
        
        Args:
            current_coords: Current coordinates [N, 2]
            num_frames: Prediction horizon
            
        Returns:
            Predicted coordinates [N, 2]
        """
        predicted_flow = self.predict_flow(num_frames)
        
        if predicted_flow is None:
            return current_coords.clone()
        
        # Sample flow at current positions
        N = current_coords.shape[0]
        H, W = predicted_flow.shape[-2:]
        
        # Normalize coordinates for grid_sample
        coords_norm = current_coords.clone()
        coords_norm[:, 0] = 2 * coords_norm[:, 0] / H - 1
        coords_norm[:, 1] = 2 * coords_norm[:, 1] / W - 1
        
        # Grid sample requires [B, H, W, 2]
        grid = coords_norm.view(1, N, 1, 2)
        
        # Sample flow values
        flow_at_coords = F.grid_sample(
            predicted_flow,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        flow_at_coords = flow_at_coords.view(2, N).T
        
        # Predict future positions
        future_coords = current_coords + flow_at_coords
        
        return future_coords
    
    def clear_history(self) -> None:
        """Clear motion history."""
        self.flow_history.clear()
        self.camera_history.clear()


class LatencyEstimator:
    """
    Estimate and track network latency.
    """
    
    def __init__(
        self,
        history_size: int = 100,
        ema_alpha: float = 0.9
    ):
        """
        Args:
            history_size: Number of samples to keep
            ema_alpha: EMA smoothing factor
        """
        self.history_size = history_size
        self.ema_alpha = ema_alpha
        
        self.latency_history: deque = deque(maxlen=history_size)
        self.ema_latency: float = 33.0  # Default 33ms (1 frame at 30fps)
    
    def update(self, latency_ms: float) -> None:
        """
        Update latency estimate.
        
        Args:
            latency_ms: Measured latency in milliseconds
        """
        self.latency_history.append(latency_ms)
        
        # EMA update
        self.ema_latency = (
            self.ema_alpha * self.ema_latency +
            (1 - self.ema_alpha) * latency_ms
        )
    
    def get_latency(self) -> float:
        """
        Get current latency estimate.
        
        Returns:
            Estimated latency in milliseconds
        """
        return self.ema_latency
    
    def get_prediction_frames(self, frame_rate: int = 30) -> int:
        """
        Get number of frames to predict ahead.
        
        Args:
            frame_rate: Game frame rate
            
        Returns:
            Number of frames for prediction
        """
        frame_duration = 1000.0 / frame_rate
        prediction_frames = int(self.ema_latency / frame_duration) + 1
        
        # Clamp to reasonable range
        return min(max(prediction_frames, 1), 5)


class LatencyCompensator(nn.Module):
    """
    Complete Latency Compensator for SGAPS-MAE.
    Predicts future sampling coordinates to account for network delay.
    """
    
    def __init__(
        self,
        feature_dim: int = 64,
        max_history: int = 10,
        default_prediction_frames: int = 2
    ):
        """
        Args:
            feature_dim: Feature dimension for neural components
            max_history: Maximum motion history length
            default_prediction_frames: Default prediction horizon
        """
        super().__init__()
        
        self.default_prediction_frames = default_prediction_frames
        
        # Components
        self.flow_estimator = OpticalFlowEstimator(feature_dim)
        self.camera_extractor = CameraMotionExtractor(feature_dim)
        self.motion_predictor = MotionPredictor(max_history)
        self.latency_estimator = LatencyEstimator()
        
        # Previous frame for flow computation
        self.prev_frame: Optional[torch.Tensor] = None
    
    def update_motion(self, frame: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Update motion estimate with new frame.
        
        Args:
            frame: Current frame [B, 3, H, W]
            
        Returns:
            Estimated optical flow or None
        """
        if self.prev_frame is None:
            self.prev_frame = frame.detach()
            return None
        
        # Estimate optical flow
        flow = self.flow_estimator(self.prev_frame, frame)
        
        # Extract camera motion
        camera_motion = self.camera_extractor(flow)
        
        # Update history
        self.motion_predictor.update_history(flow, camera_motion)
        
        # Store current frame
        self.prev_frame = frame.detach()
        
        return flow
    
    def compensate(
        self,
        current_coords: torch.Tensor,
        prediction_frames: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compensate coordinates for network latency.
        
        Args:
            current_coords: Current sampling coordinates [N, 2]
            prediction_frames: Override prediction horizon
            
        Returns:
            Latency-compensated coordinates [N, 2]
        """
        if prediction_frames is None:
            prediction_frames = self.latency_estimator.get_prediction_frames()
        
        # Predict future positions
        future_coords = self.motion_predictor.predict_positions(
            current_coords,
            prediction_frames
        )
        
        return future_coords
    
    def update_latency(self, latency_ms: float) -> None:
        """
        Update latency estimate.
        
        Args:
            latency_ms: Measured round-trip latency
        """
        self.latency_estimator.update(latency_ms)
    
    def forward(
        self,
        frame: torch.Tensor,
        coordinates: torch.Tensor,
        latency_ms: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: update motion and compensate coordinates.
        
        Args:
            frame: Current frame [B, 3, H, W]
            coordinates: Current sampling coordinates [N, 2]
            latency_ms: Optional latency measurement
            
        Returns:
            Dictionary with compensated coordinates and flow
        """
        # Update latency if provided
        if latency_ms is not None:
            self.update_latency(latency_ms)
        
        # Update motion model
        flow = self.update_motion(frame)
        
        # Get prediction horizon
        prediction_frames = self.latency_estimator.get_prediction_frames()
        
        # Compensate coordinates
        compensated_coords = self.compensate(coordinates, prediction_frames)
        
        result = {
            'compensated_coordinates': compensated_coords,
            'prediction_frames': torch.tensor([prediction_frames]),
            'estimated_latency': torch.tensor([self.latency_estimator.get_latency()])
        }
        
        if flow is not None:
            result['optical_flow'] = flow
        
        return result
    
    def reset(self) -> None:
        """Reset all state."""
        self.prev_frame = None
        self.motion_predictor.clear_history()
