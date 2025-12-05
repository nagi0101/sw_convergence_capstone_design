"""
HDF5 Storage for SGAPS-MAE Server.

Stores collected frame data in HDF5 format, organized by
checkpoint_key for per-game/level training.
"""

import logging
import asyncio
import os
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import time

try:
    import h5py
    import numpy as np
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    logging.warning("h5py not available, storage disabled")


logger = logging.getLogger(__name__)


class HDF5Storage:
    """
    HDF5-based storage for frame data.
    
    Organizes data by checkpoint_key, with each session stored
    in a separate group within the HDF5 file.
    
    File structure:
        {checkpoint_key}.h5
        ├── {session_id}/
        │   ├── frames/
        │   │   ├── 0/
        │   │   │   ├── pixels (N, 3) - u, v, value
        │   │   │   └── state_vector (M,)
        │   │   ├── 1/
        │   │   └── ...
        │   └── metadata
        │       ├── resolution
        │       ├── created_at
        │       └── frame_count
    """
    
    def __init__(
        self,
        base_path: str,
        checkpoint_key: str,
        session_id: str
    ):
        """
        Initialize storage.
        
        Args:
            base_path: Base directory for HDF5 files
            checkpoint_key: Game/level identifier
            session_id: Unique session identifier
        """
        self.base_path = Path(base_path)
        self.checkpoint_key = checkpoint_key
        self.session_id = session_id
        
        self.file_path = self.base_path / f"{checkpoint_key}.h5"
        self.h5file: Optional[h5py.File] = None
        self.session_group = None
        self.frame_count = 0
        
        self._lock = asyncio.Lock()
        
        logger.info(f"HDF5Storage initialized: {self.file_path}")
    
    async def initialize(self):
        """Initialize storage and create necessary groups."""
        if not HAS_H5PY:
            logger.error("h5py not available")
            return
        
        # Create directory if needed
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        async with self._lock:
            # Open or create HDF5 file
            self.h5file = h5py.File(self.file_path, 'a')
            
            # Create session group
            if self.session_id not in self.h5file:
                self.session_group = self.h5file.create_group(self.session_id)
                self.session_group.create_group("frames")
                
                # Store metadata
                meta = self.session_group.create_group("metadata")
                meta.attrs["created_at"] = time.time()
                meta.attrs["frame_count"] = 0
            else:
                self.session_group = self.h5file[self.session_id]
                self.frame_count = self.session_group["metadata"].attrs.get("frame_count", 0)
            
            logger.info(f"Storage initialized for session {self.session_id}")
    
    async def store_frame(
        self,
        frame_id: int,
        pixels: List[Dict],
        state_vector: List[float],
        resolution: Tuple[int, int]
    ):
        """
        Store a single frame's data.
        
        Args:
            frame_id: Frame identifier
            pixels: List of pixel data dicts
            state_vector: State vector values
            resolution: Frame resolution
        """
        if not HAS_H5PY or self.h5file is None:
            return
        
        async with self._lock:
            frames_group = self.session_group["frames"]
            
            # Create frame group
            frame_key = str(frame_id)
            if frame_key in frames_group:
                del frames_group[frame_key]
            
            frame_group = frames_group.create_group(frame_key)
            
            # Store pixels as (N, 3) array: u, v, value
            if pixels:
                pixel_data = np.array([
                    [p.get("u", 0), p.get("v", 0), p.get("value", 0)]
                    for p in pixels
                ], dtype=np.float32)
                frame_group.create_dataset("pixels", data=pixel_data, compression="gzip")
            
            # Store state vector
            if state_vector:
                state_data = np.array(state_vector, dtype=np.float32)
                frame_group.create_dataset("state_vector", data=state_data)
            
            # Store resolution
            frame_group.attrs["resolution"] = resolution
            frame_group.attrs["timestamp"] = time.time()
            
            # Update frame count
            self.frame_count += 1
            self.session_group["metadata"].attrs["frame_count"] = self.frame_count
            
            # Flush periodically
            if self.frame_count % 100 == 0:
                self.h5file.flush()
                logger.debug(f"Stored frame {frame_id}, total: {self.frame_count}")
    
    async def get_frame(self, frame_id: int) -> Optional[Dict]:
        """
        Retrieve a stored frame.
        
        Args:
            frame_id: Frame identifier
            
        Returns:
            Dictionary with pixels and state_vector, or None
        """
        if not HAS_H5PY or self.h5file is None:
            return None
        
        async with self._lock:
            frames_group = self.session_group["frames"]
            frame_key = str(frame_id)
            
            if frame_key not in frames_group:
                return None
            
            frame_group = frames_group[frame_key]
            
            result = {
                "frame_id": frame_id,
                "resolution": tuple(frame_group.attrs.get("resolution", (640, 480))),
                "timestamp": frame_group.attrs.get("timestamp", 0)
            }
            
            if "pixels" in frame_group:
                result["pixels"] = frame_group["pixels"][:]
            
            if "state_vector" in frame_group:
                result["state_vector"] = frame_group["state_vector"][:]
            
            return result
    
    async def get_frame_count(self) -> int:
        """Get the number of stored frames."""
        return self.frame_count
    
    async def close(self):
        """Close the HDF5 file."""
        if self.h5file is not None:
            async with self._lock:
                self.h5file.flush()
                self.h5file.close()
                self.h5file = None
                logger.info(f"Storage closed for session {self.session_id}")
    
    def __del__(self):
        """Ensure file is closed on deletion."""
        if self.h5file is not None:
            try:
                self.h5file.close()
            except:
                pass
