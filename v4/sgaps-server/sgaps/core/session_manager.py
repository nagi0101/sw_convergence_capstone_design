"""
Session Manager for SGAPS-MAE Server.

Handles session lifecycle, configuration, and state management
for connected clients.
"""

import logging
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import time

from sgaps.data.storage import HDF5Storage


logger = logging.getLogger(__name__)


@dataclass
class Session:
    """
    Represents an active client session.
    
    Attributes:
        client_id: Unique identifier for the client
        checkpoint_key: Model checkpoint identifier (game/level specific)
        max_state_dim: Maximum state vector dimension
        resolution: Capture resolution (width, height)
        created_at: Session creation timestamp
        frame_count: Number of frames processed
        storage: HDF5 storage instance for this session
        sampler: UV sampler instance (set after session_start)
    """
    client_id: str
    checkpoint_key: str
    max_state_dim: int
    resolution: Tuple[int, int]
    created_at: float = field(default_factory=time.time)
    frame_count: int = 0
    storage: Optional[HDF5Storage] = None
    sampler: Any = None  # FixedUVSampler, set later
    
    @property
    def duration_seconds(self) -> float:
        """Get session duration in seconds."""
        return time.time() - self.created_at


class SessionManager:
    """
    Manages all active sessions.
    
    Provides session creation, retrieval, and cleanup functionality.
    """
    
    def __init__(self, storage_base_path: str = "./data"):
        self.sessions: Dict[str, Session] = {}
        self.storage_base_path = storage_base_path
        self._lock = asyncio.Lock()
        
        logger.info(f"SessionManager initialized with storage at {storage_base_path}")
    
    async def create_session(
        self,
        client_id: str,
        checkpoint_key: str,
        max_state_dim: int,
        resolution: Tuple[int, int]
    ) -> Session:
        """
        Create a new session for a client.
        
        Args:
            client_id: Unique client identifier
            checkpoint_key: Model checkpoint key for this session
            max_state_dim: Maximum state vector dimension
            resolution: Frame resolution (width, height)
            
        Returns:
            Newly created Session instance
        """
        async with self._lock:
            # Create storage for this session
            storage = HDF5Storage(
                base_path=self.storage_base_path,
                checkpoint_key=checkpoint_key,
                session_id=client_id
            )
            await storage.initialize()
            
            # Create session
            session = Session(
                client_id=client_id,
                checkpoint_key=checkpoint_key,
                max_state_dim=max_state_dim,
                resolution=resolution,
                storage=storage
            )
            
            self.sessions[client_id] = session
            logger.info(f"Created session for {client_id} with checkpoint {checkpoint_key}")
            
            return session
    
    async def get_session(self, client_id: str) -> Optional[Session]:
        """Get a session by client ID."""
        return self.sessions.get(client_id)
    
    async def end_session(self, session: Session):
        """
        End a session and cleanup resources.
        
        Args:
            session: Session to end
        """
        async with self._lock:
            if session.client_id in self.sessions:
                # Close storage
                if session.storage:
                    await session.storage.close()
                
                # Remove from active sessions
                del self.sessions[session.client_id]
                
                logger.info(
                    f"Ended session {session.client_id}: "
                    f"{session.frame_count} frames, "
                    f"{session.duration_seconds:.1f}s duration"
                )
    
    async def cleanup_expired_sessions(self, max_age_seconds: float = 3600):
        """
        Cleanup sessions that have been inactive for too long.
        
        Args:
            max_age_seconds: Maximum session age in seconds
        """
        current_time = time.time()
        expired = []
        
        async with self._lock:
            for client_id, session in self.sessions.items():
                if current_time - session.created_at > max_age_seconds:
                    expired.append(session)
        
        for session in expired:
            await self.end_session(session)
            logger.warning(f"Cleaned up expired session: {session.client_id}")
    
    @property
    def active_session_count(self) -> int:
        """Get the number of active sessions."""
        return len(self.sessions)
    
    @property
    def total_frames_processed(self) -> int:
        """Get total frames processed across all sessions."""
        return sum(s.frame_count for s in self.sessions.values())
