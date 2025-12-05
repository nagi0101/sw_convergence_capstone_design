"""
WebSocket API for SGAPS-MAE Server.

Handles real-time bidirectional communication with Unity clients
for frame data streaming and UV coordinate distribution.
"""

import json
import logging
import asyncio
from typing import Dict, Any, Optional, Set
from dataclasses import dataclass, field
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from omegaconf import DictConfig

from sgaps.core.session_manager import SessionManager, Session
from sgaps.core.sampler import FixedUVSampler
from sgaps.core.reconstructor import OpenCVReconstructor
from sgaps.data.storage import HDF5Storage


router = APIRouter()
logger = logging.getLogger(__name__)

# Server configuration (set by main.py during startup)
_server_config: Optional[DictConfig] = None


def set_server_config(cfg: DictConfig):
    """Set server configuration from main.py."""
    global _server_config
    _server_config = cfg
    logger.info(f"Server config set: max_state_dim={cfg.max_state_dim}, "
                f"sample_count={cfg.sampling.default_sample_count}, "
                f"target_fps={cfg.target_fps}, sentinel_value={cfg.sentinel_value}")


def get_server_config() -> DictConfig:
    """Get server configuration."""
    if _server_config is None:
        raise RuntimeError("Server configuration not initialized. Call set_server_config first.")
    return _server_config


class ConnectionManager:
    """
    Manages WebSocket connections and session lifecycle.
    
    Handles connection tracking, message routing, and cleanup.
    """
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.sessions: Dict[str, Session] = {}
        self.session_manager = SessionManager()
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, client_id: str) -> bool:
        """Accept a new WebSocket connection."""
        try:
            await websocket.accept()
            async with self._lock:
                self.active_connections[client_id] = websocket
            logger.info(f"Client {client_id} connected")
            
            # Send connection acknowledgment
            await self.send_message(client_id, {
                "type": "connection_ack",
                "payload": {
                    "client_id": client_id,
                    "server_version": "0.1.0"
                }
            })
            return True
        except Exception as e:
            logger.error(f"Failed to accept connection from {client_id}: {e}")
            return False
    
    async def disconnect(self, client_id: str):
        """Handle client disconnection and cleanup."""
        async with self._lock:
            if client_id in self.active_connections:
                del self.active_connections[client_id]
            
            if client_id in self.sessions:
                session = self.sessions[client_id]
                await self.session_manager.end_session(session)
                del self.sessions[client_id]
        
        logger.info(f"Client {client_id} disconnected")
    
    async def send_message(self, client_id: str, message: Dict[str, Any]):
        """Send a JSON message to a specific client."""
        if client_id not in self.active_connections:
            logger.warning(f"Cannot send message: client {client_id} not connected")
            return
        
        websocket = self.active_connections[client_id]
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send message to {client_id}: {e}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients."""
        for client_id in list(self.active_connections.keys()):
            await self.send_message(client_id, message)
    
    def get_session(self, client_id: str) -> Optional[Session]:
        """Get the session for a client."""
        return self.sessions.get(client_id)
    
    async def create_session(self, client_id: str, config: Dict[str, Any]) -> Session:
        """Create a new session for a client."""
        session = await self.session_manager.create_session(
            client_id=client_id,
            checkpoint_key=config.get("checkpoint_key", "default"),
            max_state_dim=config.get("max_state_dim", 64),
            resolution=tuple(config.get("resolution", [640, 480]))
        )
        
        async with self._lock:
            self.sessions[client_id] = session
        
        return session


# Global connection manager instance
manager = ConnectionManager()


@router.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint for frame streaming.
    
    Protocol:
    1. Client connects and receives connection_ack
    2. Client sends session_start with configuration
    3. Server responds with session_start_ack and initial UV coordinates
    4. Client sends frame_data with sampled pixels
    5. Server processes and sends updated UV coordinates
    6. Repeat steps 4-5 until disconnect
    """
    # Generate unique client ID
    client_id = f"client_{id(websocket)}_{int(time.time() * 1000)}"
    
    # Accept connection
    if not await manager.connect(websocket, client_id):
        return
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            await handle_message(client_id, data)
            
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected normally")
    except Exception as e:
        logger.error(f"Error handling client {client_id}: {e}")
        await manager.send_message(client_id, {
            "type": "error",
            "payload": {
                "code": "INTERNAL_ERROR",
                "message": str(e)
            }
        })
    finally:
        await manager.disconnect(client_id)


async def handle_message(client_id: str, data: Dict[str, Any]):
    """Route incoming messages to appropriate handlers."""
    message_type = data.get("type")
    payload = data.get("payload", {})
    
    handlers = {
        "session_start": handle_session_start,
        "frame_data": handle_frame_data,
        "heartbeat": handle_heartbeat,
    }
    
    handler = handlers.get(message_type)
    if handler:
        await handler(client_id, payload)
    else:
        logger.warning(f"Unknown message type from {client_id}: {message_type}")
        await manager.send_message(client_id, {
            "type": "error",
            "payload": {
                "code": "UNKNOWN_MESSAGE_TYPE",
                "message": f"Unknown message type: {message_type}"
            }
        })


async def handle_session_start(client_id: str, payload: Dict[str, Any]):
    """
    Handle session_start message.
    
    Creates a new session and sends initial UV coordinates.
    Server controls sample_count and max_state_dim - client receives these values.
    """
    logger.info(f"Session start from {client_id}: {payload}")
    
    # Get server configuration (sample_count and max_state_dim are server-controlled)
    cfg = get_server_config()
    sample_count = cfg.sampling.default_sample_count
    max_state_dim = cfg.max_state_dim
    target_fps = cfg.target_fps
    sentinel_value = cfg.sentinel_value
    
    # Resolution comes from client (screen resolution)
    resolution = tuple(payload.get("resolution", [640, 480]))
    
    # Create session config with server-controlled parameters
    session_config = {
        "checkpoint_key": payload.get("checkpoint_key", "default"),
        "max_state_dim": max_state_dim,
        "resolution": resolution
    }
    
    # Create session
    session = await manager.create_session(client_id, session_config)
    
    # Generate initial UV coordinates using server-controlled sample_count
    sampler = FixedUVSampler(
        sample_count=sample_count,
        resolution=resolution
    )
    initial_coords = sampler.generate_uniform_grid()
    
    # Store sampler in session
    session.sampler = sampler
    
    # Send acknowledgment with server-controlled parameters
    # Client MUST use these values for sampling and state vector collection
    await manager.send_message(client_id, {
        "type": "session_start_ack",
        "payload": {
            "checkpoint_key": session.checkpoint_key,
            "checkpoint_loaded": False,  # Phase 1: No model loading
            "model_version": "opencv_inpaint_v1",
            "sample_count": sample_count,
            "max_state_dim": max_state_dim,
            "target_fps": target_fps,
            "sentinel_value": sentinel_value,
            "resolution": list(resolution)
        }
    })
    
    # Send initial UV coordinates
    await manager.send_message(client_id, {
        "type": "uv_coordinates",
        "payload": {
            "target_frame_id": 0,
            "coordinates": [{"u": u, "v": v} for u, v in initial_coords]
        }
    })
    
    logger.info(f"Session created for {client_id}: sample_count={sample_count}, "
                f"max_state_dim={max_state_dim}, target_fps={target_fps}, "
                f"sent {len(initial_coords)} UV coordinates")


async def handle_frame_data(client_id: str, payload: Dict[str, Any]):
    """
    Handle frame_data message.
    
    Stores the frame data, performs reconstruction (Phase 1: placeholder),
    and sends UV coordinates for the next frame.
    """
    session = manager.get_session(client_id)
    if not session:
        await manager.send_message(client_id, {
            "type": "error",
            "payload": {
                "code": "NO_SESSION",
                "message": "No active session. Send session_start first."
            }
        })
        return
    
    frame_id = payload.get("frame_id", 0)
    pixels = payload.get("pixels", [])
    state_vector = payload.get("state_vector", [])
    resolution = tuple(payload.get("resolution", [640, 480]))
    
    # Store frame data
    await session.storage.store_frame(
        frame_id=frame_id,
        pixels=pixels,
        state_vector=state_vector,
        resolution=resolution
    )
    
    # Update session stats
    session.frame_count += 1
    
    # Phase 1: Fixed UV coordinates (no adaptive sampling yet)
    # Just resend the same coordinates for next frame
    if session.sampler:
        coords = session.sampler.get_current_coordinates()
        await manager.send_message(client_id, {
            "type": "uv_coordinates",
            "payload": {
                "target_frame_id": frame_id + 1,
                "coordinates": [{"u": u, "v": v} for u, v in coords]
            }
        })
    
    # Log progress periodically
    if session.frame_count % 100 == 0:
        logger.info(f"Session {client_id}: processed {session.frame_count} frames")


async def handle_heartbeat(client_id: str, payload: Dict[str, Any]):
    """Handle heartbeat message to keep connection alive."""
    await manager.send_message(client_id, {
        "type": "heartbeat_ack",
        "payload": {
            "timestamp": time.time()
        }
    })
