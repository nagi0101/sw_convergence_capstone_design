"""
WebSocket API for SGAPS-MAE Server.

Handles real-time bidirectional communication with Unity clients
for frame data streaming and UV coordinate distribution.
"""

import json
import logging
import time
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
from PIL import Image

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from omegaconf import DictConfig

from sgaps.core.session_manager import SessionManager, Session
from sgaps.core.sampler import FixedUVSampler
from sgaps.core.reconstructor import FrameReconstructor
from sgaps.utils.metrics import compute_all_metrics

router = APIRouter()
logger = logging.getLogger(__name__)

# --- Global Objects (Initialized from main.py) ---
_server_config: Optional[DictConfig] = None
_reconstructor: Optional[FrameReconstructor] = None
try:
    import wandb
    USE_WANDB = True
except ImportError:
    USE_WANDB = False
# ----------------------------------------------------

def set_server_config(cfg: DictConfig):
    """Sets the global server configuration."""
    global _server_config
    _server_config = cfg
    logger.info("Server config set for WebSocket API.")

def set_reconstructor(reconstructor: FrameReconstructor):
    """Sets the global frame reconstructor instance."""
    global _reconstructor
    _reconstructor = reconstructor
    logger.info("Frame reconstructor set for WebSocket API.")

def get_server_config() -> DictConfig:
    """Gets the server configuration."""
    if _server_config is None:
        raise RuntimeError("Server configuration not initialized.")
    return _server_config

def get_reconstructor() -> FrameReconstructor:
    """Gets the frame reconstructor."""
    if _reconstructor is None:
        raise RuntimeError("Frame reconstructor not initialized.")
    return _reconstructor

class ConnectionManager:
    """Manages WebSocket connections and session data."""
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.sessions: Dict[str, Session] = {}
        self.session_manager = SessionManager()
        self._lock = asyncio.Lock()
        self.global_wandb_step = 0  # Global monotonic step counter for WandB logging

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        async with self._lock:
            self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected.")

    async def disconnect(self, client_id: str):
        async with self._lock:
            if client_id in self.active_connections:
                del self.active_connections[client_id]
            if client_id in self.sessions:
                session = self.sessions.pop(client_id)
                await self.session_manager.end_session(session)
        logger.info(f"Client {client_id} disconnected and session ended.")

    async def send_json(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)

    async def create_session(self, client_id: str, payload: dict) -> Session:
        cfg = get_server_config()

        # Use server-configured resolution for reconstruction
        # UV coordinates are normalized (0-1), so client resolution doesn't matter
        server_resolution = tuple(cfg.data.resolution)

        # Log client resolution for debugging
        client_resolution = payload.get("resolution", None)
        if client_resolution:
            logger.info(f"Client resolution: {client_resolution}, Server reconstruction: {server_resolution}")

        session = await self.session_manager.create_session(
            client_id=client_id,
            checkpoint_key=payload.get("checkpoint_key", "default"),
            max_state_dim=cfg.model.input_constraints.max_state_dim,
            resolution=server_resolution
        )
        async with self._lock:
            self.sessions[client_id] = session
        return session

    def get_session(self, client_id: str) -> Optional[Session]:
        return self.sessions.get(client_id)

manager = ConnectionManager()

@router.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    client_id = f"client_{id(websocket)}"
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_json()
            await handle_message(client_id, data)
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected normally.")
    except Exception as e:
        logger.error(f"Error with client {client_id}: {e}", exc_info=True)
    finally:
        await manager.disconnect(client_id)

async def handle_message(client_id: str, data: dict):
    message_type = data.get("type")
    payload = data.get("payload", {})
    
    if message_type == "session_start":
        await handle_session_start(client_id, payload)
    elif message_type == "frame_data":
        await handle_frame_data(client_id, payload)
    elif message_type == "heartbeat":
        await manager.send_json(client_id, {"type": "heartbeat_ack", "payload": {"timestamp": time.time()}})
    else:
        logger.warning(f"Unknown message type from {client_id}: {message_type}")
        await manager.send_json(client_id, {"type": "error", "payload": {"message": "Unknown message type"}})

async def handle_session_start(client_id: str, payload: dict):
    logger.info(f"Starting session for {client_id} with payload: {payload}")
    session = await manager.create_session(client_id, payload)
    cfg = get_server_config()
    reconstructor = get_reconstructor()

    # The new reconstructor handles model loading internally via get_model
    # We can check if the model exists to set the 'checkpoint_loaded' flag
    model_path = Path(cfg.paths.checkpoint_dir) / session.checkpoint_key / "best.pth"
    checkpoint_loaded = model_path.exists()
    
    # Generate initial UV coordinates
    sampler = FixedUVSampler(sample_count=cfg.sampling.default_sample_count, resolution=session.resolution)
    initial_coords = sampler.generate_uniform_grid()
    session.sampler = sampler

    await manager.send_json(client_id, {
        "type": "session_start_ack",
        "payload": {
            "checkpoint_key": session.checkpoint_key,
            "checkpoint_loaded": checkpoint_loaded,
            "model_version": f"spt_{cfg.model.name}",
            "sample_count": cfg.sampling.default_sample_count,
            "max_state_dim": cfg.model.input_constraints.max_state_dim,
            "target_fps": cfg.target_fps,
            "resolution": list(session.resolution)
        }
    })
    await manager.send_json(client_id, {
        "type": "uv_coordinates",
        "payload": {"target_frame_id": 0, "coordinates": [{"u": u, "v": v} for u, v in initial_coords]}
    })
    logger.info(f"Session for {client_id} configured and initial coordinates sent.")

async def handle_frame_data(client_id: str, payload: dict):
    session = manager.get_session(client_id)
    if not session:
        return

    frame_id = payload.get("frame_id", 0)

    # Allocate global step for WandB (thread-safe increment)
    async with manager._lock:
        global_step = manager.global_wandb_step
        manager.global_wandb_step += 1

    # 1. Parse data into NumPy arrays
    pixels = np.array([(p['u'], p['v'], p['value']) for p in payload.get("pixels", [])], dtype=np.float32)
    state_vector = np.array(payload.get("state_vector", []), dtype=np.float32)

    # 1.5. Pad state vector to max_state_dim for model compatibility
    cfg = get_server_config()
    sentinel_value = cfg.model.sentinel_value
    max_state_dim = session.max_state_dim

    padded_state_vector = np.full(max_state_dim, sentinel_value, dtype=np.float32)
    if len(state_vector) > 0:
        copy_len = min(len(state_vector), max_state_dim)
        padded_state_vector[:copy_len] = state_vector[:copy_len]

    state_mask = (padded_state_vector != sentinel_value).astype(np.float32)
    state_vector = padded_state_vector  # Replace with padded version

    # 2. Store frame data to HDF5 (non-blocking background task)
    if session.storage:
        asyncio.create_task(
            session.storage.store_frame(
                frame_id=frame_id,
                pixels=payload.get("pixels", []),
                state_vector=state_vector.tolist(),
                resolution=session.resolution
            )
        )

    # 3. Perform reconstruction
    reconstructor = get_reconstructor()
    reconstructed_frame, attn_weights = await reconstructor.reconstruct(
        pixels, state_vector, session.resolution, session.checkpoint_key
    )

    # 4. Log to WandB (if enabled)
    if USE_WANDB:
        log_payload = {
            "Session/Frame_ID": frame_id,           # Per-session frame number
            "Session/Client_ID": client_id,         # Session identifier
            "Session/Global_Step": global_step      # Global step for reference
        }
        if reconstructed_frame is not None:
            log_payload["Live/Reconstruction"] = wandb.Image(Image.fromarray(reconstructed_frame))
        # Add metrics and other visualizations later
        wandb.log(log_payload, step=global_step)

    # 5. Send back next UV coordinates (currently fixed)
    if session.sampler:
        coords = session.sampler.get_current_coordinates()
        await manager.send_json(client_id, {
            "type": "uv_coordinates",
            "payload": {"target_frame_id": frame_id + 1, "coordinates": [{"u": u, "v": v} for u, v in coords]}
        })

    session.frame_count += 1
    if session.frame_count % 100 == 0:
        logger.info(f"Session {client_id}: processed {session.frame_count} frames.")