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
import base64
from io import BytesIO

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from omegaconf import DictConfig

from sgaps.core.session_manager import SessionManager, Session
from sgaps.core.sampler import FixedUVSampler
from sgaps.core.reconstructor import FrameReconstructor
from sgaps.utils.metrics import compute_all_metrics, calculate_psnr, calculate_ssim, calculate_mse

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

# Debug mode state
_debug_visualizer = None
_frame_counter_per_client: Dict[str, int] = {}
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
        """Handle client disconnection and trigger post-session processing."""
        session_to_process = None
        async with self._lock:
            if client_id in self.active_connections:
                del self.active_connections[client_id]
            if client_id in self.sessions:
                session_to_process = self.sessions.pop(client_id)

        if session_to_process:
            # End the session (closes storage, etc.)
            await self.session_manager.end_session(session_to_process)
            logger.info(f"Client {client_id} disconnected and session ended.")

            # If debug mode was enabled, process the buffered data asynchronously
            if session_to_process.debug_enabled and session_to_process.debug_buffer:
                logger.info(f"Scheduling post-session debug visualization for {client_id}.")
                asyncio.create_task(process_debug_visualizations(session_to_process))
        else:
            logger.info(f"Client {client_id} disconnected without an active session.")

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

async def process_debug_visualizations(session: Session):
    """
    Asynchronously process and log all debug visualizations for a completed session.
    Uses run_in_executor to prevent blocking the event loop during matplotlib rendering.
    """
    global _debug_visualizer
    if _debug_visualizer is None or not USE_WANDB:
        logger.warning("Cannot process debug visualizations: Visualizer or WandB not available.")
        return

    # Verify WandB run exists (should be initialized in main.py)
    if wandb.run is None:
        logger.error("WandB run is not active. Cannot log debug visualizations.")
        return

    logger.info(f"Processing {len(session.debug_buffer)} buffered debug frames for session {session.client_id}.")

    # Get event loop for executor
    loop = asyncio.get_running_loop()
    
    for i, data in enumerate(session.debug_buffer):
        # Use pre-allocated step from buffer time (ensures monotonically increasing order)
        global_step = data.get('global_step', i)  # Fallback to index if not present
        
        try:
            # Run matplotlib visualization in thread pool to avoid blocking event loop
            composite_img = await loop.run_in_executor(
                None,  # Use default ThreadPoolExecutor
                lambda d=data: _debug_visualizer.create_composite_dashboard(
                    original_frame=d["original_frame"],
                    reconstructed_frame=d["reconstructed_frame"],
                    sampled_pixels=d["sampled_pixels"],
                    state_vector=d["state_vector"],
                    importance_map=d["importance_map"],
                    attention_weights=d["attention_weights"],
                    metadata=d["metadata"]
                )
            )

            # Log to WandB using global step (monotonically increasing)
            wandb.log({
                f"Debug/{session.client_id}/Dashboard": wandb.Image(
                    composite_img,
                    caption=f"Frame {data['metadata']['frame_id']} | PSNR={data['psnr']:.2f} SSIM={data['ssim']:.3f}"
                ),
                f"Metrics/{session.client_id}/PSNR": data['psnr'],
                f"Metrics/{session.client_id}/SSIM": data['ssim'],
                f"Metrics/{session.client_id}/MSE": data['mse'],
            }, step=global_step)
            logger.info(f"Logged post-session viz for frame {data['metadata']['frame_id']} at step {global_step}")
        except Exception as e:
            logger.error(f"Error creating/logging debug visualization for frame {data.get('metadata', {}).get('frame_id', 'unknown')}: {e}", exc_info=True)
    
    logger.info(f"Finished processing debug visualizations for session {session.client_id}.")


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
    elif message_type == "frame_data_debug":
        await handle_frame_data_debug(client_id, payload)
    elif message_type == "heartbeat":
        await manager.send_json(client_id, {"type": "heartbeat_ack", "payload": {"timestamp": time.time()}})
    else:
        logger.warning(f"Unknown message type from {client_id}: {message_type}")
        await manager.send_json(client_id, {"type": "error", "payload": {"message": "Unknown message type"}})

async def handle_session_start(client_id: str, payload: dict):
    global _debug_visualizer, _frame_counter_per_client

    logger.info(f"Starting session for {client_id} with payload: {payload}")
    session = await manager.create_session(client_id, payload)
    cfg = get_server_config()
    reconstructor = get_reconstructor()

    # The new reconstructor handles model loading internally via get_model
    # We can check if the model exists to set the 'checkpoint_loaded' flag
    model_path = Path(cfg.paths.checkpoint_dir) / session.checkpoint_key / "best.pth"
    checkpoint_loaded = model_path.exists()

    # NEW: Check if client requested debug mode
    client_debug_request = payload.get("debug", False)

    # NEW: Determine if debug mode should be enabled for this session
    # Honor client request only if server has debug capabilities enabled
    server_debug_available = cfg.debug.enabled
    debug_enabled = client_debug_request and server_debug_available

    # Log decision
    if client_debug_request and not server_debug_available:
        logger.warning(f"Client {client_id} requested debug mode but server debug is disabled in config.")
    elif debug_enabled:
        logger.info(f"Debug mode enabled for session {client_id} (client requested, server available).")

    # Store debug mode in session for later use
    session.debug_enabled = debug_enabled

    # Initialize debug visualizer if enabled and not already initialized
    if debug_enabled and _debug_visualizer is None:
        from sgaps.utils.visualization import DebugVisualizer
        _debug_visualizer = DebugVisualizer(cfg)
        logger.info("Debug visualizer initialized")

    # Initialize frame counter for this client
    if debug_enabled:
        _frame_counter_per_client[client_id] = 0

    # Generate initial UV coordinates based on sampling strategy
    if cfg.sampling.pattern in ["adaptive_importance", "hybrid"]:
        from sgaps.core.sampler import AdaptiveUVSampler
        sampler = AdaptiveUVSampler(
            config=cfg.sampling,
            resolution=session.resolution
        )
        logger.info(f"Using AdaptiveUVSampler for session {client_id}")
    else:
        # Default to uniform sampling
        sampler = FixedUVSampler(
            sample_count=cfg.sampling.default_sample_count,
            resolution=session.resolution
        )
        logger.info(f"Using FixedUVSampler for session {client_id}")

    initial_coords = sampler.get_current_coordinates()
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
            "resolution": list(session.resolution),
            "debug_mode": debug_enabled  # Send the session-specific debug status
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

    # 3.5. Update adaptive sampler with importance map (if applicable)
    if attn_weights is not None and hasattr(session.sampler, 'update_from_importance'):
        from sgaps.core.importance import AttentionEntropyImportanceCalculator

        try:
            # Calculate importance map
            importance_calc = AttentionEntropyImportanceCalculator(cfg)
            importance_map = importance_calc.calculate(
                attn_weights,
                session.resolution
            )[0]  # First batch item

            # Update sampler for next frame
            session.sampler.update_from_importance(importance_map)

            # Log importance stats to WandB (if enabled)
            if USE_WANDB:
                importance_stats = importance_calc.compute_statistics(importance_map)
                wandb.log({
                    f"Importance/{client_id}/mean": importance_stats["mean"],
                    f"Importance/{client_id}/max": importance_stats["max"],
                    f"Importance/{client_id}/std": importance_stats["std"],
                    f"Importance/{client_id}/entropy": importance_stats["importance_entropy"]
                }, step=global_step)

        except Exception as e:
            logger.error(f"Error calculating importance map: {e}. Using static sampling.")

    # 4. Log to WandB (if enabled) - organized by client_id folder
    if USE_WANDB and wandb.run:
        log_payload = {
            f"Session/{client_id}/Frame_ID": frame_id,
        }
        if reconstructed_frame is not None:
             # Only log reconstruction for non-debug sessions here
            if not session.debug_enabled:
                 log_payload[f"Live/{client_id}/Reconstruction"] = wandb.Image(Image.fromarray(reconstructed_frame))
        
        wandb.log(log_payload, step=global_step)

    # 5. Send back next UV coordinates (updated by adaptive sampler if applicable)
    if session.sampler:
        coords = session.sampler.get_current_coordinates()
        await manager.send_json(client_id, {
            "type": "uv_coordinates",
            "payload": {"target_frame_id": frame_id + 1, "coordinates": [{"u": u, "v": v} for u, v in coords]}
        })

    session.frame_count += 1
    if session.frame_count % 100 == 0:
        logger.info(f"Session {client_id}: processed {session.frame_count} frames.")


async def handle_frame_data_debug(client_id: str, payload: dict):
    """Handle frame_data_debug messages by buffering data for post-session processing."""
    global _frame_counter_per_client

    session = manager.get_session(client_id)
    if not session:
        logger.error(f"No session found for client {client_id}")
        return

    if not session.debug_enabled:
        logger.warning(f"Received debug frame data but debug mode not enabled for session {client_id}")
        return await handle_frame_data(client_id, payload)

    cfg = get_server_config()
    frame_id = payload.get("frame_id", 0)

    # --- Real-time processing (Reconstruction & Sampling) ---
    pixels = np.array([(p['u'], p['v'], p['value']) for p in payload.get("pixels", [])], dtype=np.float32)
    state_vector_raw = np.array(payload.get("state_vector", []), dtype=np.float32)

    sentinel_value = cfg.model.sentinel_value
    max_state_dim = cfg.model.input_constraints.max_state_dim
    state_vector = np.full(max_state_dim, sentinel_value, dtype=np.float32)
    if len(state_vector_raw) > 0:
        copy_len = min(len(state_vector_raw), max_state_dim)
        state_vector[:copy_len] = state_vector_raw[:copy_len]
        
    reconstructor = get_reconstructor()
    reconstructed_frame, attn_weights = await reconstructor.reconstruct(
        pixels, state_vector, session.resolution, session.checkpoint_key
    )

    importance_map = None
    if attn_weights is not None and hasattr(session.sampler, 'update_from_importance'):
        from sgaps.core.importance import AttentionEntropyImportanceCalculator
        try:
            importance_calc = AttentionEntropyImportanceCalculator(cfg)
            importance_map = importance_calc.calculate(attn_weights, session.resolution)[0]
            session.sampler.update_from_importance(importance_map)
        except Exception as e:
            logger.error(f"Error calculating importance for debug frame: {e}", exc_info=True)

    # --- Image Decoding & Storage ---
    original_frame = None
    full_frame_base64 = payload.get("full_frame_base64", None)
    if full_frame_base64:
        try:
            full_frame_bytes = base64.b64decode(full_frame_base64)
            pil_image = Image.open(BytesIO(full_frame_bytes))
            original_frame = np.array(pil_image)
            original_frame = np.flipud(original_frame)
        except Exception as e:
            logger.error(f"Failed to decode full frame: {e}")

    # Store frame data (including image if available) to HDF5
    if session.storage:
        asyncio.create_task(
            session.storage.store_frame(
                frame_id=frame_id,
                pixels=payload.get("pixels", []),
                state_vector=state_vector.tolist(),
                resolution=session.resolution,
                image=original_frame
            )
        )

    # --- Buffering for post-session visualization ---
    current_frame_count = _frame_counter_per_client.get(client_id, 0) + 1
    _frame_counter_per_client[client_id] = current_frame_count
    log_every_n = cfg.debug.visualization.log_every_n_frames
    should_buffer_visualization = (current_frame_count % log_every_n == 0)

    if should_buffer_visualization:
        # Pre-allocate global step for WandB at buffer time (not at log time)
        async with manager._lock:
            global_step = manager.global_wandb_step
            manager.global_wandb_step += 1
        
        # original_frame is already decoded above

        if original_frame is not None and reconstructed_frame is not None:
            # Resize original_frame to match reconstructed_frame for metrics
            if original_frame.shape != reconstructed_frame.shape:
                pil_orig = Image.fromarray(original_frame)
                pil_resized = pil_orig.resize(reconstructed_frame.shape[::-1], Image.BILINEAR)
                original_frame_resized = np.array(pil_resized)
            else:
                original_frame_resized = original_frame

            psnr = calculate_psnr(original_frame_resized, reconstructed_frame)
            ssim = calculate_ssim(original_frame_resized, reconstructed_frame)
            mse = calculate_mse(original_frame_resized, reconstructed_frame)

            importance_map_np = None
            if importance_map is not None:
                import torch
                if isinstance(importance_map, torch.Tensor):
                    importance_map_np = importance_map.cpu().numpy()
                else:
                    importance_map_np = importance_map
            
            # Detach attention weights to CPU/Numpy to prevent GPU OOM
            attn_weights_np = None
            if attn_weights is not None:
                import torch
                if isinstance(attn_weights, torch.Tensor):
                    attn_weights_np = attn_weights.detach().cpu().numpy()
                elif isinstance(attn_weights, (list, tuple)):
                    attn_weights_np = [t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t for t in attn_weights]
                else:
                    attn_weights_np = attn_weights

            session.debug_buffer.append({
                "global_step": global_step,  # Pre-allocated step for WandB logging
                "original_frame": original_frame,
                "reconstructed_frame": reconstructed_frame,
                "sampled_pixels": pixels,
                "state_vector": state_vector,
                "importance_map": importance_map_np,
                "attention_weights": attn_weights_np,
                "metadata": {"frame_id": frame_id, "timestamp": payload.get("timestamp", 0.0), "client_id": client_id},
                "psnr": psnr,
                "ssim": ssim,
                "mse": mse,
            })
            logger.info(f"[Debug] Buffered frame {frame_id} for post-session visualization.")

    # --- Always send back next coordinates ---
    if session.sampler:
        coords = session.sampler.get_current_coordinates()
        await manager.send_json(client_id, {
            "type": "uv_coordinates",
            "payload": {"target_frame_id": frame_id + 1, "coordinates": [{"u": u, "v": v} for u, v in coords]}
        })

    session.frame_count += 1
    if session.frame_count % 100 == 0:
        logger.info(f"Session {client_id}: processed {session.frame_count} frames.")