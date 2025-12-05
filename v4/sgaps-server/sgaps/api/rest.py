"""
REST API endpoints for SGAPS-MAE Server.

Provides health checks, status information, and model management endpoints.
"""

from fastapi import APIRouter, Response
from pydantic import BaseModel
from typing import Dict, Any, Optional
import time


router = APIRouter()

# Server start time for uptime calculation
_start_time = time.time()


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    uptime_seconds: float
    version: str


class StatusResponse(BaseModel):
    """Server status response model."""
    status: str
    uptime_seconds: float
    active_sessions: int
    total_frames_processed: int
    checkpoints_loaded: Dict[str, bool]


# Global counters (will be updated by session manager)
_active_sessions = 0
_total_frames = 0
_checkpoints: Dict[str, bool] = {}


def update_stats(active_sessions: int, total_frames: int, checkpoints: Dict[str, bool]):
    """Update global statistics (called by session manager)."""
    global _active_sessions, _total_frames, _checkpoints
    _active_sessions = active_sessions
    _total_frames = total_frames
    _checkpoints = checkpoints


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns server status and uptime information.
    """
    return HealthResponse(
        status="healthy",
        uptime_seconds=time.time() - _start_time,
        version="0.1.0"
    )


@router.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    """
    Get detailed server status.
    
    Returns information about active sessions, processed frames,
    and loaded model checkpoints.
    """
    return StatusResponse(
        status="running",
        uptime_seconds=time.time() - _start_time,
        active_sessions=_active_sessions,
        total_frames_processed=_total_frames,
        checkpoints_loaded=_checkpoints
    )


@router.get("/ready")
async def readiness_check() -> Dict[str, str]:
    """
    Readiness check for container orchestration.
    
    Returns 200 if the server is ready to accept connections.
    """
    return {"status": "ready"}
