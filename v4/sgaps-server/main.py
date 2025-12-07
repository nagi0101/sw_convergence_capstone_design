"""
SGAPS-MAE Server - Main Entry Point

FastAPI server with WebSocket support for real-time game frame streaming
and sparse pixel reconstruction.

Usage:
    # With Hydra config
    python main.py
    python main.py server.port=8080
    
    # With uvicorn (for development with auto-reload)
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from omegaconf import DictConfig, OmegaConf

from sgaps.api.websocket import router as ws_router, set_server_config
from sgaps.api.rest import router as rest_router


# Global config holder
_config: DictConfig = None


def get_config() -> DictConfig:
    """Get the current configuration."""
    return _config


def load_default_config() -> DictConfig:
    """Load default configuration from yaml files."""
    from omegaconf import OmegaConf
    
    config_path = os.path.join(os.path.dirname(__file__), "conf")
    
    # Load base config
    base_config = OmegaConf.load(os.path.join(config_path, "config.yaml"))
    
    # Load server config
    defaults = base_config.get("defaults", [])
    server_config_name = "development"
    sampling_config_name = "uniform"
    
    for default in defaults:
        if isinstance(default, dict):
            if "server" in default:
                server_config_name = default["server"]
            if "sampling" in default:
                sampling_config_name = default["sampling"]
    
    server_config_path = os.path.join(config_path, "server", f"{server_config_name}.yaml")
    if os.path.exists(server_config_path):
        server_config = OmegaConf.load(server_config_path)
        base_config.server = OmegaConf.merge(base_config.get("server", {}), server_config)
    
    # Load sampling config
    sampling_config_path = os.path.join(config_path, "sampling", f"{sampling_config_name}.yaml")
    if os.path.exists(sampling_config_path):
        sampling_config = OmegaConf.load(sampling_config_path)
        base_config.sampling = OmegaConf.merge(base_config.get("sampling", {}), sampling_config)
    
    return base_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    logging.info("SGAPS-MAE Server starting up...")
    if _config:
        logging.info(f"Configuration:\n{OmegaConf.to_yaml(_config)}")
    yield
    logging.info("SGAPS-MAE Server shutting down...")


def create_app(cfg: DictConfig = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    global _config
    
    if cfg is None:
        cfg = load_default_config()
    _config = cfg
    
    # Set server config for WebSocket handlers
    set_server_config(cfg)
    
    # Logging is configured by uvicorn, which is passed the log_level from config.
    # BasicConfig can interfere with uvicorn's more sophisticated logging setup.
    
    application = FastAPI(
        title="SGAPS-MAE Server",
        description="Sparse Game-Aware Pixel Sampling with Masked Autoencoder",
        version="0.1.0",
        lifespan=lifespan
    )
    
    # CORS middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    application.include_router(rest_router, prefix="/api", tags=["api"])
    application.include_router(ws_router, prefix="/ws", tags=["websocket"])
    
    return application


# Global app instance for uvicorn direct import
app = create_app()


def main() -> None:
    """Main entry point - runs with Hydra configuration if available."""
    import uvicorn
    
    try:
        import hydra
        
        @hydra.main(version_base=None, config_path="conf", config_name="config")
        def run_with_hydra(cfg: DictConfig) -> None:
            global app
            app = create_app(cfg)
            uvicorn.run(
                app,
                host=cfg.server.host,
                port=cfg.server.port,
                log_level=cfg.server.log_level.lower()
            )
        
        run_with_hydra()
    except ImportError:
        # Hydra not available, use default config
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )


if __name__ == "__main__":
    main()
