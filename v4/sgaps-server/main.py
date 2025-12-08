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
from omegaconf import DictConfig

from sgaps.api.websocket import router as ws_router, set_server_config, set_reconstructor
from sgaps.api.rest import router as rest_router
from sgaps.core.reconstructor import FrameReconstructor


# Global config holder
_config: DictConfig = None


def get_config() -> DictConfig:
    """Get the current configuration."""
    return _config


def load_default_config() -> DictConfig:
    """Load default configuration using Hydra's compose API."""
    from hydra import compose, initialize_config_dir

    # Get absolute path to config directory
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf")

    # Initialize Hydra with the config directory
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        # Compose configuration - Hydra handles all the defaults merging
        cfg = compose(config_name="config")

    return cfg


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    logging.info("SGAPS-MAE Server starting up...")

    # Initialize and set the reconstructor for the WebSocket API
    try:
        reconstructor = FrameReconstructor(_config)
        set_reconstructor(reconstructor)
        logging.info("FrameReconstructor initialized and set for WebSocket API.")
    except Exception as e:
        logging.error(f"Failed to initialize FrameReconstructor: {e}", exc_info=True)

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

    # Initialize Weights & Biases if configured
    if cfg.logging.type == "wandb":
        try:
            import wandb
            wandb.init(
                project=cfg.logging.wandb.project,
                entity=cfg.logging.wandb.entity,
                name=cfg.logging.wandb.name,
                config=dict(cfg) if cfg else {}
            )
            logging.info("Weights & Biases initialized successfully.")
        except ImportError:
            logging.warning("WandB not installed. Skipping initialization.")
        except Exception as e:
            logging.error(f"Failed to initialize Weights & Biases: {e}")
    
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
