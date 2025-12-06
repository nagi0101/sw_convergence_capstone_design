# SGAPS-MAE v4 Implementation Plan

## Overview

This document outlines the phased implementation plan for SGAPS-MAE (Sparse Game-Aware Pixel Sampling with Masked Autoencoder).

## Project Structure

```
v4/
├── sgaps-server/                   # Python FastAPI Server
│   ├── main.py                     # Entry point
│   ├── requirements.txt
│   ├── conf/                       # Hydra configuration
│   │   ├── config.yaml
│   │   ├── server/
│   │   ├── sampling/
│   │   └── mask_update/
│   └── sgaps/
│       ├── api/                    # WebSocket & REST handlers
│       ├── core/                   # Session, sampler, reconstructor
│       ├── data/                   # HDF5 storage, dataset
│       └── utils/                  # Metrics, helpers
│
└── unity-client/                   # Unity UPM Package
    ├── package.json
    ├── README.md
    ├── Runtime/
    │   ├── SGAPS.Runtime.asmdef
    │   ├── Scripts/
    │   │   ├── Core/               # Main components
    │   │   ├── Data/               # Data structures
    │   │   └── Utilities/
    │   └── Shaders/
    ├── Editor/
    │   ├── SGAPS.Editor.asmdef
    │   └── Scripts/
    └── Samples~/
```

## Phase 1: Infrastructure (Current Phase)

### Server (Python)

-   [x] FastAPI application with Hydra configuration
-   [x] WebSocket endpoint for real-time streaming
-   [x] Session management for multiple clients
-   [x] HDF5 storage for data collection
-   [x] Basic OpenCV inpainting reconstruction (placeholder)
-   [x] Fixed uniform grid UV sampling

### Client (Unity)

-   [x] UPM package structure
-   [x] RenderTexture frame capture with grayscale shader
-   [x] Pixel sampling at UV coordinates
-   [x] WebSocket client for server communication
-   [x] State vector collection with sentinel values
-   [x] SGAPSManager MonoBehaviour for coordination

### Key Specifications

| Parameter            | Value   |
| -------------------- | ------- |
| MAX_STATE_DIM        | 64      |
| SENTINEL_VALUE       | -999.0  |
| Default Resolution   | 640x480 |
| Default Sample Count | 500     |
| Target FPS           | 10      |

## Phase 2: MAE Training (Future)

-   Implement MAE encoder/decoder architecture
-   Training pipeline with collected data
-   Integration with reconstruction module
-   Model checkpointing per game/level

## Phase 3: Adaptive Sampling (Future)

-   **Attention Entropy-based importance calculation** (replaces Dropout-based uncertainty)
-   Dynamic UV coordinate generation based on importance maps
-   Quality feedback loop

### Key Design Decisions

#### Loss Function: Sampled Pixel L2 Loss

For fast convergence and PoC validation, we use simple MSE only on sampled pixel positions:

$$L_{total} = \frac{1}{N_{sampled}} \sum_{i \in P_{sampled}} || I_{pred}(x_i) - I_{gt}(x_i) ||_2^2$$

This ensures the model accurately reconstructs at least the sampled positions, enabling generalization to unsampled regions.

#### Importance Calculation: Attention Entropy

Instead of computationally expensive Monte Carlo Dropout, we leverage the model's existing **Decoder Cross-Attention** weights:

-   **Low Entropy**: Model confidently references specific sparse pixels → Low importance
-   **High Entropy**: Model uncertain about which sparse pixels to reference → High importance (needs more samples)

This provides importance maps with near-zero additional computation cost.

## Running the System

### Server

```bash
cd v4/sgaps-server
pip install -r requirements.txt
python main.py
```

Server will start at `ws://localhost:8000/ws/stream`

### Unity Client

1. Open Unity Package Manager
2. Add package from disk: `v4/unity-client/package.json`
3. Add `SGAPSManager` component to a GameObject
4. Configure server endpoint and capture settings
5. Press Play and click Connect

## WebSocket Protocol

### Client → Server

```json
// Session initialization
{
  "type": "session_start",
  "payload": {
    "checkpoint_key": "game_level_1",
    "max_state_dim": 64,
    "resolution": [640, 480]
  }
}

// Frame data
{
  "type": "frame_data",
  "payload": {
    "frame_id": 0,
    "timestamp": 1.234,
    "resolution": [640, 480],
    "num_pixels": 500,
    "pixels": [{"u": 0.5, "v": 0.5, "value": 128}, ...],
    "state_vector": [0.1, 0.2, ...]
  }
}
```

### Server → Client

```json
// UV coordinates for next frame
{
  "type": "uv_coordinates",
  "payload": {
    "target_frame_id": 1,
    "coordinates": [{"u": 0.1, "v": 0.2}, ...]
  }
}
```

## Dependencies

### Python Server

-   Python 3.10+
-   FastAPI, uvicorn
-   PyTorch 2.0+
-   Hydra-core
-   h5py
-   OpenCV (cv2)
-   numpy

### Unity Client

-   Unity 2021.3 LTS+
-   Newtonsoft.Json (com.unity.nuget.newtonsoft-json)
