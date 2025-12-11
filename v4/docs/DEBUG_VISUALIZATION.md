# Debug Visualization System for SGAPS-MAE

## Overview

The debug visualization system provides comprehensive monitoring of the SGAPS-MAE pipeline through WandB (Weights & Biases). When enabled, it creates a high-resolution 2x4 grid dashboard (~3000×1500 pixels) using **matplotlib subplots** - the standard approach for ML visualization.

**Key Benefits:**
- **High-resolution output**: 20×10 inches at 150 DPI = ~3000×1500 pixels
- **Clear titles & labels**: Each panel has proper matplotlib titles
- **Colorbars**: All heatmaps include colorbar scales
- **Quality monitoring**: PSNR, SSIM, MSE displayed in figure suptitle
- **State vector chart**: Proper bar chart with axes labels

## Visualizations

The dashboard contains 8 panels arranged in a 2x4 grid:

### Row 1: Frame Comparison
1. **Original Frame**: Ground truth grayscale frame
2. **Reconstructed Frame**: SPT model output
3. **Difference Map**: Absolute difference with colorbar (colormap: viridis)
4. **Sampled Pixels**: Red scatter dots showing sampled pixel locations

### Row 2: Analysis Heatmaps
5. **Importance Heatmap**: Attention entropy-based importance with colorbar
6. **Loss Map**: Pixel-wise MSE heatmap with colorbar
7. **Attention Visualization**: Average cross-attention weights with colorbar
8. **State Vector**: Bar chart (green=positive, red=negative) with axis labels

## Configuration

### Enabling Debug Mode

**Server Configuration** (`sgaps-server/conf/config.yaml`):
```yaml
defaults:
  - server: development
  - sampling: adaptive
  - debug: enabled  # Set to 'disabled' to turn off
  - ...
```

**Unity Client** (`SGAPSManager` Inspector):
- Check **"Debug Mode"** checkbox in the Debug section

### Debug Configuration Options

Edit `sgaps-server/conf/debug/enabled.yaml`:

```yaml
enabled: true

# Visualization settings - Matplotlib-based dashboard
visualization:
  # Log every N frames (1 = every frame, 10 = every 10th frame)
  log_every_n_frames: 10

  # Matplotlib figure settings for high-resolution output
  figure:
    figsize: [20, 10]  # inches (width, height) - ~3000x1500px
    dpi: 150           # High DPI for WandB clarity
    style: "dark_background"  # Matplotlib style
  
  # Layout: 2 rows, 4 columns
  layout:
    rows: 2
    cols: 4
  
  # Colormap for heatmaps (viridis, plasma, inferno, magma, jet, hot)
  colormap: "viridis"
  
  # Font sizes
  font:
    suptitle: 16
    title: 14
    label: 11
    tick: 9

# Storage settings (optional disk save)
storage:
  save_to_disk: false
  output_dir: "./debug_output"

# WandB logging settings
wandb:
  log_composite: true
  log_individual: false  # Also log individual components separately
  log_metrics: true  # Log PSNR, SSIM, MSE
```

## How It Works

### Data Flow

```
Unity Client (Debug Mode ON)
  ↓ Full grayscale frame (PNG/Base64) + sparse pixels + state vector
Server (websocket.py)
  ↓ Decode PNG from Base64
  ↓ Reconstruct from sparse pixels → reconstructed_frame
  ↓ Calculate importance map from attention weights
  ↓ Compute loss map (original vs reconstructed)
  ↓ Create 7 individual visualizations
  ↓ Composite into 2x4 grid
  ↓ WandB.log() every N frames
```

### Debug Mode Activation Logic

Debug mode only activates when **BOTH** client and server have debug enabled:

```
Client Debug=ON  + Server Debug=ON  → Full frames transmitted
Client Debug=ON  + Server Debug=OFF → Only sparse pixels (warning logged)
Client Debug=OFF + Server Debug=ON  → Only sparse pixels
Client Debug=OFF + Server Debug=OFF → Only sparse pixels
```

### HDF5 Storage Behavior

**Important**: Debug mode does **NOT** change HDF5 storage behavior. Only sparse pixels and state vectors are stored, regardless of debug mode. Full frames are used **only** for visualization and are never stored to disk.

## Usage

### 1. Start Server with Debug Mode

```bash
cd sgaps-server
python main.py debug=enabled
```

### 2. Connect Unity Client

1. Open Unity project
2. Select `SGAPSManager` GameObject in Hierarchy
3. In Inspector, check **"Debug Mode"** under Debug section
4. Click **"Connect On Start"** (or call `ConnectToServer()` via script)
5. Enter Play Mode

### 3. Monitor WandB Dashboard

Navigate to your WandB dashboard at `https://wandb.ai/<your-project>`

**Key Panels to Monitor:**
- `Debug/<client_id>/Dashboard`: 2x4 composite grid (logged every N frames)
- `Metrics/<client_id>/PSNR`: Peak Signal-to-Noise Ratio (higher = better, target > 39.2 dB)
- `Metrics/<client_id>/SSIM`: Structural Similarity Index (higher = better, target > 0.95)
- `Metrics/<client_id>/MSE`: Mean Squared Error (lower = better)
- `Importance/<client_id>/mean`: Average importance across frame
- `Importance/<client_id>/entropy`: Importance distribution entropy

## Performance Impact

### Overhead Breakdown (per logged frame)

| Operation | Time |
|-----------|------|
| PNG Encoding (Unity) | ~10-20ms |
| Base64 Encoding | ~5ms |
| Network Transmission (224x224 PNG) | ~100ms @ 1Mbps |
| PNG Decoding (Server) | ~10ms |
| Visualization Creation | ~50-100ms |
| WandB Upload (async) | ~0ms (non-blocking) |
| **Total** | ~175-235ms |

### Mitigation Strategies

1. **Log Every N Frames**: Default setting of `log_every_n_frames: 10` reduces overhead by 90%
2. **Resolution Scaling**: Target resolution of 224x224 keeps PNG size manageable (~50-100KB)
3. **Async Operations**: HDF5 storage and WandB uploads run in background
4. **Conditional Activation**: Debug mode can be toggled without code changes

### Expected Performance

- **Normal Mode** (sparse pixels only): ~2-5ms overhead per frame
- **Debug Mode** (every 10th frame): ~20-30ms overhead per frame (amortized)
- **Network Bandwidth**:
  - Normal: ~2-3 KB/frame
  - Debug: ~50-100 KB/frame (only for logged frames)

## Troubleshooting

### Issue: "Received debug frame data but debug mode is disabled"
**Cause**: Client has debug=true but server has debug=false
**Solution**: Enable debug on server side (`conf/config.yaml` → `debug: enabled`)

### Issue: Dashboard images are blank/corrupted
**Cause**: PNG encoding/decoding failure
**Solution**: Check Unity console for errors in `CaptureFullFrameAsPNG()`. Verify `TextureFormat.R8` is supported.

### Issue: State vector bars all show as zero
**Cause**: State vector contains only sentinel values (-999.0)
**Solution**: Verify `StateVectorCollector.SetState()` is being called with valid data

### Issue: Importance heatmap is all black
**Cause**: Attention weights not being returned from model
**Solution**: Check model forward pass returns `attn_weights` in addition to reconstruction

### Issue: Visualization creation is very slow (>200ms)
**Cause**: Matplotlib figure creation overhead
**Solution**: Reduce `target_resolution` to [112, 112] in debug config

### Issue: WandB dashboard shows "Out of sync" warning
**Cause**: Multiple clients logging to same step
**Solution**: This is expected behavior with multiple clients. Use `Session/Client_ID` filter to view per-client data.

## Advanced Usage

### Custom Colormap

Edit `conf/debug/enabled.yaml`:
```yaml
visualization:
  colormap: "plasma"  # Options: viridis, plasma, inferno, magma, jet, hot, cool
```

### Per-Client Logging Frequency

Currently global. For per-client control, modify `websocket.py` to check session-specific config.

### Logging Additional Metrics

Add custom metrics in `handle_frame_data_debug()`:

```python
wandb.log({
    f"Custom/{client_id}/my_metric": value,
    ...
}, step=global_step)
```

### Saving Visualizations to Disk

Enable in `conf/debug/enabled.yaml`:
```yaml
storage:
  save_to_disk: true
  output_dir: "./debug_output"
```

Then modify `handle_frame_data_debug()` to save composite_img:
```python
composite_img.save(f"{cfg.debug.storage.output_dir}/frame_{frame_id:06d}.png")
```

## API Reference

### DebugVisualizer Class

**Location**: `sgaps-server/sgaps/utils/visualization.py`

#### `create_composite_dashboard()`

Creates a 2x4 grid composite visualization.

**Parameters:**
- `original_frame: np.ndarray` - Ground truth grayscale [H, W]
- `reconstructed_frame: np.ndarray` - SPT output [H, W]
- `sampled_pixels: np.ndarray` - UV coordinates and values [N, 3] (u, v, value)
- `state_vector: np.ndarray` - Game state [max_state_dim]
- `importance_map: np.ndarray` - Entropy-based importance [H, W]
- `attention_weights: Optional[torch.Tensor]` - Cross-attention weights
- `metadata: Optional[Dict]` - Additional info (frame_id, timestamp, etc.)

**Returns:**
- `Image.Image` - PIL Image of composite dashboard

### Configuration Schema

**Debug Config** (`conf/debug/enabled.yaml`):
```python
{
  "enabled": bool,
  "visualization": {
    "log_every_n_frames": int,
    "layout": str,
    "show_*": bool,
    "target_resolution": [int, int],
    "colormap": str,
    "state_vector": {
      "method": str,
      "scaling": str,
      "show_sentinel": bool,
      "max_display_dims": int
    }
  },
  "storage": {
    "save_to_disk": bool,
    "output_dir": str
  },
  "wandb": {
    "log_composite": bool,
    "log_individual": bool,
    "log_metrics": bool
  }
}
```

## Best Practices

1. **Start with N=10**: Default `log_every_n_frames: 10` is a good balance between insight and performance
2. **Monitor PSNR/SSIM**: These metrics should improve during training. Target PSNR > 39.2 dB, SSIM > 0.95
3. **Check Importance Map**: Verify adaptive sampling focuses on high-entropy regions (edges, text, motion)
4. **Validate State Vector**: Ensure non-sentinel values align with game state (position, velocity, flags)
5. **Disable in Production**: Always use `debug: disabled` for production deployment

## Examples

### Example WandB Dashboard Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ Debug/client_abc123/Dashboard                                   │
│ ┌───────────┬───────────┬───────────┬───────────┐               │
│ │ Original  │Reconstruct│ Diff Map  │  Sampled  │               │
│ │  Frame    │   Frame   │           │  Pixels   │               │
│ ├───────────┼───────────┼───────────┼───────────┤               │
│ │Importance │ Loss Map  │ Attention │   State   │               │
│ │  Heatmap  │           │           │  Vector   │               │
│ └───────────┴───────────┴───────────┴───────────┘               │
│ Caption: Frame 1234 | PSNR=40.5 SSIM=0.96                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Metrics/client_abc123                                           │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                │
│ │ PSNR: 40.5  │ │ SSIM: 0.96  │ │ MSE: 8.2    │                │
│ └─────────────┘ └─────────────┘ └─────────────┘                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Importance/client_abc123                                        │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                │
│ │ Mean: 0.42  │ │ Max: 0.95   │ │Entropy: 3.2 │                │
│ └─────────────┘ └─────────────┘ └─────────────┘                │
└─────────────────────────────────────────────────────────────────┘
```

### Example Debug Output Log

```
[INFO] Debug visualizer initialized
[INFO] Session for client_abc123 configured with debug_mode=True
[INFO] Client resolution: [1920, 1080], Server reconstruction: (224, 224)
[INFO] [Debug] Logged visualization for frame 10 (PSNR=39.8, SSIM=0.94)
[INFO] [Debug] Logged visualization for frame 20 (PSNR=40.2, SSIM=0.95)
[INFO] [Debug] Logged visualization for frame 30 (PSNR=40.5, SSIM=0.96)
```

## See Also

- [IMPLEMENTATION_STATUS.md](./IMPLEMENTATION_STATUS.md) - Current implementation status
- [PHASE3_ADAPTIVE_SAMPLING.md](./PHASE3_ADAPTIVE_SAMPLING.md) - Adaptive sampling details
- [API_SPECIFICATION.md](./API_SPECIFICATION.md) - WebSocket protocol
- [CONFIGURATION.md](./CONFIGURATION.md) - Full configuration reference
