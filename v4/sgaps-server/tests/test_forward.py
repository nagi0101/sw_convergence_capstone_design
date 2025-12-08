"""
SPT Forward Pass Test Script
Tests the forward method with dummy data to verify implementation.
"""
import torch
from sgaps.models.spt import SparsePixelTransformer

def test_forward():
    print("=" * 60)
    print("Testing SPT Forward Pass")
    print("=" * 60)

    # 1. Create minimal config
    config = {
        "model": {
            "architecture": {
                "embed_dim": 256,
                "num_heads": 8,
                "num_encoder_layers": 2,  # Reduced for testing
                "num_decoder_layers": 2,  # Reduced for testing
                "feedforward_dim": 1024,
                "dropout": 0.1
            },
            "input_constraints": {
                "max_state_dim": 64
            },
            "sentinel_value": -999.0,
            "positional_encoding": {
                "max_freq": 10
            },
            "refinement_head": {
                "channels": [128, 64, 32]
            }
        }
    }

    # 2. Initialize model
    print("\n[1/5] Initializing model...")
    model = SparsePixelTransformer(config)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model on device: {device}")

    # 3. Create dummy input
    print("\n[2/5] Creating dummy input...")
    B = 2  # Batch size
    N = 100  # Number of sparse pixels
    max_state_dim = 64
    H, W = 224, 224  # Resolution

    # Random sparse pixels (u, v, value)
    sparse_pixels = torch.rand(B, N, 3, device=device)

    # Random state vector with some sentinels
    state_vector = torch.rand(B, max_state_dim, device=device)
    state_vector[:, 32:] = -999.0  # Half are sentinel

    # State mask
    state_mask = (state_vector != -999.0).float()

    resolution = (H, W)

    print(f"  sparse_pixels: {sparse_pixels.shape}")
    print(f"  state_vector: {state_vector.shape}")
    print(f"  state_mask: {state_mask.shape}")
    print(f"  resolution: {resolution}")

    # 4. Forward pass without attention
    print("\n[3/5] Running forward pass (no attention)...")
    with torch.no_grad():
        output = model(sparse_pixels, state_vector, state_mask, resolution, return_attention=False)

    print(f"  Output shape: {output.shape}")
    print(f"  Expected: [{B}, 1, {H}, {W}]")
    assert output.shape == (B, 1, H, W), f"Shape mismatch! Got {output.shape}"
    print("  ✓ Shape correct!")

    # 5. Forward pass with attention
    print("\n[4/5] Running forward pass (with attention)...")
    with torch.no_grad():
        output, attn_weights = model(sparse_pixels, state_vector, state_mask, resolution, return_attention=True)

    print(f"  Output shape: {output.shape}")
    print(f"  Attention weights: {type(attn_weights)}")
    print("  ✓ Forward pass successful!")

    # 6. Value range check
    print("\n[5/5] Checking output value range...")
    print(f"  Min value: {output.min().item():.4f}")
    print(f"  Max value: {output.max().item():.4f}")
    print(f"  Mean value: {output.mean().item():.4f}")

    if output.min() >= 0 and output.max() <= 1:
        print("  ✓ Values in [0, 1] range (Sigmoid working)")
    else:
        print("  ⚠ Values outside [0, 1] range!")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    test_forward()
