# SGAPS-MAE Unity Client

A lightweight Unity client for the Server-Guided Adaptive Pixel Sampling MAE (SGAPS-MAE) game session replay system.

## Overview

This package provides minimal overhead pixel extraction and compression for game frame recording. The client is designed to use less than 0.1% CPU overhead by offloading all intelligent processing to the server.

## Installation

### Using Unity Package Manager

1. Open Package Manager (Window > Package Manager)
2. Click the "+" button and select "Add package from disk..."
3. Navigate to the `com.sgapsmae.client` folder and select `package.json`

### Manual Installation

Copy the `com.sgapsmae.client` folder to your project's `Packages` directory.

## Quick Start

```csharp
using SGAPSMAEClient;
using UnityEngine;

public class GameRecorder : MonoBehaviour
{
    private SGAPSMAEGameClient _client;
    
    void Start()
    {
        _client = gameObject.AddComponent<SGAPSMAEGameClient>();
        
        // Set initial random coordinates
        var initialCoords = GenerateRandomCoordinates(500);
        _client.SetInitialCoordinates(initialCoords);
        
        // Start recording
        _client.StartRecording();
    }
    
    void OnRenderImage(RenderTexture src, RenderTexture dest)
    {
        // Process frame
        byte[] packet = _client.ProcessFrame(src);
        
        if (packet != null)
        {
            // Send packet to server
            SendToServer(packet);
        }
        
        Graphics.Blit(src, dest);
    }
    
    Vector2Int[] GenerateRandomCoordinates(int count)
    {
        var coords = new Vector2Int[count];
        for (int i = 0; i < count; i++)
        {
            coords[i] = new Vector2Int(
                Random.Range(0, 224),
                Random.Range(0, 224)
            );
        }
        return coords;
    }
}
```

## Components

### SGAPSMAEGameClient

Main component for game frame processing.

**Properties:**
- `IsRecording`: Whether the client is currently recording
- `FrameCount`: Number of frames processed
- `AverageProcessingTimeMs`: Average processing time per frame

**Methods:**
- `StartRecording()`: Begin recording session
- `StopRecording()`: End recording session
- `SetInitialCoordinates(Vector2Int[])`: Set initial sampling coordinates
- `ReceiveCoordinates(byte[])`: Process new coordinates from server
- `ProcessFrame(Texture2D)`: Process a game frame
- `ProcessFrame(RenderTexture)`: Process from render texture

### ClientConfig

Configuration settings for the client.

```csharp
var config = new ClientConfig
{
    serverHost = "localhost",
    serverPort = 8888,
    frameWidth = 256,
    frameHeight = 240,
    targetWidth = 224,
    targetHeight = 224,
    compressionLevel = 1
};
```

### PixelExtractor

Low-level pixel extraction utility.

### PacketCompressor

Compression and decompression for network packets.

## Performance

- **CPU Usage**: < 0.1%
- **Memory**: ~10MB additional
- **Packet Size**: ~2KB per frame (500 pixels)
- **Processing Time**: < 1ms per frame

## Requirements

- Unity 2021.3 or later
- .NET Standard 2.1

## License

MIT License
