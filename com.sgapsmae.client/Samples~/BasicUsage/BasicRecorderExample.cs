using SGAPSMAEClient;
using UnityEngine;

/// <summary>
/// Basic example of using SGAPS-MAE client for game recording.
/// Attach this to your main camera or a game manager object.
/// </summary>
public class BasicRecorderExample : MonoBehaviour
{
    [Header("Settings")]
    [SerializeField] private int pixelBudget = 500;
    [SerializeField] private bool autoStartRecording = true;
    
    [Header("References")]
    [SerializeField] private Camera targetCamera;
    
    private SGAPSMAEGameClient _client;
    private RenderTexture _captureTexture;
    
    void Start()
    {
        // Get or add the client component
        _client = GetComponent<SGAPSMAEGameClient>();
        if (_client == null)
        {
            _client = gameObject.AddComponent<SGAPSMAEGameClient>();
        }
        
        // Setup camera reference
        if (targetCamera == null)
        {
            targetCamera = Camera.main;
        }
        
        // Create render texture for capturing
        _captureTexture = new RenderTexture(256, 240, 24);
        
        // Set initial random coordinates
        SetupInitialCoordinates();
        
        // Subscribe to events
        _client.OnPacketReady += OnPacketReady;
        _client.OnCoordinatesReceived += OnCoordinatesReceived;
        
        // Auto-start if enabled
        if (autoStartRecording)
        {
            _client.StartRecording();
        }
    }
    
    void SetupInitialCoordinates()
    {
        var coords = new Vector2Int[pixelBudget];
        
        // Generate random sampling coordinates
        for (int i = 0; i < pixelBudget; i++)
        {
            coords[i] = new Vector2Int(
                Random.Range(0, 224),
                Random.Range(0, 224)
            );
        }
        
        _client.SetInitialCoordinates(coords);
    }
    
    void LateUpdate()
    {
        if (!_client.IsRecording || targetCamera == null)
        {
            return;
        }
        
        // Capture frame from camera
        RenderTexture prevTarget = targetCamera.targetTexture;
        targetCamera.targetTexture = _captureTexture;
        targetCamera.Render();
        targetCamera.targetTexture = prevTarget;
        
        // Process the captured frame
        byte[] packet = _client.ProcessFrame(_captureTexture);
        
        // In a real application, you would send this packet to the server
        // For this example, we just log the size
        if (packet != null && _client.FrameCount % 30 == 0)
        {
            Debug.Log($"[BasicRecorder] Frame {_client.FrameCount}, Packet size: {packet.Length} bytes");
        }
    }
    
    void OnPacketReady(byte[] packet)
    {
        // This is called when a packet is ready to send
        // In a real application, send this to your server
        
        // Example: Send via your networking system
        // NetworkManager.Instance.SendToServer(packet);
    }
    
    void OnCoordinatesReceived(Vector2Int[] coordinates)
    {
        // This is called when new coordinates are received from the server
        Debug.Log($"[BasicRecorder] Received {coordinates.Length} new coordinates");
    }
    
    void OnDestroy()
    {
        if (_client != null)
        {
            _client.OnPacketReady -= OnPacketReady;
            _client.OnCoordinatesReceived -= OnCoordinatesReceived;
        }
        
        if (_captureTexture != null)
        {
            _captureTexture.Release();
            Destroy(_captureTexture);
        }
    }
    
    // Public methods for UI buttons
    public void StartRecording() => _client?.StartRecording();
    public void StopRecording() => _client?.StopRecording();
    public void ResetClient() => _client?.Reset();
}
