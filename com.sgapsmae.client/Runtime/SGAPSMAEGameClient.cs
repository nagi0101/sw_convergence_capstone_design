using System;
using System.Collections.Generic;
using UnityEngine;

namespace SGAPSMAEClient
{
    /// <summary>
    /// Minimal game client for SGAPS-MAE.
    /// Designed for extremely low CPU overhead.
    /// </summary>
    public class SGAPSMAEGameClient : MonoBehaviour
    {
        [Header("Configuration")]
        [SerializeField] private ClientConfig _config;
        
        [Header("Status")]
        [SerializeField] private bool _isRecording;
        [SerializeField] private int _frameCount;
        [SerializeField] private float _averageProcessingTimeMs;
        
        private PixelExtractor _pixelExtractor;
        private PacketCompressor _compressor;
        
        private Vector2Int[] _currentCoordinates;
        private Queue<Vector2Int[]> _coordinateBuffer;
        
        private int _frameIdx;
        private float _totalProcessingTime;
        
        // Events
        public event Action<byte[]> OnPacketReady;
        public event Action<Vector2Int[]> OnCoordinatesReceived;
        
        /// <summary>
        /// Is the client currently recording?
        /// </summary>
        public bool IsRecording => _isRecording;
        
        /// <summary>
        /// Number of frames processed.
        /// </summary>
        public int FrameCount => _frameCount;
        
        /// <summary>
        /// Average processing time per frame in milliseconds.
        /// </summary>
        public float AverageProcessingTimeMs => _averageProcessingTimeMs;
        
        /// <summary>
        /// Current configuration.
        /// </summary>
        public ClientConfig Config => _config;
        
        private void Awake()
        {
            if (_config == null)
            {
                _config = ClientConfig.Default;
            }
            
            Initialize();
        }
        
        private void Initialize()
        {
            _pixelExtractor = new PixelExtractor(_config);
            _compressor = new PacketCompressor(_config.compressionLevel);
            _coordinateBuffer = new Queue<Vector2Int[]>(_config.coordinateBufferSize);
            
            _frameIdx = 0;
            _frameCount = 0;
            _totalProcessingTime = 0f;
        }
        
        /// <summary>
        /// Start recording session.
        /// </summary>
        public void StartRecording()
        {
            _isRecording = true;
            _frameIdx = 0;
            _frameCount = 0;
            _totalProcessingTime = 0f;
            
            Debug.Log("[SGAPS-MAE] Recording started");
        }
        
        /// <summary>
        /// Stop recording session.
        /// </summary>
        public void StopRecording()
        {
            _isRecording = false;
            Debug.Log($"[SGAPS-MAE] Recording stopped. Frames: {_frameCount}, Avg time: {_averageProcessingTimeMs:F2}ms");
        }
        
        /// <summary>
        /// Set initial coordinates for first frame.
        /// </summary>
        /// <param name="coordinates">Initial sampling coordinates</param>
        public void SetInitialCoordinates(Vector2Int[] coordinates)
        {
            _currentCoordinates = coordinates;
            Debug.Log($"[SGAPS-MAE] Initial coordinates set: {coordinates.Length} pixels");
        }
        
        /// <summary>
        /// Receive new coordinates from server.
        /// </summary>
        /// <param name="compressedData">Compressed coordinate packet</param>
        public void ReceiveCoordinates(byte[] compressedData)
        {
            try
            {
                var coordinates = _compressor.DecompressCoordinates(compressedData);
                
                // Buffer coordinates
                _coordinateBuffer.Enqueue(coordinates);
                
                // Keep buffer size limited
                while (_coordinateBuffer.Count > _config.coordinateBufferSize)
                {
                    _coordinateBuffer.Dequeue();
                }
                
                // Use buffered coordinates
                if (_coordinateBuffer.Count > 0)
                {
                    _currentCoordinates = _coordinateBuffer.Peek();
                }
                
                OnCoordinatesReceived?.Invoke(coordinates);
            }
            catch (Exception e)
            {
                Debug.LogError($"[SGAPS-MAE] Failed to decompress coordinates: {e.Message}");
            }
        }
        
        /// <summary>
        /// Process a game frame.
        /// </summary>
        /// <param name="frameTexture">Current frame texture</param>
        /// <returns>Compressed packet ready for transmission, or null if not recording</returns>
        public byte[] ProcessFrame(Texture2D frameTexture)
        {
            if (!_isRecording || _currentCoordinates == null)
            {
                return null;
            }
            
            float startTime = Time.realtimeSinceStartup;
            
            try
            {
                // Extract pixels
                _pixelExtractor.Extract(
                    frameTexture,
                    _currentCoordinates,
                    out Vector2Int[] validCoords,
                    out Color32[] pixelValues
                );
                
                // Compress packet
                byte[] packet = _compressor.Compress(_frameIdx, validCoords, pixelValues);
                
                // Update statistics
                _frameIdx++;
                _frameCount++;
                
                float processingTime = (Time.realtimeSinceStartup - startTime) * 1000f;
                _totalProcessingTime += processingTime;
                _averageProcessingTimeMs = _totalProcessingTime / _frameCount;
                
                // Advance coordinate buffer
                if (_coordinateBuffer.Count > 1)
                {
                    _coordinateBuffer.Dequeue();
                    _currentCoordinates = _coordinateBuffer.Peek();
                }
                
                OnPacketReady?.Invoke(packet);
                
                return packet;
            }
            catch (Exception e)
            {
                Debug.LogError($"[SGAPS-MAE] Frame processing error: {e.Message}");
                return null;
            }
        }
        
        /// <summary>
        /// Process frame from RenderTexture.
        /// </summary>
        /// <param name="renderTexture">Source render texture</param>
        /// <returns>Compressed packet</returns>
        public byte[] ProcessFrame(RenderTexture renderTexture)
        {
            // Create temporary texture
            var tempTexture = new Texture2D(renderTexture.width, renderTexture.height, TextureFormat.RGB24, false);
            
            RenderTexture.active = renderTexture;
            tempTexture.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
            tempTexture.Apply();
            RenderTexture.active = null;
            
            byte[] result = ProcessFrame(tempTexture);
            
            Destroy(tempTexture);
            
            return result;
        }
        
        /// <summary>
        /// Get client statistics.
        /// </summary>
        public ClientStatistics GetStatistics()
        {
            return new ClientStatistics
            {
                FrameCount = _frameCount,
                AverageProcessingTimeMs = _averageProcessingTimeMs,
                IsRecording = _isRecording,
                CoordinateBufferCount = _coordinateBuffer?.Count ?? 0
            };
        }
        
        /// <summary>
        /// Reset client state.
        /// </summary>
        public void Reset()
        {
            _frameIdx = 0;
            _frameCount = 0;
            _totalProcessingTime = 0f;
            _averageProcessingTimeMs = 0f;
            _currentCoordinates = null;
            _coordinateBuffer?.Clear();
        }
    }
    
    /// <summary>
    /// Client statistics structure.
    /// </summary>
    [Serializable]
    public struct ClientStatistics
    {
        public int FrameCount;
        public float AverageProcessingTimeMs;
        public bool IsRecording;
        public int CoordinateBufferCount;
    }
}
