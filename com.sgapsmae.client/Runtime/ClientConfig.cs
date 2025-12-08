using System;
using UnityEngine;

namespace SGAPSMAEClient
{
    /// <summary>
    /// Configuration for the SGAPS-MAE client.
    /// </summary>
    [Serializable]
    public class ClientConfig
    {
        [Header("Server Settings")]
        [Tooltip("Server hostname or IP address")]
        public string serverHost = "localhost";
        
        [Tooltip("Server port number")]
        public int serverPort = 8888;
        
        [Header("Frame Settings")]
        [Tooltip("Source frame width")]
        public int frameWidth = 256;
        
        [Tooltip("Source frame height")]
        public int frameHeight = 240;
        
        [Tooltip("Target resolution width for server")]
        public int targetWidth = 224;
        
        [Tooltip("Target resolution height for server")]
        public int targetHeight = 224;
        
        [Header("Compression Settings")]
        [Tooltip("Compression level (1-9, lower = faster)")]
        [Range(1, 9)]
        public int compressionLevel = 1;
        
        [Header("Buffer Settings")]
        [Tooltip("Number of coordinate sets to buffer")]
        public int coordinateBufferSize = 3;
        
        /// <summary>
        /// Create default configuration.
        /// </summary>
        public static ClientConfig Default => new ClientConfig();
        
        /// <summary>
        /// Get the scale factor from frame to target resolution.
        /// </summary>
        public Vector2 ScaleFactor => new Vector2(
            (float)frameHeight / targetHeight,
            (float)frameWidth / targetWidth
        );
    }
}
