using UnityEngine;
using System.Collections.Generic;

namespace SGAPS.Runtime.Data
{
    /// <summary>
    /// Frame data message sent to the server.
    /// </summary>
    [System.Serializable]
    public class FrameDataMessage
    {
        /// <summary>
        /// Unique frame identifier.
        /// </summary>
        public ulong FrameId;

        /// <summary>
        /// Timestamp when frame was captured.
        /// </summary>
        public float Timestamp;

        /// <summary>
        /// Frame resolution (width, height).
        /// </summary>
        public Vector2Int Resolution;

        /// <summary>
        /// Array of sampled pixel data.
        /// </summary>
        public PixelData[] Pixels;

        /// <summary>
        /// State vector with game context.
        /// </summary>
        public float[] StateVector;
    }

    /// <summary>
    /// Session start message sent to server.
    /// </summary>
    [System.Serializable]
    public class SessionStartMessage
    {
        /// <summary>
        /// Model checkpoint identifier.
        /// </summary>
        public string CheckpointKey;

        /// <summary>
        /// Maximum state vector dimension.
        /// </summary>
        public int MaxStateDim;

        /// <summary>
        /// Capture resolution.
        /// </summary>
        public Vector2Int Resolution;

        /// <summary>
        /// Number of samples to request.
        /// </summary>
        public int SampleCount;
    }

    /// <summary>
    /// Wire format for WebSocket messages.
    /// </summary>
    [System.Serializable]
    public class WebSocketMessage
    {
        public string type;
        public object payload;
    }

    /// <summary>
    /// UV coordinates payload received from server.
    /// </summary>
    [System.Serializable]
    public class UVCoordinatesPayload
    {
        public ulong target_frame_id;
        public List<UVCoord> coordinates;

        [System.Serializable]
        public class UVCoord
        {
            public float u;
            public float v;
        }
    }

    /// <summary>
    /// Session start acknowledgment from server.
    /// </summary>
    [System.Serializable]
    public class SessionStartAckPayload
    {
        public string checkpoint_key;
        public bool checkpoint_loaded;
        public string model_version;
    }

    /// <summary>
    /// Connection acknowledgment from server.
    /// </summary>
    [System.Serializable]
    public class ConnectionAckPayload
    {
        public string client_id;
        public string server_version;
    }

    /// <summary>
    /// Error payload from server.
    /// </summary>
    [System.Serializable]
    public class ErrorPayload
    {
        public string code;
        public string message;
    }
}
