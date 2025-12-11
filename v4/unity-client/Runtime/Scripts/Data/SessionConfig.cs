using UnityEngine;

namespace SGAPS.Runtime.Data
{
    /// <summary>
    /// Session configuration received from the server.
    /// Contains server-controlled parameters that the client MUST use.
    /// Note: sentinel_value is server-internal (used for padding state vectors) and not included here.
    /// </summary>
    [System.Serializable]
    public class SessionConfig
    {
        /// <summary>
        /// Checkpoint key for this session.
        /// </summary>
        public string CheckpointKey;

        /// <summary>
        /// Whether a model checkpoint was loaded on the server.
        /// </summary>
        public bool CheckpointLoaded;

        /// <summary>
        /// Model version string from the server.
        /// </summary>
        public string ModelVersion;

        /// <summary>
        /// Number of pixels to sample per frame (server-controlled).
        /// </summary>
        public int SampleCount;

        /// <summary>
        /// Maximum dimension for state vectors (server-controlled).
        /// </summary>
        public int MaxStateDim;

        /// <summary>
        /// Target frames per second for capture (server-controlled).
        /// </summary>
        public int TargetFPS;

        /// <summary>
        /// Screen resolution to use for capture.
        /// </summary>
        public Vector2Int Resolution;

        /// <summary>
        /// Whether debug mode is enabled on the server.
        /// When true, client should send full frames for visualization.
        /// </summary>
        public bool DebugMode;

        /// <summary>
        /// Creates an empty session config.
        /// </summary>
        public SessionConfig()
        {
            CheckpointKey = "default";
            CheckpointLoaded = false;
            ModelVersion = "unknown";
            SampleCount = 500;
            MaxStateDim = 64;
            TargetFPS = 10;
            Resolution = new Vector2Int(640, 480);
            DebugMode = false;
        }

        /// <summary>
        /// Creates a session config with specified values.
        /// </summary>
        public SessionConfig(
            string checkpointKey,
            bool checkpointLoaded,
            string modelVersion,
            int sampleCount,
            int maxStateDim,
            int targetFPS,
            Vector2Int resolution,
            bool debugMode = false)
        {
            CheckpointKey = checkpointKey ?? "default";
            CheckpointLoaded = checkpointLoaded;
            ModelVersion = modelVersion ?? "unknown";
            SampleCount = sampleCount;
            MaxStateDim = maxStateDim;
            TargetFPS = targetFPS;
            Resolution = resolution;
            DebugMode = debugMode;
        }

        public override string ToString()
        {
            return $"SessionConfig(SampleCount={SampleCount}, MaxStateDim={MaxStateDim}, " +
                   $"TargetFPS={TargetFPS}, Resolution={Resolution.x}x{Resolution.y})";
        }
    }
}
