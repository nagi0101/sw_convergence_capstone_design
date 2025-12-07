using System;
using System.Collections;
using UnityEngine;
using SGAPS.Runtime.Data;

namespace SGAPS.Runtime.Core
{
    /// <summary>
    /// Main MonoBehaviour entry point for the SGAPS system.
    /// Coordinates frame capture, pixel sampling, and server communication.
    /// </summary>
    public class SGAPSManager : MonoBehaviour
    {
        #region Inspector Fields

        [Header("Server Connection")]
        [Tooltip("WebSocket server endpoint")]
        [SerializeField]
        private string serverEndpoint = "ws://localhost:8000/ws/stream";

        [Tooltip("Model checkpoint identifier (game/map specific)")]
        [SerializeField]
        private string checkpointKey = "default";

        [Header("Behavior")]
        [Tooltip("Automatically connect to server on Start")]
        [SerializeField]
        private bool connectOnStart = false;

        [Tooltip("Show debug panel with connection status")]
        [SerializeField]
        private bool showDebugPanel = false;

        #endregion

        #region Runtime Fields (set by server)

        // These values are controlled by the server and received from session_start_ack
        private int sampleCount = 500;
        private int maxStateDim = 64;
        private int targetFPS = 10;

        // Sentinel value is server-internal for state vector padding
        // Client uses a constant default for uninitialized state (if needed locally)
        private const float DEFAULT_STATE_VALUE = 0f;

        #endregion

        #region Private Fields

        private FrameCaptureHandler frameCaptureHandler;
        private PixelSampler pixelSampler;
        private NetworkClient networkClient;
        private StateVectorCollector stateVectorCollector;

        private UVCoordinates currentUVCoordinates;
        private ulong frameCounter;
        private Coroutine captureCoroutine;

        private bool isInitialized = false;
        private bool isCapturing = false;
        private string lastError = string.Empty;

        private float lastCaptureTime;
        private float averageCaptureTime;
        private int captureTimeCount;

        #endregion

        #region Properties

        /// <summary>Returns true if connected to the server.</summary>
        public bool IsConnected => networkClient?.IsConnected ?? false;

        /// <summary>Returns the total number of frames captured.</summary>
        public ulong FrameCount => frameCounter;

        /// <summary>Returns the current server endpoint.</summary>
        public string ServerEndpoint => serverEndpoint;

        /// <summary>Returns the current screen resolution.</summary>
        public Vector2Int ScreenResolution => new Vector2Int(Screen.width, Screen.height);

        #region Debug Properties
        public bool IsCapturing => isCapturing;
        public int TargetFPS => targetFPS;
        public int SampleCount => sampleCount;
        public float LastCaptureTimeMs => lastCaptureTime;
        public float AverageCaptureTimeMs => averageCaptureTime;
        public string LastError => lastError;
        #endregion

        #endregion

        #region Unity Lifecycle

        private void Awake()
        {
            ValidateSettings();
            InitializeComponents();
        }

        private void Start()
        {
            if (connectOnStart)
            {
                ConnectToServer();
            }
        }

        private void Update()
        {
            MainThreadDispatcher.ProcessQueue();
        }

        private void OnDestroy()
        {
            Cleanup();
        }

        private void OnApplicationQuit()
        {
            Cleanup();
        }

        #endregion

        #region Initialization

        private void ValidateSettings()
        {
            if (string.IsNullOrEmpty(serverEndpoint))
            {
                serverEndpoint = "ws://localhost:8000/ws/stream";
            }
        }

        private void InitializeComponents()
        {
            try
            {
                // FrameCaptureHandler uses ScreenCapture API - no camera needed
                frameCaptureHandler = new FrameCaptureHandler();

                // PixelSampler and StateVectorCollector will be initialized after receiving server config
                pixelSampler = null;
                stateVectorCollector = null;
                currentUVCoordinates = null;

                // NetworkClient - no longer needs maxStateDim (server-controlled)
                networkClient = new NetworkClient(serverEndpoint, checkpointKey);
                networkClient.OnConnected += HandleConnected;
                networkClient.OnDisconnected += HandleDisconnected;
                networkClient.OnSessionStarted += HandleSessionStarted;
                networkClient.OnUVCoordinatesReceived += HandleUVCoordinatesReceived;
                networkClient.OnError += HandleError;

                isInitialized = true;
                Debug.Log("[SGAPS] Components initialized. Waiting for server config to complete setup.");
            }
            catch (Exception ex)
            {
                Debug.LogError($"[SGAPS] Initialization failed: {ex.Message}");
                isInitialized = false;
            }
        }

        #endregion

        #region Public API

        /// <summary>
        /// Sets a value in the state vector at the specified index.
        /// </summary>
        public void SetState(int index, float value)
        {
            stateVectorCollector?.SetState(index, value);
        }

        /// <summary>
        /// Initiates connection to the SGAPS server.
        /// </summary>
        public async void ConnectToServer()
        {
            if (!isInitialized)
            {
                Debug.LogError("[SGAPS] Cannot connect: not initialized.");
                return;
            }

            if (IsConnected)
            {
                Debug.LogWarning("[SGAPS] Already connected to server.");
                return;
            }

            Debug.Log($"[SGAPS] Connecting to {serverEndpoint}...");
            lastError = string.Empty;

            bool success = await networkClient.ConnectAsync();

            if (success)
            {
                // Send session start - server will respond with sample_count and max_state_dim
                networkClient.SendSessionStart();
                // Note: Capturing will start after receiving session_start_ack
            }
            else
            {
                Debug.LogError("[SGAPS] Failed to connect to server.");
            }
        }

        /// <summary>
        /// Disconnects from the SGAPS server.
        /// </summary>
        public void DisconnectFromServer()
        {
            StopCapturing();
            networkClient?.Disconnect();
        }

        #endregion

        #region Frame Capture

        private void StartCapturing()
        {
            if (frameCaptureHandler == null)
            {
                Debug.LogError("[SGAPS] Cannot start capturing: FrameCaptureHandler not initialized.");
                return;
            }

            if (isCapturing)
            {
                Debug.LogWarning("[SGAPS] Already capturing.");
                return;
            }

            isCapturing = true;
            frameCounter = 0;
            captureCoroutine = StartCoroutine(CaptureLoop());
            Debug.Log("[SGAPS] Started capturing frames.");
        }

        private void StopCapturing()
        {
            if (captureCoroutine != null)
            {
                StopCoroutine(captureCoroutine);
                captureCoroutine = null;
            }
            isCapturing = false;
            Debug.Log("[SGAPS] Stopped capturing frames.");
        }

        /// <summary>
        /// Coroutine-based capture loop. ScreenCapture must be called after WaitForEndOfFrame.
        /// </summary>
        private IEnumerator CaptureLoop()
        {
            float captureInterval = 1f / targetFPS;
            float lastCapture = 0f;

            while (isCapturing && IsConnected)
            {
                // Wait until end of frame - required for ScreenCapture
                yield return new WaitForEndOfFrame();

                // Check timing
                if (Time.time - lastCapture < captureInterval)
                {
                    continue;
                }
                lastCapture = Time.time;

                CaptureAndSendFrame();
            }

            isCapturing = false;
        }

        private void CaptureAndSendFrame()
        {
            if (frameCaptureHandler == null || pixelSampler == null || networkClient == null)
            {
                return;
            }

            float startTime = Time.realtimeSinceStartup;

            try
            {
                // CaptureScreen() uses ScreenCapture API - captures final rendered screen
                RenderTexture grayscaleRT = frameCaptureHandler.CaptureScreen();

                PixelData[] sampledPixels = pixelSampler.SamplePixelsFast(grayscaleRT, currentUVCoordinates);

                float[] stateVector = stateVectorCollector.GetUsedStates();

                // Use current screen resolution
                Vector2Int screenRes = frameCaptureHandler.ScreenResolution;

                var frameData = new FrameDataMessage
                {
                    FrameId = frameCounter,
                    Timestamp = Time.time,
                    Resolution = screenRes,
                    Pixels = sampledPixels,
                    StateVector = stateVector
                };

                networkClient.SendFrameData(frameData);

                frameCounter++;

                stateVectorCollector.Reset();

                float captureTime = (Time.realtimeSinceStartup - startTime) * 1000f;
                lastCaptureTime = captureTime;
                averageCaptureTime = (averageCaptureTime * captureTimeCount + captureTime) / (captureTimeCount + 1);
                captureTimeCount++;
            }
            catch (Exception ex)
            {
                Debug.LogError($"[SGAPS] Error during frame capture: {ex.Message}");
            }
        }

        #endregion

        #region Event Handlers

        private void HandleConnected()
        {
            Debug.Log("[SGAPS] Connected to server. Waiting for session configuration...");
            lastError = string.Empty;
        }

        private void HandleDisconnected()
        {
            Debug.Log("[SGAPS] Disconnected from server.");
            StopCapturing();
        }

        /// <summary>
        /// Handles session_start_ack from server. Initializes components with server-provided values.
        /// </summary>
        private void HandleSessionStarted(SessionConfig config)
        {
            Debug.Log($"[SGAPS] Session started with server config: {config}");

            // Use server-controlled parameters
            sampleCount = config.SampleCount;
            maxStateDim = config.MaxStateDim;
            targetFPS = config.TargetFPS;
            // Note: sentinel_value is server-internal for padding, not sent to client

            // Initialize components with server-provided values
            pixelSampler = new PixelSampler(sampleCount);
            currentUVCoordinates = pixelSampler.GenerateInitialPattern(SamplingPattern.UniformGrid);

            // StateVectorCollector uses default value for unset states (0 by default)
            stateVectorCollector = new StateVectorCollector(maxStateDim);

            Debug.Log($"[SGAPS] Initialized with sampleCount={sampleCount}, maxStateDim={maxStateDim}, " +
                      $"targetFPS={targetFPS}");

            // Now we can start capturing
            StartCapturing();
        }

        private void HandleUVCoordinatesReceived(UVCoordinates uvCoords)
        {
            currentUVCoordinates = uvCoords;
            Debug.Log($"[SGAPS] Received {uvCoords.Count} UV coordinates for frame {uvCoords.TargetFrameId}");
        }

        private void HandleError(string error)
        {
            lastError = error;
            Debug.LogError($"[SGAPS] Error: {error}");
        }

        #endregion

        #region Cleanup

        private void Cleanup()
        {
            StopCapturing();

            if (networkClient != null)
            {
                networkClient.OnConnected -= HandleConnected;
                networkClient.OnDisconnected -= HandleDisconnected;
                networkClient.OnSessionStarted -= HandleSessionStarted;
                networkClient.OnUVCoordinatesReceived -= HandleUVCoordinatesReceived;
                networkClient.OnError -= HandleError;
                networkClient.Dispose();
                networkClient = null;
            }

            frameCaptureHandler?.Dispose();
            frameCaptureHandler = null;

            pixelSampler?.Dispose();
            pixelSampler = null;

            isInitialized = false;
            Debug.Log("[SGAPS] Cleanup complete.");
        }

        #endregion

        #region Debug UI

        private Rect windowRect = new Rect(10, 10, 320, 100); // Height is auto-adjusted by GUILayout

        private void OnGUI()
        {
            if (!showDebugPanel) return;

            // Use GUILayout.Window for an auto-sizing, draggable window.
            windowRect = GUILayout.Window(0, windowRect, DrawDebugWindow, "SGAPS Debug Panel");
        }

        private void DrawDebugWindow(int windowID)
        {
            GUIStyle richTextStyle = CreateRichTextStyle();

            string connectionStatus = IsConnected ? "<color=green>Connected</color>" : "<color=red>Disconnected</color>";
            GUILayout.Label($"Status: {connectionStatus}", richTextStyle);

            GUILayout.Label($"Endpoint: {TruncateString(serverEndpoint, 40)}");
            GUILayout.Label($"Frames Sent: {frameCounter}");

            string captureStatus = isCapturing ? "<color=green>Active</color>" : "<color=yellow>Inactive</color>";
            GUILayout.Label($"Capturing: {captureStatus}", richTextStyle);

            GUILayout.Label($"Target FPS: {targetFPS}");
            GUILayout.Label($"Sampled Pixels: {currentUVCoordinates?.Count ?? 0}");
            GUILayout.Label($"Last Capture: {lastCaptureTime:F2}ms");
            GUILayout.Label($"Avg Capture: {averageCaptureTime:F2}ms");

            if (!string.IsNullOrEmpty(lastError))
            {
                GUIStyle errorStyle = CreateRichTextStyle();
                errorStyle.wordWrap = true;
                GUILayout.Label($"<color=red>Error: {lastError}</color>", errorStyle);
            }

            // Allow the window to be dragged by its title bar.
            GUI.DragWindow(new Rect(0, 0, 10000, 20));
        }

        private GUIStyle CreateRichTextStyle()
        {
            GUIStyle style = new GUIStyle(GUI.skin.label);
            style.richText = true;
            return style;
        }

        private string TruncateString(string text, int maxLength)
        {
            if (string.IsNullOrEmpty(text) || text.Length <= maxLength)
                return text;
            return text.Substring(0, maxLength) + "...";
        }

        #endregion
    }
}
