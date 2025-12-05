using System;
using System.Collections.Generic;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using SGAPS.Runtime.Data;

namespace SGAPS.Runtime.Core
{
    /// <summary>
    /// WebSocket client for SGAPS server communication.
    /// Handles connection management, message sending/receiving, and event dispatching.
    /// </summary>
    public class NetworkClient : IDisposable
    {
        #region Events

        /// <summary>Fired when successfully connected to the server.</summary>
        public event Action OnConnected;

        /// <summary>Fired when disconnected from the server.</summary>
        public event Action OnDisconnected;

        /// <summary>
        /// Fired when session is acknowledged by the server.
        /// Parameters: sampleCount, maxStateDim, resolution
        /// Client must use these server-provided values.
        /// </summary>
        public event Action<SessionConfig> OnSessionStarted;

        /// <summary>Fired when UV coordinates are received from the server.</summary>
        public event Action<UVCoordinates> OnUVCoordinatesReceived;

        /// <summary>Fired when an error occurs.</summary>
        public event Action<string> OnError;

        #endregion

        #region Private Fields

        private readonly string endpoint;
        private readonly string checkpointKey;

        private ClientWebSocket webSocket;
        private CancellationTokenSource cancellationTokenSource;
        private Task receiveLoopTask;

        private readonly object lockObject = new object();
        private bool isDisposed = false;

        #endregion

        #region Properties

        /// <summary>Returns true if currently connected to the server.</summary>
        public bool IsConnected
        {
            get
            {
                lock (lockObject)
                {
                    return webSocket != null && webSocket.State == WebSocketState.Open;
                }
            }
        }

        #endregion

        #region Constructor

        /// <summary>
        /// Creates a new NetworkClient instance.
        /// Note: sample_count and max_state_dim are server-controlled parameters.
        /// Client will receive them from session_start_ack.
        /// </summary>
        public NetworkClient(string endpoint, string checkpointKey)
        {
            this.endpoint = endpoint;
            this.checkpointKey = checkpointKey;
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Connects to the WebSocket server asynchronously.
        /// </summary>
        public async Task<bool> ConnectAsync()
        {
            if (IsConnected)
            {
                Debug.LogWarning("[SGAPS.NetworkClient] Already connected.");
                return true;
            }

            try
            {
                lock (lockObject)
                {
                    webSocket = new ClientWebSocket();
                    cancellationTokenSource = new CancellationTokenSource();
                }

                Uri uri = new Uri(endpoint);
                await webSocket.ConnectAsync(uri, cancellationTokenSource.Token);

                if (webSocket.State == WebSocketState.Open)
                {
                    Debug.Log($"[SGAPS.NetworkClient] Connected to {endpoint}");

                    // Start background receive loop
                    receiveLoopTask = Task.Run(() => ReceiveLoop(cancellationTokenSource.Token));

                    // Fire connected event on main thread
                    MainThreadDispatcher.Enqueue(() => OnConnected?.Invoke());

                    return true;
                }
                else
                {
                    Debug.LogError($"[SGAPS.NetworkClient] Connection failed. State: {webSocket.State}");
                    return false;
                }
            }
            catch (Exception ex)
            {
                string errorMessage = $"Connection failed: {ex.Message}";
                Debug.LogError($"[SGAPS.NetworkClient] {errorMessage}");
                MainThreadDispatcher.Enqueue(() => OnError?.Invoke(errorMessage));
                return false;
            }
        }

        /// <summary>
        /// Disconnects from the server.
        /// </summary>
        public void Disconnect()
        {
            if (isDisposed) return;

            try
            {
                lock (lockObject)
                {
                    cancellationTokenSource?.Cancel();

                    if (webSocket != null && webSocket.State == WebSocketState.Open)
                    {
                        var closeTask = webSocket.CloseAsync(
                            WebSocketCloseStatus.NormalClosure,
                            "Client disconnecting",
                            CancellationToken.None
                        );

                        if (!closeTask.Wait(TimeSpan.FromSeconds(2)))
                        {
                            Debug.LogWarning("[SGAPS.NetworkClient] Close timeout, aborting.");
                            webSocket.Abort();
                        }
                    }

                    webSocket?.Dispose();
                    webSocket = null;
                    cancellationTokenSource?.Dispose();
                    cancellationTokenSource = null;
                }

                Debug.Log("[SGAPS.NetworkClient] Disconnected.");
                MainThreadDispatcher.Enqueue(() => OnDisconnected?.Invoke());
            }
            catch (Exception ex)
            {
                Debug.LogError($"[SGAPS.NetworkClient] Error during disconnect: {ex.Message}");
            }
        }

        /// <summary>
        /// Sends session_start message to the server.
        /// Uses current screen resolution. Server will respond with sample_count and max_state_dim.
        /// </summary>
        public void SendSessionStart()
        {
            if (!IsConnected)
            {
                Debug.LogWarning("[SGAPS.NetworkClient] Cannot send session_start: not connected.");
                return;
            }

            var message = new
            {
                type = "session_start",
                payload = new
                {
                    checkpoint_key = checkpointKey,
                    resolution = new int[] { Screen.width, Screen.height }
                }
            };

            SendJsonMessage(message);
            Debug.Log($"[SGAPS.NetworkClient] Sent session_start message (resolution: {Screen.width}x{Screen.height}).");
        }

        /// <summary>
        /// Sends frame data to the server.
        /// </summary>
        public void SendFrameData(FrameDataMessage message)
        {
            if (!IsConnected)
            {
                Debug.LogWarning("[SGAPS.NetworkClient] Cannot send frame_data: not connected.");
                return;
            }

            var wireMessage = new
            {
                type = "frame_data",
                payload = new
                {
                    frame_id = message.FrameId,
                    timestamp = message.Timestamp,
                    resolution = new int[] { message.Resolution.x, message.Resolution.y },
                    num_pixels = message.Pixels.Length,
                    pixels = ConvertPixelsToWireFormat(message.Pixels),
                    state_vector = message.StateVector
                }
            };

            SendJsonMessage(wireMessage);
        }

        /// <summary>
        /// Sends a heartbeat to keep connection alive.
        /// </summary>
        public void SendHeartbeat()
        {
            if (!IsConnected) return;

            var message = new
            {
                type = "heartbeat",
                payload = new { timestamp = Time.time }
            };

            SendJsonMessage(message);
        }

        #endregion

        #region Private Methods

        private object[] ConvertPixelsToWireFormat(PixelData[] pixels)
        {
            var result = new object[pixels.Length];
            for (int i = 0; i < pixels.Length; i++)
            {
                result[i] = new
                {
                    u = pixels[i].U,
                    v = pixels[i].V,
                    value = (int)pixels[i].Value
                };
            }
            return result;
        }

        private async void SendJsonMessage(object message)
        {
            if (!IsConnected) return;

            try
            {
                string json = JsonConvert.SerializeObject(message);
                byte[] bytes = Encoding.UTF8.GetBytes(json);

                await webSocket.SendAsync(
                    new ArraySegment<byte>(bytes),
                    WebSocketMessageType.Text,
                    true,
                    cancellationTokenSource.Token
                );
            }
            catch (Exception ex)
            {
                string errorMessage = $"Send error: {ex.Message}";
                Debug.LogError($"[SGAPS.NetworkClient] {errorMessage}");
                MainThreadDispatcher.Enqueue(() => OnError?.Invoke(errorMessage));
            }
        }

        private async Task ReceiveLoop(CancellationToken cancellationToken)
        {
            byte[] buffer = new byte[1024 * 64]; // 64 KB buffer
            StringBuilder messageBuilder = new StringBuilder();

            try
            {
                while (!cancellationToken.IsCancellationRequested && IsConnected)
                {
                    messageBuilder.Clear();
                    WebSocketReceiveResult result;

                    do
                    {
                        result = await webSocket.ReceiveAsync(
                            new ArraySegment<byte>(buffer),
                            cancellationToken
                        );

                        if (result.MessageType == WebSocketMessageType.Close)
                        {
                            Debug.Log("[SGAPS.NetworkClient] Server initiated close.");
                            await webSocket.CloseAsync(
                                WebSocketCloseStatus.NormalClosure,
                                string.Empty,
                                CancellationToken.None
                            );
                            MainThreadDispatcher.Enqueue(() => OnDisconnected?.Invoke());
                            return;
                        }

                        string chunk = Encoding.UTF8.GetString(buffer, 0, result.Count);
                        messageBuilder.Append(chunk);

                    } while (!result.EndOfMessage);

                    string message = messageBuilder.ToString();
                    HandleReceivedMessage(message);
                }
            }
            catch (OperationCanceledException)
            {
                Debug.Log("[SGAPS.NetworkClient] Receive loop cancelled.");
            }
            catch (WebSocketException ex)
            {
                string errorMessage = $"WebSocket error: {ex.Message}";
                Debug.LogError($"[SGAPS.NetworkClient] {errorMessage}");
                MainThreadDispatcher.Enqueue(() =>
                {
                    OnError?.Invoke(errorMessage);
                    OnDisconnected?.Invoke();
                });
            }
            catch (Exception ex)
            {
                string errorMessage = $"Receive error: {ex.Message}";
                Debug.LogError($"[SGAPS.NetworkClient] {errorMessage}");
                MainThreadDispatcher.Enqueue(() =>
                {
                    OnError?.Invoke(errorMessage);
                    OnDisconnected?.Invoke();
                });
            }
        }

        private void HandleReceivedMessage(string message)
        {
            try
            {
                JObject json = JObject.Parse(message);
                string messageType = json["type"]?.ToString();

                switch (messageType)
                {
                    case "connection_ack":
                        Debug.Log("[SGAPS.NetworkClient] Received connection acknowledgment.");
                        break;

                    case "session_start_ack":
                        HandleSessionStartAck(json["payload"]);
                        break;

                    case "uv_coordinates":
                        HandleUVCoordinates(json["payload"]);
                        break;

                    case "heartbeat_ack":
                        break;

                    case "error":
                        HandleServerError(json["payload"]);
                        break;

                    default:
                        Debug.LogWarning($"[SGAPS.NetworkClient] Unknown message type: {messageType}");
                        break;
                }
            }
            catch (JsonException ex)
            {
                Debug.LogError($"[SGAPS.NetworkClient] Failed to parse message: {ex.Message}");
            }
        }

        private void HandleSessionStartAck(JToken payload)
        {
            string checkpointKey = payload?["checkpoint_key"]?.ToString();
            bool loaded = payload?["checkpoint_loaded"]?.ToObject<bool>() ?? false;
            string version = payload?["model_version"]?.ToString();

            // Server-controlled parameters - client MUST use these values
            // Note: sentinel_value is server-internal (for padding) and NOT sent to client
            int sampleCount = payload?["sample_count"]?.ToObject<int>() ?? 500;
            int maxStateDim = payload?["max_state_dim"]?.ToObject<int>() ?? 64;
            int targetFPS = payload?["target_fps"]?.ToObject<int>() ?? 10;

            int[] resolutionArray = payload?["resolution"]?.ToObject<int[]>() ?? new int[] { Screen.width, Screen.height };
            Vector2Int resolution = new Vector2Int(resolutionArray[0], resolutionArray[1]);

            Debug.Log($"[SGAPS.NetworkClient] Session started - Checkpoint: {checkpointKey}, Loaded: {loaded}, " +
                      $"Version: {version}, SampleCount: {sampleCount}, MaxStateDim: {maxStateDim}, " +
                      $"TargetFPS: {targetFPS}, Resolution: {resolution.x}x{resolution.y}");

            var config = new SessionConfig(
                checkpointKey: checkpointKey,
                checkpointLoaded: loaded,
                modelVersion: version,
                sampleCount: sampleCount,
                maxStateDim: maxStateDim,
                targetFPS: targetFPS,
                resolution: resolution
            );

            MainThreadDispatcher.Enqueue(() => OnSessionStarted?.Invoke(config));
        }

        private void HandleUVCoordinates(JToken payload)
        {
            try
            {
                ulong targetFrameId = payload?["target_frame_id"]?.ToObject<ulong>() ?? 0;
                JArray coordsArray = payload?["coordinates"] as JArray;

                if (coordsArray == null)
                {
                    Debug.LogWarning("[SGAPS.NetworkClient] Received uv_coordinates with no coordinates array.");
                    return;
                }

                var coords = new Vector2[coordsArray.Count];
                for (int i = 0; i < coordsArray.Count; i++)
                {
                    float u = coordsArray[i]["u"]?.ToObject<float>() ?? 0f;
                    float v = coordsArray[i]["v"]?.ToObject<float>() ?? 0f;
                    coords[i] = new Vector2(u, v);
                }

                var uvCoords = new UVCoordinates(targetFrameId, coords);

                Debug.Log($"[SGAPS.NetworkClient] Received {coords.Length} UV coordinates for frame {targetFrameId}");
                MainThreadDispatcher.Enqueue(() => OnUVCoordinatesReceived?.Invoke(uvCoords));
            }
            catch (Exception ex)
            {
                Debug.LogError($"[SGAPS.NetworkClient] Error parsing UV coordinates: {ex.Message}");
            }
        }

        private void HandleServerError(JToken payload)
        {
            string code = payload?["code"]?.ToString() ?? "UNKNOWN";
            string errorMessage = payload?["message"]?.ToString() ?? "Unknown error";

            string fullError = $"Server error [{code}]: {errorMessage}";
            Debug.LogError($"[SGAPS.NetworkClient] {fullError}");
            MainThreadDispatcher.Enqueue(() => OnError?.Invoke(fullError));
        }

        #endregion

        #region IDisposable

        public void Dispose()
        {
            if (isDisposed) return;
            isDisposed = true;
            Disconnect();
        }

        #endregion
    }

    /// <summary>
    /// Helper class to dispatch actions to the Unity main thread.
    /// </summary>
    public static class MainThreadDispatcher
    {
        private static readonly Queue<Action> actionQueue = new Queue<Action>();
        private static readonly object queueLock = new object();

        /// <summary>Enqueues an action to be executed on the main thread.</summary>
        public static void Enqueue(Action action)
        {
            if (action == null) return;

            lock (queueLock)
            {
                actionQueue.Enqueue(action);
            }
        }

        /// <summary>Processes all queued actions. Should be called from Update().</summary>
        public static void ProcessQueue()
        {
            lock (queueLock)
            {
                while (actionQueue.Count > 0)
                {
                    Action action = actionQueue.Dequeue();
                    try
                    {
                        action?.Invoke();
                    }
                    catch (Exception ex)
                    {
                        Debug.LogError($"[MainThreadDispatcher] Error: {ex.Message}");
                    }
                }
            }
        }
    }
}
