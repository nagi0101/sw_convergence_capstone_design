# Client Implementation: Unity UPM Plugin

## 개요

Unity 게임에서 실행되는 경량 클라이언트 플러그인으로, 게임 렌더링 결과를 캡처하여 서버로 전송하고, 서버로부터 받은 UV 좌표를 사용해 픽셀을 샘플링합니다.

---

## 패키지 구조

### UPM Package Layout

```
com.sgaps.client/
├── package.json
├── README.md
├── CHANGELOG.md
├── LICENSE
├── Runtime/
│   ├── SGAPS.Runtime.asmdef
│   ├── Scripts/
│   │   ├── Core/
│   │   │   ├── SGAPSManager.cs          # 메인 관리 클래스
│   │   │   ├── FrameCaptureHandler.cs   # RenderTexture 캡처
│   │   │   ├── PixelSampler.cs          # UV 좌표 기반 샘플링
│   │   │   └── NetworkClient.cs         # WebSocket 통신
│   │   ├── Data/
│   │   │   ├── PixelData.cs             # 데이터 구조체
│   │   │   ├── UVCoordinates.cs
│   │   │   └── SessionConfig.cs
│   │   ├── Utils/
│   │   │   ├── GrayscaleConverter.cs    # 흑백 변환
│   │   │   ├── CompressionUtil.cs       # 데이터 압축
│   │   │   └── PerformanceMonitor.cs    # 성능 측정
│   │   └── UI/
│   │       ├── SGAPSDebugPanel.cs       # 디버그 UI
│   │       └── ConnectionStatusUI.cs    # 연결 상태 표시
│   └── Shaders/
│       └── GrayscaleCapture.shader      # GPU 흑백 변환
├── Editor/
│   ├── SGAPS.Editor.asmdef
│   └── Scripts/
│       ├── SGAPSManagerEditor.cs        # Inspector 커스터마이징
│       └── MenuItems/
│           └── SGAPSMenu.cs             # Unity 메뉴 아이템
├── Samples~/
│   └── BasicSetup/
│       ├── Scenes/
│       │   └── SGAPSDemo.unity
│       └── Scripts/
│           └── DemoController.cs
└── Documentation~/
    ├── manual.md
    └── api-reference.md
```

### package.json

```json
{
    "name": "com.sgaps.client",
    "version": "0.1.0",
    "displayName": "SGAPS Client",
    "description": "Server-Guided Adaptive Pixel Sampling client for Unity",
    "unity": "2021.3",
    "keywords": ["replay", "streaming", "adaptive-sampling"],
    "author": {
        "name": "SGAPS Team",
        "email": "sgaps@example.com"
    },
    "dependencies": {
        "com.unity.nuget.newtonsoft-json": "3.0.2"
    },
    "samples": [
        {
            "displayName": "Basic Setup",
            "description": "Basic scene setup with SGAPS",
            "path": "Samples~/BasicSetup"
        }
    ]
}
```

---

## 핵심 컴포넌트

### 1. SGAPSManager.cs

메인 진입점이자 모든 기능을 조율하는 MonoBehaviour 컴포넌트.

```csharp
using UnityEngine;
using System;
using System.Collections;

namespace SGAPS.Runtime
{
    /// <summary>
    /// SGAPS 시스템의 메인 관리 클래스
    /// Scene에 추가하여 서버와 통신하며 프레임 캡처/전송을 관리
    /// </summary>
    public class SGAPSManager : MonoBehaviour
    {
        #region Inspector Fields

        [Header("Server Connection")]
        [Tooltip("WebSocket server endpoint (e.g., ws://server.example.com:8080)")]
        public string serverEndpoint = "ws://localhost:8080";

        [Tooltip("자동으로 시작 시 서버 연결 시도")]
        public bool connectOnStart = true;

        [Header("Capture Settings")]
        [Tooltip("캡처할 카메라 (null이면 Camera.main 사용)")]
        public Camera targetCamera;

        [Tooltip("캡처 해상도 (width, height)")]
        public Vector2Int captureResolution = new Vector2Int(640, 480);

        [Tooltip("목표 캡처 프레임레이트 (FPS)")]
        [Range(1, 60)]
        public int targetFPS = 10;

        [Header("Sampling Settings")]
        [Tooltip("초기 샘플링 패턴 (서버로부터 받기 전까지 사용)")]
        public SamplingPattern initialPattern = SamplingPattern.UniformGrid;

        [Tooltip("초기 샘플링 픽셀 개수")]
        [Range(100, 5000)]
        public int initialSampleCount = 400;

        [Header("State Vector")]
        [Tooltip("상태 벡터 최대 길이 (서버와 동일해야 함)")]
        public int maxStateDim = 64;

        [Tooltip("미사용 인덱스의 sentinel 값")]
        public float sentinelValue = -999.0f;

        [Header("Checkpoint")]
        [Tooltip("모델 체크포인트 식별자 (게임/맵별로 다른 모델 사용 시 변경)")]
        public string checkpointKey = "default";

        [Header("Performance")]
        [Tooltip("성능 모니터링 활성화")]
        public bool enablePerformanceMonitoring = true;

        [Tooltip("디버그 패널 표시")]
        public bool showDebugPanel = false;

        #endregion

        #region Private Fields

        private FrameCaptureHandler captureHandler;
        private PixelSampler pixelSampler;
        private NetworkClient networkClient;
        private PerformanceMonitor performanceMonitor;
        private StateVectorCollector stateVectorCollector;

        private UVCoordinates currentUVCoordinates;
        private float nextCaptureTime;
        private ulong frameCounter;

        private bool isInitialized = false;
        private bool isCapturing = false;

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
            if (!isInitialized || !isCapturing) return;

            // FPS 제어
            if (Time.time >= nextCaptureTime)
            {
                CaptureAndSendFrame();
                nextCaptureTime = Time.time + (1f / targetFPS);
            }
        }

        private void OnDestroy()
        {
            Cleanup();
        }

        #endregion

        #region Initialization

        private void ValidateSettings()
        {
            if (targetCamera == null)
            {
                targetCamera = Camera.main;
                if (targetCamera == null)
                {
                    Debug.LogError("[SGAPS] No camera found! Please assign a camera.");
                }
            }

            if (string.IsNullOrEmpty(serverEndpoint))
            {
                Debug.LogWarning("[SGAPS] Server endpoint not set. Using default: ws://localhost:8080");
                serverEndpoint = "ws://localhost:8080";
            }
        }

        private void InitializeComponents()
        {
            // 1. Frame Capture Handler
            captureHandler = new FrameCaptureHandler(
                targetCamera,
                captureResolution
            );

            // 2. Pixel Sampler
            pixelSampler = new PixelSampler(initialSampleCount);
            currentUVCoordinates = pixelSampler.GenerateInitialPattern(initialPattern);

            // 3. State Vector Collector
            stateVectorCollector = new StateVectorCollector(maxStateDim, sentinelValue);

            // 4. Network Client
            networkClient = new NetworkClient(serverEndpoint, checkpointKey, maxStateDim, captureResolution);
            networkClient.OnUVCoordinatesReceived += HandleUVCoordinatesReceived;
            networkClient.OnConnectionStatusChanged += HandleConnectionStatusChanged;

            // 5. Performance Monitor
            if (enablePerformanceMonitoring)
            {
                performanceMonitor = new PerformanceMonitor();
            }

            isInitialized = true;
            Debug.Log("[SGAPS] Components initialized successfully.");
        }

        #endregion

        #region Public API

        /// <summary>
        /// 서버에 연결 시도
        /// </summary>
        public async void ConnectToServer()
        {
            if (networkClient == null)
            {
                Debug.LogError("[SGAPS] NetworkClient not initialized!");
                return;
            }

            Debug.Log($"[SGAPS] Connecting to {serverEndpoint}...");
            bool success = await networkClient.ConnectAsync();

            if (success)
            {
                Debug.Log("[SGAPS] Connected to server successfully.");
                StartCapture();
            }
            else
            {
                Debug.LogError("[SGAPS] Failed to connect to server.");
            }
        }

        /// <summary>
        /// 서버 연결 해제
        /// </summary>
        public void DisconnectFromServer()
        {
            StopCapture();
            networkClient?.Disconnect();
        }

        /// <summary>
        /// 캡처 시작
        /// </summary>
        public void StartCapture()
        {
            if (!isInitialized)
            {
                Debug.LogError("[SGAPS] Not initialized!");
                return;
            }

            isCapturing = true;
            frameCounter = 0;
            nextCaptureTime = Time.time;
            Debug.Log("[SGAPS] Capture started.");
        }

        /// <summary>
        /// 캡처 중지
        /// </summary>
        public void StopCapture()
        {
            isCapturing = false;
            Debug.Log("[SGAPS] Capture stopped.");
        }

        #endregion

        #region Frame Capture & Send

        private void CaptureAndSendFrame()
        {
            performanceMonitor?.BeginSample("TotalFrameTime");

            try
            {
                // 1. RenderTexture 캡처
                performanceMonitor?.BeginSample("CaptureFrame");
                RenderTexture grayscaleRT = captureHandler.CaptureFrame();
                performanceMonitor?.EndSample("CaptureFrame");

                // 2. 픽셀 샘플링
                performanceMonitor?.BeginSample("SamplePixels");
                PixelData[] sampledPixels = pixelSampler.SamplePixels(
                    grayscaleRT,
                    currentUVCoordinates
                );
                performanceMonitor?.EndSample("SamplePixels");

                // 3. 상태 벡터 가져오기
                float[] stateVector = stateVectorCollector.GetUsedStates();

                // 4. 메타데이터 준비
                var frameData = new FrameData
                {
                    frameId = frameCounter++,
                    timestamp = Time.time,
                    resolution = captureResolution,
                    pixels = sampledPixels,
                    stateVector = stateVector
                };

                // 5. 서버로 전송
                performanceMonitor?.BeginSample("SendData");
                networkClient.SendFrameData(frameData);
                performanceMonitor?.EndSample("SendData");

                // 6. 상태 벡터 리셋 (다음 프레임을 위해)
                stateVectorCollector.Reset();
            }
            catch (Exception ex)
            {
                Debug.LogError($"[SGAPS] Error during frame capture: {ex.Message}");
            }
            finally
            {
                performanceMonitor?.EndSample("TotalFrameTime");
            }
        }

        #endregion

        #region Event Handlers

        private void HandleUVCoordinatesReceived(UVCoordinates newCoords)
        {
            // 서버로부터 새로운 UV 좌표 수신
            currentUVCoordinates = newCoords;

            Debug.Log($"[SGAPS] Received {newCoords.coordinates.Count} UV coordinates for frame {newCoords.targetFrameId}");
        }

        private void HandleConnectionStatusChanged(bool isConnected)
        {
            if (isConnected)
            {
                Debug.Log("[SGAPS] Connection established.");
            }
            else
            {
                Debug.LogWarning("[SGAPS] Connection lost. Attempting to reconnect...");
                StopCapture();

                // 재연결 시도
                StartCoroutine(ReconnectCoroutine());
            }
        }

        private IEnumerator ReconnectCoroutine()
        {
            yield return new WaitForSeconds(5f);

            if (!networkClient.IsConnected)
            {
                ConnectToServer();
            }
        }

        #endregion

        #region Cleanup

        private void Cleanup()
        {
            StopCapture();
            networkClient?.Disconnect();
            captureHandler?.Dispose();
            performanceMonitor?.PrintReport();
        }

        #endregion

        #region Debug UI

        private void OnGUI()
        {
            if (!showDebugPanel) return;

            GUI.Box(new Rect(10, 10, 300, 200), "SGAPS Debug Panel");

            GUILayout.BeginArea(new Rect(20, 40, 280, 160));

            GUILayout.Label($"Status: {(networkClient?.IsConnected == true ? "Connected" : "Disconnected")}");
            GUILayout.Label($"Frames Sent: {frameCounter}");
            GUILayout.Label($"Current FPS: {(int)(1f / Time.deltaTime)}");
            GUILayout.Label($"Sampled Pixels: {currentUVCoordinates?.coordinates.Count ?? 0}");

            if (performanceMonitor != null)
            {
                GUILayout.Label($"Capture Time: {performanceMonitor.GetAverage("CaptureFrame"):F2}ms");
                GUILayout.Label($"Sample Time: {performanceMonitor.GetAverage("SamplePixels"):F2}ms");
                GUILayout.Label($"Send Time: {performanceMonitor.GetAverage("SendData"):F2}ms");
            }

            GUILayout.EndArea();
        }

        #endregion
    }

    #region Enums

    public enum SamplingPattern
    {
        UniformGrid,
        RandomUniform,
        Checkerboard
    }

    #endregion
}
```

---

### 2. FrameCaptureHandler.cs

RenderTexture를 캡처하고 흑백으로 변환하는 클래스.

```csharp
using UnityEngine;
using UnityEngine.Rendering;

namespace SGAPS.Runtime
{
    /// <summary>
    /// 카메라 렌더링 결과를 Grayscale RenderTexture로 캡처
    /// </summary>
    public class FrameCaptureHandler
    {
        private Camera targetCamera;
        private Vector2Int resolution;

        private RenderTexture grayscaleRT;
        private Material grayscaleMaterial;

        public FrameCaptureHandler(Camera camera, Vector2Int resolution)
        {
            this.targetCamera = camera;
            this.resolution = resolution;

            Initialize();
        }

        private void Initialize()
        {
            // 1. Grayscale RenderTexture 생성
            grayscaleRT = new RenderTexture(
                resolution.x,
                resolution.y,
                0,  // No depth buffer
                RenderTextureFormat.R8  // 단일 채널 (8-bit grayscale)
            );
            grayscaleRT.name = "SGAPS_GrayscaleRT";

            // 2. Grayscale 변환 머티리얼 로드
            Shader grayscaleShader = Shader.Find("Hidden/SGAPS/GrayscaleConvert");
            if (grayscaleShader == null)
            {
                Debug.LogError("[SGAPS] GrayscaleConvert shader not found!");
                return;
            }

            grayscaleMaterial = new Material(grayscaleShader);
        }

        /// <summary>
        /// 현재 프레임을 캡처하여 Grayscale RenderTexture 반환
        /// </summary>
        public RenderTexture CaptureFrame()
        {
            // 1. 카메라 렌더링 결과를 임시 RenderTexture에 저장
            RenderTexture tempRT = RenderTexture.GetTemporary(
                resolution.x,
                resolution.y,
                24,
                RenderTextureFormat.ARGB32
            );

            RenderTexture previousRT = targetCamera.targetTexture;
            targetCamera.targetTexture = tempRT;
            targetCamera.Render();
            targetCamera.targetTexture = previousRT;

            // 2. 흑백 변환 (GPU에서 처리)
            Graphics.Blit(tempRT, grayscaleRT, grayscaleMaterial);

            // 3. 임시 텍스처 해제
            RenderTexture.ReleaseTemporary(tempRT);

            return grayscaleRT;
        }

        public void Dispose()
        {
            if (grayscaleRT != null)
            {
                grayscaleRT.Release();
                Object.Destroy(grayscaleRT);
            }

            if (grayscaleMaterial != null)
            {
                Object.Destroy(grayscaleMaterial);
            }
        }
    }
}
```

---

### 3. PixelSampler.cs

UV 좌표 리스트를 사용하여 RenderTexture에서 픽셀을 샘플링.

```csharp
using UnityEngine;
using System.Collections.Generic;

namespace SGAPS.Runtime
{
    /// <summary>
    /// UV 좌표 기반으로 RenderTexture에서 픽셀 샘플링
    /// </summary>
    public class PixelSampler
    {
        private int sampleCount;
        private Texture2D readbackTexture;

        public PixelSampler(int sampleCount)
        {
            this.sampleCount = sampleCount;
        }

        /// <summary>
        /// 초기 샘플링 패턴 생성
        /// </summary>
        public UVCoordinates GenerateInitialPattern(SamplingPattern pattern)
        {
            var uvCoords = new UVCoordinates
            {
                targetFrameId = 0,
                coordinates = new List<Vector2>()
            };

            switch (pattern)
            {
                case SamplingPattern.UniformGrid:
                    uvCoords.coordinates = GenerateUniformGrid();
                    break;

                case SamplingPattern.RandomUniform:
                    uvCoords.coordinates = GenerateRandomUniform();
                    break;

                case SamplingPattern.Checkerboard:
                    uvCoords.coordinates = GenerateCheckerboard();
                    break;
            }

            return uvCoords;
        }

        private List<Vector2> GenerateUniformGrid()
        {
            int gridSize = Mathf.CeilToInt(Mathf.Sqrt(sampleCount));
            var coords = new List<Vector2>();

            for (int y = 0; y < gridSize; y++)
            {
                for (int x = 0; x < gridSize; x++)
                {
                    float u = (x + 0.5f) / gridSize;
                    float v = (y + 0.5f) / gridSize;
                    coords.Add(new Vector2(u, v));
                }
            }

            return coords;
        }

        private List<Vector2> GenerateRandomUniform()
        {
            var coords = new List<Vector2>();

            for (int i = 0; i < sampleCount; i++)
            {
                coords.Add(new Vector2(Random.value, Random.value));
            }

            return coords;
        }

        private List<Vector2> GenerateCheckerboard()
        {
            // Checkerboard pattern 구현
            var coords = new List<Vector2>();
            int gridSize = Mathf.CeilToInt(Mathf.Sqrt(sampleCount * 2));

            for (int y = 0; y < gridSize; y++)
            {
                for (int x = 0; x < gridSize; x++)
                {
                    if ((x + y) % 2 == 0)
                    {
                        float u = (x + 0.5f) / gridSize;
                        float v = (y + 0.5f) / gridSize;
                        coords.Add(new Vector2(u, v));

                        if (coords.Count >= sampleCount) break;
                    }
                }
                if (coords.Count >= sampleCount) break;
            }

            return coords;
        }

        /// <summary>
        /// RenderTexture에서 UV 좌표 위치의 픽셀 샘플링
        /// </summary>
        public PixelData[] SamplePixels(RenderTexture rt, UVCoordinates uvCoords)
        {
            // 1. RenderTexture → Texture2D (CPU 읽기 가능하게)
            if (readbackTexture == null || readbackTexture.width != rt.width || readbackTexture.height != rt.height)
            {
                readbackTexture = new Texture2D(rt.width, rt.height, TextureFormat.R8, false);
            }

            RenderTexture.active = rt;
            readbackTexture.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
            readbackTexture.Apply();
            RenderTexture.active = null;

            // 2. UV 좌표별로 픽셀 값 추출
            PixelData[] samples = new PixelData[uvCoords.coordinates.Count];

            for (int i = 0; i < uvCoords.coordinates.Count; i++)
            {
                Vector2 uv = uvCoords.coordinates[i];
                int x = Mathf.Clamp((int)(uv.x * rt.width), 0, rt.width - 1);
                int y = Mathf.Clamp((int)(uv.y * rt.height), 0, rt.height - 1);

                Color pixel = readbackTexture.GetPixel(x, y);
                byte value = (byte)(pixel.r * 255);

                samples[i] = new PixelData
                {
                    u = uv.x,
                    v = uv.y,
                    value = value
                };
            }

            return samples;
        }
    }
}
```

---

### 4. NetworkClient.cs

WebSocket을 통한 서버 통신.

```csharp
using System;
using System.Threading.Tasks;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using UnityEngine;
using Newtonsoft.Json;

namespace SGAPS.Runtime
{
    /// <summary>
    /// WebSocket 기반 서버 통신 클라이언트
    /// </summary>
    public class NetworkClient
    {
        private string serverEndpoint;
        private ClientWebSocket webSocket;
        private CancellationTokenSource cancellationTokenSource;

        public bool IsConnected => webSocket?.State == WebSocketState.Open;

        public event Action<UVCoordinates> OnUVCoordinatesReceived;
        public event Action<bool> OnConnectionStatusChanged;

        public NetworkClient(string endpoint)
        {
            this.serverEndpoint = endpoint;
        }

        /// <summary>
        /// 서버에 연결
        /// </summary>
        public async Task<bool> ConnectAsync()
        {
            try
            {
                webSocket = new ClientWebSocket();
                cancellationTokenSource = new CancellationTokenSource();

                Uri uri = new Uri(serverEndpoint);
                await webSocket.ConnectAsync(uri, cancellationTokenSource.Token);

                OnConnectionStatusChanged?.Invoke(true);

                // 수신 루프 시작
                _ = ReceiveLoop();

                return true;
            }
            catch (Exception ex)
            {
                Debug.LogError($"[SGAPS] Connection failed: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// 연결 해제
        /// </summary>
        public void Disconnect()
        {
            if (webSocket != null && webSocket.State == WebSocketState.Open)
            {
                cancellationTokenSource?.Cancel();
                webSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "Client disconnecting", CancellationToken.None).Wait();
            }

            OnConnectionStatusChanged?.Invoke(false);
        }

        /// <summary>
        /// 프레임 데이터 전송
        /// </summary>
        public async void SendFrameData(FrameData frameData)
        {
            if (!IsConnected)
            {
                Debug.LogWarning("[SGAPS] Not connected to server.");
                return;
            }

            try
            {
                // JSON 직렬화
                string json = JsonConvert.SerializeObject(frameData);
                byte[] bytes = Encoding.UTF8.GetBytes(json);

                // 전송
                await webSocket.SendAsync(
                    new ArraySegment<byte>(bytes),
                    WebSocketMessageType.Text,
                    true,
                    cancellationTokenSource.Token
                );
            }
            catch (Exception ex)
            {
                Debug.LogError($"[SGAPS] Send error: {ex.Message}");
            }
        }

        /// <summary>
        /// 서버로부터 메시지 수신 (백그라운드 루프)
        /// </summary>
        private async Task ReceiveLoop()
        {
            byte[] buffer = new byte[1024 * 64];  // 64 KB buffer

            try
            {
                while (webSocket.State == WebSocketState.Open)
                {
                    var result = await webSocket.ReceiveAsync(
                        new ArraySegment<byte>(buffer),
                        cancellationTokenSource.Token
                    );

                    if (result.MessageType == WebSocketMessageType.Close)
                    {
                        await webSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "", CancellationToken.None);
                        OnConnectionStatusChanged?.Invoke(false);
                        break;
                    }

                    // 메시지 파싱
                    string message = Encoding.UTF8.GetString(buffer, 0, result.Count);
                    HandleReceivedMessage(message);
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"[SGAPS] Receive error: {ex.Message}");
                OnConnectionStatusChanged?.Invoke(false);
            }
        }

        private void HandleReceivedMessage(string message)
        {
            try
            {
                // UV 좌표 메시지 파싱
                var uvCoords = JsonConvert.DeserializeObject<UVCoordinates>(message);
                OnUVCoordinatesReceived?.Invoke(uvCoords);
            }
            catch (Exception ex)
            {
                Debug.LogError($"[SGAPS] Failed to parse message: {ex.Message}");
            }
        }
    }
}
```

---

### 5. Data Structures

```csharp
using System;
using System.Collections.Generic;
using UnityEngine;

namespace SGAPS.Runtime
{
    [Serializable]
    public struct PixelData
    {
        public float u;      // UV 좌표 X (0~1)
        public float v;      // UV 좌표 Y (0~1)
        public byte value;   // Grayscale 값 (0~255)
    }

    [Serializable]
    public class FrameData
    {
        public ulong frameId;
        public float timestamp;
        public Vector2Int resolution;
        public PixelData[] pixels;
        public float[] stateVector;  // 게임 상태 벡터 (가변 길이)
    }

    [Serializable]
    public class UVCoordinates
    {
        public ulong targetFrameId;
        public List<Vector2> coordinates;
    }

    [Serializable]
    public class SessionConfig
    {
        public string sessionId;
        public string checkpointKey;  // 모델 체크포인트 식별자
        public int maxStateDim;       // 상태 벡터 최대 길이
        public int targetFPS;
        public Vector2Int resolution;
    }

    /// <summary>
    /// 게임 상태를 고정 길이 벡터로 수집
    /// </summary>
    public class StateVectorCollector
    {
        public readonly int MaxStateDim;
        public readonly float SentinelValue;

        private float[] stateVector;

        public StateVectorCollector(int maxStateDim = 64, float sentinelValue = -999.0f)
        {
            MaxStateDim = maxStateDim;
            SentinelValue = sentinelValue;
            stateVector = new float[MaxStateDim];
            Reset();
        }

        /// <summary>
        /// 모든 인덱스를 sentinel 값으로 초기화
        /// </summary>
        public void Reset()
        {
            for (int i = 0; i < MaxStateDim; i++)
                stateVector[i] = SentinelValue;
        }

        /// <summary>
        /// 특정 인덱스에 상태 값 설정
        /// </summary>
        public void SetState(int index, float value)
        {
            if (index >= 0 && index < MaxStateDim)
                stateVector[index] = value;
        }

        /// <summary>
        /// 사용된 길이만큼만 반환 (전송 최적화)
        /// </summary>
        public float[] GetUsedStates()
        {
            // sentinel이 아닌 마지막 인덱스 찾기
            int lastUsedIndex = -1;
            for (int i = MaxStateDim - 1; i >= 0; i--)
            {
                if (stateVector[i] != SentinelValue)
                {
                    lastUsedIndex = i;
                    break;
                }
            }

            // 모두 sentinel이면 빈 배열 반환
            if (lastUsedIndex < 0) return new float[0];

            // 사용된 부분만 복사
            float[] result = new float[lastUsedIndex + 1];
            Array.Copy(stateVector, result, lastUsedIndex + 1);
            return result;
        }

        /// <summary>
        /// 전체 벡터 반환 (디버깅용)
        /// </summary>
        public float[] GetFullVector() => (float[])stateVector.Clone();
    }
}
```

---

## Shader: GrayscaleConvert.shader

GPU에서 RGB → Grayscale 변환을 수행하는 쉐이더.

```hlsl
Shader "Hidden/SGAPS/GrayscaleConvert"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            // Luminance 기반 흑백 변환
            fixed4 frag (v2f i) : SV_Target
            {
                fixed4 col = tex2D(_MainTex, i.uv);

                // ITU-R BT.709 표준 luminance 계산
                float gray = dot(col.rgb, float3(0.2126, 0.7152, 0.0722));

                return fixed4(gray, gray, gray, 1.0);
            }
            ENDCG
        }
    }
}
```

---

## 성능 최적화

### 1. RenderTexture Readback 최적화

**문제**: `Texture2D.ReadPixels()`는 GPU→CPU 동기화로 인해 느림.

**해결**:

```csharp
// AsyncGPUReadback 사용 (Unity 2018.2+)
using UnityEngine.Rendering;

public class AsyncPixelSampler
{
    private AsyncGPUReadbackRequest currentRequest;
    private bool isRequestPending;

    public void SamplePixelsAsync(RenderTexture rt, UVCoordinates uvCoords, Action<PixelData[]> callback)
    {
        if (isRequestPending) return;

        isRequestPending = true;
        currentRequest = AsyncGPUReadback.Request(rt);

        StartCoroutine(WaitForReadback(currentRequest, uvCoords, callback));
    }

    private IEnumerator WaitForReadback(AsyncGPUReadbackRequest request, UVCoordinates uvCoords, Action<PixelData[]> callback)
    {
        yield return new WaitUntil(() => request.done);

        if (!request.hasError)
        {
            var data = request.GetData<byte>();
            PixelData[] samples = ExtractPixelsFromBuffer(data, uvCoords);
            callback(samples);
        }

        isRequestPending = false;
    }
}
```

### 2. 메모리 풀링

```csharp
public class PixelDataPool
{
    private Queue<PixelData[]> pool = new Queue<PixelData[]>();

    public PixelData[] Rent(int size)
    {
        if (pool.Count > 0 && pool.Peek().Length == size)
        {
            return pool.Dequeue();
        }
        return new PixelData[size];
    }

    public void Return(PixelData[] array)
    {
        pool.Enqueue(array);
    }
}
```

### 3. 네트워크 압축

```csharp
using System.IO.Compression;

public static class CompressionUtil
{
    public static byte[] Compress(byte[] data)
    {
        using (var output = new MemoryStream())
        {
            using (var gzip = new GZipStream(output, CompressionMode.Compress))
            {
                gzip.Write(data, 0, data.Length);
            }
            return output.ToArray();
        }
    }

    // MsgPack 직렬화 (JSON보다 빠르고 작음)
    public static byte[] SerializePixelData(PixelData[] pixels)
    {
        // MessagePack-CSharp 사용
        return MessagePackSerializer.Serialize(pixels);
    }
}
```

---

## 에러 핸들링

### 1. 연결 끊김 처리

```csharp
public class ReconnectionManager
{
    private const int MaxRetries = 5;
    private const float RetryDelay = 5f;

    private int retryCount = 0;

    public async Task<bool> ReconnectWithRetry(NetworkClient client)
    {
        while (retryCount < MaxRetries)
        {
            Debug.Log($"[SGAPS] Reconnection attempt {retryCount + 1}/{MaxRetries}");

            bool success = await client.ConnectAsync();
            if (success)
            {
                retryCount = 0;
                return true;
            }

            retryCount++;
            await Task.Delay((int)(RetryDelay * 1000));
        }

        Debug.LogError("[SGAPS] Max reconnection attempts reached.");
        return false;
    }
}
```

### 2. 프레임 버퍼링

```csharp
public class FrameBuffer
{
    private Queue<FrameData> buffer = new Queue<FrameData>();
    private const int MaxBufferSize = 30;  // 3초 @ 10 FPS

    public void Enqueue(FrameData frame)
    {
        if (buffer.Count >= MaxBufferSize)
        {
            buffer.Dequeue();  // 가장 오래된 프레임 제거
        }

        buffer.Enqueue(frame);
    }

    public FrameData[] FlushAll()
    {
        var frames = buffer.ToArray();
        buffer.Clear();
        return frames;
    }
}
```

---

## 테스트

### Unit Tests (Unity Test Framework)

```csharp
using NUnit.Framework;
using SGAPS.Runtime;

public class PixelSamplerTests
{
    [Test]
    public void GenerateUniformGrid_CreatesCorrectCount()
    {
        var sampler = new PixelSampler(400);
        var uvCoords = sampler.GenerateInitialPattern(SamplingPattern.UniformGrid);

        Assert.GreaterOrEqual(uvCoords.coordinates.Count, 400);
    }

    [Test]
    public void SamplePixels_ReturnsCorrectDataStructure()
    {
        var sampler = new PixelSampler(100);
        var uvCoords = sampler.GenerateInitialPattern(SamplingPattern.RandomUniform);

        // Mock RenderTexture
        RenderTexture rt = new RenderTexture(640, 480, 0, RenderTextureFormat.R8);

        var samples = sampler.SamplePixels(rt, uvCoords);

        Assert.AreEqual(uvCoords.coordinates.Count, samples.Length);
        rt.Release();
    }
}
```

---

## 빌드 및 배포

### UPM Package 빌드

```bash
# 1. Git 리포지토리 생성
git init
git add .
git commit -m "Initial commit"

# 2. Tag 버전
git tag v0.1.0

# 3. GitHub에 푸시
git remote add origin https://github.com/yourusername/com.sgaps.client.git
git push -u origin master --tags
```

### Unity에서 설치

```json
// Packages/manifest.json
{
    "dependencies": {
        "com.sgaps.client": "https://github.com/yourusername/com.sgaps.client.git#v0.1.0"
    }
}
```

---

## 다음 문서

-   ✅ CLIENT_IMPLEMENTATION.md (현재 문서)
-   ⏭️ SERVER_IMPLEMENTATION.md - 서버 및 모델 구현
-   ⏭️ API_SPECIFICATION.md - 통신 프로토콜 명세
