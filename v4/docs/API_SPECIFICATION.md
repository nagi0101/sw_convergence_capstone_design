# API Specification

## 개요

SGAPS 클라이언트와 서버 간 통신 프로토콜 명세서입니다. REST API와 WebSocket을 하이브리드 방식으로 사용합니다.

---

## 통신 프로토콜 선택

### 역할 분담

| 프로토콜      | 용도                                    | 특징                      |
| ------------- | --------------------------------------- | ------------------------- |
| **REST API**  | 세션 관리, 설정 변경, 상태 조회         | Request-Response, 멱등성  |
| **WebSocket** | 실시간 프레임 데이터 전송, UV 좌표 수신 | 양방향, 저지연, 지속 연결 |

### WebSocket 선택 이유

-   **낮은 오버헤드**: HTTP handshake 이후 순수 데이터만 전송
-   **양방향 통신**: 서버 → 클라이언트 푸시 가능 (UV 좌표 전송)
-   **실시간성**: 30 FPS 스트리밍에 적합

---

## Base URL

```
Development: ws://localhost:8080
Production:  ws://your-school-server.edu:8080
```

---

## 1. REST API Endpoints

### 1.1 세션 관리

#### POST /api/sessions/create

새로운 캡처 세션을 생성합니다.

**Request**:

```json
{
    "client_name": "UnityClient_001",
    "capture_settings": {
        "resolution": [640, 480],
        "target_fps": 10,
        "grayscale": true
    }
}
```

**Response**:

```json
{
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "ws_endpoint": "ws://localhost:8080/ws/stream",
    "created_at": "2024-01-15T10:30:00Z",
    "expires_at": "2024-01-15T22:30:00Z"
}
```

**Status Codes**:

-   `201`: 세션 생성 성공
-   `400`: 잘못된 요청 (필수 필드 누락)
-   `429`: Too Many Requests (동시 세션 제한 초과)

---

#### GET /api/sessions/{session_id}

세션 정보를 조회합니다.

**Response**:

```json
{
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "active", // "active" | "paused" | "ended"
    "frames_received": 12450,
    "last_activity": "2024-01-15T11:00:00Z",
    "stats": {
        "avg_reconstruction_psnr": 26.5,
        "avg_num_pixels": 485,
        "total_data_received_mb": 125.3
    }
}
```

---

#### DELETE /api/sessions/{session_id}

세션을 종료합니다.

**Response**:

```json
{
    "message": "Session ended successfully",
    "final_stats": {
        "total_frames": 12450,
        "duration_seconds": 1245,
        "avg_fps": 10.0
    }
}
```

---

### 1.2 설정 관리

#### PATCH /api/sessions/{session_id}/config

세션 설정을 업데이트합니다.

**Request**:

```json
{
    "target_fps": 15,
    "mask_update": {
        "mode": "quality_based",
        "quality_based": {
            "low_quality_threshold": 0.75,
            "high_quality_threshold": 0.92
        }
    }
}
```

**Response**:

```json
{
    "message": "Configuration updated",
    "updated_fields": ["target_fps", "mask_update"]
}
```

---

### 1.3 데이터 조회

#### GET /api/sessions/{session_id}/frames

세션의 프레임 목록을 조회합니다.

**Query Parameters**:

-   `offset` (int): 시작 프레임 번호
-   `limit` (int): 최대 프레임 개수 (기본값: 100)

**Response**:

```json
{
  "frames": [
    {
      "frame_id": 0,
      "timestamp": 1234567890.123,
      "num_pixels": 400,
      "reconstruction_psnr": 25.3
    },
    ...
  ],
  "total_count": 12450,
  "offset": 0,
  "limit": 100
}
```

---

#### GET /api/sessions/{session_id}/frames/{frame_id}/download

특정 프레임의 재구성 이미지를 다운로드합니다.

**Response**:

-   **Content-Type**: `image/png`
-   **Body**: PNG 이미지 바이너리

---

## 2. WebSocket Protocol

### 2.1 연결

#### Endpoint

```
ws://server:8080/ws/stream
```

#### Handshake

WebSocket 연결 시 `session_id`를 query parameter로 전달:

```
ws://server:8080/ws/stream?session_id=550e8400-e29b-41d4-a716-446655440000
```

서버는 연결 수락 후 확인 메시지 전송:

```json
{
    "type": "connection_ack",
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "server_time": "2024-01-15T10:30:00Z"
}
```

#### Session Start (연결 직후 1회 전송)

클라이언트는 연결 직후 세션 설정을 서버에 전송:

```json
{
    "type": "session_start",
    "payload": {
        "checkpoint_key": "mario_world1",
        "resolution": [640, 480]
    }
}
```

**필드 설명**:

-   `checkpoint_key` (string): 모델 체크포인트 식별자. 같은 키는 같은 모델을 사용하며, 미등록 시 "default" 사용
-   `resolution` (int[2]): [width, height] - 클라이언트 화면 해상도

> **Note**: `sample_count`, `max_state_dim`, `target_fps`는 서버에서 제어하는 파라미터입니다.
> 클라이언트는 `session_start_ack`에서 이 값들을 받아 사용합니다.
> `sentinel_value`는 서버 내부에서 상태 벡터 패딩에 사용되며, 클라이언트에 전송되지 않습니다.

서버 응답:

```json
{
    "type": "session_start_ack",
    "payload": {
        "checkpoint_key": "mario_world1",
        "checkpoint_loaded": true,
        "model_version": "v1.0.0",
        "sample_count": 500,
        "max_state_dim": 64,
        "target_fps": 10,
        "resolution": [640, 480]
    }
}
```

**응답 필드 설명**:

-   `checkpoint_key` (string): 사용될 체크포인트 키
-   `checkpoint_loaded` (bool): 체크포인트 로드 성공 여부
-   `model_version` (string): 모델 버전
-   `sample_count` (int): 프레임당 샘플링할 픽셀 수 **(서버 제어)**
-   `max_state_dim` (int): 상태 벡터 최대 차원 **(서버 제어)**
-   `target_fps` (int): 목표 캡처 FPS **(서버 제어)**
-   `resolution` (int[2]): 확인된 해상도

> **Note**: `sentinel_value`는 클라이언트에 전송되지 않습니다. 서버에서 상태 벡터 저장 시 `max_state_dim`까지 패딩하는 데 사용하는 내부 값입니다.

````

---

### 2.2 메시지 타입

모든 WebSocket 메시지는 JSON 형식이며, `type` 필드로 구분합니다.

```json
{
  "type": "message_type",
  "payload": { ... }
}
````

---

### 2.3 클라이언트 → 서버 메시지

#### 2.3.1 frame_data

프레임의 희소 픽셀 데이터와 상태 벡터를 전송합니다.

```json
{
  "type": "frame_data",
  "payload": {
    "frame_id": 12345,
    "timestamp": 1234567890.123,
    "resolution": [640, 480],
    "num_pixels": 400,
    "pixels": [
      {"u": 0.0, "v": 0.0, "value": 128},
      {"u": 0.05, "v": 0.0, "value": 200},
      ...
    ],
    "state_vector": [0.5, -0.3, 45.0, 0.8, 1.0]
  }
}
```

**필드 설명**:

-   `frame_id` (int): 클라이언트 측 프레임 번호 (단조 증가)
-   `timestamp` (float): Unity `Time.time` 타임스탬프
-   `resolution` (int[2]): [width, height]
-   `num_pixels` (int): 픽셀 개수
-   `pixels` (array): 픽셀 데이터 배열
    -   `u` (float): 가로 좌표 [0, 1]
    -   `v` (float): 세로 좌표 [0, 1]
    -   `value` (int): 그레이스케일 값 [0, 255]
-   `state_vector` (float[]): 게임 상태 벡터 (가변 길이, 최대 64)
    -   클라이언트가 사용한 길이만큼만 전송
    -   서버에서 sentinel(-999.0)으로 패딩하여 고정 길이로 정규화
    -   빈 배열 `[]` 전송 시 전체 sentinel으로 처리

**상태 벡터 예시** (게임 개발자가 정의):

-   인덱스 0-1: 이동 입력 (horizontal, vertical)
-   인덱스 2: 카메라 Y 회전
-   인덱스 3: 정규화된 HP (0~1)
-   인덱스 4+: 게임별 추가 상태

**최적화**: 대량 픽셀 전송 시 압축 사용 (Phase 3에서 구현):

```json
{
    "type": "frame_data",
    "payload": {
        "frame_id": 12345,
        "timestamp": 1234567890.123,
        "resolution": [640, 480],
        "num_pixels": 400,
        "pixels_compressed": {
            "format": "msgpack+lz4",
            "data": "<base64 encoded binary>"
        }
    }
}
```

---

#### 2.3.2 heartbeat

연결 유지를 위한 Ping 메시지.

```json
{
    "type": "heartbeat",
    "payload": {
        "client_time": 1234567890.123
    }
}
```

서버 응답:

```json
{
    "type": "heartbeat_ack",
    "payload": {
        "server_time": 1234567890.145,
        "latency_ms": 22
    }
}
```

---

### 2.4 서버 → 클라이언트 메시지

#### 2.4.1 uv_coordinates

다음 프레임 샘플링에 사용할 UV 좌표 리스트.

```json
{
  "type": "uv_coordinates",
  "payload": {
    "target_frame_id": 12346,
    "num_coordinates": 485,
    "coordinates": [
      {"u": 0.123, "v": 0.456},
      {"u": 0.789, "v": 0.012},
      ...
    ]
  }
}
```

**필드 설명**:

-   `target_frame_id` (int): 이 좌표를 사용할 프레임 번호
-   `num_coordinates` (int): 좌표 개수 (가변)
-   `coordinates` (array): UV 좌표 배열

**최적화 (Phase 3)**: 양자화된 좌표:

```json
{
    "type": "uv_coordinates",
    "payload": {
        "target_frame_id": 12346,
        "num_coordinates": 485,
        "coordinates_quantized": {
            "format": "uint16",
            "data": "<base64 encoded binary>"
        }
    }
}
```

---

#### 2.4.2 reconstruction_feedback

재구성 결과 피드백 (디버깅/모니터링용).

```json
{
    "type": "reconstruction_feedback",
    "payload": {
        "frame_id": 12345,
        "psnr": 26.8,
        "ssim": 0.87,
        "reconstruction_time_ms": 28,
        "importance_map_summary": {
            "max": 0.95,
            "mean": 0.42,
            "std": 0.18
        }
    }
}
```

---

#### 2.4.3 error

서버 측 오류 메시지.

```json
{
    "type": "error",
    "payload": {
        "code": "INVALID_FRAME_DATA",
        "message": "Frame 12345 has invalid pixel coordinates",
        "details": {
            "invalid_pixels": [{ "u": 1.5, "v": 0.5, "reason": "u > 1.0" }]
        }
    }
}
```

**Error Codes**:

-   `INVALID_FRAME_DATA`: 프레임 데이터 형식 오류
-   `SESSION_EXPIRED`: 세션 만료
-   `SERVER_OVERLOAD`: 서버 과부하
-   `MODEL_ERROR`: 모델 추론 오류

---

## 3. 데이터 포맷 최적화 (Phase 3)

### 3.1 Pixel Data 압축

#### MsgPack + LZ4 압축

**장점**:

-   JSON 대비 ~3-5배 압축
-   빠른 직렬화/역직렬화

**Python (서버)**:

```python
import msgpack
import lz4.frame

# 압축
pixels = [(u, v, value), ...]
packed = msgpack.packb(pixels)
compressed = lz4.frame.compress(packed)
base64_data = base64.b64encode(compressed).decode('utf-8')

# 해제
compressed = base64.b64decode(base64_data)
packed = lz4.frame.decompress(compressed)
pixels = msgpack.unpackb(packed)
```

**C# (Unity)**:

```csharp
using MessagePack;
using K4os.Compression.LZ4;

// 압축
var pixels = new List<(float u, float v, byte value)>();
byte[] packed = MessagePackSerializer.Serialize(pixels);
byte[] compressed = LZ4Pickler.Pickle(packed);
string base64Data = Convert.ToBase64String(compressed);

// 해제
byte[] compressed = Convert.FromBase64String(base64Data);
byte[] packed = LZ4Pickler.Unpickle(compressed);
var pixels = MessagePackSerializer.Deserialize<List<(float, float, byte)>>(packed);
```

---

### 3.2 UV Coordinates 양자화

#### Float32 → UInt16 변환

**압축률**: 4배 (8 bytes → 2 bytes per coordinate)

**Python (서버)**:

```python
import numpy as np

def quantize_uv(uv_coords):
    """[N, 2] float32 → [N, 2] uint16"""
    uv_quantized = (uv_coords * 65535).astype(np.uint16)
    return uv_quantized

def dequantize_uv(uv_quantized):
    """[N, 2] uint16 → [N, 2] float32"""
    uv_coords = uv_quantized.astype(np.float32) / 65535.0
    return uv_coords

# 전송
uv_quantized = quantize_uv(uv_coords)
bytes_data = uv_quantized.tobytes()
base64_data = base64.b64encode(bytes_data).decode('utf-8')
```

**C# (Unity)**:

```csharp
public static List<Vector2> DequantizeUV(byte[] bytes)
{
    var uvCoords = new List<Vector2>();

    for (int i = 0; i < bytes.Length; i += 4)
    {
        ushort u_quantized = BitConverter.ToUInt16(bytes, i);
        ushort v_quantized = BitConverter.ToUInt16(bytes, i + 2);

        float u = u_quantized / 65535f;
        float v = v_quantized / 65535f;

        uvCoords.Add(new Vector2(u, v));
    }

    return uvCoords;
}
```

**정확도 손실**:

-   양자화 간격: 1 / 65535 ≈ 0.0000153
-   640×480 해상도 기준: 약 0.01 픽셀 오차 (무시 가능)

---

## 4. 대역폭 분석

### 4.1 Phase 1 (JSON, 미압축)

#### 클라이언트 → 서버 (상행)

```
단일 프레임:
  - 메타데이터: ~100 bytes
  - 픽셀 데이터: 400 pixels × 40 bytes/pixel (JSON) = 16,000 bytes
  - 총: ~16 KB/frame

@ 10 FPS: 16 KB × 10 = 160 KB/s
```

#### 서버 → 클라이언트 (하행)

```
UV 좌표:
  - 400 coords × 50 bytes/coord (JSON) = 20,000 bytes
  - 총: ~20 KB/frame

@ 10 FPS: 20 KB × 10 = 200 KB/s
```

**총 대역폭**: ~360 KB/s @ 10 FPS

---

### 4.2 Phase 3 (MsgPack+LZ4, 양자화)

#### 클라이언트 → 서버 (상행)

```
단일 프레임:
  - 메타데이터: ~50 bytes (MsgPack)
  - 픽셀 데이터: 400 pixels × 5 bytes/pixel (MsgPack) = 2,000 bytes
  - LZ4 압축: ~1,000 bytes (압축률 2x)
  - 총: ~1 KB/frame

@ 30 FPS: 1 KB × 30 = 30 KB/s
```

#### 서버 → 클라이언트 (하행)

```
UV 좌표 (양자화):
  - 400 coords × 4 bytes/coord (uint16×2) = 1,600 bytes
  - 총: ~1.6 KB/frame

@ 30 FPS: 1.6 KB × 30 = 48 KB/s
```

**총 대역폭**: ~78 KB/s @ 30 FPS

**개선율**: 360 KB/s (10 FPS) → 78 KB/s (30 FPS) = **3배 FPS 증가, 5배 대역폭 절감**

---

## 5. 에러 핸들링

### 5.1 재연결 프로토콜

#### 클라이언트 측 재연결 전략

```csharp
private const int MaxRetries = 5;
private const float BaseRetryDelay = 2f;

public async Task ReconnectWithExponentialBackoff()
{
    for (int retry = 0; retry < MaxRetries; retry++)
    {
        float delay = BaseRetryDelay * Mathf.Pow(2, retry);  // 2, 4, 8, 16, 32초

        Debug.Log($"Reconnecting in {delay}s (attempt {retry + 1}/{MaxRetries})");
        await Task.Delay((int)(delay * 1000));

        bool success = await ConnectAsync();
        if (success) return;
    }

    Debug.LogError("Max reconnection attempts reached. Giving up.");
}
```

---

### 5.2 프레임 손실 처리

#### 서버 측 Gap Detection

```python
class FrameGapDetector:
    def __init__(self):
        self.last_frame_id = -1
        self.gaps = []

    def check_gap(self, frame_id):
        if self.last_frame_id == -1:
            self.last_frame_id = frame_id
            return None

        expected = self.last_frame_id + 1
        if frame_id != expected:
            gap = (expected, frame_id - 1)
            self.gaps.append(gap)
            print(f"[WARNING] Frame gap detected: {gap}")

        self.last_frame_id = frame_id
        return gap
```

서버는 gap 발견 시 클라이언트에 알림:

```json
{
    "type": "frame_gap_detected",
    "payload": {
        "missing_frames": [12340, 12341, 12342],
        "action": "continue" // "continue" | "resend_request"
    }
}
```

---

### 5.3 타임아웃

#### WebSocket 타임아웃 설정

```python
# FastAPI WebSocket 타임아웃
@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            # 60초 타임아웃
            data = await asyncio.wait_for(
                websocket.receive_json(),
                timeout=60.0
            )
            # 처리...

    except asyncio.TimeoutError:
        await websocket.send_json({
            "type": "error",
            "payload": {
                "code": "TIMEOUT",
                "message": "No data received for 60 seconds"
            }
        })
        await websocket.close()
```

---

## 6. 보안

### 6.1 인증 (Optional)

#### Bearer Token 방식

클라이언트는 REST API로 토큰 발급:

```http
POST /api/auth/login
Content-Type: application/json

{
  "username": "client001",
  "password": "secret"
}
```

응답:

```json
{
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "bearer",
    "expires_in": 3600
}
```

WebSocket 연결 시 헤더에 포함:

```
ws://server:8080/ws/stream?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

---

### 6.2 Rate Limiting

```python
from fastapi import WebSocket, HTTPException
from collections import defaultdict
import time

class RateLimiter:
    def __init__(self, max_requests=100, window_seconds=60):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests = defaultdict(list)

    def check_rate_limit(self, client_id):
        now = time.time()

        # 윈도우 밖 요청 제거
        self.requests[client_id] = [
            t for t in self.requests[client_id]
            if now - t < self.window
        ]

        # 제한 확인
        if len(self.requests[client_id]) >= self.max_requests:
            raise HTTPException(status_code=429, detail="Too many requests")

        self.requests[client_id].append(now)
```

---

## 7. 모니터링

### 7.1 Health Check

```http
GET /api/health

Response:
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "uptime_seconds": 86400,
  "active_connections": 3,
  "gpu_utilization": 0.75,
  "gpu_memory_used_mb": 3072
}
```

---

### 7.2 Metrics Endpoint (Prometheus 형식)

```http
GET /api/metrics

Response (text/plain):
# HELP sgaps_frames_received_total Total frames received
# TYPE sgaps_frames_received_total counter
sgaps_frames_received_total 124500

# HELP sgaps_reconstruction_psnr_avg Average PSNR
# TYPE sgaps_reconstruction_psnr_avg gauge
sgaps_reconstruction_psnr_avg 26.5

# HELP sgaps_inference_time_seconds Inference time
# TYPE sgaps_inference_time_seconds histogram
sgaps_inference_time_seconds_bucket{le="0.01"} 120
sgaps_inference_time_seconds_bucket{le="0.03"} 980
sgaps_inference_time_seconds_bucket{le="0.05"} 1200
```

---

## 다음 문서

-   ✅ API_SPECIFICATION.md (현재 문서)
-   ⏭️ CONFIGURATION.md - 설정 시스템 설계
-   ⏭️ DEVELOPMENT_ROADMAP.md - 개발 로드맵
