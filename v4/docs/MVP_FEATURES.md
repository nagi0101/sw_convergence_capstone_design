# MVP 기능 명세서

## MVP 개발 철학

### 핵심 원칙

1. **빠른 검증**: 각 Phase마다 동작하는 End-to-End 시스템 구축
2. **점진적 개선**: 단순한 것부터 시작하여 복잡도 증가
3. **측정 가능**: 모든 Phase에 명확한 성공 기준 설정
4. **학습 중심**: 초기에는 재구성 품질보다 데이터 수집에 집중

---

## Phase 1: 기본 통신 및 고정 샘플링 (Foundation)

### 목표

클라이언트와 서버 간 기본 통신을 확립하고, 고정된 위치의 픽셀을 샘플링하여 재구성하는 기준선(Baseline) 구축

### 기간

2주 (Week 1-2)

### 구현 기능

#### 1.1 Unity Client (UPM Plugin)

##### 1.1.1 RenderTexture 캡처

```csharp
// 기능
- Main Camera의 렌더링 결과를 Grayscale RenderTexture로 캡처
- 해상도: 설정 가능 (기본값 640×480)
- 프레임레이트: 설정 가능 (기본값 10 FPS - 부하 테스트용)

// 성공 기준
- RenderTexture가 Unity Scene View에서 미리보기 가능
- 흑백 변환이 올바르게 적용됨 (RGB → Y 변환)
- FPS 영향 < 0.5 fps
```

##### 1.1.2 고정 픽셀 샘플링

```csharp
// 기능
- 미리 정의된 UV 좌표 리스트에서 픽셀 값 추출
- 샘플링 패턴: 균등 그리드 (예: 20×20 = 400 pixels)
- CPU에서 Texture2D.GetPixel() 사용

// 성공 기준
- 정확한 UV 좌표에서 픽셀 값 추출 확인
- 샘플링 시간 < 5ms per frame
```

##### 1.1.3 WebSocket 클라이언트

```csharp
// 기능
- 서버에 WebSocket 연결
- 샘플링된 픽셀 데이터 + 상태 벡터 전송 (JSON 형식)
- 연결 상태 모니터링 (Inspector에 표시)

// 세션 시작 메시지 (연결 직후 1회 전송)
{
  "type": "session_start",
  "checkpoint_key": "mario_world1",  // 모델 체크포인트 식별자
  "max_state_dim": 64,               // 상태 벡터 최대 길이
  "resolution": [640, 480]
}

// 프레임 데이터 포맷
{
  "type": "frame",
  "frame_id": 12345,
  "timestamp": 1234567890.123,
  "pixels": [
    {"u": 0.0, "v": 0.0, "value": 128},
    {"u": 0.05, "v": 0.0, "value": 200},
    ...
  ],
  "state_vector": [0.5, -0.3, 45.0, 0.8]  // 사용된 길이만큼만 전송
}

// 성공 기준
- 서버에 초당 10개 프레임 전송 성공
- 패킷 손실률 < 1%
- 연결 끊김 시 자동 재연결
- 세션 시작 시 checkpoint_key가 서버에 전달됨
```

##### 1.1.4 상태 벡터 수집

```csharp
// 기능
- 게임 상태를 고정 길이 float 배열로 수집
- 최대 길이: 64 (설정 가능, 서버와 동일해야 함)
- 미사용 인덱스는 sentinel 값(-999.0)으로 초기화
- 클라이언트가 필요한 만큼만 값 설정 (나머지는 sentinel 유지)

// 예시 사용법 (게임 개발자가 정의)
public class GameStateCollector : MonoBehaviour {
    [SerializeField] private SGAPSClient sgapsClient;

    void Update() {
        // 게임별로 필요한 상태만 기록
        sgapsClient.SetState(0, Input.GetAxis("Horizontal"));  // 이동 입력
        sgapsClient.SetState(1, Input.GetAxis("Vertical"));
        sgapsClient.SetState(2, Camera.main.transform.eulerAngles.y);  // 카메라 방향
        sgapsClient.SetState(3, player.health / 100f);  // 정규화된 HP
        // 인덱스 4~63은 sentinel(-999.0)으로 유지됨
    }
}

// StateVectorCollector 내부 구현
public class StateVectorCollector {
    public const int MAX_STATE_DIM = 64;
    public const float SENTINEL_VALUE = -999.0f;

    private float[] stateVector = new float[MAX_STATE_DIM];

    public StateVectorCollector() {
        Reset();  // sentinel로 초기화
    }

    public void Reset() {
        for (int i = 0; i < MAX_STATE_DIM; i++)
            stateVector[i] = SENTINEL_VALUE;
    }

    public void SetState(int index, float value) {
        if (index >= 0 && index < MAX_STATE_DIM)
            stateVector[index] = value;
    }

    public float[] GetUsedStates() {
        // sentinel이 아닌 마지막 인덱스까지만 반환 (전송 최적화)
        int lastUsedIndex = -1;
        for (int i = MAX_STATE_DIM - 1; i >= 0; i--) {
            if (stateVector[i] != SENTINEL_VALUE) {
                lastUsedIndex = i;
                break;
            }
        }
        if (lastUsedIndex < 0) return new float[0];

        float[] result = new float[lastUsedIndex + 1];
        Array.Copy(stateVector, result, lastUsedIndex + 1);
        return result;
    }
}

// 성공 기준
- 상태 벡터가 프레임 데이터와 함께 전송됨
- 미사용 인덱스는 sentinel 값 유지
- 상태 수집 오버헤드 < 0.1ms per frame
```

##### 1.1.5 설정 UI (Inspector)

```csharp
// 노출 파라미터
- Server Endpoint: string (예: "ws://server.example.com:8080")
- Target FPS: int (기본값 10)
- Resolution: Vector2Int (기본값 640×480)
- Checkpoint Key: string (기본값 "default") // 모델 체크포인트 식별자
- Enable Logging: bool

// Checkpoint Key 사용 예시
// - 같은 게임/맵: 동일한 키 사용 → 같은 모델로 추론
// - 다른 게임/맵: 다른 키 사용 → 별도 모델 학습/로드
// 예: "mario_world1", "mario_world2", "tetris_main"

// 성공 기준
- Unity Inspector에서 실시간 값 변경 가능
- 값 변경 시 즉시 반영 (재시작 불필요)
- Checkpoint Key가 세션 시작 시 서버에 전달됨
```

#### 1.2 Server Backend

##### 1.2.1 FastAPI + WebSocket 서버

```python
# 기능
- WebSocket 엔드포인트 `/ws/stream` 구현
- 클라이언트로부터 픽셀 데이터 + 상태 벡터 수신
- 세션별 checkpoint_key 관리
- 수신 데이터 콘솔 로깅 (디버깅용)

# 세션 관리
class SessionManager:
    def __init__(self):
        self.sessions = {}  # websocket_id → SessionInfo

    def register_session(self, ws_id, session_start_msg):
        self.sessions[ws_id] = SessionInfo(
            checkpoint_key=session_start_msg['checkpoint_key'],
            max_state_dim=session_start_msg.get('max_state_dim', 64),
            resolution=tuple(session_start_msg['resolution'])
        )

    def get_checkpoint_key(self, ws_id) -> str:
        return self.sessions[ws_id].checkpoint_key

# 상태 벡터 정규화 (수신 시)
MAX_STATE_DIM = 64
SENTINEL_VALUE = -999.0

def normalize_state_vector(state_vector: list) -> np.ndarray:
    """가변 길이 상태 벡터를 고정 길이로 패딩"""
    normalized = np.full(MAX_STATE_DIM, SENTINEL_VALUE, dtype=np.float32)
    if state_vector:
        normalized[:len(state_vector)] = state_vector
    return normalized

# 성공 기준
- 동시에 3개 클라이언트 연결 가능
- 메시지 처리 지연 < 10ms
- 서버 크래시 없이 24시간 연속 동작
- checkpoint_key별로 세션 분리 관리
```

##### 1.2.2 희소 픽셀 → 프레임 복원 (단순 버전)

```python
# 알고리즘
- Nearest Neighbor Interpolation (NN)
- 또는 Bilinear Interpolation

def reconstruct_frame_simple(sparse_pixels, resolution):
    # 1. 빈 프레임 생성
    frame = np.zeros(resolution)

    # 2. 샘플링된 픽셀 채우기
    for pixel in sparse_pixels:
        x = int(pixel.u * resolution[0])
        y = int(pixel.v * resolution[1])
        frame[y, x] = pixel.value

    # 3. 보간법으로 나머지 채우기
    frame = cv2.inpaint(frame, mask, cv2.INPAINT_TELEA)

    return frame

# 성공 기준
- PSNR > 20 dB (매우 낮은 기준, 품질보다 동작 확인)
- 복원 시간 < 50ms per frame
- 복원된 프레임을 PNG로 저장 가능
```

##### 1.2.3 UV 좌표 생성 (고정 패턴)

```python
# 기능
- 클라이언트에게 전송할 UV 좌표 리스트 생성
- 패턴: 균등 그리드 (현재는 고정, Phase 2에서 동적으로)

def generate_fixed_uv_grid(num_samples):
    # 예: 20×20 그리드
    grid_size = int(np.sqrt(num_samples))
    uv_coords = []
    for i in range(grid_size):
        for j in range(grid_size):
            u = i / grid_size
            v = j / grid_size
            uv_coords.append({"u": u, "v": v})
    return uv_coords

# 성공 기준
- WebSocket으로 클라이언트에 UV 좌표 전송
- 클라이언트가 수신하여 다음 프레임 샘플링에 사용
```

##### 1.2.4 데이터 저장

```python
# 기능
- 수신한 픽셀 데이터 + 상태 벡터를 파일 시스템에 저장
- 포맷: 에피소드별 HDF5 파일
- 체크포인트 키별로 디렉토리 분리

# 디렉토리 구조
data/
  {checkpoint_key}/           # 예: mario_world1/
    episode_001.h5
    episode_002.h5
    ...

# HDF5 파일 구조
episode_001.h5
  /metadata
    checkpoint_key: string    # 모델 식별자
    max_state_dim: int        # 상태 벡터 최대 길이 (64)
    resolution: [int, int]    # 해상도
  /frames
    /frame_0000
      /pixels (N×3 array: u, v, value)
      /state_vector (M array: float, M <= max_state_dim)
      /timestamp (float)
    /frame_0001
      ...

# 상태 벡터 저장 시 주의사항
- 클라이언트가 전송한 길이만 저장 (가변 길이)
- 로딩 시 sentinel(-999.0)으로 패딩하여 고정 길이로 정규화

def load_state_vector(frame_group, max_dim=64, sentinel=-999.0):
    raw = frame_group['state_vector'][:]
    padded = np.full(max_dim, sentinel, dtype=np.float32)
    padded[:len(raw)] = raw
    return padded

# 성공 기준
- 30분 게임플레이 (~18,000 프레임) 저장 성공
- 파일 크기 < 500 MB
- 데이터 로딩 시 손상 없음
- 상태 벡터가 올바르게 저장/복원됨
```

#### 1.3 Monitoring & Debugging

##### 1.3.1 서버 대시보드 (Streamlit)

```python
# 기능
- 실시간 연결된 클라이언트 수
- 초당 수신 프레임 수 (FPS)
- 최근 10개 재구성 프레임 미리보기
- 네트워크 대역폭 사용량

# 성공 기준
- 브라우저에서 실시간 업데이트 확인
- 재구성 프레임 품질 육안 확인 가능
```

### Phase 1 성공 기준 (전체)

| 메트릭               | 목표                     | 측정 방법          |
| -------------------- | ------------------------ | ------------------ |
| End-to-End 동작      | ✅ Unity → 서버 → 재구성 | 수동 테스트        |
| 클라이언트 FPS 영향  | < 1 fps 감소             | Unity Profiler     |
| 네트워크 상행 대역폭 | < 50 KB/s @ 10 FPS       | Wireshark          |
| 서버 처리 지연       | < 100ms                  | 타임스탬프 비교    |
| 재구성 PSNR          | > 20 dB                  | cv2.PSNR()         |
| 데이터 수집          | 30분 에피소드 10개       | 파일 확인          |
| 상태 벡터 수집       | ✅ 프레임과 함께 저장    | HDF5 파일 확인     |
| 체크포인트 키        | ✅ 세션별 분리 저장      | 디렉토리 구조 확인 |

### Phase 1 Deliverables

-   [ ] Unity UPM Package (`.tgz` 파일)
-   [ ] FastAPI 서버 코드 (`server/` 디렉토리)
-   [ ] Docker Compose 설정 (서버 실행용)
-   [ ] README: 설치 및 실행 가이드
-   [ ] 샘플 데이터셋 (1개 에피소드)

---

## Phase 2: 적응적 마스크 업데이트

### 목표

고정 샘플링 패턴에서 벗어나, 서버가 재구성 품질에 따라 **동적으로** UV 좌표를 결정하는 시스템 구축

### 기간

2주 (Week 3-4)

### 구현 기능

#### 2.1 Importance Map 계산

##### 2.1.1 재구성 오류 기반 중요도

```python
class ImportanceCalculator:
    def compute(self, reconstructed_frame, sparse_pixels):
        # 1. Ground Truth 근사
        # (완전한 GT는 없으므로, 이전 프레임 + 샘플 픽셀로 추정)
        gt_approx = self._approximate_ground_truth(
            reconstructed_frame, sparse_pixels
        )

        # 2. 픽셀별 오류 계산
        error_map = np.abs(reconstructed_frame - gt_approx)

        # 3. 엣지 검출 (고주파 영역 강조)
        edges = cv2.Sobel(reconstructed_frame, cv2.CV_64F, 1, 1)

        # 4. 가중 합산
        importance_map = 0.6 * error_map + 0.4 * np.abs(edges)

        # 5. 정규화
        importance_map /= importance_map.max()

        return importance_map

# 성공 기준
- Importance map이 시각적으로 합리적 (엣지/변화 영역이 밝음)
- 계산 시간 < 20ms
```

##### 2.1.2 시간적 변화 추적

```python
class TemporalTracker:
    def __init__(self):
        self.previous_frame = None

    def update(self, current_frame):
        if self.previous_frame is None:
            temporal_importance = np.zeros_like(current_frame)
        else:
            # 프레임 간 차이 (모션 영역)
            temporal_importance = np.abs(
                current_frame - self.previous_frame
            )

        self.previous_frame = current_frame.copy()
        return temporal_importance

# 성공 기준
- 움직이는 객체 영역에서 높은 중요도
- 정적 배경에서 낮은 중요도
```

#### 2.2 적응적 UV 좌표 생성

##### 2.2.1 중요도 기반 샘플링

```python
def generate_adaptive_uv(importance_map, num_samples, strategy="hybrid"):
    if strategy == "importance_only":
        # 100% 중요도 기반
        uv_coords = weighted_random_sample(
            importance_map, num_samples
        )

    elif strategy == "hybrid":
        # 70% 중요도 + 30% 균등
        important_count = int(num_samples * 0.7)
        uniform_count = num_samples - important_count

        uv_important = weighted_random_sample(
            importance_map, important_count
        )
        uv_uniform = uniform_random_sample(uniform_count)

        uv_coords = uv_important + uv_uniform

    # 최소 거리 제약 (클러스터링 방지)
    uv_coords = enforce_min_distance(
        uv_coords, min_distance=0.02  # 2% of image
    )

    return uv_coords

# 성공 기준
- 중요 영역에 샘플 밀집도 증가 확인 (시각화)
- 균등 샘플이 전체 프레임 커버리지 보장
```

#### 2.3 마스크 업데이트 주기 설정

##### 2.3.1 설정 기반 업데이트

```python
# 설정 파일 (config.yaml)
mask_update:
  mode: "fixed"  # "fixed" | "adaptive" | "quality_based"

  # Fixed mode
  fixed_interval: 5  # 5 프레임마다 업데이트

  # Adaptive mode
  adaptive:
    min_interval: 3
    max_interval: 10
    quality_threshold: 0.85  # SSIM

  # Quality-based mode
  quality_based:
    low_quality_threshold: 0.7  # SSIM < 0.7 → 즉시 업데이트
    high_quality_threshold: 0.9  # SSIM > 0.9 → 업데이트 주기 2배로

# 구현
class MaskUpdateScheduler:
    def should_update(self, frame_idx, reconstruction_quality):
        if self.config.mode == "fixed":
            return (frame_idx % self.config.fixed_interval) == 0

        elif self.config.mode == "adaptive":
            if reconstruction_quality < self.config.adaptive.quality_threshold:
                return True
            else:
                return (frame_idx % self.config.adaptive.min_interval) == 0

        elif self.config.mode == "quality_based":
            if reconstruction_quality < self.config.quality_based.low_quality_threshold:
                return True  # 즉시
            elif reconstruction_quality > self.config.quality_based.high_quality_threshold:
                return (frame_idx % 10) == 0  # 느리게
            else:
                return (frame_idx % 5) == 0  # 보통

# 성공 기준
- 각 모드별 동작 확인
- Wandb에 업데이트 이벤트 로깅
- 품질이 낮을 때 업데이트 빈도 증가 확인
```

#### 2.4 Client-Side 업데이트 수신

##### 2.4.1 동적 UV 좌표 수신

```csharp
// Unity 클라이언트 수정
void OnUVCoordinatesReceived(string json) {
    var uvData = JsonConvert.DeserializeObject<UVCoordinates>(json);

    // 다음 프레임 샘플링 시 사용할 좌표 업데이트
    this.nextFrameUVs = uvData.coordinates;

    // 로그
    Debug.Log($"Received {uvData.coordinates.Count} UV coords for frame {uvData.target_frame_id}");
}

// 성공 기준
- 서버에서 전송한 UV 좌표를 올바르게 파싱
- 다음 프레임 샘플링에 반영 확인
```

#### 2.5 품질 개선 평가

##### 2.5.1 A/B 테스트 프레임워크

```python
# 실험 설정
experiments = {
    "baseline": {
        "sampling": "uniform_grid",
        "num_samples": 400
    },
    "adaptive_v1": {
        "sampling": "importance_only",
        "num_samples": 400,
        "update_interval": 5
    },
    "adaptive_v2": {
        "sampling": "hybrid",
        "num_samples": 400,
        "update_interval": "quality_based"
    }
}

# 각 실험마다 동일한 에피소드로 평가
# Wandb에 실험별 메트릭 기록

# 성공 기준
- Adaptive v2가 Baseline 대비 PSNR +3dB 이상
- 샘플링 효율: 동일 픽셀 수로 더 높은 품질
```

### Phase 2 성공 기준 (전체)

| 메트릭                   | 목표                | 측정 방법              |
| ------------------------ | ------------------- | ---------------------- |
| 재구성 PSNR              | > 25 dB             | cv2.PSNR()             |
| 재구성 SSIM              | > 0.85              | skimage.metrics.ssim() |
| 적응적 샘플링 효과       | Baseline 대비 +3dB  | A/B 테스트             |
| 마스크 업데이트 오버헤드 | < 10ms              | Python profiler        |
| 네트워크 하행 대역폭     | < 100 KB/s @ 10 FPS | Wireshark              |
| UV 좌표 압축률           | > 2x (vs. JSON)     | MsgPack 사용           |

### Phase 2 Deliverables

-   [ ] Importance Calculator 모듈
-   [ ] Adaptive UV Generator
-   [ ] Mask Update Scheduler (3가지 모드)
-   [ ] A/B 테스트 결과 리포트 (Wandb)
-   [ ] 업데이트된 Unity Plugin
-   [ ] 설정 파일 예시 (`config/adaptive_v2.yaml`)

---

## Phase 3: 학습 및 최적화

### 목표

단순 보간법을 넘어, 딥러닝 모델(Sparse Pixel Transformer)을 학습하여 재구성 품질을 비약적으로 향상

### 기간

3-4주 (Week 5-8)

### 구현 기능

#### 3.1 데이터 전처리 파이프라인

##### 3.1.1 학습 데이터셋 생성

```python
# Phase 1-2에서 수집한 데이터 활용
class SGAPSDataset(torch.utils.data.Dataset):
    def __init__(self, episode_dir, config):
        self.episodes = self._load_episodes(episode_dir)
        self.config = config
        self.max_state_dim = config.max_state_dim  # 64
        self.sentinel_value = config.sentinel_value  # -999.0

    def __getitem__(self, idx):
        # 1. 에피소드 + 프레임 선택
        episode, frame_idx = self._get_episode_frame(idx)

        # 2. 희소 픽셀 데이터 로드
        sparse_pixels = episode.get_frame_pixels(frame_idx)

        # 3. 상태 벡터 로드 (고정 길이로 패딩됨)
        state_vector = episode.get_state_vector(frame_idx)  # [max_state_dim]

        # 4. Ground Truth 프레임 (필요 시 원본 게임에서 재생성)
        gt_frame = episode.get_ground_truth(frame_idx)

        # 5. 상태 벡터 마스크 생성 (sentinel이 아닌 위치 = 1)
        state_mask = (state_vector != self.sentinel_value).astype(np.float32)

        # 6. 텐서 변환
        return {
            "sparse_pixels": torch.tensor(sparse_pixels),  # [N, 3] (u, v, value)
            "num_pixels": len(sparse_pixels),
            "state_vector": torch.tensor(state_vector),    # [max_state_dim]
            "state_mask": torch.tensor(state_mask),        # [max_state_dim] (0 or 1)
            "gt_frame": torch.tensor(gt_frame),            # [H, W, 1]
            "resolution": episode.resolution
        }

# 성공 기준
- 10개 에피소드 (각 30분) → ~180,000 프레임
- 데이터 로딩 속도 > 100 samples/sec
- Train/Val/Test 스플릿: 70/15/15
- 상태 벡터와 마스크가 올바르게 로드됨
```

##### 3.1.2 데이터 증강

```python
# 학습 시 적용
def augment(sparse_pixels, gt_frame):
    # 1. 랜덤 플립 (좌우)
    if random.random() > 0.5:
        sparse_pixels = flip_horizontal(sparse_pixels)
        gt_frame = flip_horizontal(gt_frame)

    # 2. 밝기 조정 (±20%)
    brightness_factor = random.uniform(0.8, 1.2)
    sparse_pixels[:, 2] *= brightness_factor  # value 채널
    gt_frame *= brightness_factor

    # 3. 노이즈 추가 (샘플 픽셀에만)
    noise = np.random.normal(0, 5, sparse_pixels[:, 2].shape)
    sparse_pixels[:, 2] += noise

    return sparse_pixels, gt_frame

# 성공 기준
- 증강 후에도 UV 좌표 일관성 유지
- 밝기/노이즈 범위가 합리적
```

#### 3.2 Sparse Pixel Transformer 구현

##### 3.2.1 모델 아키텍처

```python
class SparsePixelTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Pixel Embedding
        self.pixel_embed = nn.Linear(3, config.embed_dim)  # (u, v, value) → 256D

        # 2. State Vector Encoder (고정 길이 입력)
        self.state_encoder = StateVectorEncoder(
            max_state_dim=config.max_state_dim,  # 64
            embed_dim=config.embed_dim,          # 256
            sentinel_value=config.sentinel_value # -999.0
        )

        # 3. Positional Encoding (Continuous)
        self.pos_encoder = ContinuousPositionalEncoding(config.embed_dim)

        # 4. Sparse Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=8,
            dim_feedforward=1024,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # 5. State-Pixel Cross-Attention (상태 벡터와 픽셀 특징 결합)
        self.state_pixel_attention = nn.MultiheadAttention(
            embed_dim=config.embed_dim,
            num_heads=8,
            batch_first=True
        )

        # 6. Cross-Attention Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.embed_dim,
            nhead=8,
            dim_feedforward=1024,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

        # 7. CNN Refinement Head
        self.refine_head = nn.Sequential(
            nn.Conv2d(config.embed_dim, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),  # Grayscale output
            nn.Sigmoid()
        )

    def forward(self, sparse_pixels, state_vector, state_mask, resolution):
        # sparse_pixels: [B, N, 3] (batch, num_pixels, (u,v,value))
        # state_vector: [B, max_state_dim] (고정 길이, sentinel 포함)
        # state_mask: [B, max_state_dim] (0: sentinel, 1: valid)
        B, N, _ = sparse_pixels.shape
        H, W = resolution

        # 1. 픽셀 임베딩
        pixel_embeds = self.pixel_embed(sparse_pixels)  # [B, N, 256]
        pixel_embeds = self.pos_encoder(pixel_embeds, sparse_pixels[:, :, :2])

        # 2. 상태 벡터 임베딩 (마스킹 적용)
        state_embeds = self.state_encoder(state_vector, state_mask)  # [B, 1, 256]

        # 3. Encoder: 희소 픽셀 간 관계 학습
        encoded = self.encoder(pixel_embeds)  # [B, N, 256]

        # 4. 상태 벡터와 픽셀 특징 결합 (Cross-Attention)
        # 상태 벡터가 Query, 인코딩된 픽셀이 Key/Value
        state_conditioned, _ = self.state_pixel_attention(
            query=state_embeds,
            key=encoded,
            value=encoded
        )  # [B, 1, 256]

        # 상태 정보를 모든 픽셀 임베딩에 broadcast하여 결합
        encoded = encoded + state_conditioned.expand(-1, N, -1)  # [B, N, 256]

        # 5. Query 생성: 전체 프레임의 각 픽셀 위치
        query_positions = self._generate_query_grid(B, H, W)  # [B, H*W, 2]
        query_embeds = self.pos_encoder(
            torch.zeros(B, H*W, self.config.embed_dim, device=sparse_pixels.device),
            query_positions
        )

        # 6. Decoder: Cross-attention으로 각 위치의 값 예측
        decoded = self.decoder(query_embeds, encoded)  # [B, H*W, 256]

        # 7. Reshape + CNN Refinement
        decoded = decoded.view(B, H, W, self.config.embed_dim).permute(0, 3, 1, 2)
        output = self.refine_head(decoded)  # [B, 1, H, W]

        return output


class StateVectorEncoder(nn.Module):
    """고정 길이 상태 벡터를 임베딩으로 변환 (sentinel 마스킹 적용)"""

    def __init__(self, max_state_dim, embed_dim, sentinel_value=-999.0):
        super().__init__()
        self.max_state_dim = max_state_dim
        self.sentinel_value = sentinel_value

        # 상태 벡터 → 임베딩
        self.linear = nn.Linear(max_state_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, state_vector, state_mask):
        # state_vector: [B, max_state_dim]
        # state_mask: [B, max_state_dim] (0: sentinel, 1: valid)

        # sentinel 위치를 0으로 마스킹 (학습에 영향 없도록)
        masked_state = state_vector * state_mask

        # 임베딩 생성
        embeds = self.linear(masked_state)  # [B, embed_dim]
        embeds = self.norm(embeds)

        return embeds.unsqueeze(1)  # [B, 1, embed_dim]


# 성공 기준
- 모델 파라미터 < 50M (추론 속도 고려)
- 단일 프레임 추론 < 33ms @ RTX 3090
- Gradient checkpointing으로 메모리 < 8GB
- 상태 벡터 활용으로 재구성 품질 향상 (A/B 테스트)
```

##### 3.2.2 손실 함수

```python
class SGAPSLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mse_weight = config.loss.mse_weight
        self.perceptual_weight = config.loss.perceptual_weight
        self.sparsity_weight = config.loss.sparsity_weight

        # VGG for perceptual loss (optional)
        if self.perceptual_weight > 0:
            self.vgg = torchvision.models.vgg16(pretrained=True).features[:16]
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def forward(self, pred, gt, num_pixels):
        # 1. 재구성 손실 (MSE)
        mse_loss = F.mse_loss(pred, gt)

        # 2. 지각 손실 (선택 사항)
        if self.perceptual_weight > 0:
            # Grayscale → RGB (VGG 입력용)
            pred_rgb = pred.repeat(1, 3, 1, 1)
            gt_rgb = gt.repeat(1, 3, 1, 1)

            pred_features = self.vgg(pred_rgb)
            gt_features = self.vgg(gt_rgb)

            perceptual_loss = F.mse_loss(pred_features, gt_features)
        else:
            perceptual_loss = 0.0

        # 3. 희소성 정규화 (픽셀 개수가 너무 많아지지 않도록)
        # 목표: 평균 500 pixels
        sparsity_loss = F.relu(num_pixels.float().mean() - 500) / 500

        # 총 손실
        total_loss = (
            self.mse_weight * mse_loss +
            self.perceptual_weight * perceptual_loss +
            self.sparsity_weight * sparsity_loss
        )

        return {
            "total": total_loss,
            "mse": mse_loss,
            "perceptual": perceptual_loss,
            "sparsity": sparsity_loss
        }

# 성공 기준
- 학습 초기: MSE 손실 감소 확인
- Perceptual loss 추가 시 시각적 품질 향상 확인
```

#### 3.3 학습 파이프라인

##### 3.3.1 Trainer 구현

```python
class SGAPSTrainer:
    def __init__(self, model, dataloader, config):
        self.model = model
        self.dataloader = dataloader
        self.config = config

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.num_epochs
        )

        # Mixed Precision
        self.scaler = torch.cuda.amp.GradScaler()

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        for batch in tqdm(self.dataloader):
            sparse_pixels = batch["sparse_pixels"].cuda()
            gt_frame = batch["gt_frame"].cuda()
            resolution = batch["resolution"][0]

            with torch.cuda.amp.autocast():
                pred = self.model(sparse_pixels, resolution)
                losses = self.criterion(pred, gt_frame, batch["num_pixels"])

            # Backward
            self.optimizer.zero_grad()
            self.scaler.scale(losses["total"]).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += losses["total"].item()

            # Wandb 로깅
            wandb.log({
                "train/loss": losses["total"].item(),
                "train/mse": losses["mse"].item(),
                "train/lr": self.optimizer.param_groups[0]["lr"]
            })

        return total_loss / len(self.dataloader)

# 성공 기준
- 100 에폭 학습 완료 (RTX 3090 기준 ~24시간)
- 학습 loss 곡선이 수렴
- Validation PSNR > 28 dB
```

##### 3.3.2 평가 메트릭

```python
def evaluate(model, val_loader):
    model.eval()
    psnrs, ssims = [], []

    with torch.no_grad():
        for batch in val_loader:
            sparse_pixels = batch["sparse_pixels"].cuda()
            gt_frame = batch["gt_frame"].cuda()

            pred = model(sparse_pixels, batch["resolution"][0])

            # 메트릭 계산
            psnr = calculate_psnr(pred, gt_frame)
            ssim = calculate_ssim(pred, gt_frame)

            psnrs.append(psnr)
            ssims.append(ssim)

    return {
        "psnr": np.mean(psnrs),
        "ssim": np.mean(ssims)
    }

# 성공 기준
- PSNR > 28 dB
- SSIM > 0.90
- 육안으로 구분 어려운 품질
```

#### 3.4 서버 통합 및 추론 최적화

##### 3.4.1 학습된 모델 배포 및 체크포인트 관리

```python
# FastAPI 서버 업데이트
class CheckpointManager:
    """체크포인트 키 기반 모델 관리"""

    def __init__(self, checkpoint_dir: str, config):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config = config
        self.loaded_models = {}  # checkpoint_key → model
        self.default_key = "default"

    def get_model(self, checkpoint_key: str) -> nn.Module:
        """체크포인트 키에 해당하는 모델 반환 (lazy loading)"""
        if checkpoint_key not in self.loaded_models:
            model_path = self.checkpoint_dir / checkpoint_key / "best.pth"

            if not model_path.exists():
                # 해당 키의 체크포인트가 없으면 default 사용
                print(f"Checkpoint '{checkpoint_key}' not found, using default")
                model_path = self.checkpoint_dir / self.default_key / "best.pth"

            model = self._load_model(model_path)
            self.loaded_models[checkpoint_key] = model

        return self.loaded_models[checkpoint_key]

    def _load_model(self, model_path: Path) -> nn.Module:
        model = SparsePixelTransformer(self.config)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        model.cuda()
        return torch.jit.script(model)

    def register_new_checkpoint(self, checkpoint_key: str, model: nn.Module):
        """새로운 체크포인트 저장"""
        save_dir = self.checkpoint_dir / checkpoint_key
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "best.pth")
        self.loaded_models[checkpoint_key] = model


class InferenceEngine:
    def __init__(self, checkpoint_dir, config):
        self.checkpoint_manager = CheckpointManager(checkpoint_dir, config)
        self.config = config
        self.max_state_dim = config.max_state_dim
        self.sentinel_value = config.sentinel_value

    @torch.no_grad()
    def reconstruct(self, sparse_pixels, state_vector, checkpoint_key, resolution):
        # 1. 체크포인트 키에 해당하는 모델 가져오기
        model = self.checkpoint_manager.get_model(checkpoint_key)

        # 2. 상태 벡터 정규화 및 마스크 생성
        state_vector = self._normalize_state_vector(state_vector)
        state_mask = (state_vector != self.sentinel_value).float()

        # 3. 추론
        with torch.cuda.amp.autocast():
            pred = model(sparse_pixels, state_vector, state_mask, resolution)

        return pred

    def _normalize_state_vector(self, state_vector):
        """가변 길이 → 고정 길이 (sentinel 패딩)"""
        if state_vector.shape[-1] < self.max_state_dim:
            padding = torch.full(
                (*state_vector.shape[:-1], self.max_state_dim - state_vector.shape[-1]),
                self.sentinel_value,
                device=state_vector.device
            )
            state_vector = torch.cat([state_vector, padding], dim=-1)
        return state_vector

# 체크포인트 디렉토리 구조
# checkpoints/
#   default/
#     best.pth
#   mario_world1/
#     best.pth
#   mario_world2/
#     best.pth
#   tetris_main/
#     best.pth

# 성공 기준
- 추론 시간 < 30ms per frame
- 배치 처리 시 throughput > 100 fps
- checkpoint_key별로 적절한 모델 로드 확인
- 미등록 키는 default 모델 사용
```

##### 3.4.2 모델 자동 업데이트

```python
# 주기적으로 새로운 데이터로 재학습
class ModelUpdateService:
    def __init__(self):
        self.last_training_time = time.time()
        self.training_interval = 7 * 24 * 3600  # 1주일

    async def check_and_update(self):
        if time.time() - self.last_training_time > self.training_interval:
            # 새로운 데이터로 재학습
            await self.trigger_training_job()

            # 새 모델 배포
            await self.deploy_new_model()

            self.last_training_time = time.time()

# 성공 기준
- 자동 재학습 파이프라인 동작
- A/B 테스트로 새 모델 검증 후 배포
```

#### 3.5 압축 최적화

##### 3.5.1 UV 좌표 양자화

```python
# Float32 → Int16 변환으로 4배 압축
def quantize_uv_coords(uv_coords, resolution):
    # [0, 1] → [0, 65535] (uint16)
    u_quantized = (uv_coords[:, 0] * 65535).astype(np.uint16)
    v_quantized = (uv_coords[:, 1] * 65535).astype(np.uint16)

    return np.stack([u_quantized, v_quantized], axis=1)

def dequantize_uv_coords(uv_quantized):
    # [0, 65535] → [0, 1]
    uv_coords = uv_quantized.astype(np.float32) / 65535.0
    return uv_coords

# 성공 기준
- 대역폭 4배 감소 (120 KB/s → 30 KB/s)
- 정확도 손실 < 0.1% (육안 확인 불가)
```

##### 3.5.2 픽셀 데이터 압축

```python
# MsgPack + LZ4 압축
import msgpack
import lz4.frame

def compress_pixel_data(pixel_data):
    # 1. MsgPack 직렬화
    packed = msgpack.packb(pixel_data)

    # 2. LZ4 압축
    compressed = lz4.frame.compress(packed)

    return compressed

# 성공 기준
- 압축률 > 3x (vs. JSON)
- 압축/해제 시간 < 5ms
```

### Phase 3 성공 기준 (전체)

| 메트릭               | 목표                  | 측정 방법        |
| -------------------- | --------------------- | ---------------- |
| 재구성 PSNR          | > 28 dB               | 평가 스크립트    |
| 재구성 SSIM          | > 0.90                | 평가 스크립트    |
| 추론 시간            | < 30ms @ RTX 3090     | PyTorch Profiler |
| 학습 시간            | < 48시간 (100 epochs) | 실측             |
| 모델 크기            | < 200 MB              | Checkpoint 파일  |
| 네트워크 대역폭 (총) | < 50 KB/s @ 30 FPS    | Wireshark        |
| 압축률               | > 100x vs. 원본       | 계산             |
| 상태 벡터 효과       | +1 dB vs. 미사용      | A/B 테스트       |
| 체크포인트 분리      | ✅ 키별 모델 로드     | 기능 테스트      |

### Phase 3 Deliverables

-   [ ] Sparse Pixel Transformer 구현
-   [ ] 학습 코드 (`train_spt.py`)
-   [ ] 평가 코드 (`evaluate_spt.py`)
-   [ ] 학습된 체크포인트 (Best model)
-   [ ] 서버 추론 엔진 통합
-   [ ] 압축 최적화 모듈
-   [ ] 실험 결과 리포트 (Wandb)
-   [ ] 최종 성능 데모 영상

---

## MVP 이후 확장 계획 (Optional)

### Phase 4: 컬러 영상 지원

-   YCbCr 색공간 활용
-   Luminance 채널 우선, Chrominance 서브샘플링
-   목표: 샘플링 < 3% 픽셀, PSNR > 30 dB

### Phase 5: 실시간 스트리밍

-   지연시간 < 100ms (End-to-End)
-   30 FPS 동작 보장
-   클라이언트 다중 접속 (동시 10명)

### Phase 6: t+2 미래 예측 (선택 사항)

-   Optical Flow 모델 통합
-   네트워크 지연 보상
-   예측 오류 시 Fallback 전략

### Phase 7: 게임 메타데이터 수집

-   플레이어 위치, HP, 스킬 사용 등
-   영상 + 메타데이터 동기화
-   리플레이 시 UI 오버레이

---

## 성공 지표 요약

| Phase   | 핵심 목표      | PSNR    | SSIM   | 완료 기준           |
| ------- | -------------- | ------- | ------ | ------------------- |
| Phase 1 | 기본 통신 구축 | > 20 dB | > 0.70 | End-to-End 동작     |
| Phase 2 | 적응적 샘플링  | > 25 dB | > 0.85 | Baseline 대비 +3dB  |
| Phase 3 | 딥러닝 재구성  | > 28 dB | > 0.90 | 육안 구분 불가 품질 |

---

## 리스크 및 대응

| 리스크            | 완화 전략                       |
| ----------------- | ------------------------------- |
| 학습 데이터 부족  | 공개 게임 영상으로 Pre-training |
| 재구성 품질 부족  | 샘플링 픽셀 수 동적 증가        |
| 서버 과부하       | 배치 처리 + GPU 오토스케일링    |
| 네트워크 불안정   | 프레임 버퍼링 + 재연결 로직     |
| Unity 버전 호환성 | 최소 버전 명시 + 자동 테스트    |

---

## 다음 문서

-   ✅ MVP_FEATURES.md (현재 문서)
-   ⏭️ CLIENT_IMPLEMENTATION.md - Unity 플러그인 상세 설계
-   ⏭️ SERVER_IMPLEMENTATION.md - 서버 및 모델 구현
-   ⏭️ API_SPECIFICATION.md - 통신 프로토콜 명세
