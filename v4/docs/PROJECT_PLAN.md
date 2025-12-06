# SGAPS-MAE v4: Server-Guided Adaptive Pixel Sampling for Game Replay

## 프로젝트 개요

### 목적

Unity 게임의 게임플레이 영상을 극도로 효율적으로 수집하고 재구성하는 시스템 개발

### 핵심 가치 제안

-   **극한 효율성**: 전체 프레임 대신 선택된 픽셀만 전송 (0.5-2% 목표)
-   **서버 주도**: 모든 지능형 결정을 서버에서 수행, 클라이언트 부하 최소화
-   **적응적 학습**: 게임 데이터가 축적됨에 따라 자동으로 성능 향상
-   **빠른 프로토타이핑**: 흑백 영상으로 개념 검증 후 컬러로 확장

---

## 시스템 아키텍처

### 전체 구조 다이어그램

```
┌─────────────────────────────────────────────────────────────────┐
│                          Unity Game                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              SGAPS Client (UPM Plugin)                   │  │
│  │                                                           │  │
│  │  1. RenderTexture Capture (Grayscale)                    │  │
│  │  2. Receive UV Coordinates from Server                   │  │
│  │  3. Sample Pixels at UV positions                        │  │
│  │  4. Send Pixel Data + Metadata                           │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↕ HTTP/WebSocket
┌─────────────────────────────────────────────────────────────────┐
│                      GPU Server (School)                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    SGAPS Server                          │  │
│  │                                                           │  │
│  │  A. Receive Sparse Pixel Data                            │  │
│  │  B. Reconstruct Frame (Neural Network)                   │  │
│  │  C. Compute Importance Map                               │  │
│  │  D. Generate UV Coordinates for Next Frame               │  │
│  │  E. Adaptive Learning & Mask Update                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │            Data Storage & Training Loop                  │  │
│  │  - Episode Database (Pixel Data + Metadata)              │  │
│  │  - Periodic Model Retraining                             │  │
│  │  - A/B Testing for Sampling Strategies                   │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 컴포넌트 역할

#### 1. Unity Client (UPM Plugin)

-   **입력**: 게임 렌더링 결과 (RenderTexture)
-   **출력**: 희소 픽셀 데이터 (UV 좌표 + 픽셀 값)
-   **CPU 사용량 목표**: < 1%
-   **GPU 사용량 목표**: < 5% (RenderTexture 읽기만)

#### 2. SGAPS Server

-   **입력**: 클라이언트로부터 받은 픽셀 데이터
-   **출력**:
    -   재구성된 프레임 (저장용)
    -   다음 프레임용 UV 좌표 (클라이언트로 전송)
-   **GPU 집약적**: 모든 딥러닝 연산 수행

#### 3. Configuration System

-   **마스크 업데이트 주기**: 설정 가능 (매 프레임 / N 프레임마다 / 적응적)
-   **샘플링 픽셀 개수**: 동적 조정 가능 (최소/최대 범위 설정)
-   **서버 엔드포인트**: 환경별로 변경 가능

---

## 데이터 플로우

### Timeline: 단일 프레임 처리 과정

```
Time  │ Client (Unity)                │ Network      │ Server (GPU)
──────┼───────────────────────────────┼──────────────┼─────────────────────────
t=0   │ Capture Frame 0               │              │
      │ Wait for UV coords...         │              │
──────┼───────────────────────────────┼──────────────┼─────────────────────────
t=1   │                               │ ← UV coords  │ (Previously computed
      │ Receive UV coords for Frame 0 │   for F0     │  from Frame -1)
      │ Sample pixels at UV positions │              │
──────┼───────────────────────────────┼──────────────┼─────────────────────────
t=2   │ Send sampled pixels →         │ Pixels F0 →  │
      │ Capture Frame 1               │              │ Receive pixels F0
      │ Wait for UV coords...         │              │ Reconstruct F0
      │                               │              │ Compute importance map
      │                               │              │ Generate UV for F1
──────┼───────────────────────────────┼──────────────┼─────────────────────────
t=3   │                               │ ← UV coords  │ Send UV coords for F1
      │ Receive UV coords for Frame 1 │   for F1     │ (Optional) Update mask
      │ Sample pixels at UV positions │              │
──────┼───────────────────────────────┼──────────────┼─────────────────────────
```

### 주요 특징

1. **파이프라인 구조**: 클라이언트와 서버가 동시에 작업 (지연시간 상쇄)
2. **동기화 불필요**: 각 프레임은 독립적으로 처리
3. **적응적 마스크**: 서버가 결정한 주기에 따라 UV 좌표 갱신

---

## 핵심 설계 결정

### 1. 클라이언트: Unity UPM Plugin 선택

#### 비교: UPM Plugin vs 별도 프로세스

| 항목            | Unity UPM Plugin                     | 별도 프로세스 (Screen Capture)       |
| --------------- | ------------------------------------ | ------------------------------------ |
| **성능**        | ⭐⭐⭐⭐⭐ RenderTexture 직접 접근   | ⭐⭐ Windows API 오버헤드            |
| **정확성**      | ⭐⭐⭐⭐⭐ 렌더링 결과 직접 획득     | ⭐⭐⭐ UI 오버레이, 윈도우 가림 문제 |
| **통합성**      | ⭐⭐⭐⭐⭐ Unity 에디터 통합         | ⭐⭐ 별도 프로그램 실행 필요         |
| **개발 편의성** | ⭐⭐⭐⭐ C# 스크립트, Inspector 설정 | ⭐⭐ C++/C# P/Invoke, 복잡한 후킹    |
| **범용성**      | ⭐⭐ Unity 게임만 지원               | ⭐⭐⭐⭐⭐ 모든 Windows 게임         |
| **배포**        | ⭐⭐⭐⭐⭐ Package Manager로 간편    | ⭐⭐⭐ 설치 프로그램 필요            |

**결정**: Unity UPM Plugin 선택

-   **이유**:
    -   프로젝트 목표가 Unity 게임 대상
    -   RenderTexture를 통한 제로 카피(zero-copy) 픽셀 접근
    -   Unity 에디터에서 실시간 디버깅 가능
    -   향후 게임 내 메타데이터 (플레이어 위치, 상태 등) 수집 용이

### 2. 화면 캡처: RenderTexture 방식

```csharp
// 매우 효율적인 방식
Camera mainCamera;
RenderTexture grayscaleRT;  // 흑백 1채널

void CaptureFrame() {
    // GPU에서 렌더링 결과를 직접 읽음 (CPU 복사 최소화)
    RenderTexture.active = grayscaleRT;

    // UV 좌표 리스트만큼만 픽셀 읽기
    foreach (Vector2 uv in serverUVCoordinates) {
        int x = (int)(uv.x * width);
        int y = (int)(uv.y * height);
        byte pixel = ReadPixel(x, y);  // 단일 픽셀만 읽기
        sampledPixels.Add(new PixelData(uv, pixel));
    }
}
```

**장점**:

-   DirectX/Vulkan 렌더링 파이프라인에 직접 접근
-   Post-processing 이후 최종 결과 획득
-   흑백 변환을 GPU에서 처리 (쉐이더 사용)

### 3. 통신 프로토콜: HTTP + WebSocket 하이브리드

-   **HTTP REST**: 세션 시작/종료, 설정 변경
-   **WebSocket**: 실시간 픽셀 데이터 전송 및 UV 좌표 수신

**대역폭 추정** (30 FPS 기준):

```
클라이언트 → 서버 (상행):
  - 픽셀 데이터: 500 pixels/frame × 1 byte × 30 fps = 15 KB/s
  - 메타데이터: ~5 KB/s (프레임 번호, 타임스탬프 등)
  - 총: ~20 KB/s

서버 → 클라이언트 (하행):
  - UV 좌표: 500 coords × 8 bytes (float×2) × 30 fps = 120 KB/s
  - 압축 시: ~60 KB/s (좌표 양자화)
```

---

## 기술 스택

### Client (Unity)

```yaml
언어: C# 9.0+
Unity 버전: 2021.3 LTS 이상
주요 라이브러리:
    - UnityEngine.Rendering (RenderTexture)
    - System.Net.WebSockets (통신)
    - Newtonsoft.Json (직렬화)
패키지 형태: UPM Git Package
```

### Server (Python)

```yaml
언어: Python 3.10+
프레임워크:
    - FastAPI (REST API + WebSocket)
    - PyTorch 2.0+ (딥러닝)
    - Hydra (설정 관리)
주요 라이브러리:
    - torchvision (이미지 처리)
    - opencv-python (전처리)
    - wandb (실험 추적)
    - numpy, pillow
배포:
    - Docker 컨테이너
    - NVIDIA GPU 지원 (CUDA 11.8+)
```

### 데이터베이스

```yaml
메타데이터: PostgreSQL
    - 에피소드 정보, 설정, 통계
픽셀 데이터: 파일 시스템
    - HDF5 또는 MsgPack 형식
    - 에피소드별 압축 저장
```

---

## 새로운 모델 아키텍처 (v3 VideoMAE 대체)

### 설계 방향

#### VideoMAE의 한계

1. **고정 패치 마스킹**: Tube masking은 무작위 선택
2. **정사각 패치**: 16×16 블록 단위로만 처리
3. **대칭성 가정**: 마스킹/복원이 균등하게 분포

#### 새로운 접근: Sparse Pixel Transformer (SPT)

> **Note**: 아래 하이퍼파라미터들은 `conf/config.yaml`에서 설정 가능합니다.

```
입력: 희소 픽셀 집합 (N개, N은 가변)
  - Position: (u, v) 좌표 [0, 1] 범위
  - Value: 그레이스케일 값 [0, 255] (Phase 1)

출력: 재구성된 프레임 (H × W × 1)

하이퍼파라미터 (configurable):
  - embed_dim: 256 (기본값)
  - num_heads: 8
  - num_encoder_layers: 6
  - num_decoder_layers: 4

아키텍처:
  1. Pixel Embedding Layer
     - 각 픽셀을 (u, v, value) → embed_dim 차원 벡터로 변환
     - Continuous positional encoding

  2. Sparse Transformer Encoder (num_encoder_layers)
     - Self-attention on sparse pixels only
     - O(N²) complexity (N은 작으므로 매우 빠름)

  3. Cross-Attention Decoder (num_decoder_layers)
     - Query: 전체 프레임의 각 픽셀 위치 (H×W)
     - Key/Value: Sparse pixel embeddings (N)
     - 각 출력 픽셀이 인근 샘플 픽셀들을 참조

  4. State-Pixel Cross-Attention
     - 상태 벡터(게임 컨텍스트)와 픽셀 특징 결합

  5. CNN Refinement Head
     - 부드러운 재구성을 위한 후처리
     - 3-layer ConvNet: embed_dim → 128 → 64 → 1
```

#### 핵심 혁신점

1. **가변 입력 크기**: 500개든 2000개든 동일 모델 사용
2. **연속 좌표**: 정수 그리드에 국한되지 않음
3. **적응적 샘플링 친화적**: 중요 영역에 픽셀 집중 가능

### 손실 함수 (Sampled Pixel L2 Loss)

초기 단계의 빠른 수렴과 개념 검증(PoC)을 위해 **샘플링된 픽셀에 대한 MSE(Mean Squared Error)**만을 사용합니다.

```python
# 샘플링된 위치에서만 손실 계산
L_total = (1 / N_sampled) * sum(||I_pred(x_i) - I_gt(x_i)||^2 for x_i in P_sampled)

# P_sampled: 샘플링된 픽셀 좌표 집합
# N_sampled: 샘플링된 픽셀 수 (예: 500)
# I(x_i): 해당 좌표의 픽셀 값 (Grayscale 0~1)
```

**장점:**

-   빠른 수렴: 복잡한 Perceptual/Temporal Loss 없이 직관적
-   희소 픽셀 정확도 보장: 모델이 최소한 주어진 힌트는 정확히 복원하도록 강제
-   확장 가능: 향후 Perceptual Loss 등 추가 가능

---

## 적응적 샘플링 알고리즘

### Importance Map 계산 (Attention Entropy)

모델이 이미 계산하는 **Decoder Cross-Attention Map의 엔트로피**를 활용하여 추가 연산 비용 없이 중요도를 판단합니다.

**핵심 아이디어:**

-   Decoder Cross-Attention은 '복원해야 할 전체 픽셀(Query)'이 '주어진 희소 픽셀(Key)' 중 어디를 참조할지 결정
-   **낮은 엔트로피**: 특정 희소 픽셀에 강하게 집중 → "확실한 단서가 있다" → 중요도 낮음
-   **높은 엔트로피**: 여러 희소 픽셀을 두루뭉술하게 참조 → "어디를 봐야 할지 모른다" → 중요도 높음 (추가 샘플링 필요)

```python
def compute_importance_from_attention(attention_weights, resolution):
    """
    Decoder Cross-Attention 엔트로피 기반 중요도 계산

    Args:
        attention_weights: [B, num_heads, H*W, N] - Decoder 마지막 레이어에서 추출
        resolution: (H, W)
    """
    H, W = resolution
    epsilon = 1e-9

    # 1. Head 평균: [B, num_heads, H*W, N] → [B, H*W, N]
    attn_avg = attention_weights.mean(dim=1)

    # 2. 픽셀별 엔트로피 계산
    # Importance_i = -∑_j A(i,j) * log(A(i,j) + ε)
    entropy = -torch.sum(attn_avg * torch.log(attn_avg + epsilon), dim=-1)  # [B, H*W]

    # 3. 이미지 형태로 변환 및 정규화
    importance_map = entropy.mean(dim=0).view(H, W)  # [H, W]
    importance_map = (importance_map - importance_map.min()) / (importance_map.max() - importance_map.min() + epsilon)

    return importance_map
```

**장점:**

-   추가 연산 비용 거의 없음 (attention 이미 계산됨)
-   Monte Carlo Dropout 대비 빠르고 결정적(deterministic)
-   모델의 실제 불확실성을 직접 반영

### UV 좌표 생성 전략

```python
def generate_uv_coordinates(importance_map, num_samples, config):
    """
    중요도 맵 기반으로 샘플링 위치 결정
    """
    # 1. 확률 분포로 변환
    prob_map = importance_map / importance_map.sum()

    # 2. 전략적 샘플링
    if config.sampling_strategy == "importance":
        # 중요도 높은 곳 위주로 샘플링
        uv_coords = weighted_random_sample(prob_map, num_samples)

    elif config.sampling_strategy == "hybrid":
        # 70% 중요도 기반, 30% 균등 분포
        important_samples = weighted_random_sample(prob_map, int(num_samples * 0.7))
        uniform_samples = uniform_random_sample(num_samples - len(important_samples))
        uv_coords = important_samples + uniform_samples

    # 3. 최소 거리 제약 (샘플들이 너무 몰리지 않도록)
    uv_coords = enforce_min_distance(uv_coords, min_distance=0.01)

    return uv_coords
```

### 마스크 업데이트 주기 결정

```python
class AdaptiveMaskUpdater:
    def __init__(self, config):
        self.base_interval = config.mask_update_interval  # 예: 5 frames
        self.adaptive = config.adaptive_update

    def should_update_mask(self, frame_idx, reconstruction_quality):
        # 고정 주기
        if not self.adaptive:
            return (frame_idx % self.base_interval) == 0

        # 적응적 주기
        if reconstruction_quality < 0.8:  # PSNR이 낮으면
            return True  # 즉시 업데이트
        elif reconstruction_quality > 0.95:  # 매우 좋으면
            return (frame_idx % (self.base_interval * 2)) == 0  # 주기 2배로
        else:
            return (frame_idx % self.base_interval) == 0
```

---

## 흑백 영상 최적화

### 왜 흑백으로 시작하는가?

1. **데이터 크기 1/3**: RGB → Grayscale로 즉시 대역폭 절감
2. **학습 속도 3배**: 채널 수가 적어 모델 파라미터 감소
3. **디버깅 용이**: 시각적으로 문제 파악이 쉬움
4. **핵심 검증**: 적응적 샘플링 알고리즘의 효과를 빠르게 검증

### 컬러로의 전환 계획

```yaml
Phase 1 (흑백):
  - 모델: Sparse Pixel Transformer (1-channel)
  - 목표 PSNR: > 25 dB
  - 목표 샘플링: < 2% 픽셀

Phase 2 (컬러 확장):
  - 방법 1: 각 채널 독립적으로 처리 (3배 느림)
  - 방법 2: 채널 간 상관성 활용 (Y-Cb-Cr 색공간)
    → Luminance(Y)는 높은 해상도
    → Chrominance(Cb, Cr)는 낮은 해상도
  - 예상 성능: 샘플링 2-3% 픽셀로 유지 가능
```

---

## 성능 벤치마크 목표

### Client-Side (Unity)

| 메트릭        | 목표         | 측정 방법        |
| ------------- | ------------ | ---------------- |
| CPU 사용률    | < 1%         | Unity Profiler   |
| GPU 사용률    | < 5%         | RenderDoc        |
| 메모리 증가   | < 50 MB      | Profiler         |
| FPS 영향      | < 1 fps 감소 | 프레임 타임 측정 |
| 네트워크 상행 | < 30 KB/s    | Wireshark        |
| 네트워크 하행 | < 100 KB/s   | Wireshark        |

### Server-Side (GPU)

| 메트릭                  | 목표             | 측정 방법         |
| ----------------------- | ---------------- | ----------------- |
| 추론 시간 (단일 프레임) | < 33ms (30 FPS)  | PyTorch profiler  |
| GPU 메모리              | < 4 GB           | nvidia-smi        |
| 배치 처리량             | > 100 frames/sec | 벤치마크 스크립트 |
| 재구성 품질 (PSNR)      | > 25 dB          | 평가 메트릭       |
| 재구성 품질 (SSIM)      | > 0.85           | 평가 메트릭       |

### End-to-End

| 메트릭        | 목표        | 비고                           |
| ------------- | ----------- | ------------------------------ |
| 왕복 지연시간 | < 100ms     | 클라이언트 → 서버 → 클라이언트 |
| 압축률        | > 50x       | vs. 전체 프레임 전송           |
| 스토리지 효율 | < 10 MB/min | 1920×1080 @ 30fps 기준         |

---

## 리스크 및 완화 전략

### 1. 네트워크 불안정성

**리스크**: WebSocket 연결 끊김, 패킷 손실
**완화**:

-   자동 재연결 로직
-   프레임 버퍼링 (최대 5 프레임)
-   연결 끊김 시 로컬 캐싱

### 2. 재구성 품질 저하

**리스크**: 초기 학습 데이터 부족으로 품질 나쁨
**완화**:

-   Pre-training: 공개 게임 영상으로 기본 모델 학습
-   Progressive training: 쉬운 장면부터 어려운 장면으로
-   Fallback: 품질이 낮으면 샘플링 픽셀 수 자동 증가

### 3. 서버 과부하

**리스크**: 다수 클라이언트 동시 접속 시 처리 지연
**완화**:

-   배치 처리: 여러 클라이언트 요청을 배치로 묶어 처리
-   우선순위 큐: 학습용 데이터 수집 vs 실시간 재구성
-   Auto-scaling: GPU 서버 수평 확장 (쿠버네티스)

### 4. Unity 버전 호환성

**리스크**: 다양한 Unity 버전에서 RenderTexture API 차이
**완화**:

-   최소 지원 버전 명시 (Unity 2021.3 LTS)
-   Compatibility layer 구현
-   자동 테스트 파이프라인 (Unity Test Framework)

---

## 다음 단계

이 문서를 기반으로 다음 상세 문서들을 작성합니다:

1. ✅ **PROJECT_PLAN.md** (현재 문서)
2. ⏭️ **MVP_FEATURES.md** - 단계별 구현 기능 명세
3. ⏭️ **CLIENT_IMPLEMENTATION.md** - Unity 플러그인 상세 설계
4. ⏭️ **SERVER_IMPLEMENTATION.md** - 서버 및 모델 구현 상세
5. ⏭️ **API_SPECIFICATION.md** - REST/WebSocket API 명세
6. ⏭️ **CONFIGURATION.md** - 설정 시스템 및 파라미터
7. ⏭️ **DEVELOPMENT_ROADMAP.md** - 개발 일정 및 마일스톤

---

## 검토 체크리스트

이 문서를 검토할 때 다음 사항을 확인해주세요:

-   [ ] 시스템 아키텍처가 명확하고 실현 가능한가?
-   [ ] 클라이언트 구현 방식(UPM Plugin) 선택이 타당한가?
-   [ ] 새로운 모델(Sparse Pixel Transformer)이 VideoMAE보다 적합한가?
-   [ ] 적응적 샘플링 알고리즘이 구체적이고 구현 가능한가?
-   [ ] 흑백 → 컬러 전환 계획이 합리적인가?
-   [ ] 성능 목표가 현실적인가?
-   [ ] 기술 스택이 적절한가?
-   [ ] 리스크 완화 전략이 충분한가?
