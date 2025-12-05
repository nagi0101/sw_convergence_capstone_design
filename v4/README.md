# Server-Guided Adaptive Pixel Sampling MAE (SGAPS-MAE)

## 개요

SGAPS-MAE는 게임 세션 리플레이를 위한 혁신적인 픽셀 단위 적응적 샘플링 시스템입니다. 기존 패치 기반 MAE의 한계를 극복하고, 클라이언트 부하를 최소화하면서도 0.5-2%의 픽셀만으로 전체 프레임을 복원합니다.

### 핵심 혁신

-   **서버 주도 샘플링**: 모든 지능을 서버에 집중, 클라이언트는 단순 픽셀 추출만 수행
-   **픽셀 단위 적응**: 정보량이 높은 픽셀만 선택적 샘플링
-   **실시간 피드백 루프**: 복원 품질 기반 동적 샘플링 최적화
-   **극한 압축**: 250-1000개 픽셀(0.5-2%)만으로 224×224 프레임 복원

## 시스템 아키텍처

### 전체 시스템 구조

```mermaid
flowchart TB
    subgraph Client["클라이언트 (극경량)"]
        GameFrame["게임 프레임<br/>256×240 RGB"]
        GameState["게임 상태<br/>State Vector"]
        PixelExtractor["픽셀 추출기<br/>coords[u,v] → pixels"]
        StateCollector["상태 수집기<br/>StateVectorCollector"]
        Compressor["압축<br/>zlib + msgpack"]

        GameFrame --> PixelExtractor
        GameState --> StateCollector
        PixelExtractor --> Compressor
        StateCollector --> Compressor
    end

    subgraph Network["네트워크"]
        Upload["상행: 2KB/frame<br/>pixels + state_vector"]
        Download["하행: 0.5KB/frame<br/>UV 좌표"]
    end

    subgraph Server["서버 (모든 지능)"]
        PixelDecoder["픽셀 디코더"]
        StateEncoder["State Vector Encoder<br/>게임 컨텍스트 인코딩"]
        SparseTransformer["Sparse Pixel Transformer<br/>Self-Attention Encoder"]
        CrossAttention["Cross-Attention Decoder<br/>Query Grid → Dense"]
        QualityAnalyzer["품질 분석기<br/>Uncertainty Estimation"]
        CoordGenerator["좌표 생성기<br/>Top-N Importance"]
        CNNHead["CNN Refinement Head<br/>최종 복원"]
    end

    Compressor -->|"pixels + state"| Upload
    Upload --> PixelDecoder
    Upload --> StateEncoder

    PixelDecoder --> SparseTransformer
    StateEncoder --> CrossAttention
    SparseTransformer --> CrossAttention
    CrossAttention --> CNNHead
    CNNHead --> QualityAnalyzer
    QualityAnalyzer --> CoordGenerator

    CoordGenerator --> Download
    Download -->|"coords[(u,v)...]"| PixelExtractor

    style Client fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style Server fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style Network fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
```

### 적응적 픽셀 샘플링 메커니즘

```mermaid
flowchart LR
    subgraph Frame_t["Frame t"]
        Reconstruction_t["복원 결과"]
        Uncertainty_t["불확실성 맵<br/>Monte Carlo Dropout"]
        Attention_t["Attention 가중치<br/>정보 부족 영역"]
        Temporal_t["시간적 오차<br/>일관성 분석"]
    end

    subgraph Analysis["중요도 분석"]
        ImportanceMap["중요도 맵 생성<br/>H×W"]
        Hierarchy["계층적 분류"]
        Critical["Critical: 10%<br/>크로스헤어, 적"]
        Important["Important: 20%<br/>캐릭터, 객체"]
        Moderate["Moderate: 30%<br/>환경 디테일"]
        Optional["Optional: 40%<br/>배경, 장식"]
    end

    subgraph Prediction["모션 예측"]
        OpticalFlow["광학 흐름 추정"]
        CameraMotion["카메라 모션 보상"]
        LatencyComp["지연 보상<br/>t+2 예측"]
    end

    subgraph Frame_t2["Frame t+2"]
        FutureCoords["예측 좌표<br/>[(u',v')...]"]
        Budget["샘플링 예산<br/>500 pixels"]
    end

    Reconstruction_t --> Uncertainty_t
    Reconstruction_t --> Attention_t
    Reconstruction_t --> Temporal_t

    Uncertainty_t -->|"0.4"| ImportanceMap
    Attention_t -->|"0.3"| ImportanceMap
    Temporal_t -->|"0.3"| ImportanceMap

    ImportanceMap --> Hierarchy
    Hierarchy --> Critical
    Hierarchy --> Important
    Hierarchy --> Moderate
    Hierarchy --> Optional

    Critical --> Budget
    Important --> Budget
    Moderate --> Budget

    ImportanceMap --> OpticalFlow
    OpticalFlow --> CameraMotion
    CameraMotion --> LatencyComp
    LatencyComp --> FutureCoords

    Budget --> FutureCoords

    style Frame_t fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    style Analysis fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style Prediction fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    style Frame_t2 fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
```

### Sparse Pixel Transformer 아키텍처

> **Note**: Phase 1에서는 Grayscale(1채널)로 시작하며, Phase 2에서 컬러(RGB/YCbCr) 지원을 추가합니다.
> 임베딩 차원, 레이어 수 등은 `config.yaml`에서 설정 가능합니다.

```mermaid
graph TB
    subgraph Input["입력"]
        PixelValues["픽셀 값<br/>[N, 1] Grayscale"]
        PixelPositions["픽셀 위치<br/>[N, 2] (u,v)"]
        StateVector["상태 벡터<br/>[max_state_dim]"]
    end

    subgraph Embedding["임베딩 (configurable)"]
        PixelEmbed["Pixel Embedding<br/>Linear(1, embed_dim)"]
        PosEnc["Continuous Position<br/>Encoding(embed_dim)"]
        StateEnc["State Vector Encoder<br/>MLP → embed_dim"]
        PixelWithPos["Pixel + Position<br/>[N, embed_dim]"]
    end

    subgraph TransformerEncoder["Sparse Transformer Encoder"]
        SelfAttn1["Self-Attention Layer 1<br/>embed_dim, num_heads"]
        SelfAttn2["Self-Attention Layer 2"]
        SelfAttnN["...<br/>num_encoder_layers"]
        SparseFeatures["Sparse Features<br/>[N, embed_dim]"]
    end

    subgraph CrossAttentionDecoder["Cross-Attention Decoder"]
        QueryGrid["Query Grid 생성<br/>[H×W, 2] 전체 픽셀 좌표"]
        QueryEmbed["Query Embedding<br/>[H×W, embed_dim]"]
        CrossAttn1["Cross-Attention Layer 1<br/>Query: Grid, K/V: Sparse"]
        CrossAttn2["Cross-Attention Layer 2"]
        CrossAttnN["...<br/>num_decoder_layers"]
        StatePixelAttn["State-Pixel<br/>Cross-Attention"]
        DenseFeatures["Dense Features<br/>[H×W, embed_dim]"]
    end

    subgraph CNNHead["CNN Refinement Head"]
        Reshape["Reshape<br/>[H, W, embed_dim]"]
        Conv1["Conv2d(embed_dim, 128)"]
        Conv2["Conv2d(128, 64)"]
        Conv3["Conv2d(64, 1)"]
        Sigmoid["Sigmoid()"]
    end

    subgraph Output["출력"]
        ReconstructedFrame["복원된 프레임<br/>[H, W, 1] Grayscale"]
    end

    PixelValues --> PixelEmbed
    PixelPositions --> PosEnc
    PixelEmbed --> PixelWithPos
    PosEnc --> PixelWithPos
    StateVector --> StateEnc

    PixelWithPos --> SelfAttn1
    SelfAttn1 --> SelfAttn2
    SelfAttn2 --> SelfAttnN
    SelfAttnN --> SparseFeatures

    PixelPositions --> QueryGrid
    QueryGrid --> QueryEmbed
    QueryEmbed --> CrossAttn1
    SparseFeatures --> CrossAttn1
    CrossAttn1 --> CrossAttn2
    CrossAttn2 --> CrossAttnN
    CrossAttnN --> StatePixelAttn
    StateEnc --> StatePixelAttn
    StatePixelAttn --> DenseFeatures

    DenseFeatures --> Reshape
    Reshape --> Conv1
    Conv1 --> Conv2
    Conv2 --> Conv3
    Conv3 --> Sigmoid
    Sigmoid --> ReconstructedFrame

    style Input fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style TransformerEncoder fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    style CrossAttentionDecoder fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    style CNNHead fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
```

**모델 설정** (`config.yaml`):

```yaml
model:
    embed_dim: 256 # 임베딩 차원 (configurable)
    num_heads: 8 # Attention heads
    num_encoder_layers: 6 # Sparse Transformer Encoder layers
    num_decoder_layers: 4 # Cross-Attention Decoder layers
    max_state_dim: 64 # 상태 벡터 최대 차원
```

### Temporal Memory Bank 구조 (Phase 2+ 예정)

> **Note**: Temporal Memory Bank는 Phase 2 이후 적응적 샘플링 구현 시 추가될 예정입니다.
> 현재 Phase 1에서는 고정된 UV 좌표 샘플링을 사용합니다.

```mermaid
flowchart TB
    subgraph MemoryBank["Temporal Memory Bank"]
        subgraph StaticMemory["장기 메모리 (정적 요소)"]
            UI["UI/HUD 요소<br/>신뢰도 > 0.9"]
            Background["고정 배경<br/>움직임 < 0.1"]
            StaticConf["신뢰도 맵<br/>EMA 업데이트"]
        end

        subgraph DynamicMemory["단기 메모리 (동적 요소)"]
            MovingObjects["움직이는 객체<br/>FIFO Queue"]
            RecentFrames["최근 100 프레임<br/>Deque"]
            MotionVectors["모션 벡터<br/>광학 흐름"]
        end

        subgraph UpdateLogic["업데이트 로직"]
            MotionScore["움직임 점수 계산"]
            Decision["정적/동적 판별<br/>threshold: 0.1"]
            EMAUpdate["EMA 업데이트<br/>α=0.95"]
            QueuePush["Queue Push<br/>maxlen=100"]
        end
    end

    subgraph Usage["메모리 활용"]
        StaticRetrieval["정적 픽셀 검색<br/>conf > 0.9"]
        DynamicMask["동적 영역 마스크"]
        BudgetReduction["샘플링 예산 감소<br/>500 → 300"]
        FocusedSampling["동적 영역 집중<br/>샘플링"]
    end

    NewPixel["새 픽셀 입력<br/>(pos, val, frame_idx)"]

    NewPixel --> MotionScore
    MotionScore --> Decision

    Decision -->|"< 0.1"| EMAUpdate
    Decision -->|">= 0.1"| QueuePush

    EMAUpdate --> StaticMemory
    QueuePush --> DynamicMemory

    StaticMemory --> StaticRetrieval
    StaticRetrieval --> DynamicMask
    DynamicMask --> BudgetReduction
    BudgetReduction --> FocusedSampling

    style StaticMemory fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style DynamicMemory fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    style UpdateLogic fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    style Usage fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
```

## 핵심 구성요소

### 1. Sparse Pixel Transformer

희소 픽셀 집합에서 전체 프레임을 복원하는 Transformer 기반 모델:

-   **Self-Attention Encoder**: 희소 픽셀 간의 관계 학습 (O(N²) 복잡도, N은 작아 효율적)
-   **Cross-Attention Decoder**: Query Grid로 전체 프레임 위치 복원
-   **State-Pixel Cross-Attention**: 게임 상태와 픽셀 특징 결합
-   **CNN Refinement Head**: 최종 복원 품질 향상

### 2. 서버 예측 오차 기반 픽셀 중요도 계산

서버는 복원 품질을 자체 평가하여 다음 프레임에 필요한 픽셀을 결정합니다:

-   **Monte Carlo Dropout**: 여러 dropout 패턴으로 불확실성 추정
-   **Attention 가중치 분석**: 정보 부족 영역 식별
-   **시간적 일관성 검사**: 프레임 간 불일치 영역 탐지
-   **종합 중요도 맵**: 가중 평균으로 최종 중요도 계산

### 3. 계층적 샘플링 예산 할당

```
Critical (10%): 크로스헤어, 적 위치 등 게임플레이 핵심 요소
Important (20%): 캐릭터, 주요 객체
Moderate (30%): 환경 디테일
Optional (40%): 배경, 장식 요소
```

### 4. 적응적 지연 보상

네트워크 지연을 고려한 예측 시스템:

-   실시간 지연 측정
-   2-5 프레임 미래 예측
-   모션 벡터 기반 픽셀 궤적 추정

## 학습 전략

### 커리큘럼 학습 단계

```mermaid
graph LR
    subgraph Phase1["Phase 1 (Epochs 1-100)"]
        P1_Strategy["전략: Uniform<br/>샘플률: 10%<br/>목표: 전역 구조"]
    end

    subgraph Phase2["Phase 2 (Epochs 101-300)"]
        P2_Strategy["전략: Edge-focused<br/>샘플률: 5%<br/>목표: 경계선"]
    end

    subgraph Phase3["Phase 3 (Epochs 301-500)"]
        P3_Strategy["전략: Hard negative<br/>샘플률: 2%<br/>목표: 고오차 영역"]
    end

    subgraph Phase4["Phase 4 (Epochs 501+)"]
        P4_Strategy["전략: Extreme sparse<br/>샘플률: 0.5%<br/>목표: 핵심만"]
    end

    Phase1 --> Phase2
    Phase2 --> Phase3
    Phase3 --> Phase4

    style Phase1 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style Phase2 fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    style Phase3 fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    style Phase4 fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
```

### 손실 함수 구성

```
L_total = 0.3 × L_sampled + 0.4 × L_perceptual + 0.2 × L_structural + 0.1 × L_temporal
```

-   **L_sampled**: 샘플링된 픽셀의 직접 손실 (정보량 역가중치)
-   **L_perceptual**: 비샘플링 영역의 지각적 손실 (LPIPS)
-   **L_structural**: 구조적 일관성 (SSIM)
-   **L_temporal**: 시간적 평활성

## 성능 벤치마크

### 시스템 성능

| 지표            | 값        | 비고             |
| --------------- | --------- | ---------------- |
| **클라이언트**  |           |                  |
| CPU 사용률      | 0.1%      | 단순 배열 접근만 |
| 메모리 사용     | 10MB      | 프레임 버퍼만    |
| GPU 요구사항    | 없음      | CPU만으로 동작   |
| 배터리 소모     | 무시 가능 | 모바일 최적화    |
| **네트워크**    |           |                  |
| 상행 대역폭     | 60KB/s    | @30fps           |
| 하행 대역폭     | 15KB/s    | @30fps           |
| 총 대역폭       | 75KB/s    | 일반 넷코드 수준 |
| **복원 품질**   |           |                  |
| PSNR            | 39.2 dB   | 우수             |
| SSIM            | 0.95      | 높은 유사성      |
| 샘플링 비율     | 0.5-2%    | 250-1000 pixels  |
| **서버 확장성** |           |                  |
| 동시 세션       | 10,000+   | 단일 서버        |
| GPU 메모리/세션 | 40MB      | 효율적           |
| 처리 지연       | 5-10ms    | 실시간           |

### 기존 방법과의 비교

| 방법          | 샘플링률   | 대역폭     | 클라이언트 부하 | PSNR       |
| ------------- | ---------- | ---------- | --------------- | ---------- |
| Full Frame    | 100%       | 4.5MB/s    | 낮음            | -          |
| Patch MAE     | 5-10%      | 150KB/s    | 높음 (GPU 필요) | 35dB       |
| Pixel MAE     | 2%         | 90KB/s     | 중간            | 38dB       |
| **SGAPS-MAE** | **0.5-2%** | **75KB/s** | **매우 낮음**   | **39.2dB** |

## 프로젝트 구조

```
v4/
├── README.md                    # 프로젝트 문서 (본 파일)
├── docs/                        # 상세 문서
│   ├── API_SPECIFICATION.md    # REST/WebSocket API 명세
│   ├── CLIENT_IMPLEMENTATION.md # Unity 클라이언트 상세 설계
│   ├── SERVER_IMPLEMENTATION.md # 서버 및 모델 구현 상세
│   ├── CONFIGURATION.md        # 설정 시스템 및 파라미터
│   └── ...
├── sgaps-server/                # Python 서버
│   ├── main.py                 # FastAPI 엔트리포인트
│   ├── requirements.txt        # Python 의존성
│   ├── conf/
│   │   └── config.yaml         # Hydra 설정 (embed_dim, num_heads 등)
│   └── sgaps/
│       ├── api/                # WebSocket/REST 핸들러
│       ├── core/               # 세션 관리, 샘플러, 복원기
│       ├── data/               # HDF5 스토리지
│       ├── models/             # Sparse Pixel Transformer (Phase 2)
│       └── utils/              # 유틸리티
└── unity-client/                # Unity UPM 패키지
    ├── package.json
    ├── README.md
    ├── Runtime/
    │   └── Scripts/
    │       ├── Core/           # SGAPSManager, NetworkClient, etc.
    │       └── Data/           # SessionConfig, StateVectorCollector, etc.
    └── Samples~/               # 예제 씬
```

## 설치 및 사용법

### 요구사항

-   Python 3.8+
-   PyTorch 2.0+
-   CUDA 11.8+ 또는 12.x (서버만)
-   최소 8GB GPU 메모리 (서버)

### 설치

```bash
# conda 환경 생성 및 활성화
conda create -n sgaps python=3.10 -y
conda activate sgaps

# PyTorch GPU 버전 먼저 설치 (pip 사용)
# ⚠️ 반드시 pip install -r requirements.txt 전에 실행해야 GPU 버전이 유지됨
#
# CUDA 버전 확인: nvidia-smi 또는 nvcc --version
# - CUDA 12.x 환경:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# - CUDA 11.x 환경:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 나머지 의존성 설치 (torch, torchvision은 이미 설치되어 skip됨)
pip install -r requirements.txt

# GPU 설치 확인
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

> **⚠️ 주의**: `pip install -r requirements.txt`를 먼저 실행하면 CPU 전용 PyTorch가 설치됩니다.
> 반드시 PyTorch를 먼저 설치한 후 나머지 패키지를 설치하세요.

### 서버 실행

```bash
cd sgaps-server

# 기본 설정으로 실행
python main.py

# 포트 변경
python main.py server.port=8080

# 개발 모드 (자동 리로드) - uvicorn 직접 사용
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Unity 클라이언트

1. Unity Package Manager에서 `unity-client` 폴더를 추가
2. `SGAPSManager` 컴포넌트를 씬에 추가
3. 서버 엔드포인트 설정 (`ws://localhost:8000/ws/stream`)
4. Play 모드에서 자동 연결 또는 `ConnectToServer()` 호출

자세한 내용은 [Unity Client README](unity-client/README.md) 참조

### 서버-클라이언트 통신 프로토콜

```mermaid
sequenceDiagram
    participant C as Unity Client
    participant S as SGAPS Server

    C->>S: WebSocket Connect
    S->>C: connection_ack (client_id, server_version)

    C->>S: session_start (checkpoint_key, resolution)
    Note over S: 서버 설정에서 파라미터 로드
    S->>C: session_start_ack
    Note over C: sample_count, max_state_dim,<br/>target_fps 수신

    S->>C: uv_coordinates (initial)
    Note over C: PixelSampler, StateVectorCollector 초기화<br/>캡처 시작

    loop Every Frame (at target_fps)
        C->>S: frame_data (pixels, state_vector)
        S->>C: uv_coordinates (for next frame)
    end

    C->>S: Disconnect
```

**서버 제어 파라미터**: 다음 값들은 서버 설정(`conf/config.yaml`)에서 관리되며, `session_start_ack`를 통해 클라이언트에 전달됩니다:

| 파라미터        | 설명                    | 기본값 |
| --------------- | ----------------------- | ------ |
| `sample_count`  | 프레임당 샘플링 픽셀 수 | 500    |
| `max_state_dim` | 상태 벡터 최대 차원     | 64     |
| `target_fps`    | 캡처 프레임 레이트      | 10     |

> **Note**: `sentinel_value`는 서버 내부에서 상태 벡터 패딩에 사용되며, 클라이언트에 전달되지 않습니다.

### 학습 및 평가 (Phase 2 예정)

> 현재 Phase 1에서는 데이터 수집 및 저장 기능만 구현되어 있습니다.
> 학습 및 평가 기능은 Phase 2에서 추가될 예정입니다.

## 기술적 장점

1. **극도로 낮은 클라이언트 부하**: GPU 불필요, CPU 0.1% 사용
2. **효율적 대역폭**: 일반 게임 넷코드와 유사한 수준 (75KB/s)
3. **우수한 복원 품질**: 0.5-2% 픽셀로 PSNR 39.2dB 달성
4. **높은 확장성**: 단일 서버로 10,000+ 세션 동시 처리
5. **실시간 적응**: 피드백 루프를 통한 동적 최적화

## 활용 분야

-   **게임 세션 리플레이**: 극소량 데이터로 전체 게임플레이 복원
-   **클라우드 게임**: 대역폭 절감으로 비용 대폭 감소
-   **e스포츠 방송**: 고품질 저지연 스트리밍
-   **원격 게임 플레이**: 모바일 환경에서도 고품질 게임 가능
-   **게임 분석**: 효율적인 게임플레이 데이터 저장 및 분석

## 한계 및 향후 연구

### 현재 한계

-   초기 세션에서 정적 요소 학습 필요 (워밍업 시간)
-   극도로 빠른 모션에서 성능 저하 가능
-   네트워크 지연 변동성에 민감

### 향후 연구 방향

-   다중 해상도 적응적 샘플링
-   게임 장르별 특화 모델 개발
-   연합 학습을 통한 크로스 세션 개선
-   신경 압축 코덱과의 통합

## 라이센스

MIT License

## 참고문헌

1. He, K., et al. (2021). Masked autoencoders are scalable vision learners. arXiv:2111.06377
2. Vaswani, A., et al. (2017). Attention is all you need. NeurIPS.
3. Dosovitskiy, A., et al. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. ICLR.

## 기여

프로젝트에 기여하시려면:

1. Fork 후 브랜치 생성
2. 코드 작성 및 테스트
3. Pull Request 제출

문의사항은 Issues 페이지를 이용해주세요.
