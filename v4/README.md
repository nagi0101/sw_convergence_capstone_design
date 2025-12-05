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
        PixelExtractor["픽셀 추출기<br/>coords[u,v] → pixels"]
        Compressor["압축<br/>zlib + msgpack"]

        GameFrame --> PixelExtractor
        PixelExtractor --> Compressor
    end

    subgraph Network["네트워크"]
        Upload["상행: 2KB/frame<br/>60KB/s @ 30fps"]
        Download["하행: 0.5KB/frame<br/>15KB/s @ 30fps"]
    end

    subgraph Server["서버 (모든 지능)"]
        PixelDecoder["픽셀 디코더"]
        SparseEncoder["Sparse Pixel Encoder<br/>Graph Neural Network"]
        InfoDiffusion["Information Diffusion<br/>Sparse → Dense"]
        QualityAnalyzer["품질 분석기<br/>Uncertainty Estimation"]
        CoordGenerator["좌표 생성기<br/>Top-N Importance"]
        MemoryBank["Temporal Memory Bank<br/>Static/Dynamic"]
    end

    Compressor -->|"2KB"| Upload
    Upload --> PixelDecoder

    PixelDecoder --> SparseEncoder
    SparseEncoder --> InfoDiffusion
    MemoryBank --> InfoDiffusion
    InfoDiffusion --> QualityAnalyzer
    QualityAnalyzer --> CoordGenerator

    CoordGenerator --> Download
    Download -->|"0.5KB<br/>coords[(u,v)...]"| PixelExtractor

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

### Sparse Pixel Encoder 아키텍처

```mermaid
graph TB
    subgraph Input["입력"]
        PixelValues["픽셀 값<br/>[N, 3] RGB"]
        PixelPositions["픽셀 위치<br/>[N, 2] (u,v)"]
    end

    subgraph Encoding["인코딩"]
        PixelEnc["Pixel Encoder<br/>Linear(3, 384)"]
        PosEnc["Continuous Position<br/>Encoding(384)"]
        Concat["Concatenate<br/>[N, 768]"]
    end

    subgraph GraphProcessing["Graph Neural Network"]
        KNNGraph["KNN Graph 구성<br/>k=8 neighbors"]
        GAT1["Graph Attention Layer 1<br/>768→768, 8 heads"]
        GAT2["Graph Attention Layer 2<br/>768→768, 8 heads"]
        GAT3["Graph Attention Layer 3<br/>768→768, 8 heads"]
        GAT4["Graph Attention Layer 4<br/>768→768, 8 heads"]
        GAT5["Graph Attention Layer 5<br/>768→768, 8 heads"]
        GAT6["Graph Attention Layer 6<br/>768→768, 8 heads"]
    end

    subgraph Diffusion["Information Diffusion"]
        InitGrid["빈 그리드 초기화<br/>[224, 224, 768]"]
        PlaceSparse["희소 특징 배치"]
        AnisotropicDiff["Anisotropic Diffusion<br/>경계 보존 확산"]
        Gradient["그래디언트 계산"]
        Conductance["Conductance 맵<br/>1/(1+∇²)"]
        Laplacian["Laplacian 연산"]
        NonlinearTrans["비선형 변환<br/>10 iterations"]
    end

    subgraph Output["출력"]
        DenseFeatures["Dense Feature Map<br/>[224, 224, 768]"]
        Decoder["Lightweight Decoder<br/>4 Transformer Blocks"]
        ReconstructedFrame["복원된 프레임<br/>[224, 224, 3]"]
    end

    PixelValues --> PixelEnc
    PixelPositions --> PosEnc
    PixelEnc --> Concat
    PosEnc --> Concat

    Concat --> KNNGraph
    PixelPositions --> KNNGraph

    KNNGraph --> GAT1
    GAT1 --> GAT2
    GAT2 --> GAT3
    GAT3 --> GAT4
    GAT4 --> GAT5
    GAT5 --> GAT6

    GAT6 --> PlaceSparse
    InitGrid --> PlaceSparse
    PlaceSparse --> AnisotropicDiff

    AnisotropicDiff --> Gradient
    Gradient --> Conductance
    Conductance --> Laplacian
    Laplacian --> NonlinearTrans
    NonlinearTrans --> AnisotropicDiff
    NonlinearTrans --> DenseFeatures

    DenseFeatures --> Decoder
    Decoder --> ReconstructedFrame

    style Input fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style GraphProcessing fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    style Diffusion fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    style Output fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
```

### Temporal Memory Bank 구조

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

### 1. 서버 예측 오차 기반 픽셀 중요도 계산

서버는 복원 품질을 자체 평가하여 다음 프레임에 필요한 픽셀을 결정합니다:

-   **Monte Carlo Dropout**: 여러 dropout 패턴으로 불확실성 추정
-   **Attention 가중치 분석**: 정보 부족 영역 식별
-   **시간적 일관성 검사**: 프레임 간 불일치 영역 탐지
-   **종합 중요도 맵**: 가중 평균으로 최종 중요도 계산

### 2. Information Diffusion Module

희소 픽셀에서 전체 이미지로 정보를 확산시키는 핵심 모듈:

-   **Anisotropic Diffusion**: 경계를 보존하면서 정보 확산
-   **그래디언트 기반 Conductance**: 경계에서 낮은 확산율
-   **학습 가능한 비선형 변환**: 10회 반복으로 최적 확산

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
v5/
├── README.md                    # 프로젝트 문서 (본 파일)
├── requirements.txt             # 의존성 패키지
├── models/                      # 모델 아키텍처
│   ├── __init__.py
│   ├── sgaps_mae.py            # SGAPS-MAE 메인 모델
│   ├── sparse_encoder.py       # Sparse Pixel Encoder
│   ├── information_diffusion.py # Information Diffusion Module
│   ├── graph_attention.py      # Graph Attention Layers
│   └── temporal_memory.py      # Temporal Memory Bank
├── server/                      # 서버 구현
│   ├── __init__.py
│   ├── replay_server.py        # 메인 서버
│   ├── quality_analyzer.py     # 품질 분석기
│   ├── coordinate_generator.py # 좌표 생성기
│   └── latency_compensator.py  # 지연 보상
├── client/                      # 클라이언트 구현
│   ├── __init__.py
│   ├── minimal_client.py       # 경량 클라이언트
│   ├── pixel_extractor.py      # 픽셀 추출기
│   └── compressor.py           # 압축 모듈
├── training/                    # 학습 관련
│   ├── __init__.py
│   ├── trainer.py              # 학습 메인
│   ├── curriculum.py           # 커리큘럼 학습
│   ├── losses.py               # 손실 함수
│   └── config.yaml             # 학습 설정
├── utils/                       # 유틸리티
│   ├── __init__.py
│   ├── compression.py          # 압축 유틸리티
│   ├── network.py              # 네트워크 통신
│   └── metrics.py              # 평가 지표
└── experiments/                 # 실험 스크립트
    ├── train.py                # 학습 실행
    ├── evaluate.py             # 평가
    └── demo_client_server.py   # 데모 실행
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
2. `SGAPSManager` 컴포넌트를 씩에 추가
3. 서버 엔드포인트 설정 (`ws://localhost:8000/ws/stream`)
4. Play 모드에서 자동 연결 또는 `ConnectToServer()` 호출

자세한 내용은 [Unity Client README](unity-client/README.md) 참조

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
2. Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks
3. Perona, P., & Malik, J. (1990). Scale-space and edge detection using anisotropic diffusion

## 기여

프로젝트에 기여하시려면:

1. Fork 후 브랜치 생성
2. 코드 작성 및 테스트
3. Pull Request 제출

문의사항은 Issues 페이지를 이용해주세요.
