# SGAPS-MAE v4 프로젝트 요약

## 프로젝트 개요
**SGAPS-MAE (Server-Guided Adaptive Pixel Sampling MAE)**는 게임 세션 리플레이를 위한 혁신적인 픽셀 단위 적응적 샘플링 시스템입니다. 전체 프레임의 0.5-2% 픽셀만으로 224×224 프레임을 복원합니다.

## 핵심 혁신
- **서버 주도 샘플링**: 모든 지능을 서버에 집중, 클라이언트는 단순 픽셀 추출만 수행
- **픽셀 단위 적응**: 정보량이 높은 픽셀만 선택적 샘플링
- **실시간 피드백 루프**: 복원 품질 기반 동적 샘플링 최적화
- **극한 압축**: 250-1000개 픽셀(0.5-2%)만으로 프레임 복원

## 시스템 아키텍처

### 전체 구조
```
Unity Client (극경량)
  ↓ 2KB/frame (pixels + state_vector)
Server (모든 지능)
  - Sparse Pixel Transformer
  - Attention Entropy 기반 중요도 계산
  - 적응적 UV 좌표 생성
  ↓ 0.5KB/frame (UV coordinates)
Unity Client
```

### 주요 컴포넌트

#### 1. Unity Client (unity-client/)
- **FrameCaptureHandler**: 화면 캡처 및 Grayscale 변환
- **PixelSampler**: UV 좌표 기반 픽셀 샘플링
- **NetworkClient**: WebSocket 통신
- **StateVectorCollector**: 게임 상태 수집 (가변 길이, sentinel 패딩)
- **SGAPSManager**: 전체 시스템 조율

#### 2. Python Server (sgaps-server/)
- **FastAPI + WebSocket**: 실시간 양방향 통신
- **Sparse Pixel Transformer**: 희소 픽셀 → 전체 프레임 복원
  - Self-Attention Encoder (희소 픽셀 간 관계)
  - State Token Integration (게임 상태를 토큰으로 통합하여 Encoder 입력)
  - **Skip Connection** (Encoder-Decoder Skip, 기본 활성화)
  - CrossAttentionDecoderLayer (메모리 효율적, Self-Attention 제거)
  - CNN Refinement Head (최종 품질 향상)
- **Attention Entropy Importance Calculator**: 중요도 맵 생성
- **Adaptive UV Sampler**: 중요도 기반 샘플링
- **HDF5 Storage**: 데이터 저장 (checkpoint_key/session_id 구조)

### 핵심 메커니즘

#### Attention Entropy 기반 중요도 계산
- Decoder Cross-Attention 가중치의 엔트로피로 불확실성 측정
- 낮은 엔트로피: 특정 픽셀에 집중 → 중요도 낮음
- 높은 엔트로피: 불확실성 높음 → 추가 샘플링 필요

#### 적응적 샘플링 전략
- 60% importance-based (높은 엔트로피 영역)
- 40% uniform (전역 커버리지 유지)

#### 서버-클라이언트 프로토콜
```
1. Client → Server: session_start {checkpoint_key, resolution}
2. Server → Client: session_start_ack {sample_count, max_state_dim, target_fps}
3. Server → Client: uv_coordinates (초기 좌표)
4. Loop:
   - Client → Server: frame_data {pixels, state_vector}
   - Server → Client: uv_coordinates (다음 프레임용)
```

## 설정 시스템 (Hydra)
- **model/**: spt_small, spt_base, spt_large
- **training/**: phase1, phase2, phase3
- **sampling/**: uniform, importance, hybrid, adaptive_importance
- **mask_update/**: fixed, adaptive, quality_based
- **skip/**: enabled (기본), disabled - Encoder-Decoder Skip Connection
- **debug/**: enabled, disabled (시각화 시스템)
- **server/**: development, production, school_gpu

### 서버 제어 파라미터
다음 값들은 서버 설정(conf/config.yaml)에서 관리되며 클라이언트에 전달:
- `sample_count`: 프레임당 샘플 픽셀 수 (기본: 500)
- `max_state_dim`: 상태 벡터 최대 차원 (기본: 64)
- `target_fps`: 캡처 프레임 레이트 (기본: 10)
- `sentinel_value`: -999.0 (서버 내부 전용, 상태 벡터 패딩용)

## 프로젝트 구조
```
v4/
├── README.md                   # 프로젝트 문서
├── docs/                       # 상세 문서
│   ├── API_SPECIFICATION.md
│   ├── SERVER_IMPLEMENTATION.md
│   ├── CLIENT_IMPLEMENTATION.md
│   ├── CONFIGURATION.md
│   ├── IMPLEMENTATION_STATUS.md
│   ├── DEVELOPMENT_ROADMAP.md
│   ├── PHASE3_ADAPTIVE_SAMPLING.md
│   └── DEBUG_VISUALIZATION.md
├── sgaps-server/               # Python 서버
│   ├── main.py
│   ├── requirements.txt
│   ├── conf/                   # Hydra 설정
│   └── sgaps/
│       ├── api/                # WebSocket/REST 핸들러
│       ├── core/               # 세션 관리, 샘플러, 복원기
│       ├── data/               # HDF5 스토리지, Dataset
│       ├── models/             # Sparse Pixel Transformer
│       └── utils/              # 메트릭, 유틸리티
└── unity-client/               # Unity UPM 패키지
    ├── package.json
    ├── Runtime/Scripts/
    │   ├── Core/               # SGAPSManager, NetworkClient
    │   └── Data/               # PixelData, StateVectorCollector
    └── Samples~/               # 예제 씬
```

## 개발 상태 (2025-12-09)

### ✅ Phase 1 완료 (인프라)
- Unity ↔ Server WebSocket 통신
- 프레임 캡처 및 픽셀 샘플링
- HDF5 데이터 저장
- 서버 설정 시스템 (Hydra)
- 고정 패턴 샘플링

### ✅ Phase 2 완료 (핵심 ML 파이프라인) - 100%
- ✅ **Sparse Pixel Transformer 모델**: 완전 구현 (100%)
  - Self-Attention Encoder, Cross-Attention Decoder 완전 동작
  - State Vector Encoder 완전 구현
  - CNN Refinement Head 완전 구현
  - Forward pass 11단계 완전 구현
  - Attention weights 반환 지원
- ✅ **학습 파이프라인**: 완전 구현 (100%)
  - SGAPSDataset (HDF5 데이터 로딩)
  - SGAPSTrainer (train/validation loop)
  - Checkpoint 저장/로드
  - AMP (Mixed Precision) 지원
- ✅ **서버 통합**: 완전 구현 (100%)
  - FrameReconstructor (실시간 추론)
  - 체크포인트 키 기반 모델 관리
  - 해상도 독립성 (Resolution Independence)
- ✅ **모니터링 시스템**: WandB 통합 완료 (100%)
  - 학습 과정 모니터링
  - 실시간 추론 결과 로깅
  - Multi-session 지원

### ✅ Phase 3 완료 (적응적 샘플링) - 100%
- ✅ **Attention Entropy Importance Calculator**: 완전 구현
  - `sgaps/core/importance.py` 신규 생성
  - Shannon Entropy 기반 중요도 맵 생성
  - Cross-Attention 가중치 집계 및 분석
- ✅ **Adaptive UV Sampler**: 완전 구현
  - `sgaps/core/sampler.py`에 AdaptiveUVSampler 클래스 추가
  - 60% 중요도 기반 + 40% 균등 샘플링 하이브리드 전략
  - 웜업 메커니즘 (기본 10 프레임)
  - 기본 충돌 회피 알고리즘
- ✅ **서버 파이프라인 통합**: 완전 구현
  - `sgaps/api/websocket.py` 적응형 샘플링 통합
  - 실시간 중요도 맵 계산 및 UV 좌표 생성
  - WandB 중요도 메트릭 로깅 (mean, max, std, entropy)
- ✅ **설정 시스템**: 완전 구현
  - `conf/sampling/adaptive.yaml` 신규 생성
  - Hydra 명령줄 오버라이드 지원

### 🔄 현재 시스템 기능
- ✅ **오프라인 학습**: `python scripts/train.py` 실행 가능
- ✅ **실시간 추론**: Unity 클라이언트 연결 후 프레임 재구성 동작
- ✅ **WandB 모니터링**: 학습 및 추론 로그 실시간 기록
- ✅ **Multi-session**: 여러 클라이언트 동시 연결 지원
- ✅ **적응형 샘플링**: Attention Entropy 기반 동적 UV 좌표 생성
- ✅ **폐쇄 루프 피드백**: 실시간 중요도 맵 업데이트 및 샘플링 조정
- ✅ **디버그 시각화**: 2x4 그리드 통합 대시보드 (원본/복원/차이/샘플링/중요도/손실/Attention/State)
- ✅ **설정 가능 로깅**: N 프레임마다 WandB 로깅 (성능 최적화)
- ✅ **품질 메트릭**: PSNR, SSIM, MSE 자동 계산 및 로깅

### 🚧 Phase 4 예정 (고급 기능 및 최적화)
- ⏳ **Optical Flow & Motion Prediction**: 시간적 일관성 강화
- ⏳ **Multi-Checkpoint Management**: 동적 모델 전환 및 A/B 테스팅
- ⏳ **Reconstruction Quality Feedback Loop**: 품질 기반 샘플링 조정
- ⏳ **Unity Debug UI**: 실시간 모니터링 인게임 UI (현재는 WandB 대시보드 사용)

### 📊 주요 버그 수정 완료
- ✅ Resolution Independence (CUDA OOM 해결)
- ✅ Gradient Tracking 오류 수정
- ✅ Checkpoint Caching 개선
- ✅ WandB Multi-Session Step Conflict 해결

**전체 진행률: ~90% (핵심 기능 및 디버그 시스템 완전 동작, 고급 최적화 및 부가 기능 미구현)**

## 기술 스택
- **Client**: Unity 2021.3 LTS, C#, WebSocket (NativeWebSocket)
- **Server**: Python 3.10, FastAPI, PyTorch 2.0+, Hydra
- **Storage**: HDF5
- **Monitoring**: Weights & Biases (WandB)
- **Communication**: WebSocket (실시간), REST API (세션 관리)

## 성능 목표
- **재구성 품질**: PSNR > 39.2 dB, SSIM > 0.95
- **샘플링 비율**: 0.5-2% (250-1000 pixels)
- **클라이언트 부하**: CPU 0.1%, GPU 불필요
- **네트워크 대역폭**: 75 KB/s @ 30 FPS
- **서버 추론 시간**: 5-10ms @ RTX 3090

## 실행 환경 제약
- **sgaps-server/**는 원격 GPU 서버에서 실행됩니다
- 로컬에서 Python 실행은 무의미하며, 원격 서버에서 실행할 명령어를 요청해야 합니다
- Unity 프로젝트는 로컬에서 실행 가능합니다

## 주요 용어
- **UV 좌표**: 정규화된 2D 좌표 (0.0-1.0 범위)
- **Sparse Pixels**: 샘플링된 희소 픽셀 집합
- **State Vector**: 게임 상태 벡터 (가변 길이, sentinel 패딩)
- **Checkpoint Key**: 모델 식별자 (게임/맵별 다른 모델 사용)
- **Importance Map**: 픽셀별 중요도 점수 (0-1 범위)
- **Sentinel Value**: -999.0 (미사용 상태 벡터 요소 표시)
