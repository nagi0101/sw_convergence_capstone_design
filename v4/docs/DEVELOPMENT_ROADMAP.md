# Development Roadmap

## 개요

SGAPS v4 프로젝트의 개발 로드맵입니다. 초기 8주 계획을 넘어, Phase 1, 2, 3를 성공적으로 완료하였으며, 현재는 시스템의 성능 최적화 및 고급 기능 구현을 목표로 다음 단계로 진입합니다. 각 Phase마다 동작하는 End-to-End 시스템을 구축하는 원칙은 지속됩니다.

---

## 전체 타임라인

```
Week 1-2: Phase 0 - 문서화 및 환경 구축 (완료)
Week 3-4: Phase 1 - 기본 통신 및 고정 샘플링 (완료)
Week 5-6: Phase 2 - 핵심 ML 파이프라인 (완료)
Week 7-8: Phase 3 - 적응형 샘플링 구현 (완료)
Week 9+:  Phase 4 - 성능 최적화 및 고급 기능 (진행 예정)
```

---

## ✅ Phase 0: 문서화 및 환경 구축 (Week 1-2)

### 목표
프로젝트의 기반을 다지고, 개발 환경을 완벽히 구축하여 이후 원활한 개발 진행

**결과:**
-   ✅ 확정된 프로젝트 계획서 (7개 문서)
-   ✅ 기술 스택 학습 완료
-   ✅ Unity 개발 환경 설정 완료
-   ✅ Python 서버 개발 환경 설정 완료 (GPU 서버 접속 및 PyTorch 테스트 포함)
-   ✅ CI/CD 파이프라인 및 Pre-commit hooks 설정 완료

---

## ✅ Phase 1: 기본 통신 및 고정 샘플링 (Week 3-4)

### 목표
Unity 클라이언트 ↔ 서버 간 기본 통신을 확립하고, 고정 패턴 샘플링으로 프레임 재구성

**결과:**
-   ✅ Unity 클라이언트 기본 기능 (RenderTexture 캡처, 픽셀 샘플링, WebSocket 통신) 구현 및 UPM 패키지 빌드
-   ✅ FastAPI 서버 기본 기능 (WebSocket 엔드포인트, 클라이언트 연결 관리, OpenCV 기반 재구성, HDF5 데이터 저장) 구현
-   ✅ Unity 클라이언트와 FastAPI 서버 간 End-to-End 통신 및 기본 프레임 재구성 성공
-   ✅ **Milestone 1 체크리스트 달성:**
    -   ✅ Unity → 서버 프레임 전송 성공
    -   ✅ 서버 → Unity UV 좌표 수신 성공
    -   ✅ 재구성 PSNR > 20 dB (OpenCV Inpainting 기준)
    -   ✅ 클라이언트 FPS 영향 < 1 fps
    -   ✅ 30분 에피소드 데이터 수집 완료

---

## ✅ Phase 2: 핵심 ML 파이프라인 (Week 5-6)

### 목표
Sparse Pixel Transformer (SPT) 모델을 구현하고, 이를 활용한 학습 및 추론 파이프라인 구축

**결과:**
-   ✅ **Sparse Pixel Transformer (SPT) 모델 구현:** `sgaps/models/spt.py` (`CrossAttentionDecoderLayer`, `SparsePixelTransformer` 등), `sgaps/models/positional_encoding.py`, `sgaps/models/losses.py` (L2, Perceptual, Structural Loss). `return_attention` 기능 포함.
-   ✅ **학습 파이프라인 구축:** `sgaps/training/trainer.py` (`SGAPSTrainer`), `scripts/train.py` (Hydra 연동 학습 스크립트). AMP (Mixed Precision) 지원 및 WandB 로깅 통합.
-   ✅ **추론 파이프라인 구축:** `sgaps/core/reconstructor.py` (모델 로딩, GPU 추론, AMP, `attention_weights` 반환). 체크포인트 키 기반 모델 관리.
-   ✅ **Milestone 2 체크리스트 달성:**
    -   ✅ SPT 모델 구현 및 학습 가능
    -   ✅ 기본 학습 파이프라인 동작
    -   ✅ 수집된 데이터로 학습 가능
    -   ✅ Attention 기반 중요도 맵 생성 기능 확보
    -   ✅ 적응형 샘플링 구현 준비 완료

---

## ✅ Phase 3: 적응형 샘플링 구현 (Week 7-8)

### 목표
Attention 메커니즘을 활용한 동적 적응 샘플링을 구현하여 시스템의 효율성과 복원 품질 극대화

**결과:**
-   ✅ **Attention Entropy 기반 중요도 계산기 구현:** `sgaps/core/importance.py` (`AttentionEntropyImportanceCalculator`). Cross-Attention 가중치로부터 Shannon Entropy 기반 중요도 맵 생성.
-   ✅ **적응형 UV 샘플러 구현:** `sgaps/core/sampler.py` (`AdaptiveUVSampler` 확장). 60% 중요도 기반 + 40% 균등 샘플링 전략, 웜업 메커니즘, 기본적인 충돌 회피 알고리즘 구현.
-   ✅ **서버 파이프라인 통합:** `sgaps/api/websocket.py`에서 `FrameReconstructor`, `AttentionEntropyImportanceCalculator`, `AdaptiveUVSampler`를 연동하여 매 프레임마다 동적으로 UV 좌표를 생성 및 클라이언트에 전송. WandB 로깅을 통해 중요도 맵 통계 모니터링.
-   ✅ **설정 시스템 확장:** `conf/sampling/adaptive.yaml` 추가. Hydra를 통해 적응형 샘플링 파라미터 유연하게 설정 및 오버라이드 가능.
-   ✅ **Milestone 3 체크리스트 달성:**
    -   ✅ 재구성 PSNR > 28 dB
    -   ✅ 재구성 SSIM > 0.90
    -   ✅ 추론 시간 < 30ms @ RTX 3090
    -   ✅ 네트워크 대역폭 < 50 KB/s @ 30 FPS
    -   ✅ Docker 컨테이너로 배포 성공 (모델 통합)

---

## 🚧 Phase 4: 성능 최적화 및 고급 기능 (Week 9+)

### 목표
구현된 핵심 시스템의 성능을 최적화하고, 사용자 경험을 향상시키기 위한 고급 기능들을 추가

### 주요 기능 및 개선 사항 (예정)

-   **4.1 품질 기반 샘플 개수 조정:** 실시간 재구성 품질(PSNR/SSIM)에 따라 다음 프레임의 `sample_count`를 동적으로 조절하여 대역폭 효율 극대화.
-   **4.2 다중 스케일 중요도:** 여러 해상도 스케일에서 중요도를 계산하여 더욱 정교하고 풍부한 정보 기반 샘플링 구현.
-   **4.3 시간적 일관성:** 이전 프레임들의 중요도 맵 또는 모션 벡터를 활용하여 프레임 간 샘플링 일관성 강화, 깜빡임 현상 완화.
-   **4.4 광학 흐름 및 모션 예측:** 클라이언트에서 전송된 모션 벡터를 활용하여 다음 프레임의 픽셀 중요도를 예측하고 샘플링 효율 증대.
-   **4.5 Multi-Checkpoint Management 강화:** A/B 테스트, 동적 모델 스위칭 등 고급 모델 관리 기능 추가.
-   **4.6 서버/클라이언트 시각화 툴 개선:** WandB 외에 실시간 디버깅 및 시각화를 위한 추가 툴 개발 (예: Unity 인게임 디버그 UI).
-   **4.7 모델 경량화 및 추론 최적화:** `TorchScript` 컴파일 및 `ONNX` 내보내기, 양자화 등의 기법을 적용하여 추론 속도 및 리소스 사용량 추가 개선.

---

## 최종 마일스톤: SGAPS-MAE 완성

### 성공 기준 (Phase 4 완료 시)

| 항목                      | 목표               | 상태         |
| ------------------------- | ------------------ | ------------ |
| **기능**                  |                    |              |
| Unity → 서버 통신         | 동작               | ✅           |
| 서버 → Unity UV 좌표 전송 | 동작               | ✅           |
| 적응적 샘플링             | 구현               | ✅           |
| 딥러닝 재구성             | 구현               | ✅           |
| 품질 기반 샘플링          | 구현               | 🔴           |
| 시간적 일관성             | 구현               | 🔴           |
| **성능**                  |                    |              |
| 재구성 PSNR               | > 30 dB            | 🟡 (개선 중) |
| 재구성 SSIM               | > 0.95             | 🟡 (개선 중) |
| 클라이언트 FPS 영향       | < 1 fps            | ✅           |
| 서버 추론 시간            | < 10ms             | 🟡 (개선 중) |
| 네트워크 대역폭           | < 50 KB/s @ 30 FPS | ✅           |
| **개발**                  |                    |              |
| Unity UPM 패키지          | 릴리즈             | ✅           |
| FastAPI 서버              | 배포               | ✅           |
| 학습 파이프라인           | 동작               | ✅           |
| 문서화                    | 완료               | ✅           |

---

## 리스크 관리

(기존 로드맵의 리스크 관리 섹션 유지. 필요 시 업데이트)

---

## 주간 체크인

(기존 로드맵의 주간 체크인 섹션 유지)

---

## 개발 원칙

(기존 로드맵의 개발 원칙 섹션 유지)

---

## 최종 체크리스트

### 문서

-   ✅ PROJECT_PLAN.md
-   ✅ MVP_FEATURES.md
-   ✅ CLIENT_IMPLEMENTATION.md
-   ✅ SERVER_IMPLEMENTATION.md
-   ✅ API_SPECIFICATION.md
-   ✅ CONFIGURATION.md
-   ✅ DEVELOPMENT_ROADMAP.md (본 문서)
-   ✅ IMPLEMENTATION_STATUS.md
-   ✅ PHASE3_ADAPTIVE_SAMPLING.md (신규)

### 코드

-   ✅ Unity UPM 패키지
-   ✅ FastAPI 서버
-   ✅ Sparse Pixel Transformer
-   ✅ 학습 스크립트
-   ✅ 평가 스크립트
-   ✅ Attention Entropy Importance Calculator
-   ✅ Adaptive UV Sampler

### 배포

-   ✅ Docker 이미지
-   ✅ CI/CD 파이프라인
-   ✅ 학교 GPU 서버 배포

### 결과물

-   ✅ 학습된 모델 체크포인트
-   ✅ 샘플 데이터셋
-   ✅ 평가 결과 리포트
-   ✅ 데모 영상

---

## 다음 단계

1.  ✅ **모든 Phase 1, 2, 3 구현 및 문서화 완료**
2.  ⏭️ **Phase 4: 성능 최적화 및 고급 기능 개발 시작**
3.  ⏭️ **지속적인 벤치마킹 및 결과 리포트**
4.  ⏭️ **시스템 안정성 및 견고성 강화**

---

**프로젝트 시작일**: 2025년 X월 X일
**MVP 목표일**: 시작일 + 8주 (초과 달성)

이 로드맵은 살아있는 문서입니다. 진행 상황에 따라 유연하게 조정됩니다.