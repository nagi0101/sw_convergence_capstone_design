# Development Roadmap

## 개요

SGAPS v4 프로젝트의 8주 개발 로드맵입니다. 빠른 프로토타이핑을 우선시하며, 각 Phase마다 동작하는 End-to-End 시스템을 구축합니다.

---

## 전체 타임라인

```
Week 1-2: Phase 0 - 문서화 및 환경 구축
Week 3-4: Phase 1 - 기본 통신 및 고정 샘플링
Week 5-6: Phase 2 - 적응적 마스크 업데이트
Week 7-8: Phase 3 - 딥러닝 학습 및 최적화
```

---

## Phase 0: 문서화 및 환경 구축 (Week 1-2)

### 목표

프로젝트의 기반을 다지고, 개발 환경을 완벽히 구축하여 이후 원활한 개발 진행

### Week 1: 문서 작성 및 검토

#### Day 1-2: 계획서 작성

-   [x] PROJECT_PLAN.md 작성
-   [x] MVP_FEATURES.md 작성
-   [x] CLIENT_IMPLEMENTATION.md 작성
-   [x] SERVER_IMPLEMENTATION.md 작성
-   [x] API_SPECIFICATION.md 작성
-   [x] CONFIGURATION.md 작성
-   [x] DEVELOPMENT_ROADMAP.md 작성

#### Day 3-4: 문서 검토 및 수정

-   [ ] 사용자에게 7개 문서 검토 요청
-   [ ] 피드백 수렴 및 문서 수정
-   [ ] 최종 문서 확정

#### Day 5-7: 기술 스택 리서치

-   [ ] Unity UPM 패키지 개발 가이드 학습
-   [ ] FastAPI + WebSocket 튜토리얼
-   [ ] PyTorch Transformer 구현 스터디
-   [ ] Hydra 설정 시스템 학습

**Deliverables**:

-   ✅ 확정된 프로젝트 계획서 (7개 문서)
-   [ ] 기술 스택 학습 노트

---

### Week 2: 개발 환경 구축

#### Day 1-2: Unity 환경 설정

-   [ ] Unity 2021.3 LTS 설치
-   [ ] 테스트용 샘플 프로젝트 생성
-   [ ] UPM 패키지 템플릿 생성
-   [ ] Git 리포지토리 구조 설정
    ```
    sgaps/
    ├── unity-client/       # Unity UPM 패키지
    ├── python-server/      # FastAPI 서버
    ├── docs/               # 문서
    └── examples/           # 샘플 프로젝트
    ```

#### Day 3-4: Python 서버 환경 설정

-   [ ] Python 3.10 가상환경 생성
-   [ ] requirements.txt 작성 및 의존성 설치
    ```
    fastapi==0.104.1
    uvicorn==0.24.0
    websockets==12.0
    torch==2.1.0
    torchvision==0.16.0
    hydra-core==1.3.2
    opencv-python==4.8.1
    numpy==1.24.3
    pillow==10.1.0
    h5py==3.10.0
    wandb==0.16.0
    ```
-   [ ] Hydra 설정 디렉토리 구조 생성
-   [ ] 기본 FastAPI 앱 테스트 (Hello World)

#### Day 5-6: GPU 서버 접속 및 설정

-   [ ] 학교 GPU 서버 접속 확인
-   [ ] SSH 키 설정
-   [ ] Docker 설치 확인
-   [ ] CUDA/cuDNN 버전 확인
-   [ ] 테스트 PyTorch 코드 실행 (GPU 작동 확인)

#### Day 7: CI/CD 파이프라인 구축

-   [ ] GitHub Actions 설정
    -   Unity 코드 린팅 (C#)
    -   Python 코드 린팅 (Black, Flake8)
    -   단위 테스트 자동 실행
-   [ ] Pre-commit hooks 설정

**Deliverables**:

-   [ ] Unity 프로젝트 템플릿
-   [ ] FastAPI 서버 템플릿
-   [ ] Docker Compose 설정 파일
-   [ ] CI/CD 파이프라인

**Milestone 0 체크리스트**:

-   [ ] 모든 문서가 검토 및 확정됨
-   [ ] Unity 프로젝트가 빌드 가능함
-   [ ] FastAPI 서버가 로컬에서 실행됨
-   [ ] 학교 GPU 서버 접속 가능
-   [ ] Git 리포지토리 구조 완성

---

## Phase 1: 기본 통신 및 고정 샘플링 (Week 3-4)

### 목표

Unity 클라이언트 ↔ 서버 간 기본 통신을 확립하고, 고정 패턴 샘플링으로 프레임 재구성

### Week 3: Unity 클라이언트 구현

#### Day 1-2: RenderTexture 캡처 구현

-   [ ] `FrameCaptureHandler.cs` 구현
-   [ ] Grayscale 변환 쉐이더 작성
-   [ ] Inspector에서 캡처 프리뷰 표시
-   [ ] 성능 측정 (FPS 영향 < 1 fps)

#### Day 3-4: 픽셀 샘플링 구현

-   [ ] `PixelSampler.cs` 구현
-   [ ] 균등 그리드 패턴 생성
-   [ ] RenderTexture에서 픽셀 추출
-   [ ] 데이터 구조체 정의 (`PixelData`)

#### Day 5-6: WebSocket 클라이언트 구현

-   [ ] `NetworkClient.cs` 구현
-   [ ] 서버 연결 및 재연결 로직
-   [ ] JSON 직렬화/역직렬화
-   [ ] 에러 핸들링

#### Day 7: Unity 통합 및 테스트

-   [ ] `SGAPSManager.cs` 통합
-   [ ] Inspector UI 구현
-   [ ] 더미 서버로 End-to-End 테스트
-   [ ] UPM 패키지 빌드

**Deliverables**:

-   [ ] Unity UPM 패키지 v0.1.0
-   [ ] 테스트 씬 (SGAPSDemo.unity)

---

### Week 4: Python 서버 구현

#### Day 1-2: FastAPI + WebSocket 서버

-   [ ] `main.py` WebSocket 엔드포인트 구현
-   [ ] 클라이언트 연결 관리 (`ConnectionManager`)
-   [ ] 프레임 데이터 수신 및 파싱
-   [ ] 로컬에서 Unity와 연결 테스트

#### Day 3-4: 단순 프레임 재구성

-   [ ] OpenCV 기반 보간법 구현 (Inpainting)
-   [ ] PSNR, SSIM 메트릭 계산
-   [ ] 재구성된 프레임 PNG로 저장

#### Day 5-6: UV 좌표 생성 및 전송

-   [ ] 고정 그리드 UV 좌표 생성
-   [ ] 클라이언트에 좌표 전송
-   [ ] 클라이언트가 수신 확인

#### Day 7: 데이터 저장 및 테스트

-   [ ] HDF5 데이터 저장 구현
-   [ ] 30분 테스트 에피소드 수집
-   [ ] End-to-End 통합 테스트

**Deliverables**:

-   [ ] FastAPI 서버 v0.1.0
-   [ ] 샘플 데이터셋 (1개 에피소드)

**Milestone 1 체크리스트**:

-   [ ] Unity → 서버 프레임 전송 성공
-   [ ] 서버 → Unity UV 좌표 수신 성공
-   [ ] 재구성 PSNR > 20 dB
-   [ ] 클라이언트 FPS 영향 < 1 fps
-   [ ] 30분 에피소드 데이터 수집 완료

---

## Phase 2: 적응적 마스크 업데이트 (Week 5-6)

### 목표

서버가 재구성 품질에 따라 동적으로 UV 좌표를 결정하는 시스템 구축

### Week 5: Importance Map 계산 (Attention Entropy)

#### Day 1-2: AttentionEntropyImportanceCalculator 구현

-   [ ] Decoder Cross-Attention weights 추출 기능 추가
-   [ ] Head 평균 및 픽셀별 엔트로피 계산
-   [ ] Importance map 정규화 [0, 1]
-   [ ] 단위 테스트 (엔트로피 범위 검증)

#### Day 3-4: Adaptive UV Sampler 구현

-   [ ] 중요도 기반 가중 샘플링
-   [ ] Hybrid 샘플링 (70% importance + 30% uniform)
-   [ ] 최소 거리 제약 적용
-   [ ] 샘플링 전략 시각화

#### Day 5-7: 마스크 업데이트 스케줄러

-   [ ] Fixed mode 구현
-   [ ] Quality-based mode 구현
-   [ ] Adaptive mode 구현
-   [ ] Wandb에 업데이트 이벤트 로깅

**Deliverables**:

-   [ ] Importance Calculator 모듈
-   [ ] Adaptive UV Sampler
-   [ ] Mask Update Scheduler

---

### Week 6: 평가 및 최적화

#### Day 1-2: A/B 테스트 프레임워크

-   [ ] 실험 설정 파일 작성
-   [ ] 동일 에피소드로 여러 전략 비교
-   [ ] Wandb에 결과 기록

#### Day 3-4: 성능 평가

-   [ ] Baseline (균등 샘플링) vs Adaptive 비교
-   [ ] PSNR/SSIM 향상 확인
-   [ ] 샘플링 효율 분석

#### Day 5-6: 클라이언트 업데이트

-   [ ] Unity 클라이언트에 동적 UV 좌표 수신 로직 추가
-   [ ] Inspector에 현재 샘플링 전략 표시
-   [ ] 재연결 시 상태 복구

#### Day 7: 통합 테스트 및 디버깅

-   [ ] End-to-End 테스트
-   [ ] 버그 수정
-   [ ] 문서 업데이트

**Deliverables**:

-   [ ] A/B 테스트 결과 리포트
-   [ ] Unity UPM 패키지 v0.2.0
-   [ ] FastAPI 서버 v0.2.0

**Milestone 2 체크리스트**:

-   [ ] Adaptive 샘플링이 Baseline 대비 +3dB PSNR 달성
-   [ ] 재구성 PSNR > 25 dB
-   [ ] 재구성 SSIM > 0.85
-   [ ] 마스크 업데이트 오버헤드 < 10ms

---

## Phase 3: 딥러닝 학습 및 최적화 (Week 7-8)

### 목표

Sparse Pixel Transformer 모델을 학습하여 재구성 품질을 비약적으로 향상

### Week 7: 모델 구현 및 학습 시작

#### Day 1-2: Sparse Pixel Transformer 구현

-   [ ] `spt.py` 모델 아키텍처 구현
-   [ ] `positional_encoding.py` 연속 위치 인코딩
-   [ ] `losses.py` Sampled Pixel L2 Loss 구현
-   [ ] 단위 테스트 (입력/출력 shape 검증)

#### Day 3-4: 데이터 파이프라인

-   [ ] `SGAPSDataset` 구현
-   [ ] 데이터 증강 (`transforms.py`)
-   [ ] DataLoader 설정
-   [ ] Train/Val/Test 스플릿

#### Day 5-7: 학습 파이프라인

-   [ ] `SGAPSTrainer` 구현
-   [ ] Mixed Precision 적용
-   [ ] Wandb 로깅 설정
-   [ ] 학습 시작 (GPU 서버)

**Deliverables**:

-   [ ] Sparse Pixel Transformer 코드
-   [ ] 학습 스크립트

---

### Week 8: 학습 완료 및 배포

#### Day 1-3: 학습 모니터링 및 조정

-   [ ] Wandb에서 학습 곡선 모니터링
-   [ ] 하이퍼파라미터 튜닝 (필요 시)
-   [ ] 체크포인트 관리

#### Day 4-5: 평가 및 벤치마크

-   [ ] Test set 평가
-   [ ] PSNR/SSIM 계산
-   [ ] 추론 속도 벤치마크
-   [ ] 재구성 샘플 시각화

#### Day 6: 서버 통합 및 배포

-   [ ] 학습된 모델을 FastAPI 서버에 통합
-   [ ] TorchScript 컴파일 (속도 향상)
-   [ ] Docker 이미지 빌드
-   [ ] 학교 GPU 서버에 배포

#### Day 7: 최종 테스트 및 데모

-   [ ] End-to-End 통합 테스트
-   [ ] 성능 벤치마크 (모든 메트릭)
-   [ ] 데모 영상 녹화
-   [ ] 최종 문서 업데이트

**Deliverables**:

-   [ ] 학습된 모델 체크포인트
-   [ ] 평가 결과 리포트
-   [ ] Docker 이미지
-   [ ] 데모 영상

**Milestone 3 체크리스트**:

-   [ ] 재구성 PSNR > 28 dB
-   [ ] 재구성 SSIM > 0.90
-   [ ] 추론 시간 < 30ms @ RTX 3090
-   [ ] 네트워크 대역폭 < 50 KB/s @ 30 FPS
-   [ ] Docker 컨테이너로 배포 성공

---

## 최종 마일스톤: MVP 완성

### 성공 기준

| 항목                      | 목표               | 상태 |
| ------------------------- | ------------------ | ---- |
| **기능**                  |                    |      |
| Unity → 서버 통신         | 동작               | [ ]  |
| 서버 → Unity UV 좌표 전송 | 동작               | [ ]  |
| 적응적 샘플링             | 구현               | [ ]  |
| 딥러닝 재구성             | 구현               | [ ]  |
| **성능**                  |                    |      |
| 재구성 PSNR               | > 28 dB            | [ ]  |
| 재구성 SSIM               | > 0.90             | [ ]  |
| 클라이언트 FPS 영향       | < 1 fps            | [ ]  |
| 서버 추론 시간            | < 30ms             | [ ]  |
| 네트워크 대역폭           | < 50 KB/s @ 30 FPS | [ ]  |
| **개발**                  |                    |      |
| Unity UPM 패키지          | 릴리즈             | [ ]  |
| FastAPI 서버              | 배포               | [ ]  |
| 학습 파이프라인           | 동작               | [ ]  |
| 문서화                    | 완료               | [ ]  |

---

## 리스크 관리

### 리스크 1: 학습 데이터 부족

**영향**: 높음
**확률**: 중간
**완화**:

-   공개 게임 영상 데이터셋으로 Pre-training
-   데이터 증강 적극 활용
-   시뮬레이션 데이터 생성

**비상 계획**:

-   단순 보간법으로 fallback
-   샘플링 픽셀 수 증가

---

### 리스크 2: GPU 서버 불안정

**영향**: 높음
**확률**: 낮음
**완화**:

-   체크포인트 자주 저장 (5 에폭마다)
-   학습 재개 로직 구현
-   로컬 GPU (개인)로 백업

**비상 계획**:

-   Google Colab Pro 사용
-   AWS SageMaker 임시 사용

---

### 리스크 3: Unity 호환성 문제

**영향**: 중간
**확률**: 중간
**완화**:

-   최소 지원 버전 명시 (Unity 2021.3 LTS)
-   자동 테스트 파이프라인 구축
-   Compatibility layer 구현

**비상 계획**:

-   지원 버전 제한
-   커뮤니티 이슈 트래킹

---

### 리스크 4: 일정 지연

**영향**: 중간
**확률**: 높음
**완화**:

-   버퍼 시간 확보 (각 Phase당 +2일)
-   핵심 기능 우선 구현
-   스프린트 회고를 통해 조정

**비상 계획**:

-   Phase 3 일부 기능 축소
-   컬러 지원 등 부가 기능 제외

---

## 주간 체크인

### 매주 금요일 오후

-   [ ] 주간 목표 달성 여부 확인
-   [ ] 다음 주 계획 검토
-   [ ] 리스크 및 이슈 논의
-   [ ] 문서 업데이트

### 체크인 템플릿

```markdown
## Week X 체크인 (YYYY-MM-DD)

### 완료한 작업

-   ...

### 진행 중인 작업

-   ...

### 다음 주 계획

-   ...

### 블로커 및 이슈

-   ...

### 메트릭

-   커밋 수: X
-   PR 수: X
-   코드 리뷰: X
```

---

## 개발 후 계획 (Optional)

### Phase 4: 컬러 영상 지원 (Week 9-10)

-   YCbCr 색공간 활용
-   Luminance 우선, Chrominance 서브샘플링

### Phase 5: 성능 최적화 (Week 11-12)

-   압축 알고리즘 적용 (MsgPack + LZ4)
-   UV 좌표 양자화
-   30 FPS @ 실시간 동작 달성

### Phase 6: 사용자 연구 (Week 13-14)

-   게임 개발자 대상 베타 테스트
-   피드백 수집 및 반영

---

## 개발 원칙

### 1. 빠른 프로토타이핑

-   "완벽함"보다 "동작함"을 먼저
-   매 Phase마다 End-to-End 시스템 구축
-   조기 피드백 루프

### 2. 점진적 개선

-   Phase 1: 단순하지만 동작
-   Phase 2: 스마트하게 개선
-   Phase 3: 최고 품질 달성

### 3. 측정 가능한 목표

-   모든 Phase에 명확한 메트릭
-   주간 진행 상황 추적
-   데이터 기반 의사결정

### 4. 문서 우선

-   코드 작성 전 문서 작성
-   설계 검토 후 구현
-   변경 사항 즉시 반영

---

## 최종 체크리스트

### 문서

-   [x] PROJECT_PLAN.md
-   [x] MVP_FEATURES.md
-   [x] CLIENT_IMPLEMENTATION.md
-   [x] SERVER_IMPLEMENTATION.md
-   [x] API_SPECIFICATION.md
-   [x] CONFIGURATION.md
-   [x] DEVELOPMENT_ROADMAP.md

### 코드 (예정)

-   [ ] Unity UPM 패키지
-   [ ] FastAPI 서버
-   [ ] Sparse Pixel Transformer
-   [ ] 학습 스크립트
-   [ ] 평가 스크립트

### 배포 (예정)

-   [ ] Docker 이미지
-   [ ] CI/CD 파이프라인
-   [ ] 학교 GPU 서버 배포

### 결과물 (예정)

-   [ ] 학습된 모델 체크포인트
-   [ ] 샘플 데이터셋
-   [ ] 평가 결과 리포트
-   [ ] 데모 영상

---

## 다음 단계

1. ✅ **모든 문서 작성 완료**
2. ⏭️ **사용자에게 문서 검토 요청**
3. ⏭️ **피드백 반영 및 문서 최종 확정**
4. ⏭️ **Phase 0 Week 2부터 개발 시작**

---

**프로젝트 시작일**: 2025년 X월 X일 (사용자 확정 후)
**MVP 목표일**: 시작일 + 8주

이 로드맵은 살아있는 문서입니다. 진행 상황에 따라 유연하게 조정됩니다.
