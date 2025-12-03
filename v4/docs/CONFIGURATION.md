# Configuration System

## 개요

SGAPS 프로젝트는 **Hydra**를 사용하여 계층적이고 유연한 설정 시스템을 구축합니다. 실험 추적, 하이퍼파라미터 튜닝, 환경별 배포를 쉽게 관리할 수 있습니다.

---

## 설정 디렉토리 구조

```
conf/
├── config.yaml                 # 메인 설정 (defaults 조합)
├── model/
│   ├── spt_small.yaml         # Sparse Pixel Transformer (소형)
│   ├── spt_base.yaml          # 기본 모델
│   └── spt_large.yaml         # 대형 모델
├── training/
│   ├── phase1.yaml            # Phase 1: 기본 통신
│   ├── phase2.yaml            # Phase 2: 적응적 샘플링
│   └── phase3.yaml            # Phase 3: 딥러닝 학습
├── sampling/
│   ├── uniform.yaml           # 균등 샘플링
│   ├── importance.yaml        # 중요도 샘플링
│   └── hybrid.yaml            # 하이브리드 샘플링
├── mask_update/
│   ├── fixed.yaml             # 고정 주기
│   ├── adaptive.yaml          # 적응적 주기
│   └── quality_based.yaml     # 품질 기반
├── server/
│   ├── development.yaml       # 로컬 개발 환경
│   ├── production.yaml        # 프로덕션 환경
│   └── school_gpu.yaml        # 학교 GPU 서버
├── data/
│   ├── base.yaml              # 기본 데이터 설정
│   └── augmentation.yaml      # 데이터 증강
└── logging/
    ├── console.yaml           # 콘솔 로깅
    ├── wandb.yaml             # Weights & Biases
    └── tensorboard.yaml       # TensorBoard
```

---

## 1. 메인 설정 (config.yaml)

```yaml
# conf/config.yaml

defaults:
  - model: spt_base
  - training: phase1
  - sampling: uniform
  - mask_update: fixed
  - server: development
  - data: base
  - logging: wandb
  - _self_  # 현재 파일의 설정이 defaults보다 우선

# 프로젝트 메타데이터
project:
  name: sgaps-v4
  version: 0.1.0
  description: "Server-Guided Adaptive Pixel Sampling for Game Replay"

# 경로 설정
paths:
  data_root: ${oc.env:DATA_ROOT,./data}
  checkpoint_dir: ${oc.env:CHECKPOINT_DIR,./checkpoints}
  output_dir: ./outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}

# 랜덤 시드
seed: 42

# Hydra 설정
hydra:
  run:
    dir: ${paths.output_dir}
  sweep:
    dir: ./multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

---

## 2. 모델 설정

### 2.1 spt_base.yaml

```yaml
# conf/model/spt_base.yaml

name: sparse_pixel_transformer
checkpoint_path: null  # 학습 시작 시 null, 추론 시 경로 지정

# 아키텍처
architecture:
  embed_dim: 256
  num_heads: 8
  num_encoder_layers: 6
  num_decoder_layers: 4
  dropout: 0.1

# Positional Encoding
positional_encoding:
  type: continuous_sinusoidal
  max_freq: 10
  num_bands: 64

# CNN Refinement Head
refinement_head:
  channels: [128, 64]
  kernel_size: 3
  activation: relu

# 입력 제약
input_constraints:
  min_num_pixels: 100
  max_num_pixels: 5000
  supported_resolutions:
    - [640, 480]
    - [1280, 720]
    - [1920, 1080]

# 추론 최적화
inference:
  use_amp: true          # Mixed Precision
  use_torch_compile: true  # PyTorch 2.0 compile
  batch_size: 1          # 실시간 추론은 배치 크기 1
```

### 2.2 spt_small.yaml

```yaml
# conf/model/spt_small.yaml
# 빠른 프로토타이핑용 소형 모델

defaults:
  - spt_base

name: sparse_pixel_transformer_small

architecture:
  embed_dim: 128         # 256 → 128
  num_heads: 4           # 8 → 4
  num_encoder_layers: 4  # 6 → 4
  num_decoder_layers: 2  # 4 → 2

refinement_head:
  channels: [64, 32]     # [128, 64] → [64, 32]
```

---

## 3. 학습 설정

### 3.1 phase1.yaml

```yaml
# conf/training/phase1.yaml
# Phase 1: 기본 통신 및 고정 샘플링

num_epochs: 50
batch_size: 16
num_workers: 4

# Optimizer
optimizer:
  type: adamw
  learning_rate: 1.0e-4
  weight_decay: 0.01
  betas: [0.9, 0.999]

# Scheduler
scheduler:
  type: cosine_annealing
  T_max: ${training.num_epochs}
  eta_min: 1.0e-6

# Loss 가중치
loss:
  mse_weight: 1.0
  perceptual_weight: 0.0     # Phase 1에서는 사용 안 함
  sparsity_weight: 0.0       # Phase 1에서는 사용 안 함

# Gradient Clipping
gradient_clip_norm: 1.0

# Mixed Precision
use_amp: true

# Checkpoint
checkpoint:
  save_interval: 5           # 5 에폭마다 저장
  keep_top_k: 3              # 상위 3개 체크포인트만 유지
  metric: val_psnr           # 저장 기준 메트릭
  mode: max                  # 높을수록 좋음

# 조기 종료
early_stopping:
  patience: 10
  min_delta: 0.1             # PSNR 0.1 dB 개선 없으면 종료
```

### 3.2 phase3.yaml

```yaml
# conf/training/phase3.yaml
# Phase 3: 딥러닝 본격 학습

defaults:
  - phase1

num_epochs: 100
batch_size: 32             # 16 → 32
num_workers: 8             # 4 → 8

optimizer:
  learning_rate: 5.0e-4    # 더 높은 LR

loss:
  mse_weight: 1.0
  perceptual_weight: 0.1   # Perceptual loss 활성화
  sparsity_weight: 0.001   # Sparsity 정규화

# Data Augmentation
data_augmentation:
  enable: true
  flip_horizontal: 0.5
  brightness_range: [0.8, 1.2]
  noise_stddev: 5.0

checkpoint:
  save_interval: 2         # 더 자주 저장
```

---

## 4. 샘플링 설정

### 4.1 uniform.yaml

```yaml
# conf/sampling/uniform.yaml
# 균등 샘플링 (Phase 1)

strategy: uniform_grid
num_samples: 400           # 고정

grid:
  type: square             # "square" | "hexagonal"
  jitter: 0.0              # 격자에 무작위성 추가 (0~0.5)

# 최소 거리 제약 (사용 안 함)
enforce_min_distance: false
min_distance: 0.0
```

### 4.2 hybrid.yaml

```yaml
# conf/sampling/hybrid.yaml
# 하이브리드 샘플링 (Phase 2+)

strategy: hybrid

num_samples:
  min: 300                 # 최소 픽셀 수
  max: 800                 # 최대 픽셀 수
  target: 500              # 목표 픽셀 수 (동적 조정)

# 샘플링 비율
importance_ratio: 0.7      # 70% 중요도 기반
uniform_ratio: 0.3         # 30% 균등 분포

# 최소 거리 제약
enforce_min_distance: true
min_distance: 0.02         # 이미지의 2%

# 동적 샘플 개수 조정
adaptive_num_samples:
  enable: true
  quality_threshold: 0.85  # SSIM < 0.85 → 샘플 증가
  increase_step: 50
  decrease_step: 25
```

---

## 5. 마스크 업데이트 설정

### 5.1 fixed.yaml

```yaml
# conf/mask_update/fixed.yaml
# 고정 주기 업데이트 (Phase 1-2)

mode: fixed
interval: 5                # 5 프레임마다 UV 좌표 갱신
```

### 5.2 quality_based.yaml

```yaml
# conf/mask_update/quality_based.yaml
# 품질 기반 동적 업데이트 (Phase 2+)

mode: quality_based

# 품질 임계값 (SSIM 기준)
thresholds:
  low_quality: 0.70        # < 0.70 → 즉시 업데이트
  high_quality: 0.90       # > 0.90 → 업데이트 주기 2배로

# 업데이트 주기 범위
interval:
  min: 3                   # 최소 3 프레임
  max: 10                  # 최대 10 프레임
  default: 5               # 기본값

# 품질 계산 방식
quality_metric: ssim       # "ssim" | "psnr" | "custom"
```

### 5.3 adaptive.yaml

```yaml
# conf/mask_update/adaptive.yaml
# 적응적 업데이트 (실험용)

mode: adaptive

# EMA 기반 품질 추적
quality_ema:
  alpha: 0.9               # EMA 가중치

# 임계값 범위
thresholds:
  quality_threshold: 0.85

# 업데이트 간격
interval:
  min: 3
  max: 10

# 예측 기반 업데이트 (실험적)
predictive:
  enable: false
  lookahead_frames: 3
```

---

## 6. 서버 설정

### 6.1 development.yaml

```yaml
# conf/server/development.yaml
# 로컬 개발 환경

host: 0.0.0.0
port: 8080
reload: true               # 코드 변경 시 자동 재시작

# WebSocket
websocket:
  heartbeat_interval: 30   # 30초마다 heartbeat
  timeout: 60              # 60초 타임아웃

# CORS
cors:
  allow_origins: ["*"]
  allow_credentials: true

# Rate Limiting
rate_limit:
  enable: false            # 개발 시 비활성화

# Logging
log_level: DEBUG

# GPU 설정
device: cuda
gpu_id: 0
```

### 6.2 school_gpu.yaml

```yaml
# conf/server/school_gpu.yaml
# 학교 GPU 서버 환경

defaults:
  - development

host: 0.0.0.0
port: 8080
reload: false              # 프로덕션에서는 비활성화

log_level: INFO

# Rate Limiting
rate_limit:
  enable: true
  max_requests: 100
  window_seconds: 60

# 동시 접속 제한
max_connections: 10

# GPU 설정 (학교 서버에 맞게 조정)
device: cuda
gpu_id: ${oc.env:CUDA_VISIBLE_DEVICES,0}

# 데이터 저장 경로
data_storage:
  root: /data/sgaps
  format: hdf5
  compression: gzip
  compression_level: 6

# 모델 체크포인트
checkpoint_path: /models/sgaps/spt_best.pth
```

---

## 7. 데이터 설정

### 7.1 base.yaml

```yaml
# conf/data/base.yaml

# 데이터 경로
data_root: ${paths.data_root}
train_episodes: ${data.data_root}/train
val_episodes: ${data.data_root}/val
test_episodes: ${data.data_root}/test

# Split 비율
split_ratios:
  train: 0.7
  val: 0.15
  test: 0.15

# DataLoader
batch_size: ${training.batch_size}
num_workers: ${training.num_workers}
pin_memory: true
persistent_workers: true
prefetch_factor: 2

# 캐싱
cache:
  enable: false
  max_size_gb: 10
```

### 7.2 augmentation.yaml

```yaml
# conf/data/augmentation.yaml
# 데이터 증강 설정

defaults:
  - base

# Augmentation 확률
augmentation:
  # Geometric
  flip_horizontal: 0.5
  flip_vertical: 0.0       # 게임 영상에서는 비현실적

  # Photometric
  brightness:
    enable: true
    range: [0.8, 1.2]
  contrast:
    enable: false
  gamma:
    enable: false

  # Noise
  gaussian_noise:
    enable: true
    stddev_range: [0, 10]  # 픽셀 값 기준

  # Spatial
  random_crop:
    enable: false          # Phase 1에서는 비활성화
```

---

## 8. 로깅 설정

### 8.1 wandb.yaml

```yaml
# conf/logging/wandb.yaml

type: wandb

wandb:
  project: sgaps-v4
  entity: null             # 팀 이름 (없으면 개인 계정)
  name: ${now:%Y%m%d_%H%M%S}_${model.name}
  tags:
    - ${training}
    - ${sampling.strategy}

  # 로깅 설정
  log_interval: 10         # 10 iteration마다
  log_gradients: true
  log_model: true          # 모델 아티팩트 저장

  # 체크포인트 업로드
  save_checkpoints: true

# 추가 메트릭
custom_metrics:
  - train/loss
  - train/mse
  - train/perceptual
  - val/psnr
  - val/ssim
  - system/gpu_utilization
  - system/gpu_memory_mb
```

### 8.2 console.yaml

```yaml
# conf/logging/console.yaml
# 간단한 콘솔 로깅 (개발용)

type: console

log_level: INFO
log_format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

log_interval: 50           # 50 iteration마다

# 파일 로깅
file_logging:
  enable: true
  log_file: ${paths.output_dir}/training.log
```

---

## 9. 설정 사용 예시

### 9.1 학습 스크립트

```python
# scripts/train.py

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # 설정 출력 (디버깅용)
    print(OmegaConf.to_yaml(cfg))

    # 모델 초기화
    from sgaps.models.spt import SparsePixelTransformer
    model = SparsePixelTransformer(cfg)

    # 데이터 로더
    from sgaps.data.dataset import create_dataloaders
    train_loader, val_loader = create_dataloaders(cfg)

    # Trainer
    from sgaps.training.trainer import SGAPSTrainer
    trainer = SGAPSTrainer(model, train_loader, val_loader, cfg)

    # 학습 시작
    for epoch in range(cfg.training.num_epochs):
        train_loss = trainer.train_epoch(epoch)
        val_psnr, val_ssim = trainer.validate(epoch)

        print(f"Epoch {epoch}: Loss={train_loss:.4f}, PSNR={val_psnr:.2f}, SSIM={val_ssim:.4f}")

        # 체크포인트 저장
        if (epoch + 1) % cfg.training.checkpoint.save_interval == 0:
            trainer.save_checkpoint(epoch, val_psnr)

if __name__ == "__main__":
    main()
```

### 9.2 CLI 사용법

#### 기본 실행
```bash
python scripts/train.py
```

#### 특정 설정 오버라이드
```bash
# Phase 3 학습, Large 모델, Hybrid 샘플링
python scripts/train.py \
  training=phase3 \
  model=spt_large \
  sampling=hybrid \
  mask_update=quality_based
```

#### 하이퍼파라미터 변경
```bash
# Learning rate 변경
python scripts/train.py \
  training.optimizer.learning_rate=1e-3 \
  training.num_epochs=200
```

#### 다중 실험 (Sweeps)
```bash
# Hydra Multirun으로 여러 실험 병렬 실행
python scripts/train.py -m \
  training=phase3 \
  model=spt_small,spt_base,spt_large \
  sampling=importance,hybrid
```

이 명령은 6개 실험을 생성합니다:
- spt_small + importance
- spt_small + hybrid
- spt_base + importance
- spt_base + hybrid
- spt_large + importance
- spt_large + hybrid

---

### 9.3 프로그래밍 방식 설정 변경

```python
from omegaconf import OmegaConf

# YAML에서 로드
cfg = OmegaConf.load("conf/config.yaml")

# 값 접근
print(cfg.model.architecture.embed_dim)  # 256

# 값 변경
cfg.model.architecture.embed_dim = 512

# 새 필드 추가
cfg.new_field = "new_value"

# YAML로 저장
OmegaConf.save(cfg, "modified_config.yaml")
```

---

## 10. 환경 변수 통합

### 10.1 .env 파일

```bash
# .env

# 데이터 경로
DATA_ROOT=/data/sgaps
CHECKPOINT_DIR=/models/sgaps

# GPU 설정
CUDA_VISIBLE_DEVICES=0

# Wandb
WANDB_API_KEY=your_api_key_here
WANDB_ENTITY=your_team_name

# 서버 설정
SERVER_HOST=0.0.0.0
SERVER_PORT=8080
```

### 10.2 설정에서 환경 변수 참조

```yaml
# config.yaml

paths:
  data_root: ${oc.env:DATA_ROOT,./data}
  checkpoint_dir: ${oc.env:CHECKPOINT_DIR,./checkpoints}

server:
  host: ${oc.env:SERVER_HOST,0.0.0.0}
  port: ${oc.env:SERVER_PORT,8080}

logging:
  wandb:
    api_key: ${oc.env:WANDB_API_KEY}
```

---

## 11. 설정 검증

### 11.1 Structured Configs (Type Safety)

```python
# sgaps/config/schema.py

from dataclasses import dataclass
from typing import List, Optional
from omegaconf import MISSING

@dataclass
class ModelConfig:
    name: str = "sparse_pixel_transformer"
    checkpoint_path: Optional[str] = None

    @dataclass
    class Architecture:
        embed_dim: int = 256
        num_heads: int = 8
        num_encoder_layers: int = 6
        num_decoder_layers: int = 4
        dropout: float = 0.1

    architecture: Architecture = Architecture()

@dataclass
class TrainingConfig:
    num_epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-4

@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()

# Hydra에 등록
from hydra.core.config_store import ConfigStore
cs = ConfigStore.instance()
cs.store(name="config_schema", node=Config)
```

사용:
```python
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: Config):  # Type hint로 자동 완성 지원
    print(cfg.model.architecture.embed_dim)  # IDE가 필드를 알고 있음
```

---

## 12. 베스트 프랙티스

### 1. 계층적 구성
- 공통 설정은 `base.yaml`에, 특화 설정은 상속하여 오버라이드
- `defaults` 리스트로 구성 합성

### 2. 환경별 분리
- `development.yaml`, `production.yaml` 등으로 환경 분리
- CI/CD에서 `--config-name` 플래그로 선택

### 3. 실험 재현성
- 모든 하이퍼파라미터를 설정 파일에 명시
- `seed` 고정
- Hydra가 자동으로 설정을 출력 디렉토리에 저장

### 4. 버전 관리
- 설정 파일도 Git에 커밋
- 중요한 실험은 태그로 표시

---

## 다음 문서

- ✅ CONFIGURATION.md (현재 문서)
- ⏭️ DEVELOPMENT_ROADMAP.md - 개발 로드맵
