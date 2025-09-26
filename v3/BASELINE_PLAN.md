# v3 Baseline VideoMAE Implementation Plan

## 프로젝트 목표
v4 HST-MAE와의 성능 비교를 위한 표준 VideoMAE 베이스라인 구현 (공식 구현 활용)

## 구현 전략
1. **공식 구현 활용**: MCG-NJU의 공식 VideoMAE 구현 사용
2. **두 가지 접근법 제공**:
   - Option A: Hugging Face Transformers 활용 (빠른 구현)
   - Option B: 원본 GitHub 구현 활용 (논문 충실)
3. **게임 데이터 적응**: SMB 데이터셋에 맞게 최소 수정

## 1. 기존 구현 활용 방안

### 1.1 공식 리소스
- **논문**: "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training" (NeurIPS 2022 Spotlight)
- **GitHub**: https://github.com/MCG-NJU/VideoMAE
- **Hugging Face**: https://huggingface.co/MCG-NJU/videomae-base
- **라이센스**: CC-BY-NC 4.0 (학술 연구용 적합)

### 1.2 VideoMAE 아키텍처 (논문 기준)
```
Input: Video clips (16×224×224×3)
├── Tubelet Embedding (2×16×16 3D patches)
├── Tube Masking (90-95% masking ratio)
├── ViT Encoder
│   ├── 12 layers (base) / 24 layers (large)
│   ├── 768 dim (base) / 1024 dim (large)
│   ├── 12 heads (base) / 16 heads (large)
│   └── MLP ratio: 4
├── Lightweight Decoder
│   ├── 4 layers
│   ├── 384 dim (base) / 512 dim (large)
│   └── 6 heads (base) / 8 heads (large)
└── Reconstruction Target: Normalized pixels
```

### 1.3 v4 HST-MAE와의 주요 차이점
| 구성요소 | VideoMAE (v3 Baseline) | HST-MAE (v4) |
|---------|------------------------|--------------|
| 아키텍처 | 단일 스케일 ViT | 계층적 Swin Transformer |
| 마스킹 | Tube masking (90-95%) | 이중 마스킹 + 지역 인식 |
| 입력 | 비디오 프레임만 | 비디오 + 상태 벡터 |
| 어텐션 | 전역 self-attention | Shifted window attention |
| 압축 | 없음 | RVQ 코드북 |
| 학습 전략 | 고정 마스킹 | 3단계 커리큘럼 |

## 2. Option A: Hugging Face Transformers 활용 (권장)

### 2.1 설치
```bash
pip install transformers>=4.30.0
pip install torch torchvision
pip install pytorchvideo av
pip install pillow numpy
```

### 2.2 구현 계획
```python
# v3/baseline_hf.py
import torch
from transformers import (
    VideoMAEForPreTraining,
    VideoMAEImageProcessor,
    VideoMAEConfig
)

# 1. 사전학습된 모델 로드 및 게임 데이터 적응
model = VideoMAEForPreTraining.from_pretrained(
    "MCG-NJU/videomae-base",
    ignore_mismatched_sizes=True
)

# 2. SMB 데이터셋용 커스텀 설정
config = VideoMAEConfig(
    image_size=224,
    patch_size=16,
    num_channels=3,
    num_frames=16,
    tubelet_size=2,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    decoder_num_hidden_layers=4,
    decoder_hidden_size=384,
    decoder_num_attention_heads=6,
    mask_ratio=0.9  # 90% masking
)

# 3. 데이터 로더 구현
class SMBVideoDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, processor):
        self.data_path = data_path
        self.processor = processor
        # Load SMB frames

    def __getitem__(self, idx):
        # Sample 16 consecutive frames
        # Apply processor
        # Return masked inputs

# 4. 학습 루프
def train_videomae():
    # MSE loss for reconstruction
    # AdamW optimizer
    # Cosine LR schedule
    # Train for 800 epochs
```

### 2.3 주요 장점
- **빠른 구현**: 이미 검증된 코드 활용
- **안정성**: Hugging Face의 유지보수
- **호환성**: 다른 Transformers 모델과 쉽게 통합
- **사전학습 가중치**: Kinetics-400 등에서 학습된 가중치 활용 가능

## 3. Option B: 원본 GitHub 구현 활용

### 3.1 설치
```bash
# Clone official repository
git clone https://github.com/MCG-NJU/VideoMAE.git
cd VideoMAE

# Install dependencies
pip install -r requirements.txt
pip install timm==0.4.12
pip install decord
```

### 3.2 구현 계획
```bash
# 1. 데이터 준비
python prepare_smb_dataset.py \
    --input_dir ../smbdataset \
    --output_dir data/smb \
    --num_frames 16

# 2. Pre-training
python run_mae_pretraining.py \
    --data_path data/smb \
    --mask_type tube \
    --mask_ratio 0.9 \
    --model pretrain_videomae_base_patch16_224 \
    --batch_size 32 \
    --num_epochs 800 \
    --save_ckpt_freq 100

# 3. Fine-tuning (if needed)
python run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_path data/smb \
    --finetune checkpoints/checkpoint.pth \
    --batch_size 16 \
    --num_epochs 100
```

### 3.3 필요한 수정사항
```python
# dataset.py - SMB 데이터 로더 추가
class SMBDataset(Dataset):
    def __init__(self, root, num_frames=16):
        # Load SMB game frames
        # Handle 256x240 → 224x224 resizing

# masking_generator.py - 게임용 마스킹 전략
class GameTubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio=0.9):
        # Tube masking for game sequences
```

## 4. 평가 계획

### 4.1 필수 평가 지표
```python
# evaluation/metrics.py
def evaluate_baseline():
    metrics = {
        'PSNR': calculate_psnr(),      # ~33-35 dB expected
        'SSIM': calculate_ssim(),      # ~0.85 expected
        'MSE': calculate_mse(),        # Reconstruction error
        'Inference_FPS': measure_fps(), # ~120 FPS expected
        'Memory_GB': measure_memory()   # ~6GB expected
    }
    return metrics
```

### 4.2 v4와의 비교 분석
```python
# comparison/analyze.py
def compare_with_v4():
    # 1. 동일한 테스트 세트 사용
    # 2. 동일한 마스킹 패턴 적용
    # 3. 정량적 지표 비교
    # 4. 시각적 결과 비교
    # 5. 실패 케이스 분석
```

## 5. 구현 일정 (1주)

### Day 1-2: 환경 설정 및 모델 준비
- [ ] Hugging Face 또는 GitHub 구현 선택
- [ ] 필요한 라이브러리 설치
- [ ] 사전학습 가중치 다운로드

### Day 3-4: 데이터 파이프라인
- [ ] SMB 데이터셋 로더 구현
- [ ] 전처리 파이프라인 구축
- [ ] 데이터 검증

### Day 5-6: 학습 실행
- [ ] 학습 스크립트 실행
- [ ] 하이퍼파라미터 조정
- [ ] 체크포인트 저장

### Day 7: 평가 및 비교
- [ ] 평가 지표 측정
- [ ] v4와 비교 분석
- [ ] 결과 문서화

## 6. 예상 결과

### 6.1 성능 목표
| 지표 | VideoMAE (v3) | HST-MAE (v4) | 차이 |
|-----|--------------|--------------|------|
| PSNR | 33-35 dB | 38.5 dB | -3.5~5.5 dB |
| SSIM | 0.83-0.87 | 0.94 | -0.07~0.11 |
| FPS | 100-150 | 85 | +15~65 |
| Memory | 5-7 GB | 8 GB | -1~3 GB |

### 6.2 주요 비교 포인트
1. **재구성 품질**: 표준 VideoMAE의 한계 확인
2. **공간 일관성**: 지역 경계에서의 아티팩트
3. **시간 일관성**: 프레임 간 떨림 현상
4. **계산 효율성**: 단순 구조의 속도 이점

## 7. 코드 구조

```
v3/
├── Option_A_HuggingFace/
│   ├── baseline_hf.py           # Main implementation
│   ├── dataset.py               # SMB dataset loader
│   ├── train.py                 # Training script
│   └── evaluate.py              # Evaluation script
├── Option_B_Original/
│   ├── VideoMAE/                # Cloned repository
│   ├── smb_dataset.py           # SMB adapter
│   └── run_experiments.sh       # Experiment scripts
├── evaluation/
│   ├── metrics.py               # Evaluation metrics
│   └── compare_v4.py           # Comparison analysis
├── results/
│   ├── checkpoints/             # Model weights
│   └── visualizations/          # Result images
└── requirements.txt             # Dependencies
```

## 8. 참고 문헌

1. Tong et al. (2022). "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training" NeurIPS 2022
2. MCG-NJU/VideoMAE GitHub Repository: https://github.com/MCG-NJU/VideoMAE
3. Hugging Face VideoMAE Documentation: https://huggingface.co/docs/transformers/model_doc/videomae
4. Wang et al. (2023). "VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking" CVPR 2023

## 9. 선택 권장사항

**Hugging Face 구현 (Option A) 권장 이유:**
1. 더 빠른 개발 및 안정성
2. 사전학습 가중치 즉시 활용 가능
3. 유지보수 및 버그 수정 지원
4. v4와의 공정한 비교 가능

**원본 구현 (Option B) 선택 시:**
- 논문 재현성 100% 보장 필요 시
- 커스텀 수정이 많이 필요한 경우
- 학습 과정 세밀한 제어 필요 시