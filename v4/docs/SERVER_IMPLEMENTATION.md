# Server Implementation: SGAPS Backend

## 개요

학교 GPU 서버에서 실행되는 Python 기반 백엔드 시스템으로, 클라이언트로부터 희소 픽셀 데이터를 수신하여 프레임을 재구성하고, 적응적 샘플링을 위한 UV 좌표를 생성합니다.

---

## 아키텍처

```
sgaps-server/
├── main.py                      # FastAPI 엔트리포인트
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── conf/                        # Hydra 설정 파일
│   ├── config.yaml
│   ├── model/
│   │   └── spt.yaml            # Sparse Pixel Transformer
│   ├── training/
│   │   ├── phase1.yaml
│   │   ├── phase2.yaml
│   │   └── phase3.yaml
│   └── server/
│       └── base.yaml
├── sgaps/
│   ├── __init__.py
│   ├── api/                    # FastAPI 엔드포인트
│   │   ├── __init__.py
│   │   ├── websocket.py       # WebSocket 핸들러
│   │   └── rest.py            # REST API
│   ├── models/                # 딥러닝 모델
│   │   ├── __init__.py
│   │   ├── spt.py             # Sparse Pixel Transformer
│   │   ├── positional_encoding.py
│   │   └── losses.py
│   ├── core/                  # 핵심 로직
│   │   ├── __init__.py
│   │   ├── reconstructor.py  # 프레임 재구성
│   │   ├── importance.py     # Importance Map
│   │   ├── sampler.py        # UV 좌표 생성
│   │   └── mask_updater.py   # 마스크 업데이트 전략
│   ├── data/                  # 데이터 처리
│   │   ├── __init__.py
│   │   ├── dataset.py        # PyTorch Dataset
│   │   ├── storage.py        # HDF5 저장
│   │   └── transforms.py     # 데이터 증강
│   ├── training/              # 학습 파이프라인
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── evaluator.py
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py        # PSNR, SSIM
│       └── visualization.py  # 시각화 도구
└── scripts/
    ├── train.py              # 학습 스크립트
    ├── evaluate.py           # 평가 스크립트
    └── inference_bench.py    # 추론 벤치마크
```

---

## 1. FastAPI 서버

### 1.1 main.py

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
from typing import Dict
import hydra
from omegaconf import DictConfig

from sgaps.api.websocket import ConnectionManager
from sgaps.api.rest import router as rest_router
from sgaps.core.reconstructor import FrameReconstructor
from sgaps.core.importance import ImportanceCalculator
from sgaps.core.sampler import AdaptiveUVSampler
from sgaps.core.mask_updater import MaskUpdateScheduler
from sgaps.models.spt import SparsePixelTransformer
from sgaps.utils.metrics import MetricsCalculator

app = FastAPI(title="SGAPS Server", version="0.1.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REST API 라우터
app.include_router(rest_router, prefix="/api")

# 전역 객체
manager = ConnectionManager()
reconstructor: FrameReconstructor = None
importance_calc: ImportanceCalculator = None
uv_sampler: AdaptiveUVSampler = None
mask_scheduler: MaskUpdateScheduler = None


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로드 및 초기화"""
    global reconstructor, importance_calc, uv_sampler, mask_scheduler

    # Hydra 설정 로드
    config = hydra.compose(config_name="config")

    # 모델 로드
    model = SparsePixelTransformer.load_from_checkpoint(
        config.model.checkpoint_path
    )
    model.eval()
    model.cuda()

    # 컴포넌트 초기화
    reconstructor = FrameReconstructor(model, config)
    importance_calc = ImportanceCalculator(config)
    uv_sampler = AdaptiveUVSampler(config)
    mask_scheduler = MaskUpdateScheduler(config)

    print("[SGAPS] Server initialized successfully.")


@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    """클라이언트 WebSocket 연결 처리"""
    client_id = await manager.connect(websocket)
    print(f"[SGAPS] Client {client_id} connected.")

    # 세션 정보 (첫 메시지에서 초기화)
    session_info = None

    try:
        while True:
            # 클라이언트로부터 메시지 수신
            data = await websocket.receive_json()

            # 세션 시작 메시지 처리
            if data.get("type") == "session_start":
                session_info = await handle_session_start(data["payload"], client_id)

                # 서버 설정값 로드
                cfg = get_server_config()

                await websocket.send_json({
                    "type": "session_start_ack",
                    "payload": {
                        "checkpoint_key": session_info.checkpoint_key,
                        "checkpoint_loaded": session_info.checkpoint_loaded,
                        "model_version": session_info.model_version,
                        # 서버 제어 파라미터 - 클라이언트는 이 값들을 사용해야 함
                        "sample_count": cfg.sampling.default_sample_count,
                        "max_state_dim": cfg.max_state_dim,
                        "target_fps": cfg.target_fps,
                        "sentinel_value": cfg.sentinel_value,
                        "resolution": list(session_info.resolution)
                    }
                })
                continue

            # 프레임 데이터 처리
            if data.get("type") == "frame_data":
                if session_info is None:
                    await websocket.send_json({
                        "type": "error",
                        "payload": {"code": "SESSION_NOT_STARTED", "message": "Send session_start first"}
                    })
                    continue

                frame_data = parse_frame_data(data["payload"], session_info)

                # 프레임 재구성 (체크포인트 키로 모델 선택)
                reconstructed_frame = await reconstructor.reconstruct(
                    frame_data.sparse_pixels,
                    frame_data.state_vector,
                    frame_data.resolution,
                    checkpoint_key=session_info.checkpoint_key
                )

                # Importance Map 계산
                importance_map = importance_calc.compute(
                    reconstructed_frame,
                    frame_data.sparse_pixels
                )

                # 마스크 업데이트 여부 결정
                should_update = mask_scheduler.should_update(
                    frame_data.frame_id,
                    reconstruction_quality=calculate_quality(reconstructed_frame)
                )

                # UV 좌표 생성 (필요 시)
                if should_update:
                    uv_coords = uv_sampler.generate(
                        importance_map,
                        num_samples=config.sampling.num_samples
                    )

                    await websocket.send_json({
                        "type": "uv_coordinates",
                        "payload": {
                            "target_frame_id": frame_data.frame_id + 1,
                            "coordinates": uv_coords.tolist()
                        }
                    })

                # 데이터 저장 (비동기)
                asyncio.create_task(
                    save_frame_data(client_id, session_info.checkpoint_key, frame_data, reconstructed_frame)
                )

    except WebSocketDisconnect:
        manager.disconnect(client_id)
        print(f"[SGAPS] Client {client_id} disconnected.")


async def handle_session_start(payload: dict, client_id: str):
    """세션 시작 메시지 처리"""
    from dataclasses import dataclass

    @dataclass
    class SessionInfo:
        checkpoint_key: str
        max_state_dim: int
        resolution: tuple
        checkpoint_loaded: bool
        model_version: str

    checkpoint_key = payload.get("checkpoint_key", "default")
    max_state_dim = payload.get("max_state_dim", 64)
    resolution = tuple(payload.get("resolution", [640, 480]))

    # 체크포인트 로드 시도
    checkpoint_loaded = reconstructor.load_checkpoint(checkpoint_key)

    return SessionInfo(
        checkpoint_key=checkpoint_key,
        max_state_dim=max_state_dim,
        resolution=resolution,
        checkpoint_loaded=checkpoint_loaded,
        model_version="v1.0.0"
    )


def parse_frame_data(data: dict, session_info):
    """JSON → FrameData 객체로 변환 (상태 벡터 정규화 포함)"""
    from sgaps.data.dataset import FrameData
    import numpy as np

    # 상태 벡터 정규화 (가변 길이 → 고정 길이)
    SENTINEL_VALUE = -999.0
    raw_state = data.get("state_vector", [])
    state_vector = np.full(session_info.max_state_dim, SENTINEL_VALUE, dtype=np.float32)
    if len(raw_state) > 0:
        state_vector[:len(raw_state)] = raw_state

    return FrameData(
        frame_id=data["frame_id"],
        timestamp=data["timestamp"],
        resolution=tuple(data["resolution"]),
        sparse_pixels=np.array([
            (p["u"], p["v"], p["value"])
            for p in data["pixels"]
        ], dtype=np.float32),
        state_vector=state_vector
    )


def calculate_quality(frame):
    """재구성 품질 간단 계산 (실제로는 GT와 비교 필요)"""
    # 임시로 픽셀 분산으로 근사
    return float(np.std(frame))


async def save_frame_data(client_id, frame_data, reconstructed_frame):
    """프레임 데이터 저장 (HDF5)"""
    from sgaps.data.storage import DataStorage

    storage = DataStorage(f"data/client_{client_id}")
    await storage.save_frame(frame_data, reconstructed_frame)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        workers=1  # WebSocket은 단일 worker
    )
```

### 1.2 WebSocket 연결 관리

```python
# sgaps/api/websocket.py

from fastapi import WebSocket
from typing import Dict
import uuid

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket) -> str:
        await websocket.accept()
        client_id = str(uuid.uuid4())
        self.active_connections[client_id] = websocket
        return client_id

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def broadcast(self, message: dict):
        for websocket in self.active_connections.values():
            await websocket.send_json(message)
```

---

## 2. Sparse Pixel Transformer 모델

### 2.1 SPT 아키텍처

```python
# sgaps/models/spt.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .positional_encoding import ContinuousPositionalEncoding

class SparsePixelTransformer(nn.Module):
    """
    희소 픽셀 집합과 상태 벡터로부터 전체 프레임을 재구성하는 Transformer 기반 모델
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 하이퍼파라미터
        self.embed_dim = config.model.embed_dim  # 256
        self.num_heads = config.model.num_heads  # 8
        self.num_encoder_layers = config.model.num_encoder_layers  # 6
        self.num_decoder_layers = config.model.num_decoder_layers  # 4
        self.max_state_dim = config.model.max_state_dim  # 64
        self.sentinel_value = config.model.sentinel_value  # -999.0

        # 1. Pixel Embedding: (u, v, value) → embed_dim
        self.pixel_embed = nn.Linear(3, self.embed_dim)

        # 2. State Vector Encoder (고정 길이 입력, sentinel 마스킹)
        self.state_encoder = StateVectorEncoder(
            max_state_dim=self.max_state_dim,
            embed_dim=self.embed_dim,
            sentinel_value=self.sentinel_value
        )

        # 3. Positional Encoding (연속 좌표)
        self.pos_encoder = ContinuousPositionalEncoding(self.embed_dim)

        # 4. Sparse Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_encoder_layers
        )

        # 5. State-Pixel Cross-Attention (상태 벡터와 픽셀 특징 결합)
        self.state_pixel_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            batch_first=True
        )

        # 6. Cross-Attention Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.num_decoder_layers
        )

        # 7. CNN Refinement Head
        self.refine_head = nn.Sequential(
            nn.Conv2d(self.embed_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),  # Grayscale
            nn.Sigmoid()
        )

        # 학습 가능한 query 좌표 임베딩
        self.register_buffer('query_grid', None)

    def _generate_query_grid(self, batch_size, height, width, device):
        """전체 프레임의 각 픽셀 위치를 query로 생성"""
        if self.query_grid is None or self.query_grid.shape[1] != height * width:
            y_coords = torch.linspace(0, 1, height, device=device)
            x_coords = torch.linspace(0, 1, width, device=device)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
            self.query_grid = grid

        return self.query_grid.unsqueeze(0).repeat(batch_size, 1, 1)

    def forward(self, sparse_pixels, state_vector, state_mask, resolution):
        """
        Args:
            sparse_pixels: [B, N, 3] (batch, num_pixels, (u, v, value))
            state_vector: [B, max_state_dim] (고정 길이, sentinel 포함)
            state_mask: [B, max_state_dim] (0: sentinel, 1: valid)
            resolution: (height, width)

        Returns:
            reconstructed: [B, 1, H, W]
        """
        B, N, _ = sparse_pixels.shape
        H, W = resolution
        device = sparse_pixels.device

        # 1. Pixel Embedding
        pixel_embeds = self.pixel_embed(sparse_pixels)  # [B, N, embed_dim]

        # 2. Positional Encoding (UV 좌표 기반)
        uv_coords = sparse_pixels[:, :, :2]  # [B, N, 2]
        pixel_embeds = self.pos_encoder(pixel_embeds, uv_coords)

        # 3. State Vector Encoding (마스킹 적용)
        state_embeds = self.state_encoder(state_vector, state_mask)  # [B, 1, embed_dim]

        # 4. Encoder: 희소 픽셀 간 관계 학습
        encoded = self.encoder(pixel_embeds)  # [B, N, embed_dim]

        # 5. State-Pixel Cross-Attention (상태 벡터가 Query, 픽셀이 Key/Value)
        state_conditioned, _ = self.state_pixel_attention(
            query=state_embeds,
            key=encoded,
            value=encoded
        )  # [B, 1, embed_dim]

        # 상태 정보를 모든 픽셀 임베딩에 broadcast하여 결합
        encoded = encoded + state_conditioned.expand(-1, N, -1)  # [B, N, embed_dim]

        # 6. Query 생성: 전체 프레임의 각 픽셀 위치
        query_positions = self._generate_query_grid(B, H, W, device)  # [B, H*W, 2]

        # Query 임베딩 초기화 (위치 정보만)
        query_embeds = torch.zeros(B, H * W, self.embed_dim, device=device)
        query_embeds = self.pos_encoder(query_embeds, query_positions)

        # 7. Decoder: Cross-attention으로 각 위치의 값 예측
        decoded = self.decoder(query_embeds, encoded)  # [B, H*W, embed_dim]

        # 8. Reshape → [B, H, W, embed_dim]
        decoded = decoded.view(B, H, W, self.embed_dim)

        # 9. Channel-first → [B, embed_dim, H, W]
        decoded = decoded.permute(0, 3, 1, 2)

        # 10. CNN Refinement
        output = self.refine_head(decoded)  # [B, 1, H, W]

        return output

    @staticmethod
    def load_from_checkpoint(checkpoint_path):
        """체크포인트에서 모델 로드"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint['config']

        model = SparsePixelTransformer(config)
        model.load_state_dict(checkpoint['model_state_dict'])

        return model


class StateVectorEncoder(nn.Module):
    """
    고정 길이 상태 벡터를 임베딩으로 변환 (sentinel 마스킹 적용)
    - 모델은 상태 벡터의 각 요소가 무엇인지 알 필요 없음
    - 게임별로 다른 의미를 가지더라도, 학습을 통해 자동으로 학습
    """

    def __init__(self, max_state_dim, embed_dim, sentinel_value=-999.0):
        super().__init__()
        self.max_state_dim = max_state_dim
        self.sentinel_value = sentinel_value

        # 상태 벡터 → 임베딩
        self.linear = nn.Linear(max_state_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, state_vector, state_mask):
        """
        Args:
            state_vector: [B, max_state_dim] (고정 길이, sentinel 포함)
            state_mask: [B, max_state_dim] (0: sentinel, 1: valid)

        Returns:
            state_embeds: [B, 1, embed_dim]
        """
        # sentinel 위치를 0으로 마스킹 (학습에 영향 없도록)
        masked_state = state_vector * state_mask

        # 임베딩 생성
        embeds = self.linear(masked_state)  # [B, embed_dim]
        embeds = self.norm(embeds)

        return embeds.unsqueeze(1)  # [B, 1, embed_dim]
```

### 2.2 Continuous Positional Encoding

```python
# sgaps/models/positional_encoding.py

import torch
import torch.nn as nn
import math

class ContinuousPositionalEncoding(nn.Module):
    """
    연속 좌표 (u, v)에 대한 Sinusoidal Positional Encoding
    """

    def __init__(self, embed_dim, max_freq=10):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_freq = max_freq

        # 주파수 밴드 생성
        num_bands = embed_dim // 4  # (u_sin, u_cos, v_sin, v_cos)
        freqs = torch.logspace(0, math.log10(max_freq), num_bands)
        self.register_buffer('freqs', freqs)

    def forward(self, embeddings, uv_coords):
        """
        Args:
            embeddings: [B, N, embed_dim]
            uv_coords: [B, N, 2] (u, v 좌표 [0, 1] 범위)

        Returns:
            embeddings + positional encoding
        """
        B, N, _ = uv_coords.shape

        # u, v 분리
        u = uv_coords[:, :, 0:1]  # [B, N, 1]
        v = uv_coords[:, :, 1:2]  # [B, N, 1]

        # 각 주파수별로 sin, cos 계산
        u_encodings = []
        v_encodings = []

        for freq in self.freqs:
            u_encodings.append(torch.sin(2 * math.pi * freq * u))
            u_encodings.append(torch.cos(2 * math.pi * freq * u))
            v_encodings.append(torch.sin(2 * math.pi * freq * v))
            v_encodings.append(torch.cos(2 * math.pi * freq * v))

        # Concatenate
        pos_encoding = torch.cat(u_encodings + v_encodings, dim=2)  # [B, N, embed_dim]

        # 원본 임베딩에 추가
        return embeddings + pos_encoding
```

### 2.3 손실 함수

```python
# sgaps/models/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class SGAPSLoss(nn.Module):
    """SGAPS 학습을 위한 복합 손실 함수"""

    def __init__(self, config):
        super().__init__()
        self.mse_weight = config.loss.mse_weight
        self.perceptual_weight = config.loss.perceptual_weight
        self.sparsity_weight = config.loss.sparsity_weight

        # VGG for perceptual loss
        if self.perceptual_weight > 0:
            vgg = torchvision.models.vgg16(pretrained=True)
            self.vgg_features = vgg.features[:16].eval()

            for param in self.vgg_features.parameters():
                param.requires_grad = False

    def forward(self, pred, target, num_pixels):
        """
        Args:
            pred: [B, 1, H, W] 예측 프레임
            target: [B, 1, H, W] Ground Truth
            num_pixels: [B] 각 샘플의 픽셀 개수

        Returns:
            dict of losses
        """
        # 1. MSE 재구성 손실
        mse_loss = F.mse_loss(pred, target)

        # 2. Perceptual 손실 (VGG features)
        perceptual_loss = 0.0
        if self.perceptual_weight > 0:
            # Grayscale → RGB (VGG 입력용)
            pred_rgb = pred.repeat(1, 3, 1, 1)
            target_rgb = target.repeat(1, 3, 1, 1)

            with torch.no_grad():
                target_features = self.vgg_features(target_rgb)

            pred_features = self.vgg_features(pred_rgb)
            perceptual_loss = F.mse_loss(pred_features, target_features)

        # 3. Sparsity 정규화 (픽셀 개수 제한)
        target_num_pixels = 500.0
        sparsity_loss = F.relu(num_pixels.float().mean() - target_num_pixels) / target_num_pixels

        # 총 손실
        total_loss = (
            self.mse_weight * mse_loss +
            self.perceptual_weight * perceptual_loss +
            self.sparsity_weight * sparsity_loss
        )

        return {
            "total": total_loss,
            "mse": mse_loss.item(),
            "perceptual": perceptual_loss.item() if isinstance(perceptual_loss, torch.Tensor) else 0.0,
            "sparsity": sparsity_loss.item()
        }
```

---

## 3. 핵심 로직

### 3.1 Frame Reconstructor

```python
# sgaps/core/reconstructor.py

import torch
import numpy as np

class FrameReconstructor:
    """희소 픽셀 + 상태 벡터 → 전체 프레임 재구성"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_state_dim = config.model.max_state_dim
        self.sentinel_value = config.model.sentinel_value

        # 체크포인트 키 기반 모델 관리
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            config=config
        )

    def load_checkpoint(self, checkpoint_key: str) -> bool:
        """체크포인트 키에 해당하는 모델 로드"""
        return self.checkpoint_manager.load_model(checkpoint_key)

    async def reconstruct(self, sparse_pixels, state_vector, resolution, checkpoint_key="default"):
        """
        Args:
            sparse_pixels: np.ndarray [N, 3] (u, v, value)
            state_vector: np.ndarray [max_state_dim] (고정 길이, sentinel 포함)
            resolution: (height, width)
            checkpoint_key: str - 모델 식별자

        Returns:
            reconstructed_frame: np.ndarray [H, W] (grayscale)
        """
        # 모델 가져오기
        model = self.checkpoint_manager.get_model(checkpoint_key)

        # NumPy → Torch Tensor
        sparse_pixels_tensor = torch.from_numpy(sparse_pixels).float()
        sparse_pixels_tensor = sparse_pixels_tensor.unsqueeze(0)  # [1, N, 3]
        sparse_pixels_tensor = sparse_pixels_tensor.to(self.device)

        # 상태 벡터 텐서 변환
        state_tensor = torch.from_numpy(state_vector).float()
        state_tensor = state_tensor.unsqueeze(0)  # [1, max_state_dim]
        state_tensor = state_tensor.to(self.device)

        # 상태 마스크 생성
        state_mask = (state_tensor != self.sentinel_value).float()

        # 추론
        with torch.no_grad():
            with torch.cuda.amp.autocast():  # Mixed Precision
                output = model(sparse_pixels_tensor, state_tensor, state_mask, resolution)

        # Tensor → NumPy
        reconstructed = output.squeeze(0).squeeze(0).cpu().numpy()  # [H, W]
        reconstructed = (reconstructed * 255).astype(np.uint8)

        return reconstructed


class CheckpointManager:
    """체크포인트 키 기반 모델 관리"""

    def __init__(self, checkpoint_dir: str, config):
        from pathlib import Path

        self.checkpoint_dir = Path(checkpoint_dir)
        self.config = config
        self.loaded_models = {}  # checkpoint_key → model
        self.default_key = "default"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, checkpoint_key: str) -> bool:
        """체크포인트 키에 해당하는 모델 로드 (성공 여부 반환)"""
        try:
            _ = self.get_model(checkpoint_key)
            return True
        except Exception as e:
            print(f"[SGAPS] Failed to load checkpoint '{checkpoint_key}': {e}")
            return False

    def get_model(self, checkpoint_key: str) -> nn.Module:
        """체크포인트 키에 해당하는 모델 반환 (lazy loading)"""
        if checkpoint_key not in self.loaded_models:
            model_path = self.checkpoint_dir / checkpoint_key / "best.pth"

            if not model_path.exists():
                # 해당 키의 체크포인트가 없으면 default 사용
                print(f"[SGAPS] Checkpoint '{checkpoint_key}' not found, using default")
                checkpoint_key = self.default_key
                model_path = self.checkpoint_dir / self.default_key / "best.pth"

            model = self._load_model(model_path)
            self.loaded_models[checkpoint_key] = model
            print(f"[SGAPS] Loaded checkpoint: {checkpoint_key}")

        return self.loaded_models[checkpoint_key]

    def _load_model(self, model_path) -> nn.Module:
        """checkpoint 파일에서 모델 로드"""
        model = SparsePixelTransformer.load_from_checkpoint(model_path)
        model.eval()
        model.to(self.device)
        # TorchScript 컴파일 (속도 향상)
        # model = torch.jit.script(model)
        return model

    def register_new_checkpoint(self, checkpoint_key: str, model: nn.Module):
        """새로운 체크포인트 저장"""
        save_dir = self.checkpoint_dir / checkpoint_key
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': self.config
        }, save_dir / "best.pth")
        self.loaded_models[checkpoint_key] = model
        print(f"[SGAPS] Registered new checkpoint: {checkpoint_key}")
```

### 3.2 Importance Calculator

```python
# sgaps/core/importance.py

import numpy as np
import cv2

class ImportanceCalculator:
    """재구성 품질 기반으로 Importance Map 계산"""

    def __init__(self, config):
        self.config = config
        self.previous_frame = None

    def compute(self, reconstructed_frame, sparse_pixels):
        """
        Args:
            reconstructed_frame: np.ndarray [H, W]
            sparse_pixels: np.ndarray [N, 3] (u, v, value)

        Returns:
            importance_map: np.ndarray [H, W] (0~1 normalized)
        """
        H, W = reconstructed_frame.shape

        # 1. Ground Truth 근사 (샘플 픽셀로부터)
        gt_approx = self._approximate_gt(reconstructed_frame, sparse_pixels, (H, W))

        # 2. 재구성 오류 맵
        error_map = np.abs(reconstructed_frame - gt_approx).astype(np.float32)

        # 3. 엣지 검출 (Sobel)
        edges = cv2.Sobel(reconstructed_frame, cv2.CV_64F, 1, 1, ksize=3)
        edges = np.abs(edges).astype(np.float32)

        # 4. 시간적 변화 (모션)
        if self.previous_frame is not None:
            temporal_diff = np.abs(reconstructed_frame - self.previous_frame).astype(np.float32)
        else:
            temporal_diff = np.zeros_like(reconstructed_frame, dtype=np.float32)

        self.previous_frame = reconstructed_frame.copy()

        # 5. 가중 합산
        importance_map = (
            0.5 * error_map +
            0.3 * edges +
            0.2 * temporal_diff
        )

        # 6. 정규화 [0, 1]
        importance_map = (importance_map - importance_map.min()) / (importance_map.max() - importance_map.min() + 1e-8)

        return importance_map

    def _approximate_gt(self, reconstructed_frame, sparse_pixels, resolution):
        """샘플 픽셀 위치는 GT 값으로 채우고 나머지는 재구성 값 사용"""
        H, W = resolution
        gt_approx = reconstructed_frame.copy()

        for pixel in sparse_pixels:
            u, v, value = pixel
            x = int(u * W)
            y = int(v * H)
            if 0 <= x < W and 0 <= y < H:
                gt_approx[y, x] = value

        return gt_approx
```

### 3.3 Adaptive UV Sampler

```python
# sgaps/core/sampler.py

import numpy as np
from scipy.spatial.distance import cdist

class AdaptiveUVSampler:
    """Importance Map 기반으로 UV 좌표 생성"""

    def __init__(self, config):
        self.config = config
        self.strategy = config.sampling.strategy  # "importance" | "hybrid"
        self.min_distance = config.sampling.min_distance  # 0.02

    def generate(self, importance_map, num_samples):
        """
        Args:
            importance_map: np.ndarray [H, W]
            num_samples: int

        Returns:
            uv_coords: np.ndarray [num_samples, 2] (u, v)
        """
        H, W = importance_map.shape

        if self.strategy == "importance_only":
            uv_coords = self._importance_sampling(importance_map, num_samples)

        elif self.strategy == "hybrid":
            # 70% 중요도, 30% 균등
            num_important = int(num_samples * 0.7)
            num_uniform = num_samples - num_important

            uv_important = self._importance_sampling(importance_map, num_important)
            uv_uniform = self._uniform_sampling(num_uniform)

            uv_coords = np.vstack([uv_important, uv_uniform])

        # 최소 거리 제약
        uv_coords = self._enforce_min_distance(uv_coords, self.min_distance)

        return uv_coords

    def _importance_sampling(self, importance_map, num_samples):
        """중요도 기반 가중 샘플링"""
        H, W = importance_map.shape

        # 확률 분포로 변환
        prob_map = importance_map.flatten()
        prob_map = prob_map / prob_map.sum()

        # 가중 랜덤 샘플링
        indices = np.random.choice(
            H * W,
            size=num_samples,
            replace=False,
            p=prob_map
        )

        # 인덱스 → (u, v) 좌표
        y_coords = indices // W
        x_coords = indices % W

        u = (x_coords + 0.5) / W
        v = (y_coords + 0.5) / H

        uv_coords = np.stack([u, v], axis=1)
        return uv_coords

    def _uniform_sampling(self, num_samples):
        """균등 랜덤 샘플링"""
        u = np.random.rand(num_samples)
        v = np.random.rand(num_samples)
        return np.stack([u, v], axis=1)

    def _enforce_min_distance(self, uv_coords, min_distance):
        """샘플 간 최소 거리 제약"""
        # Greedy 방식: 거리가 너무 가까운 샘플 제거
        filtered = [uv_coords[0]]

        for i in range(1, len(uv_coords)):
            distances = cdist([uv_coords[i]], filtered)[0]
            if np.all(distances >= min_distance):
                filtered.append(uv_coords[i])

        return np.array(filtered)
```

### 3.4 Mask Update Scheduler

```python
# sgaps/core/mask_updater.py

class MaskUpdateScheduler:
    """마스크 업데이트 주기 결정"""

    def __init__(self, config):
        self.config = config
        self.mode = config.mask_update.mode  # "fixed" | "adaptive" | "quality_based"

        if self.mode == "fixed":
            self.interval = config.mask_update.fixed_interval

        elif self.mode == "adaptive":
            self.min_interval = config.mask_update.adaptive.min_interval
            self.max_interval = config.mask_update.adaptive.max_interval
            self.quality_threshold = config.mask_update.adaptive.quality_threshold

        elif self.mode == "quality_based":
            self.low_quality = config.mask_update.quality_based.low_quality_threshold
            self.high_quality = config.mask_update.quality_based.high_quality_threshold

    def should_update(self, frame_idx, reconstruction_quality):
        """
        Args:
            frame_idx: int
            reconstruction_quality: float (예: SSIM 0~1)

        Returns:
            bool
        """
        if self.mode == "fixed":
            return (frame_idx % self.interval) == 0

        elif self.mode == "adaptive":
            if reconstruction_quality < self.quality_threshold:
                return True
            else:
                return (frame_idx % self.min_interval) == 0

        elif self.mode == "quality_based":
            if reconstruction_quality < self.low_quality:
                return True  # 즉시 업데이트
            elif reconstruction_quality > self.high_quality:
                return (frame_idx % 10) == 0  # 느리게
            else:
                return (frame_idx % 5) == 0  # 보통
```

---

## 4. 학습 파이프라인

### 4.1 Dataset

```python
# sgaps/data/dataset.py

import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class SGAPSDataset(Dataset):
    """SGAPS 학습용 데이터셋"""

    def __init__(self, episode_paths, config, transforms=None):
        self.episode_paths = episode_paths
        self.config = config
        self.transforms = transforms
        self.max_state_dim = config.model.max_state_dim
        self.sentinel_value = config.model.sentinel_value

        # 모든 에피소드의 프레임 인덱스 생성
        self.frame_indices = []
        for ep_path in episode_paths:
            with h5py.File(ep_path, 'r') as f:
                num_frames = len(f['frames'].keys())
                for i in range(num_frames):
                    self.frame_indices.append((ep_path, i))

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        ep_path, frame_idx = self.frame_indices[idx]

        with h5py.File(ep_path, 'r') as f:
            frame_group = f['frames'][f'frame_{frame_idx:04d}']

            sparse_pixels = frame_group['pixels'][:]  # [N, 3]
            gt_frame = frame_group['ground_truth'][:]  # [H, W]
            resolution = tuple(frame_group.attrs['resolution'])

            # 상태 벡터 로드 (가변 길이 → 고정 길이 패딩)
            raw_state = frame_group['state_vector'][:] if 'state_vector' in frame_group else []
            state_vector = np.full(self.max_state_dim, self.sentinel_value, dtype=np.float32)
            if len(raw_state) > 0:
                state_vector[:len(raw_state)] = raw_state

        # 상태 마스크 생성 (sentinel이 아닌 위치 = 1)
        state_mask = (state_vector != self.sentinel_value).astype(np.float32)

        # 데이터 증강
        if self.transforms:
            sparse_pixels, gt_frame = self.transforms(sparse_pixels, gt_frame)

        return {
            "sparse_pixels": torch.from_numpy(sparse_pixels).float(),
            "gt_frame": torch.from_numpy(gt_frame).float().unsqueeze(0),  # [1, H, W]
            "state_vector": torch.from_numpy(state_vector).float(),
            "state_mask": torch.from_numpy(state_mask).float(),
            "num_pixels": len(sparse_pixels),
            "resolution": resolution
        }
```

### 4.2 Trainer

```python
# sgaps/training/trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

class SGAPSTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
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

        # Loss
        from sgaps.models.losses import SGAPSLoss
        self.criterion = SGAPSLoss(config)

        # Mixed Precision
        self.scaler = torch.cuda.amp.GradScaler()

        self.device = torch.device("cuda")
        self.model.to(self.device)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        progress = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch in progress:
            sparse_pixels = batch["sparse_pixels"].to(self.device)
            gt_frame = batch["gt_frame"].to(self.device)
            state_vector = batch["state_vector"].to(self.device)
            state_mask = batch["state_mask"].to(self.device)
            num_pixels = batch["num_pixels"]
            resolution = batch["resolution"][0]

            # Forward (상태 벡터 포함)
            with torch.cuda.amp.autocast():
                pred = self.model(sparse_pixels, state_vector, state_mask, resolution)
                losses = self.criterion(pred, gt_frame, num_pixels)

            # Backward
            self.optimizer.zero_grad()
            self.scaler.scale(losses["total"]).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += losses["total"].item()

            # Wandb 로깅
            wandb.log({
                "train/loss": losses["total"].item(),
                "train/mse": losses["mse"],
                "train/perceptual": losses["perceptual"],
                "train/lr": self.optimizer.param_groups[0]["lr"]
            })

            progress.set_postfix(loss=losses["total"].item())

        return total_loss / len(self.train_loader)

    def validate(self, epoch):
        self.model.eval()
        total_psnr = 0
        total_ssim = 0

        from sgaps.utils.metrics import calculate_psnr, calculate_ssim

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                sparse_pixels = batch["sparse_pixels"].to(self.device)
                gt_frame = batch["gt_frame"].to(self.device)
                resolution = batch["resolution"][0]

                pred = self.model(sparse_pixels, resolution)

                psnr = calculate_psnr(pred, gt_frame)
                ssim = calculate_ssim(pred, gt_frame)

                total_psnr += psnr
                total_ssim += ssim

        avg_psnr = total_psnr / len(self.val_loader)
        avg_ssim = total_ssim / len(self.val_loader)

        wandb.log({
            "val/psnr": avg_psnr,
            "val/ssim": avg_ssim
        })

        return avg_psnr, avg_ssim

    def save_checkpoint(self, epoch, psnr):
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "psnr": psnr,
            "config": self.config
        }, f"checkpoints/spt_epoch{epoch}_psnr{psnr:.2f}.pth")
```

---

## 5. 배포

### 5.1 Dockerfile

```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

WORKDIR /app

# 시스템 의존성
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# 포트 노출
EXPOSE 8080

# 실행
CMD ["python", "main.py"]
```

### 5.2 docker-compose.yml

```yaml
version: "3.8"

services:
    sgaps-server:
        build: .
        ports:
            - "8080:8080"
        volumes:
            - ./data:/app/data
            - ./checkpoints:/app/checkpoints
        environment:
            - CUDA_VISIBLE_DEVICES=0
        deploy:
            resources:
                reservations:
                    devices:
                        - driver: nvidia
                          count: 1
                          capabilities: [gpu]
```

---

## 다음 문서

-   ✅ SERVER_IMPLEMENTATION.md (현재 문서)
-   ⏭️ API_SPECIFICATION.md - 통신 프로토콜 명세
-   ⏭️ CONFIGURATION.md - 설정 시스템
-   ⏭️ DEVELOPMENT_ROADMAP.md - 개발 로드맵
