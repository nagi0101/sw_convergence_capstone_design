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
│   │   ├── cleaner.py        # 데이터 클리닝
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
        # 4. Sparse Transformer Encoder (Pre-LN enabled)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=config.model.architecture.feedforward_dim,
            dropout=config.model.architecture.dropout,
            batch_first=True,
            norm_first=True  # Optimization: Pre-LN
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_encoder_layers
        )

        # 5. Skip Connection (Optional)
        if self.skip_enabled:
            self.skip_proj = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.Dropout(config.model.architecture.skip.dropout),
                nn.LayerNorm(self.embed_dim)
            )

        # 6. Cross-Attention Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=config.model.architecture.feedforward_dim,
            dropout=config.model.architecture.dropout,
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.num_decoder_layers
        )

        # 7. CNN Refinement Head (ResNet-based + Sigmoid)
        self.refine_head = nn.Sequential(
            BasicBlock(self.embed_dim, self.embed_dim),
            BasicBlock(self.embed_dim, self.embed_dim),
            nn.Conv2d(self.embed_dim, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.register_buffer('query_grid', None)

    def _generate_query_grid(self, batch_size, height, width, device):
        """전체 프레임의 각 픽셀 위치를 query로 생성 (캐싱 사용)"""
        # 해상도 변경 시 또는 초기화 시 그리드 생성
        if self.query_grid is None or self.query_grid.shape[0] != height * width:
            y_coords = torch.linspace(0, 1, height, device=device)
            x_coords = torch.linspace(0, 1, width, device=device)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            grid = torch.stack([xx.flatten(), yy.flatten()], dim=1) # [H*W, 2]
            self.query_grid = grid

        # 배치 크기만큼 확장
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

        # 1. Pixel Embedding: [B, N, embed_dim]
        pixel_embeds = self.pixel_embed(sparse_pixels)

        # 2. Positional Encoding (UV 좌표 기반)
        uv_coords = sparse_pixels[:, :, :2]
        pixel_embeds = self.pos_encoder(pixel_embeds, uv_coords)

        # 3. State Vector Encoding: [B, 1, embed_dim]
        state_embeds = self.state_encoder(state_vector, state_mask)

        # 4. State Token Integration
        # 상태 벡터를 'Global Context Token'으로 취급하여 픽셀 시퀀스 앞에 붙임
        # encoder_input: [B, N+1, embed_dim]
        encoder_input = torch.cat([state_embeds, pixel_embeds], dim=1)

        # 5. Sparse Transformer Encoder (Self-Attention)
        # 픽셀들은 서로의 정보뿐만 아니라 State Token도 함께 참조함
        encoded_full = self.encoder(encoder_input)  # [B, N+1, embed_dim]
        
        # Memory for Decoder (State + Pixels)
        memory = encoded_full

        # 6. Skip Connection (Optional)
        encoded_avg_proj = None
        if self.skip_enabled:
             # 픽셀 부분만 평균 내어 Global Context로 사용
             encoded_pixels = encoded_full[:, 1:, :] 
             encoded_avg = encoded_pixels.mean(dim=1, keepdim=True)
             encoded_avg_proj = self.skip_proj(encoded_avg)

        # 7. Query 생성: 전체 프레임의 각 픽셀 위치 [B, H*W, 2]
        query_positions = self._generate_query_grid(B, H, W, device)

        # 8. Query 임베딩 초기화 (위치 정보만)
        query_embeds = torch.zeros(B, H * W, self.embed_dim, device=device)
        query_embeds = self.pos_encoder(query_embeds, query_positions)

        # 9. Decoder: Cross-Attention
        # Query(전체 위치)가 Key/Value(Encoded Memory)를 참조
        decoded = self.decoder(
            tgt=query_embeds, 
            memory=memory
        )  # [B, H*W, embed_dim]

        # Skip Connection Injection
        if self.skip_enabled:
            decoded = decoded + self.config.model.architecture.skip.weight * encoded_avg_proj.expand(-1, decoded.shape[1], -1)

        # 10. Reshape → [B, H, W, embed_dim]
        decoded = decoded.view(B, H, W, self.embed_dim)

        # 11. Channel-first → [B, embed_dim, H, W]
        decoded = decoded.permute(0, 3, 1, 2)

        # 12. CNN Refinement
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

### 2.3 손실 함수 (Sampled Pixel L2 Loss)

초기 단계의 빠른 수렴과 개념 검증(PoC)을 위해 **샘플링된 픽셀에 대한 MSE(Mean Squared Error)**만을 사용합니다.

**수식:**

$$L_{total} = \frac{1}{N_{sampled}} \sum_{i \in P_{sampled}} || I_{pred}(x_i) - I_{gt}(x_i) ||_2^2$$

```python
# sgaps/models/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SampledPixelL2Loss(nn.Module):
    """샘플링된 픽셀 위치에서만 L2 손실 계산

    모델이 최소한 주어진 힌트(샘플)는 정확히 복원하도록 강제합니다.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target, sampled_coords):
        """
        Args:
            pred: [B, 1, H, W] 예측 프레임
            target: [B, 1, H, W] Ground Truth
            sampled_coords: [B, N, 2] 샘플링된 UV 좌표 (u, v in [0, 1])

        Returns:
            dict of losses
        """
        B, _, H, W = pred.shape
        N = sampled_coords.shape[1]

        # 1. UV 좌표를 픽셀 인덱스로 변환
        u = sampled_coords[:, :, 0]  # [B, N]
        v = sampled_coords[:, :, 1]  # [B, N]

        x = (u * (W - 1)).long().clamp(0, W - 1)  # [B, N]
        y = (v * (H - 1)).long().clamp(0, H - 1)  # [B, N]

        # 2. 샘플링된 위치에서 픽셀 값 추출
        pred_flat = pred.view(B, H * W)     # [B, H*W]
        target_flat = target.view(B, H * W) # [B, H*W]

        indices = y * W + x  # [B, N]

        pred_sampled = torch.gather(pred_flat, dim=1, index=indices)     # [B, N]
        target_sampled = torch.gather(target_flat, dim=1, index=indices) # [B, N]

        # 3. L2 Loss 계산 (샘플링된 픽셀에서만)
        l2_loss = F.mse_loss(pred_sampled, target_sampled)

        return {
            "total": l2_loss,
            "l2_sampled": l2_loss.item()
        }
```

> **Note:** 이 단순한 손실 함수는 MVP 단계에서 빠른 학습 수렴을 위해 설계되었습니다. 향후 Perceptual Loss, SSIM Loss 등을 추가하여 시각적 품질을 개선할 수 있습니다.

---

## 3. 핵심 로직

### 3.1 Frame Reconstructor

```python
# sgaps/core/reconstructor.py

import torch
import numpy as np

class FrameReconstructor:
    """희소 픽셀 + 상태 벡터 → 전체 프레임 재구성"""
    # ... (Implementation details)
```

### 3.2 Data Cleaning & Balancing

학습 데이터의 품질과 균형을 위한 클리닝 파이프라인(`sgaps/data/cleaner.py`)입니다.

1.  **Stationarity Logic (정적 구간 제거)**:
    -   상태 벡터의 속도(Velocity)를 계산하여 변화가 거의 없는 정적 프레임을 제거합니다.
    -   'Smart Thresholding'을 통해 데이터 분포에 따른 동적 임계값을 적용합니다.

2.  **Balancing Logic (K-Means Clustering)**:
    -   상태 벡터를 특징으로 하여 K-Means 클러스터링을 수행합니다.
    -   특정 클러스터에 데이터가 편중되는 것을 방지하기 위해 Downsampling을 수행합니다.

### 3.3 Checkpoint Manager
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

### 3.2 Importance Calculator (Attention Entropy)

모델이 이미 계산하는 **Decoder Cross-Attention Map의 엔트로피**를 활용하여 추가 연산 비용 없이 중요도를 판단합니다.

**핵심 아이디어:**

-   **낮은 엔트로피**: 특정 희소 픽셀에 강하게 집중 → 불확실성 낮음 (중요도 낮음)
-   **높은 엔트로피**: 여러 희소 픽셀을 두루뭉술하게 참조 → 불확실성 높음 (중요도 높음, 추가 샘플링 필요)

```python
# sgaps/core/importance.py

import torch
import numpy as np

class AttentionEntropyImportanceCalculator:
    """Decoder Cross-Attention 엔트로피 기반 Importance Map 계산"""

    def __init__(self, config):
        self.config = config
        self.epsilon = 1e-9  # log(0) 방지

    def compute(self, attention_weights, resolution):
        """
        Args:
            attention_weights: Decoder Cross-Attention 가중치
                Shape: [B, num_heads, H*W, N]
                - H*W: 전체 프레임 픽셀 수 (Query)
                - N: 샘플링된 희소 픽셀 수 (Key)
            resolution: (H, W) 출력 해상도

        Returns:
            importance_map: np.ndarray [H, W] (0~1 normalized)
        """
        H, W = resolution
        B, num_heads, L_query, L_key = attention_weights.shape

        # 1. Head 평균 (Head Averaging)
        # 여러 Head의 Attention을 평균내어 하나의 맵으로 만듬
        attn_avg = attention_weights.mean(dim=1)  # [B, H*W, N]

        # 2. 픽셀별 엔트로피 계산 (Pixel-wise Entropy)
        # 각 픽셀 위치 i에 대해, 샘플링된 픽셀들(j=1...N)에 대한
        # 확률 분포의 엔트로피를 계산
        # Importance_i = -∑_j A_avg(i,j) * log(A_avg(i,j) + ε)
        entropy = -torch.sum(
            attn_avg * torch.log(attn_avg + self.epsilon),
            dim=-1  # Key 차원에 대해 합산
        )  # [B, H*W]

        # 3. 배치 평균 및 이미지 형태로 변환
        entropy = entropy.mean(dim=0)  # [H*W]
        importance_map = entropy.view(H, W).cpu().numpy()  # [H, W]

        # 4. 정규화 [0, 1]
        importance_map = (importance_map - importance_map.min()) / (
            importance_map.max() - importance_map.min() + self.epsilon
        )

        return importance_map

    def compute_from_model(self, model, sparse_pixels, state_vector,
                           state_mask, resolution):
        """Forward pass와 함께 importance map 계산"""
        with torch.no_grad():
            # Attention weights를 반환하는 forward
            pred, attn_weights = model(
                sparse_pixels, state_vector, state_mask, resolution,
                return_attention=True
            )
        return self.compute(attn_weights, resolution), pred
```

**모델 수정 (선택적):**

```python
# SparsePixelTransformer에 attention weights 반환 기능 추가

class SparsePixelTransformer(nn.Module):
    def forward(self, sparse_pixels, state_vector, state_mask, resolution,
                return_attention=False):
        # ... (기존 로직)

        # Decoder 레이어에서 attention weights 추출
        # need_weights=True로 설정하여 attention weights 반환
        for layer in self.decoder.layers:
            decoded, attn_weights = layer(
                decoded, encoded,
                need_weights=True,
                average_attn_weights=False  # head별 weights 유지
            )

        # ... (기존 로직)

        if return_attention:
            return output, attn_weights  # 마지막 레이어의 attention
        return output
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

## 4. 모니터링 및 시각화 (WandB)

이 프로젝트는 별도의 웹 대시보드를 구축하는 대신, MLOps 플랫폼인 **Weights & Biases (WandB)**를 사용하여 학습 과정과 실시간 추론 결과를 모니터링합니다. 서버는 `wandb` 라이브러리를 통해 주요 데이터를 WandB 대시보드로 비동기적으로 전송합니다.

### 4.1 연동 방식

-   **초기화**: 서버 시작 시 (`main.py`) `wandb.init()`을 호출하여 WandB 세션을 생성합니다.
-   **로깅**: 실시간 추론 중 (`websocket.py`) 또는 학습 중 (`trainer.py`)에 `wandb.log()` API를 호출하여 데이터를 전송합니다. 이 호출은 비동기적으로 작동하여 메인 스레드의 성능에 영향을 최소화합니다.

### 4.2 주요 모니터링 항목

WandB 대시보드에서 다음과 같은 항목들을 실시간으로 확인할 수 있습니다.

-   **실시간 복원 프레임**: `wandb.Image`를 통해 매 프레임 또는 N 프레임 단위로 복원된 이미지를 시각적으로 확인합니다.
-   **중요도 맵 (Importance Map)**: Attention Entropy로부터 계산된 중요도 맵을 Heatmap 이미지로 시각화하여, 모델이 프레임의 어떤 부분에 집중하고 있는지 분석합니다.
-   **샘플링 좌표**: 적응형 샘플링에 의해 선택된 UV 좌표를 이미지 위에 점으로 표시하여 샘플링 분포를 확인합니다.
-   **성능 지표**: PSNR, SSIM 등의 품질 지표를 실시간 차트로 확인하여 성능 변화를 추적합니다.
-   **시스템 리소스**: 서버의 GPU 온도, 사용률, 메모리 사용량 등을 모니터링합니다.

```python
# websocket.py 에서의 로깅 예시
import wandb
from PIL import Image

# ... 프레임 복원 및 분석 후 ...
wandb.log({
    "Live/Reconstruction": wandb.Image(Image.fromarray(reconstructed_frame)),
    "Live/Importance Map": wandb.Image(Image.fromarray(importance_heatmap)),
    "Metrics/Live PSNR": psnr_value,
    "Session/Active Clients": manager.active_connection_count
})
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
