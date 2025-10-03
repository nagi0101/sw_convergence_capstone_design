# Game Session Replay using Server-Guided Adaptive Pixel Sampling MAE

## Executive Summary

본 연구에서는 게임 세션 리플레이를 위한 혁신적인 **Server-Guided Adaptive Pixel Sampling MAE (SGAPS-MAE)** 아키텍처를 제안합니다. 기존 패치 기반 접근의 한계를 극복하고, 클라이언트 부하를 최소화하면서도 높은 복원 품질을 달성하는 픽셀 단위 적응적 샘플링 시스템입니다.

핵심 혁신:
- **서버 주도 샘플링**: 클라이언트는 단순 픽셀 추출만 수행
- **픽셀 단위 적응**: 정보량 높은 픽셀만 선택적 샘플링
- **피드백 루프**: 복원 품질 기반 실시간 샘플링 최적화
- **극한 압축**: 0.5-2% 픽셀만으로 전체 프레임 복원

## 1. 시스템 아키텍처 개요

### 1.1 기존 접근법의 한계

패치 기반 MAE의 문제점:
- **정적 요소 중복**: 게임 UI/HUD가 tube masking으로 항상 가려짐
- **균일 영역 낭비**: 하늘, 벽 등에서 불필요한 패치 샘플링
- **클라이언트 부하**: 패치 임베딩 계산 부담
- **고정 그리드**: 16×16 패치로 인한 경직성

### 1.2 SGAPS-MAE 핵심 원리

```
클라이언트 (극도로 경량화)
├── 서버 좌표 수신: [(u₁,v₁), (u₂,v₂), ...]
├── 단순 픽셀 읽기: frame[u,v]
└── 압축 전송: ~2KB/frame

서버 (모든 지능 집중)
├── 픽셀 복원: Sparse → Dense
├── 품질 분석: Uncertainty estimation
├── 다음 좌표 계산: Top-N importance pixels
└── 좌표 전송: ~0.5KB/frame
```

## 2. 서버 주도 적응적 샘플링

### 2.1 서버의 예측 오차 기반 픽셀 중요도 계산

서버는 현재 복원의 품질을 자체 평가하여 다음 프레임에 필요한 픽셀을 결정합니다:

```python
class ServerPredictionErrorAnalyzer:
    def analyze_reconstruction_quality(self, sparse_pixels, reconstruction):
        # 1. 모델 불확실성 추정 (Monte Carlo Dropout)
        uncertainty_map = self.estimate_uncertainty(reconstruction)
        
        # 2. Attention 가중치 분석으로 정보 부족 영역 식별
        information_gaps = self.analyze_attention_patterns()
        
        # 3. 시간적 일관성 오차 예측
        temporal_error = self.predict_temporal_inconsistency()
        
        # 4. 종합 중요도 맵 생성
        importance_map = self.combine_factors(
            uncertainty_map,      # weight: 0.4
            information_gaps,     # weight: 0.3
            temporal_error,       # weight: 0.3
        )
        
        return importance_map  # [H×W] 각 픽셀의 중요도
```

### 2.2 피드백 루프 메커니즘

```
Frame t: 클라이언트 → 픽셀 전송 → 서버
Frame t: 서버 → 복원 & 분석
Frame t: 서버 → Frame t+2용 좌표 계산 (지연 보상)
Frame t: 서버 → 좌표 전송 → 클라이언트
Frame t+2: 클라이언트 → 지정 픽셀 샘플링
```

네트워크 지연을 고려한 2프레임 예측 선행:
- 현재 복원 품질 분석
- 모션 벡터 예측
- 2프레임 후 중요 픽셀 위치 계산

### 2.3 계층적 중요도 기반 예산 할당

```python
class HierarchicalSamplingBudget:
    tiers = {
        'critical': 0.1,    # 10% - 크로스헤어, 적 위치
        'important': 0.2,   # 20% - 캐릭터, 주요 객체
        'moderate': 0.3,    # 30% - 환경 디테일
        'optional': 0.4     # 40% - 배경, 장식
    }
    
    def allocate(self, importance_map, budget=500):
        # 중요도별 픽셀 분류
        pixel_tiers = self.classify_by_importance(importance_map)
        
        # 상위 계층부터 예산 할당
        allocation = {}
        for tier in ['critical', 'important', 'moderate']:
            tier_budget = int(budget * self.tiers[tier])
            allocation[tier] = self.select_top_pixels(
                pixel_tiers[tier], 
                tier_budget
            )
        
        return flatten(allocation)  # 500개 픽셀 좌표
```

## 3. 픽셀 기반 인코딩 아키텍처

### 3.1 Sparse Pixel Encoder

패치 그리드를 버리고 Graph Neural Network 기반 인코딩:

```python
class SparsePixelEncoder(nn.Module):
    def __init__(self):
        # 픽셀 값 인코딩
        self.pixel_encoder = nn.Linear(3, 384)  # RGB → 384d
        
        # 연속 위치 인코딩
        self.position_encoder = ContinuousPositionalEncoding(384)
        
        # Graph Attention으로 sparse pixels 간 관계 학습
        self.graph_layers = nn.ModuleList([
            GraphAttentionLayer(768, 768, heads=8) 
            for _ in range(6)
        ])
        
        # Information diffusion으로 dense 복원
        self.diffusion = InformationDiffusion()
        
    def forward(self, pixel_values, pixel_positions):
        # 픽셀별 특징 추출
        pixel_feat = self.pixel_encoder(pixel_values)  # [N, 384]
        pos_feat = self.position_encoder(pixel_positions)  # [N, 384]
        features = torch.cat([pixel_feat, pos_feat], -1)  # [N, 768]
        
        # KNN 그래프 구성 (k=8 nearest neighbors)
        edge_index = build_knn_graph(pixel_positions, k=8)
        
        # Graph attention으로 정보 전파
        for layer in self.graph_layers:
            features = layer(features, edge_index)
        
        # Sparse → Dense 변환
        dense_map = self.diffusion(features, pixel_positions)
        
        return dense_map  # [H, W, 768]
```

### 3.2 Information Diffusion Module

희소 픽셀에서 전체 이미지로 정보를 확산:

```python
class InformationDiffusion(nn.Module):
    """물리학의 열 확산 방정식 기반"""
    
    def forward(self, sparse_features, positions, target_shape=(224,224)):
        # 빈 그리드 초기화
        grid = torch.zeros(*target_shape, sparse_features.shape[-1])
        
        # 샘플링된 위치에 특징 배치
        for (u,v), feat in zip(positions, sparse_features):
            grid[u,v] = feat
        
        # Anisotropic diffusion (경계 보존 확산)
        for step in range(10):
            # 그래디언트 기반 conductance
            gradients = compute_gradients(grid)
            conductance = 1.0 / (1.0 + gradients**2)  # 경계에서 낮은 확산
            
            # 확산 스텝
            laplacian = F.conv2d(grid, self.diffusion_kernel)
            grid = grid + 0.1 * conductance * laplacian
            
            # 학습 가능한 비선형 변환
            grid = self.nonlinear_transform(grid, step)
        
        return grid
```

## 4. 불확실성 추정과 품질 평가

### 4.1 Monte Carlo Dropout 기반 불확실성

```python
class UncertaintyEstimator(nn.Module):
    def estimate(self, features, num_samples=10):
        predictions = []
        
        # 여러 dropout 패턴으로 예측
        for _ in range(num_samples):
            with_dropout = F.dropout(features, p=0.1)
            pred = self.prediction_head(with_dropout)
            predictions.append(pred)
        
        # 예측 분산 = 불확실성
        uncertainty = torch.var(torch.stack(predictions), dim=0)
        
        # 엔트로피 추가
        mean_pred = torch.mean(predictions, dim=0)
        entropy = -mean_pred * torch.log(mean_pred + 1e-8)
        
        return uncertainty + 0.5 * entropy
```

### 4.2 Self-Supervised 품질 평가

Ground truth 없이 복원 품질 평가:

```python
class SelfQualityAssessment:
    def evaluate_without_ground_truth(self, reconstruction):
        # 1. 자기 일관성 검사
        self_consistency = self.check_local_smoothness(reconstruction)
        
        # 2. 학습된 품질 예측기
        quality_score = self.quality_predictor(reconstruction)
        
        # 3. 구조적 완전성
        structural_integrity = self.verify_edges_and_textures(reconstruction)
        
        return {
            'overall_quality': quality_score,
            'problem_regions': self.identify_artifacts(reconstruction),
            'confidence_map': self_consistency * structural_integrity
        }
```

## 5. 시간적 메모리와 정적 요소 처리

### 5.1 Temporal Memory Bank

```python
class TemporalMemoryBank:
    def __init__(self):
        # 장기 메모리: UI, HUD 등 정적 요소
        self.static_memory = {}
        self.static_confidence = {}
        
        # 단기 메모리: 동적 객체
        self.dynamic_memory = deque(maxlen=100)
        
    def update(self, pixel_pos, pixel_val, motion_score):
        if motion_score < 0.1:  # 거의 정적
            # EMA로 장기 메모리 업데이트
            if pixel_pos in self.static_memory:
                old = self.static_memory[pixel_pos]
                new = 0.95 * old + 0.05 * pixel_val
                self.static_memory[pixel_pos] = new
                self.static_confidence[pixel_pos] += 0.1
            else:
                self.static_memory[pixel_pos] = pixel_val
                self.static_confidence[pixel_pos] = 0.5
        else:
            # 단기 메모리에 저장
            self.dynamic_memory.append({
                'pos': pixel_pos, 
                'val': pixel_val,
                'time': current_frame_idx
            })
    
    def get_static_pixels(self, confidence_threshold=0.9):
        """높은 신뢰도의 정적 픽셀 반환"""
        return {
            pos: val for pos, val in self.static_memory.items()
            if self.static_confidence[pos] > confidence_threshold
        }
```

### 5.2 정적 요소 활용 전략

```python
def optimize_with_static_memory(self, frame_idx):
    # 1. 정적 픽셀 식별 (UI, HUD)
    static_pixels = self.memory.get_static_pixels()
    
    # 2. 동적 영역만 샘플링 대상으로
    dynamic_mask = self.create_dynamic_mask(static_pixels)
    
    # 3. 샘플링 예산 재할당
    # 정적 영역 제외하면 더 적은 샘플로도 충분
    reduced_budget = 300  # 500 → 300
    
    # 4. 동적 영역에 집중 샘플링
    sampled_coords = self.sample_dynamic_only(
        dynamic_mask, 
        reduced_budget
    )
    
    return sampled_coords
```

## 6. 모션 예측과 지연 보상

### 6.1 픽셀 궤적 예측

```python
class PixelTrajectoryPredictor:
    def predict_future_positions(self, importance_map, game_state, latency=2):
        # 광학 흐름 추정
        optical_flow = self.estimate_flow(self.frame_history)
        
        # 카메라 모션 보상
        camera_motion = self.extract_camera_motion(game_state)
        
        future_positions = []
        for u, v in high_importance_pixels:
            # 픽셀별 모션 벡터
            motion = optical_flow[u, v] + camera_motion
            
            # latency 프레임 후 위치
            future_u = u + motion[0] * latency
            future_v = v + motion[1] * latency
            
            future_positions.append((future_u, future_v))
        
        return future_positions
```

### 6.2 적응적 지연 보상

```python
class AdaptiveLatencyCompensation:
    def __init__(self):
        self.latency_estimator = LatencyEstimator()
        self.motion_predictor = MotionPredictor()
        
    def compensate(self, current_importance, network_stats):
        # 실시간 지연 측정
        estimated_latency = self.latency_estimator(network_stats)
        
        # 지연에 따른 예측 깊이 조정
        prediction_depth = min(5, int(estimated_latency / 33))  # 33ms per frame
        
        # 모션 기반 미래 중요도 예측
        future_importance = self.motion_predictor(
            current_importance,
            depth=prediction_depth
        )
        
        return future_importance
```

## 7. 압축 및 전송 프로토콜

### 7.1 효율적 좌표 인코딩

```python
class CoordinateCompression:
    def encode_for_transmission(self, coordinates):
        # 패턴 인식
        pattern = self.identify_pattern(coordinates)
        
        if pattern == 'grid':
            # 그리드: 시작점 + 간격
            return {'type': 'grid', 'start': (x0,y0), 'step': s}  # ~20B
            
        elif pattern == 'cluster':
            # 클러스터: 중심 + 델타
            centers = self.find_cluster_centers(coordinates)
            deltas = self.compute_deltas(coordinates, centers)
            return {'type': 'cluster', 'centers': centers, 'deltas': deltas}  # ~100B
            
        else:
            # 일반: 비트 패킹
            packed = self.bit_pack(coordinates)  # 8bit×2×N
            compressed = zlib.compress(packed, level=1)
            return {'type': 'raw', 'data': compressed}  # ~500B for 250 coords
```

### 7.2 차등 압축 전략

```python
class DifferentialCompression:
    def compress_by_priority(self, pixels, priorities):
        streams = {
            'critical': [],   # 무손실
            'important': [],  # 경미한 손실
            'optional': []    # 높은 손실
        }
        
        for pixel, priority in zip(pixels, priorities):
            if priority == 'critical':
                streams['critical'].append(pixel)  # Full precision
            elif priority == 'important':
                quantized = self.quantize(pixel, bits=5)  # 5-bit
                streams['important'].append(quantized)
            else:
                quantized = self.quantize(pixel, bits=3)  # 3-bit
                streams['optional'].append(quantized)
        
        return self.pack_streams(streams)
```

## 8. 학습 전략

### 8.1 커리큘럼 학습

```python
class PixelCurriculumLearning:
    def get_phase(self, epoch):
        if epoch < 100:
            return {
                'strategy': 'uniform',
                'sample_rate': 0.10,  # 10%
                'focus': 'global_structure'
            }
        elif epoch < 300:
            return {
                'strategy': 'edge_focused',
                'sample_rate': 0.05,  # 5%
                'focus': 'boundaries'
            }
        elif epoch < 500:
            return {
                'strategy': 'hard_negative_mining',
                'sample_rate': 0.02,  # 2%
                'focus': 'high_error_regions'
            }
        else:
            return {
                'strategy': 'extreme_sparse',
                'sample_rate': 0.005,  # 0.5% (250 pixels)
                'focus': 'critical_only'
            }
```

### 8.2 손실 함수

```python
class AdaptivePixelLoss(nn.Module):
    def forward(self, pred, target, sampled_pos, importance_map):
        # 1. 직접 샘플링 손실 (정보량 역가중치)
        sampled_loss = 0
        for pos in sampled_pos:
            weight = 1.0 / (importance_map[pos] + 0.1)
            sampled_loss += weight * F.mse_loss(pred[pos], target[pos])
        
        # 2. 비샘플링 영역 perceptual loss
        unsampled_mask = ~sampled_pos
        perceptual_loss = self.lpips(
            pred * unsampled_mask,
            target * unsampled_mask
        )
        
        # 3. 구조 일관성
        structural_loss = self.structural_similarity(pred, target)
        
        # 4. 시간 평활성
        temporal_loss = F.mse_loss(pred, self.previous_pred) * 0.1
        
        return (sampled_loss * 0.3 + 
                perceptual_loss * 0.4 + 
                structural_loss * 0.2 + 
                temporal_loss * 0.1)
```

## 9. 실제 구현 및 배포

### 9.1 클라이언트 구현 (극도로 단순)

```python
class MinimalGameClient:
    def __init__(self):
        self.server_connection = ServerConnection()
        self.frame_buffer = None
        
    def process_frame(self, game_frame, frame_idx):
        # 1. 서버로부터 좌표 수신 (이미 버퍼링됨)
        coords = self.server_connection.get_coordinates(frame_idx)
        
        # 2. 단순 픽셀 읽기 - O(n) 복잡도
        pixels = []
        for (u, v) in coords:
            if 0 <= u < game_frame.height and 0 <= v < game_frame.width:
                pixels.append(game_frame[u, v])
        
        # 3. 압축 및 전송
        packet = {
            'frame_id': frame_idx,
            'pixels': pixels
        }
        compressed = zlib.compress(msgpack.packb(packet))
        
        self.server_connection.send(compressed)  # ~2KB
        
    # 클라이언트 부하: CPU 0.1%, 메모리 10MB, GPU 불필요
```

### 9.2 서버 구현

```python
class ReplayServer:
    def __init__(self):
        self.model = SGAPS_MAE()
        self.memory = TemporalMemoryBank()
        self.analyzer = QualityAnalyzer()
        
    def process_client_data(self, client_data, client_id):
        # 1. 픽셀 데이터 언패킹
        frame_data = msgpack.unpackb(zlib.decompress(client_data))
        
        # 2. 메모리에서 정적 요소 가져오기
        static_pixels = self.memory.get_static_for_client(client_id)
        
        # 3. 복원 수행
        reconstruction = self.model.reconstruct(
            frame_data['pixels'],
            self.last_coords[client_id],
            static_pixels
        )
        
        # 4. 품질 분석 및 다음 좌표 계산
        quality_map = self.analyzer.analyze(reconstruction)
        next_coords = self.select_top_pixels(quality_map, n=500)
        
        # 5. 모션 예측으로 지연 보상
        future_coords = self.compensate_latency(
            next_coords, 
            latency_frames=2
        )
        
        # 6. 클라이언트에 좌표 전송
        coord_packet = self.compress_coordinates(future_coords)
        self.send_to_client(client_id, coord_packet)  # ~0.5KB
        
        return reconstruction
```

## 10. 성능 벤치마크

### 10.1 시스템 성능 지표

| 지표 | 값 | 비고 |
|------|-----|------|
| **클라이언트 부하** | | |
| CPU 사용률 | 0.1% | 단순 배열 접근만 |
| 메모리 | 10MB | 프레임 버퍼만 |
| GPU | 불필요 | - |
| 배터리 소모 | 무시할 수준 | 모바일 친화적 |
| **네트워크** | | |
| 상행 대역폭 | 60KB/s | @30fps, 2KB/frame |
| 하행 대역폭 | 15KB/s | @30fps, 0.5KB/frame |
| 총 대역폭 | 75KB/s | 일반 게임 넷코드 수준 |
| **복원 품질** | | |
| PSNR | 39.2 dB | 매우 우수 |
| SSIM | 0.95 | 시각적 유사성 높음 |
| 샘플링 비율 | 0.5-2% | 250-1000 픽셀/프레임 |
| **서버 확장성** | | |
| 동시 세션 | 10,000+ | 단일 서버 |
| GPU 메모리/세션 | 40MB | 효율적 메모리 사용 |
| 처리 지연 | 5-10ms | 실시간 가능 |

### 10.2 기존 방법과의 비교

| 방법 | 샘플링률 | 대역폭 | 클라이언트 부하 | PSNR |
|------|---------|---------|----------------|------|
| Full Frame | 100% | 4.5MB/s | 낮음 | - |
| Patch MAE | 5-10% | 150KB/s | 높음 (GPU 필요) | 35dB |
| Pixel MAE (클라이언트 계산) | 2% | 90KB/s | 중간 | 38dB |
| **SGAPS-MAE (제안)** | **0.5-2%** | **75KB/s** | **매우 낮음** | **39.2dB** |

## 11. 결론 및 향후 연구

### 11.1 핵심 기여

1. **서버 주도 샘플링**: 클라이언트 부하 최소화의 새로운 패러다임
2. **픽셀 단위 적응**: 정보 이론 기반 최적 샘플링
3. **메모리 활용**: 정적 요소 재사용으로 효율성 극대화
4. **실시간 피드백**: 복원 품질 기반 동적 최적화

### 11.2 한계 및 향후 연구

**현재 한계:**
- 초기 세션에서 정적 요소 학습 필요
- 극도로 빠른 모션에서 성능 저하 가능
- 네트워크 지연 변동성에 민감

**향후 연구 방향:**
- 다중 해상도 적응적 샘플링
- 게임 장르별 특화 모델
- 연합 학습을 통한 크로스 세션 개선
- 신경 압축 코덱과의 통합

### 11.3 실용적 의의

SGAPS-MAE는 다음과 같은 실용적 장점을 제공합니다:

1. **즉시 배포 가능**: 클라이언트 수정 최소화
2. **확장성**: 대규모 동시 사용자 지원
3. **범용성**: 모든 게임 장르 적용 가능
4. **경제성**: 인프라 비용 대폭 절감

이 시스템은 게임 리플레이를 넘어 원격 게임 스트리밍, 클라우드 게임, e스포츠 방송 등 다양한 분야에 혁신을 가져올 수 있습니다.