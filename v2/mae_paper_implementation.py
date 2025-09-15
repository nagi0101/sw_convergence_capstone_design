"""
Masked Autoencoder (MAE) - 논문 구현 버전
Paper: "Masked Autoencoders Are Scalable Vision Learners" (He et al., 2021)

주요 변경사항:
1. 논문과 동일한 모델 크기 (ViT-Base, ViT-Large 옵션)
2. 효율적인 마스킹 구현 (인코더에서 mask token 제외)
3. Normalized pixel targets 옵션
4. 논문의 하이퍼파라미터 사용
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from einops import rearrange, repeat
from matplotlib.widgets import Button
import math

# 전역 시각화 상태
_viz_state = None

# Matplotlib 한글 폰트 설정
def _setup_korean_font():
    try:
        import matplotlib
        from matplotlib import font_manager as fm
        preferred_fonts = ["Malgun Gothic", "NanumGothic", "AppleGothic"]
        available = {f.name for f in fm.fontManager.ttflist}
        for name in preferred_fonts:
            if name in available:
                matplotlib.rcParams["font.family"] = name
                break
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

_setup_korean_font()


class PatchEmbed(nn.Module):
    """이미지를 패치로 분할하고 임베딩"""
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class Attention(nn.Module):
    """Multi-head Self Attention 모듈"""
    
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP 모듈"""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Transformer 블록"""
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                              attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                      act_layer=act_layer, drop=drop)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MAEEncoder(nn.Module):
    """MAE 인코더 - 논문 구현"""
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, 
                 depth=12, num_heads=12, mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        
        self.initialize_weights()
        
    def initialize_weights(self):
        # 위치 임베딩 초기화 (sin-cos 대신 간단한 초기화)
        torch.nn.init.normal_(self.pos_embed, std=.02)
        torch.nn.init.normal_(self.cls_token, std=.02)
        
        # 패치 임베딩 초기화
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
    def random_masking(self, x, mask_ratio):
        """랜덤 마스킹 수행"""
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # 각 샘플별로 정렬하여 작은 값들을 keep
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # keep할 패치만 선택
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # 마스크 생성: 0은 keep, 1은 remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
        
    def forward(self, x, mask_ratio=0.75):
        # 패치 임베딩
        x = self.patch_embed(x)
        
        # 위치 임베딩 추가 (cls token 제외)
        x = x + self.pos_embed[:, 1:, :]
        
        # 마스킹 수행
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # cls token 추가
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Transformer 블록 적용
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        return x, mask, ids_restore


class MAEDecoder(nn.Module):
    """MAE 디코더 - 논문 구현"""
    
    def __init__(self, patch_size=16, num_patches=196, embed_dim=512, 
                 depth=8, num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.decoder_embed = nn.Linear(768, embed_dim, bias=True)  # encoder dim -> decoder dim
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.decoder_blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        
        self.decoder_norm = norm_layer(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, patch_size**2 * 3, bias=True)
        
        self.initialize_weights()
        
    def initialize_weights(self):
        torch.nn.init.normal_(self.decoder_pos_embed, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        
    def forward(self, x, ids_restore):
        # 임베딩 차원 변환
        x = self.decoder_embed(x)
        
        # mask tokens 추가
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # cls token 제거
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)  # cls token 다시 추가
        
        # 위치 임베딩 추가
        x = x + self.decoder_pos_embed
        
        # Transformer 블록 적용
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # 픽셀 예측
        x = self.decoder_pred(x)
        x = x[:, 1:, :]  # cls token 제거
        
        return x


class MaskedAutoencoder(nn.Module):
    """전체 MAE 모델 - 논문 구현"""
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=True):
        super().__init__()
        
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss
        
        # 인코더
        self.encoder = MAEEncoder(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=encoder_embed_dim, depth=encoder_depth, num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio, norm_layer=norm_layer
        )
        
        num_patches = self.encoder.patch_embed.num_patches
        
        # 디코더
        self.decoder = MAEDecoder(
            patch_size=patch_size, num_patches=num_patches,
            embed_dim=decoder_embed_dim, depth=decoder_depth, num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio, norm_layer=norm_layer
        )
        
    def patchify(self, imgs):
        """이미지를 패치로 변환"""
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x
    
    def unpatchify(self, x):
        """패치를 이미지로 변환"""
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    
    def forward_loss(self, imgs, pred, mask):
        """손실 계산"""
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], 패치당 평균 손실
        
        loss = (loss * mask).sum() / mask.sum()  # 마스킹된 패치에 대해서만 평균
        return loss
    
    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.encoder(imgs, mask_ratio)
        pred = self.decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def get_model_config(model_size='base'):
    """모델 설정 반환"""
    configs = {
        'base': {  # ViT-B/16
            'encoder_embed_dim': 768,
            'encoder_depth': 12,
            'encoder_num_heads': 12,
            'decoder_embed_dim': 512,
            'decoder_depth': 8,
            'decoder_num_heads': 16,
        },
        'large': {  # ViT-L/16
            'encoder_embed_dim': 1024,
            'encoder_depth': 24,
            'encoder_num_heads': 16,
            'decoder_embed_dim': 512,
            'decoder_depth': 8,
            'decoder_num_heads': 16,
        },
        'huge': {  # ViT-H/14
            'encoder_embed_dim': 1280,
            'encoder_depth': 32,
            'encoder_num_heads': 16,
            'decoder_embed_dim': 512,
            'decoder_depth': 8,
            'decoder_num_heads': 16,
        }
    }
    return configs[model_size]


class ImageDataset(Dataset):
    """간단한 이미지 데이터셋"""
    
    def __init__(self, image_path=None, size=224):
        self.size = size
        if image_path and os.path.exists(image_path):
            img = Image.open(image_path).convert('RGB')
            img = img.resize((size, size))
            self.image = np.array(img, dtype=np.float32) / 255.0
        else:
            # 샘플 이미지 생성
            self.image = self.create_sample_image()
    
    def create_sample_image(self):
        """그라디언트 샘플 이미지 생성"""
        h = w = self.size
        x, y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        r = np.sin(2 * np.pi * x) * 0.5 + 0.5
        g = np.sin(2 * np.pi * y) * 0.5 + 0.5
        b = np.sin(2 * np.pi * (x + y)) * 0.5 + 0.5
        return np.stack([r, g, b], axis=-1).astype(np.float32)
    
    def __len__(self):
        return 1000  # 임의의 큰 수 (무한 반복 시뮬레이션)
    
    def __getitem__(self, idx):
        # 이미지를 텐서로 변환 (CHW 형식)
        img_tensor = torch.from_numpy(self.image).permute(2, 0, 1)
        return img_tensor


def visualize_reconstruction(model, dataloader, device, iteration=0):
    """재구성 결과 시각화"""
    global _viz_state
    
    model.eval()
    with torch.no_grad():
        # 한 배치 가져오기
        imgs = next(iter(dataloader)).to(device)
        
        # 추론
        loss, pred, mask = model(imgs, mask_ratio=0.75)
        
        # 재구성
        pred = model.unpatchify(pred)
        
        # 마스킹된 이미지 생성
        masked_imgs = imgs.clone()
        mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_size**2 * 3)
        mask = model.unpatchify(mask.reshape(imgs.shape[0], -1, model.patch_size**2 * 3))
        masked_imgs[mask > 0.5] = 0.5  # 회색으로 마스킹
        
        # 첫 번째 이미지만 시각화
        original = imgs[0].cpu().permute(1, 2, 0).numpy()
        masked = masked_imgs[0].cpu().permute(1, 2, 0).numpy()
        reconstructed = pred[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
        
    # 시각화
    if _viz_state is None:
        plt.ion()
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        im_orig = axes[0].imshow(original)
        axes[0].set_title("원본 이미지")
        axes[0].axis("off")
        
        im_mask = axes[1].imshow(masked)
        axes[1].set_title("마스킹된 이미지 (75%)")
        axes[1].axis("off")
        
        im_recon = axes[2].imshow(reconstructed)
        axes[2].set_title(f"재구성 이미지 (반복 {iteration})")
        axes[2].axis("off")
        
        fig.suptitle(f"손실: {loss.item():.4f}")
        plt.tight_layout()
        
        _viz_state = {
            "fig": fig,
            "axes": axes,
            "im_orig": im_orig,
            "im_mask": im_mask,
            "im_recon": im_recon,
        }
    else:
        _viz_state["im_mask"].set_data(masked)
        _viz_state["im_recon"].set_data(reconstructed)
        _viz_state["axes"][2].set_title(f"재구성 이미지 (반복 {iteration})")
        _viz_state["fig"].suptitle(f"손실: {loss.item():.4f}")
        _viz_state["fig"].canvas.draw_idle()
        plt.pause(0.001)
    
    return loss.item()


def train_mae(image_path=None, model_size='base', num_epochs=100, 
              learning_rate=1.5e-4, batch_size=32, visualize_every=5):
    """MAE 모델 훈련"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"디바이스: {device}")
    print(f"모델 크기: {model_size}")
    
    # 데이터셋 및 데이터로더
    dataset = ImageDataset(image_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 모델 생성
    config = get_model_config(model_size)
    model = MaskedAutoencoder(
        img_size=224, patch_size=16,
        **config,
        norm_pix_loss=True
    ).to(device)
    
    # 옵티마이저 (논문 설정)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, 
                            betas=(0.9, 0.95), weight_decay=0.05)
    
    # 학습률 스케줄러 (Cosine Decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    print(f"\n훈련 시작: {num_epochs} 에폭")
    print("=" * 60)
    
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        # 한 에폭당 몇 개의 배치만 처리 (데모용)
        for i, imgs in enumerate(dataloader):
            if i >= 10:  # 10 배치만 처리
                break
                
            imgs = imgs.to(device)
            
            # 순전파
            loss, _, _ = model(imgs, mask_ratio=0.75)
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            
            # 그래디언트 클리핑 (안정성)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        # 학습률 업데이트
        scheduler.step()
        
        # 시각화
        if (epoch + 1) % visualize_every == 0:
            val_loss = visualize_reconstruction(model, dataloader, device, epoch + 1)
            print(f"에폭 [{epoch+1}/{num_epochs}] - 훈련 손실: {avg_loss:.4f}, "
                  f"검증 손실: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    print("\n훈련 완료!")
    return model, losses


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MAE 논문 구현 학습")
    parser.add_argument("--image", type=str, default=None, help="학습할 이미지 경로")
    parser.add_argument("--model-size", type=str, default="base", 
                       choices=["base", "large", "huge"], help="모델 크기")
    parser.add_argument("--epochs", type=int, default=100, help="훈련 에폭 수")
    parser.add_argument("--batch-size", type=int, default=32, help="배치 크기")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="학습률")
    parser.add_argument("--visualize-every", type=int, default=5, help="시각화 주기")
    
    args = parser.parse_args()
    
    print("=== Masked Autoencoder (MAE) - 논문 구현 ===\n")
    
    if torch.cuda.is_available():
        print(f"CUDA 사용 가능: {torch.cuda.get_device_name()}")
    else:
        print("CPU 모드로 실행됩니다.")
    
    # 모델 훈련
    model, losses = train_mae(
        image_path=args.image,
        model_size=args.model_size,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        visualize_every=args.visualize_every
    )
    
    # 결과 분석
    print("\n=== 훈련 결과 분석 ===")
    print(f"최종 손실: {losses[-1]:.4f}")
    print(f"최저 손실: {min(losses):.4f}")
    print(f"손실 감소율: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
    
    # 모델 정보
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n총 파라미터: {total_params:,}")
    print(f"학습 가능 파라미터: {trainable_params:,}")
    
    plt.ioff()
    plt.show()
    
    return model, losses


if __name__ == "__main__":
    model, losses = main()
