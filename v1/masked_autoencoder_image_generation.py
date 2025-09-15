"""
Masked Autoencoder 기반 이미지 생성 프로젝트

이 프로젝트는 다음 단계를 수행한다:
1. 이미지 하나를 읽어온다
2. 이미지를 패치로 분할하고 랜덤하게 마스킹한다
3. Masked Autoencoder가 마스킹된 패치를 복원하도록 학습한다
4. 생성된 이미지의 마스킹된 패치가 원본과 같아지도록 학습한다
5. 2~4를 반복한다
6. 각 반복마다 생성 이미지를 표시하여 학습 진행을 시각적으로 확인한다
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.widgets import Button
from threadpoolctl import threadpool_limits

# 전역 시각화 상태 (단일 창을 유지하며 갱신)
_viz_state = None


# Matplotlib 한글 폰트 설정 (Windows 기본: Malgun Gothic)
def _setup_korean_font():
    try:
        import matplotlib
        from matplotlib import font_manager as fm

        preferred_fonts = [
            "Malgun Gothic",  # Windows 기본 한글 폰트
            "NanumGothic",  # 나눔고딕
            "AppleGothic",  # macOS
            "Noto Sans CJK KR",
            "Noto Sans KR",
            "Batang",
            "Gulim",
            "Dotum",
        ]
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = None
        for name in preferred_fonts:
            if name in available:
                chosen = name
                break
        if chosen:
            matplotlib.rcParams["font.family"] = chosen
            matplotlib.rcParams["font.sans-serif"] = preferred_fonts
        # 마이너스 기호가 네모로 나오지 않도록 설정
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        # 폰트 설정 실패 시에도 실행은 계속
        pass


_setup_korean_font()


def create_sample_image(size=(128, 128)):
    """테스트용 그라디언트 이미지를 생성한다"""
    h, w = size
    x, y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))

    # 그라디언트 패턴 생성
    r = np.sin(2 * np.pi * x) * 0.5 + 0.5
    g = np.sin(2 * np.pi * y) * 0.5 + 0.5
    b = np.sin(2 * np.pi * (x + y)) * 0.5 + 0.5

    img = np.stack([r, g, b], axis=-1)
    return img


def load_and_preprocess_image(image_path_or_array, target_size=(128, 128)):
    """이미지를 로딩하고 전처리한다"""
    if isinstance(image_path_or_array, str):
        img = Image.open(image_path_or_array).convert("RGB")
        img = img.resize(target_size)
        img_array = np.array(img, dtype=np.float32) / 255.0
    else:
        img_array = image_path_or_array

    return img_array


def patchify(image, patch_size=16):
    """이미지를 패치로 분할한다"""
    h, w, c = image.shape
    assert h % patch_size == 0 and w % patch_size == 0, "이미지 크기가 패치 크기로 나누어 떨어지지 않습니다"

    n_patches_h = h // patch_size
    n_patches_w = w // patch_size

    patches = image.reshape(n_patches_h, patch_size, n_patches_w, patch_size, c)
    patches = patches.transpose(0, 2, 1, 3, 4)
    patches = patches.reshape(n_patches_h * n_patches_w, patch_size, patch_size, c)

    return patches


def unpatchify(patches, n_patches_h, n_patches_w, patch_size=16):
    """패치를 다시 이미지로 조합한다"""
    num_patches, _, _, c = patches.shape
    assert num_patches == n_patches_h * n_patches_w

    patches = patches.reshape(n_patches_h, n_patches_w, patch_size, patch_size, c)
    patches = patches.transpose(0, 2, 1, 3, 4)
    image = patches.reshape(n_patches_h * patch_size, n_patches_w * patch_size, c)

    return image


def random_masking(patches, mask_ratio=0.75):
    """패치를 랜덤하게 마스킹한다"""
    num_patches = len(patches)
    num_mask = int(mask_ratio * num_patches)

    # 랜덤하게 마스킹할 패치 선택
    mask_indices = np.random.choice(num_patches, num_mask, replace=False)
    mask = np.zeros(num_patches, dtype=bool)
    mask[mask_indices] = True

    return mask, mask_indices


class MaskedAutoencoder(nn.Module):
    """기본적인 Masked Autoencoder 모델"""

    def __init__(self, patch_size=16, embed_dim=512, depth=6, num_heads=8, decoder_embed_dim=256, decoder_depth=4):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # 패치 임베딩
        self.patch_embed = nn.Linear(patch_size * patch_size * 3, embed_dim)

        # 위치 임베딩 (학습 가능)
        self.num_patches = (128 // patch_size) ** 2  # 128x128 이미지 기준
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # 마스크 토큰
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 디코더
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim))

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim,
            nhead=num_heads,
            dim_feedforward=decoder_embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)

        # 패치 복원
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * patch_size * 3)

        self.initialize_weights()

    def initialize_weights(self):
        """가중치 초기화"""
        # 위치 임베딩 초기화
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.mask_token, std=0.02)

        # 선형 레이어 초기화
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_encoder(self, x, mask):
        """인코더 순전파"""
        # 배치 차원 추가 (num_patches, embed_dim) -> (1, num_patches, embed_dim)
        x = x.unsqueeze(0)

        # 패치 임베딩
        x = self.patch_embed(x)

        # 위치 임베딩 추가
        x = x + self.pos_embed

        # 마스킹되지 않은 패치만 유지
        x = x[:, ~mask, :]  # 배치 차원 고려한 인덱싱

        # 인코더 적용
        x = self.encoder(x)

        return x

    def forward_decoder(self, x, mask):
        """디코더 순전파"""
        # 인코더 출력을 디코더 차원으로 변환
        x = self.decoder_embed(x)

        # 마스크 토큰을 디코더 차원으로 확장
        decoder_mask_token = self.decoder_embed(self.mask_token)
        mask_tokens = decoder_mask_token.repeat(1, mask.sum().item(), 1)

        # 전체 시퀀스 재구성
        full_tokens = torch.zeros(1, self.num_patches, x.shape[-1], device=x.device)
        full_tokens[:, ~mask, :] = x
        full_tokens[:, mask, :] = mask_tokens

        # 위치 임베딩 추가
        full_tokens = full_tokens + self.decoder_pos_embed

        # 디코더 적용
        x = self.decoder(full_tokens)

        # 패치 예측
        x = self.decoder_pred(x)

        return x

    def forward(self, x, mask):
        """전체 순전파"""
        latent = self.forward_encoder(x, mask)
        pred = self.forward_decoder(latent, mask)
        return pred

    def forward_loss(self, imgs, pred, mask):
        """손실 계산"""
        # imgs: (num_patches, patch_size*patch_size*3)
        # pred: (1, num_patches, patch_size*patch_size*3)
        # mask: (num_patches,)

        target = imgs.unsqueeze(0)  # (1, num_patches, patch_size*patch_size*3)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # 패치별 평균 (1, num_patches)
        loss = loss.squeeze(0)  # (num_patches,)
        loss = (loss * mask).sum() / mask.sum()  # 마스킹된 패치에 대해서만 손실 계산
        return loss


class MaskedAutoencoderImageGenerator:
    """Masked Autoencoder 기반 이미지 생성기"""

    def __init__(self, image_shape, patch_size=16, mask_ratio=0.75):
        self.image_shape = image_shape
        self.h, self.w, self.c = image_shape
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        # 패치 수 계산
        self.n_patches_h = self.h // patch_size
        self.n_patches_w = self.w // patch_size
        self.num_patches = self.n_patches_h * self.n_patches_w

        # 디바이스 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 모델 초기화
        self.model = MaskedAutoencoder(patch_size=patch_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=0.05)

        self.training_step = 0

    def train_step(self, image):
        """한 번의 훈련 스텝을 수행한다"""
        self.model.train()

        # 이미지를 패치로 분할
        patches = patchify(image, self.patch_size)
        patches_flat = patches.reshape(self.num_patches, -1)

        # 랜덤 마스킹
        mask, _ = random_masking(patches, self.mask_ratio)

        # PyTorch 텐서로 변환
        patches_tensor = torch.FloatTensor(patches_flat).to(self.device)
        mask_tensor = torch.BoolTensor(mask).to(self.device)

        # 순전파
        self.optimizer.zero_grad()
        pred = self.model(patches_tensor, mask_tensor)

        # 손실 계산
        loss = self.model.forward_loss(patches_tensor, pred.squeeze(0), mask_tensor)

        # 역전파
        loss.backward()
        self.optimizer.step()

        self.training_step += 1

        return float(loss.item()), mask

    def generate_image(self, original_image=None, mask_ratio=None):
        """이미지를 생성한다 (마스킹된 부분 복원)"""
        self.model.eval()

        if original_image is None:
            # 완전히 새로운 이미지 생성 (모든 패치 마스킹)
            patches = np.zeros((self.num_patches, self.patch_size, self.patch_size, 3))
            mask = np.ones(self.num_patches, dtype=bool)
        else:
            # 기존 이미지의 일부를 마스킹하여 복원
            patches = patchify(original_image, self.patch_size)
            if mask_ratio is None:
                mask_ratio = self.mask_ratio
            mask, _ = random_masking(patches, mask_ratio)

        patches_flat = patches.reshape(self.num_patches, -1)

        # PyTorch 텐서로 변환
        patches_tensor = torch.FloatTensor(patches_flat).to(self.device)
        mask_tensor = torch.BoolTensor(mask).to(self.device)

        with torch.no_grad():
            pred = self.model(patches_tensor, mask_tensor)
            pred = pred.squeeze(0).cpu().numpy()

        # 예측 결과를 패치 형태로 변환
        pred_patches = pred.reshape(self.num_patches, self.patch_size, self.patch_size, 3)

        # 마스킹된 부분만 예측 결과로 교체
        result_patches = patches.copy()
        result_patches[mask] = pred_patches[mask]

        # 패치를 이미지로 재조합
        generated_image = unpatchify(result_patches, self.n_patches_h, self.n_patches_w, self.patch_size)
        generated_image = np.clip(generated_image, 0, 1)

        return generated_image, mask


def visualize_training_progress(
    original_image, generated_image, mask, iteration, save_path=None
):
    """훈련 진행상황을 시각화한다 (단일 창에서 실시간 업데이트)"""
    global _viz_state

    # 초기화: 첫 호출에서만 창 생성
    if _viz_state is None:
        plt.ion()  # 인터랙티브 모드 활성화 (non-blocking)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.subplots_adjust(bottom=0.15)  # 버튼 영역 확보

        # 원본 이미지
        im_orig = axes[0, 0].imshow(original_image)
        axes[0, 0].set_title("원본 이미지")
        axes[0, 0].axis("off")

        # 마스크 시각화
        mask_vis = np.zeros_like(original_image)
        im_mask = axes[0, 1].imshow(mask_vis)
        axes[0, 1].set_title("마스크")
        axes[0, 1].axis("off")

        # 생성 이미지
        im_gen = axes[0, 2].imshow(generated_image)
        axes[0, 2].set_title(f"생성된 이미지 (반복 {iteration})")
        axes[0, 2].axis("off")

        # 차이 이미지
        diff = np.abs(original_image - generated_image)
        im_diff = axes[1, 0].imshow(diff)
        mse = float(np.mean(diff**2))
        axes[1, 0].set_title(f"차이 이미지 (MSE: {mse:.4f})")
        axes[1, 0].axis("off")

        # 마스킹된 부분만의 차이
        im_masked_diff = axes[1, 1].imshow(diff)
        axes[1, 1].set_title("마스킹된 부분 차이")
        axes[1, 1].axis("off")

        # 예측 정확도 그래프 준비
        axes[1, 2].set_title("훈련 손실")
        axes[1, 2].set_xlabel("반복")
        axes[1, 2].set_ylabel("MSE")
        loss_line, = axes[1, 2].plot([], [], 'b-')
        axes[1, 2].grid(True)

        # 버튼 UI
        ax_pause = fig.add_axes([0.30, 0.06, 0.15, 0.08])
        ax_stop = fig.add_axes([0.52, 0.06, 0.15, 0.08])
        btn_pause = Button(ax_pause, "일시정지")
        btn_stop = Button(ax_stop, "종료")

        # 상태 플래그
        state = {
            "paused": False,
            "stop_requested": False,
        }

        def on_pause(event):
            state["paused"] = not state["paused"]
            btn_pause.label.set_text("재시작" if state["paused"] else "일시정지")
            fig.canvas.draw_idle()

        def on_stop(event):
            state["stop_requested"] = True
            try:
                plt.close(fig)
            except Exception:
                pass

        btn_pause.on_clicked(on_pause)
        btn_stop.on_clicked(on_stop)

        fig.tight_layout(rect=[0, 0.12, 1, 1])

        _viz_state = {
            "fig": fig,
            "axes": axes,
            "im_orig": im_orig,
            "im_mask": im_mask,
            "im_gen": im_gen,
            "im_diff": im_diff,
            "im_masked_diff": im_masked_diff,
            "loss_line": loss_line,
            "original_image": original_image,
            "btn_pause": btn_pause,
            "btn_stop": btn_stop,
            "state": state,
            "loss_history": [],
            "iteration_history": [],
        }
    else:
        fig = _viz_state["fig"]
        axes = _viz_state["axes"]

    # 마스크 시각화 업데이트
    patch_size = 16
    n_patches_h = original_image.shape[0] // patch_size
    n_patches_w = original_image.shape[1] // patch_size

    mask_vis = np.zeros_like(original_image)
    for i, masked in enumerate(mask):
        if masked:
            patch_h = i // n_patches_w
            patch_w = i % n_patches_w
            y_start = patch_h * patch_size
            y_end = y_start + patch_size
            x_start = patch_w * patch_size
            x_end = x_start + patch_size
            mask_vis[y_start:y_end, x_start:x_end] = [1, 0, 0]  # 빨간색으로 마스킹된 부분 표시

    # 업데이트
    _viz_state["im_mask"].set_data(mask_vis)
    _viz_state["im_gen"].set_data(generated_image)

    diff = np.abs(_viz_state["original_image"] - generated_image)
    mse = float(np.mean(diff**2))
    _viz_state["im_diff"].set_data(diff)

    # 마스킹된 부분만의 차이 계산
    masked_diff = diff.copy()
    for i, masked in enumerate(mask):
        if not masked:
            patch_h = i // n_patches_w
            patch_w = i % n_patches_w
            y_start = patch_h * patch_size
            y_end = y_start + patch_size
            x_start = patch_w * patch_size
            x_end = x_start + patch_size
            masked_diff[y_start:y_end, x_start:x_end] = 0

    _viz_state["im_masked_diff"].set_data(masked_diff)

    # 손실 그래프 업데이트
    _viz_state["loss_history"].append(mse)
    _viz_state["iteration_history"].append(iteration)

    if len(_viz_state["loss_history"]) > 1:
        _viz_state["loss_line"].set_data(_viz_state["iteration_history"], _viz_state["loss_history"])
        axes[1, 2].relim()
        axes[1, 2].autoscale_view()

    # 제목 업데이트
    axes[0, 2].set_title(f"생성된 이미지 (반복 {iteration})")
    axes[1, 0].set_title(f"차이 이미지 (MSE: {mse:.4f})")
    masked_mse = float(np.mean(masked_diff**2))
    axes[1, 1].set_title(f"마스킹된 부분 차이 (MSE: {masked_mse:.4f})")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    # 렌더링 갱신 (non-blocking)
    fig.canvas.draw_idle()
    plt.pause(0.001)

    return mse


def train_masked_autoencoder(
    original_image,
    n_iterations=None,
    visualize_every=5,
    mask_ratio=0.75,
):
    """Masked Autoencoder를 훈련한다"""

    generator = MaskedAutoencoderImageGenerator(
        original_image.shape,
        patch_size=16,
        mask_ratio=mask_ratio
    )
    results = []

    if n_iterations is None or (
        isinstance(n_iterations, (int, float)) and n_iterations <= 0
    ):
        print(f"훈련 시작: 무한 반복, 마스크 비율: {mask_ratio}")
    else:
        print(
            f"훈련 시작: {int(n_iterations)}회 반복, 마스크 비율: {mask_ratio}"
        )
    print("-" * 60)

    def _check_ui_pause_stop():
        """UI 상태(일시정지/종료)를 확인하고 필요 시 대기 또는 중단 신호를 반환"""
        global _viz_state
        if _viz_state is None or "state" not in _viz_state:
            return False  # 중단 아님
        st = _viz_state["state"]
        # 일시정지 상태면 재시작/종료 버튼 입력을 기다리며 이벤트 루프 유지
        while st.get("paused") and not st.get("stop_requested"):
            try:
                plt.pause(0.05)
            except Exception:
                break
        return bool(st.get("stop_requested"))

    iteration = 0
    while True:
        # 종료 조건(유한 반복)
        if n_iterations is not None and iteration >= int(n_iterations):
            break

        # 사용자 종료/일시정지 확인 (반복 시작 시점)
        try:
            should_stop = _check_ui_pause_stop()
        except Exception:
            should_stop = False
        if should_stop:
            print("사용자 요청으로 훈련을 조기 종료합니다.")
            break

        iteration += 1

        # 1. 모델 훈련
        loss, mask = generator.train_step(original_image)

        # 2. 이미지 생성 (시각화 타이밍에만)
        if iteration % visualize_every == 1 or (
            n_iterations is not None and iteration == int(n_iterations)
        ):
            generated_image, vis_mask = generator.generate_image(original_image, mask_ratio=0.5)

            # 시각화
            mse = visualize_training_progress(
                original_image, generated_image, vis_mask, iteration
            )

            # 결과 저장
            results.append(
                {
                    "iteration": iteration,
                    "loss": loss,
                    "mse": mse,
                    "generated_image": generated_image.copy(),
                    "mask": vis_mask.copy(),
                }
            )

        print(
            f"반복 {iteration:3d}: 훈련 손실={loss:.6f}"
        )

        # UI 이벤트 처리를 위해 아주 짧게 양보
        try:
            plt.pause(0.001)
        except Exception:
            pass

        # 시각화 이후에도 일시정지/종료를 반영
        try:
            should_stop = _check_ui_pause_stop()
        except Exception:
            should_stop = False
        if should_stop:
            print("사용자 요청으로 훈련을 조기 종료합니다.")
            break

    return generator, results


def analyze_results(training_results, original_image):
    """훈련 결과를 분석한다"""

    if not training_results:
        print("\n=== 훈련 결과 분석 ===")
        print("분석할 결과가 없습니다 (훈련이 수행되지 않았습니다).")
        return

    # 성능 지표 추출
    loss_values = [r["loss"] for r in training_results]
    mse_values = [r["mse"] for r in training_results]

    print("\n=== 훈련 결과 분석 ===")
    print(f"초기 훈련 손실: {loss_values[0]:.6f}")
    print(f"최종 훈련 손실: {loss_values[-1]:.6f}")
    print(
        f"손실 개선율: {((loss_values[0] - loss_values[-1]) / loss_values[0] * 100):.1f}%"
    )
    print()
    print(f"초기 MSE: {mse_values[0]:.6f}")
    print(f"최종 MSE: {mse_values[-1]:.6f}")
    print(
        f"MSE 개선율: {((mse_values[0] - mse_values[-1]) / mse_values[0] * 100):.1f}%"
    )

    # 최종 생성 이미지 품질 분석
    final_generated = training_results[-1]["generated_image"]
    pixel_diff = np.abs(original_image - final_generated)

    print("\n=== 최종 생성 이미지 품질 ===")
    print(f"평균 픽셀 차이: {pixel_diff.mean():.4f}")
    print(f"차이 < 0.05인 픽셀: {(pixel_diff < 0.05).mean()*100:.1f}%")
    print(f"차이 < 0.10인 픽셀: {(pixel_diff < 0.10).mean()*100:.1f}%")
    print(f"차이 < 0.20인 픽셀: {(pixel_diff < 0.20).mean()*100:.1f}%")


def _choose_image_via_dialog():
    """Tk 파일 선택 대화상자를 통해 이미지 경로를 선택 (실패 시 None)"""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.update()
        path = filedialog.askopenfilename(
            title="학습할 이미지 선택",
            filetypes=[
                ("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"),
                ("All Files", "*.*"),
            ],
        )
        root.destroy()
        return path if path else None
    except Exception:
        return None


def main():
    """메인 실행 함수"""

    import argparse

    parser = argparse.ArgumentParser(description="Masked Autoencoder 기반 이미지 생성 학습")
    parser.add_argument(
        "--image", type=str, default=None, help="학습에 사용할 이미지 경로"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=128,
        help="입력 이미지를 리사이즈할 한 변 픽셀 (정사각형)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="훈련 반복 횟수 (미지정 시 무한 반복)",
    )
    parser.add_argument(
        "--visualize-every", type=int, default=5, help="시각화 주기 (반복 수)"
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=0.75,
        help="마스킹 비율 (0~1, 높을수록 더 많은 패치가 마스킹됨)",
    )
    args = parser.parse_args()

    print("=== Masked Autoencoder 기반 이미지 생성 프로젝트 ===\n")

    # PyTorch 설정
    if torch.cuda.is_available():
        print(f"CUDA 사용 가능: {torch.cuda.get_device_name()}")
    else:
        print("CPU 모드로 실행됩니다.")
    print()

    # 1. 원본 이미지 불러오기
    image_path = args.image
    if not image_path:
        print("1. 학습할 로컬 이미지를 선택하세요 (취소 시 샘플 그라디언트 사용)...")
        image_path = _choose_image_via_dialog()

    if image_path and os.path.isfile(image_path):
        try:
            print(f"1. 원본 이미지 로드: {image_path}")
            original_image = load_and_preprocess_image(
                image_path, target_size=(args.size, args.size)
            )
        except Exception as e:
            print(f"이미지 로드 실패: {e}. 샘플 그라디언트로 대체합니다.")
            original_image = create_sample_image((args.size, args.size))
    else:
        print("1. 원본 이미지 생성(샘플 그라디언트)...")
        original_image = create_sample_image((args.size, args.size))

    # 2. 모델 훈련
    print("\n2. Masked Autoencoder 훈련 시작...")
    generator, results = train_masked_autoencoder(
        original_image,
        n_iterations=args.iterations,
        visualize_every=args.visualize_every,
        mask_ratio=args.mask_ratio,
    )

    # 3. 결과 분석
    print("\n3. 결과 분석...")
    analyze_results(results, original_image)

    print("\n=== 프로젝트 완료! ===")

    # 창을 자동 종료하지 않고, 사용자가 닫을 때까지 유지
    try:
        plt.ioff()
        if plt.get_fignums():
            plt.show()
    except Exception:
        pass

    return generator, results


if __name__ == "__main__":
    generator, results = main()