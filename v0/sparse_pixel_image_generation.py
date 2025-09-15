"""
스파스 픽셀 샘플링 기반 이미지 생성 프로젝트

이 프로젝트는 다음 단계를 수행한다:
1. 이미지 하나를 읽어온다
2. 이미지의 랜덤한 UV 좌표에서 RGB 픽셀 N개를 추출한다
3. [r, g, b, u, v] * N의 벡터를 입력으로 신경망이 이미지를 생성한다
4. 생성된 이미지의 샘플링 위치 픽셀이 원본과 같아지도록 신경망을 학습한다
5. 2~4를 반복한다
6. 각 반복마다 생성 이미지를 표시하여 학습 진행을 시각적으로 확인한다
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neural_network import MLPRegressor
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


def sample_random_pixels(image, n_samples=100):
    """이미지에서 랜덤한 UV 좌표의 RGB 픽셀을 N개 추출한다"""
    h, w, c = image.shape

    # 랜덤 좌표 생성
    u_coords = np.random.randint(0, w, n_samples)  # x 좌표
    v_coords = np.random.randint(0, h, n_samples)  # y 좌표

    # UV를 0~1 범위로 정규화
    u_norm = u_coords / (w - 1)
    v_norm = v_coords / (h - 1)

    # 해당 좌표의 RGB 값 추출
    rgb_values = image[v_coords, u_coords]  # (n_samples, 3)

    # [r, g, b, u, v] 형태로 결합
    samples = np.column_stack([rgb_values, u_norm, v_norm])  # (n_samples, 5)

    return samples, (u_coords, v_coords)


class CoordinateBasedImageGenerator:
    """좌표 기반 이미지 생성기"""

    def __init__(self, image_shape):
        self.image_shape = image_shape
        self.h, self.w, self.c = image_shape

        # 좌표만을 입력으로 받는 단순한 모델 (점진적 학습 구성)
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 64, 32),  # 가벼운 모델로 변경
            activation="relu",
            solver="adam",
            learning_rate_init=0.005,
            max_iter=1,  # 매 step 당 1 epoch만 수행 (warm_start)
            warm_start=True,
            random_state=42,
        )

        # 리플레이 버퍼(누적) 설정: 너무 커지면 UI가 버벅이므로 상한을 둔다
        self.replay_buffer_maxlen = 10000
        self.batch_size = 2048
        self.X_buffer = None  # (N,2) u,v
        self.y_buffer = None  # (N,3) r,g,b

    def add_training_samples(self, sampled_pixels):
        """새로운 샘플링 데이터를 누적한다 (리플레이 버퍼로 관리)"""
        # sampled_pixels: (n_samples, 5) [r, g, b, u, v]
        coords = sampled_pixels[:, 3:5]
        colors = sampled_pixels[:, 0:3]

        if self.X_buffer is None:
            self.X_buffer = coords
            self.y_buffer = colors
        else:
            self.X_buffer = np.vstack([self.X_buffer, coords])
            self.y_buffer = np.vstack([self.y_buffer, colors])

        # 상한 유지 (최근 데이터 우선 보존)
        if len(self.X_buffer) > self.replay_buffer_maxlen:
            self.X_buffer = self.X_buffer[-self.replay_buffer_maxlen :]
            self.y_buffer = self.y_buffer[-self.replay_buffer_maxlen :]

    def train_step(self):
        """누적된 데이터로 모델을 훈련한다 (작은 미니배치, 1 epoch)"""
        if self.X_buffer is None or len(self.X_buffer) == 0:
            return

        n = len(self.X_buffer)
        if n <= self.batch_size:
            X_batch, y_batch = self.X_buffer, self.y_buffer
        else:
            idx = np.random.choice(n, size=self.batch_size, replace=False)
            X_batch = self.X_buffer[idx]
            y_batch = self.y_buffer[idx]

        # 과도한 스레딩으로 UI가 멈추지 않도록 연산 스레드 제한
        with threadpool_limits(limits=1):
            self.model.fit(X_batch, y_batch)

    def generate_image(self, scale=1.0):
        """전체 이미지를 생성한다 (옵션: 저해상도 생성으로 시각화 가속)"""
        # 해상도 스케일 적용
        h = max(1, int(self.h * scale))
        w = max(1, int(self.w * scale))

        # 전체 이미지 좌표 생성
        u_coords, v_coords = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        coords = np.stack([u_coords.flatten(), v_coords.flatten()], axis=1)

        try:
            with threadpool_limits(limits=1):
                colors = self.model.predict(coords)

            generated = colors.reshape(h, w, self.c)
            generated = np.clip(generated, 0, 1)

            # 원해상도로 업샘플 (최근접)
            if (h, w) != (self.h, self.w):
                generated = np.array(
                    Image.fromarray((generated * 255).astype(np.uint8)).resize(
                        (self.w, self.h), resample=Image.NEAREST
                    ),
                    dtype=np.float32,
                )
                generated /= 255.0
        except Exception:
            generated = np.random.rand(self.h, self.w, self.c)

        return generated

    def get_sample_count(self):
        """현재까지 누적된 샘플 수를 반환한다"""
        return 0 if self.X_buffer is None else int(len(self.X_buffer))


def visualize_training_progress(
    original_image, generated_image, sampled_coords, iteration, save_path=None
):
    """훈련 진행상황을 시각화한다 (단일 창에서 실시간 업데이트)"""
    global _viz_state

    # 초기화: 첫 호출에서만 창 생성
    if _viz_state is None:
        plt.ion()  # 인터랙티브 모드 활성화 (non-blocking)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.subplots_adjust(bottom=0.2)  # 버튼 영역 확보

        # 원본 이미지 및 샘플 좌표
        im_orig = axes[0].imshow(original_image)
        axes[0].set_title("원본 이미지")
        axes[0].axis("off")
        scat = axes[0].scatter([], [], c="red", s=10, alpha=0.7)

        # 생성 이미지
        im_gen = axes[1].imshow(generated_image)
        axes[1].set_title(f"생성된 이미지 (반복 {iteration})")
        axes[1].axis("off")

        # 차이 이미지
        diff = np.abs(original_image - generated_image)
        im_diff = axes[2].imshow(diff)
        mse = float(np.mean(diff**2))
        axes[2].set_title(f"차이 이미지 (MSE: {mse:.4f})")
        axes[2].axis("off")

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
            "im_gen": im_gen,
            "im_diff": im_diff,
            "scat": scat,
            "original_image": original_image,
            "btn_pause": btn_pause,
            "btn_stop": btn_stop,
            "state": state,
        }
    else:
        fig = _viz_state["fig"]
        axes = _viz_state["axes"]
        im_gen = _viz_state["im_gen"]
        im_diff = _viz_state["im_diff"]
        scat = _viz_state["scat"]

    # 업데이트: 샘플 좌표, 생성 이미지, 차이 이미지
    if sampled_coords is not None:
        u_coords, v_coords = sampled_coords
        # scatter 업데이트는 (N,2) 형태의 배열 필요 (x=u, y=v)
        scat.set_offsets(np.column_stack([u_coords, v_coords]))

    im_gen.set_data(generated_image)
    diff = np.abs(_viz_state["original_image"] - generated_image)
    mse = float(np.mean(diff**2))
    _viz_state["im_diff"].set_data(diff)

    axes[1].set_title(f"생성된 이미지 (반복 {iteration})")
    axes[2].set_title(f"차이 이미지 (MSE: {mse:.4f})")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    # 렌더링 갱신 (non-blocking)
    fig.canvas.draw_idle()
    plt.pause(0.001)

    return mse


def train_image_generator(
    original_image,
    n_iterations=None,
    n_samples_per_iter=50,
    visualize_every=2,
    viz_scale=0.7,
    fast_metric_sample=2048,
):
    """이미지 생성기를 훈련한다"""

    generator = CoordinateBasedImageGenerator(original_image.shape)
    results = []

    if n_iterations is None or (
        isinstance(n_iterations, (int, float)) and n_iterations <= 0
    ):
        print(f"훈련 시작: 무한 반복, 반복당 {n_samples_per_iter}개 샘플")
    else:
        print(
            f"훈련 시작: {int(n_iterations)}회 반복, 반복당 {n_samples_per_iter}개 샘플"
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
        # 주: _check_ui_pause_stop는 상단에서 정의됨
        try:
            should_stop = _check_ui_pause_stop()
        except Exception:
            should_stop = False
        if should_stop:
            print("사용자 요청으로 훈련을 조기 종료합니다.")
            break

        iteration += 1

        # 1. 새로운 픽셀 샘플링
        sampled_pixels, coords = sample_random_pixels(
            original_image, n_samples_per_iter
        )

        # 2. 샘플 누적
        generator.add_training_samples(sampled_pixels)

        # 3. 모델 훈련 (가벼운 한 스텝)
        generator.train_step()

        # 4~5. 이미지/지표 계산: 시각화 타이밍에만 전체 이미지를 생성
        do_full = (iteration % visualize_every == 1) or (
            n_iterations is not None and iteration == int(n_iterations)
        )

        if do_full:
            generated_image = generator.generate_image(scale=viz_scale)
            mse = float(np.mean((original_image - generated_image) ** 2))
            mae = float(np.mean(np.abs(original_image - generated_image)))
        else:
            # 빠른 근사 지표: 일부 픽셀만 샘플링
            h, w, _ = original_image.shape
            sample_n = min(fast_metric_sample, h * w)
            us = np.random.randint(0, w, sample_n)
            vs = np.random.randint(0, h, sample_n)
            uv = np.column_stack([us / (w - 1), vs / (h - 1)])
            with threadpool_limits(limits=1):
                preds = generator.model.predict(uv)
            gt = original_image[vs, us]
            diffs = gt - preds
            mse = float(np.mean(diffs**2))
            mae = float(np.mean(np.abs(diffs)))
            generated_image = None  # 필요 시 마지막 전체 이미지를 유지

        # 6. 결과 저장
        results.append(
            {
                "iteration": iteration,
                "mse": mse,
                "mae": mae,
                "total_samples": generator.get_sample_count(),
                "generated_image": (
                    generated_image.copy() if generated_image is not None else None
                ),
                "sampled_coords": coords,
            }
        )

        print(
            f"반복 {iteration:2d}: 누적샘플={generator.get_sample_count():4d}, "
            f"MSE={mse:.6f}, MAE={mae:.6f}"
        )

        # 7. 시각화 (전체 이미지를 생성한 경우만 호출)
        if do_full and results[-1]["generated_image"] is not None:
            visualize_training_progress(
                original_image, results[-1]["generated_image"], coords, iteration
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
    mse_values = [r["mse"] for r in training_results]
    mae_values = [r["mae"] for r in training_results]
    sample_counts = [r["total_samples"] for r in training_results]

    print("\n=== 훈련 결과 분석 ===")
    print(f"초기 MSE: {mse_values[0]:.6f}")
    print(f"최종 MSE: {mse_values[-1]:.6f}")
    print(
        f"MSE 개선율: {((mse_values[0] - mse_values[-1]) / mse_values[0] * 100):.1f}%"
    )
    print()
    print(f"초기 MAE: {mae_values[0]:.6f}")
    print(f"최종 MAE: {mae_values[-1]:.6f}")
    print(
        f"MAE 개선율: {((mae_values[0] - mae_values[-1]) / mae_values[0] * 100):.1f}%"
    )
    print()
    print(f"총 사용된 샘플: {sample_counts[-1]}개")
    print(
        f"전체 픽셀 대비: {(sample_counts[-1] / (original_image.shape[0] * original_image.shape[1]) * 100):.2f}%"
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

    parser = argparse.ArgumentParser(description="스파스 픽셀 기반 이미지 생성 학습")
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
        "--samples-per-iter", type=int, default=50, help="반복당 샘플링 픽셀 수"
    )
    parser.add_argument(
        "--visualize-every", type=int, default=2, help="시각화 주기 (반복 수)"
    )
    parser.add_argument(
        "--viz-scale",
        type=float,
        default=0.7,
        help="시각화용 저해상도 배율 (0~1, 작을수록 빠름)",
    )
    args = parser.parse_args()

    print("=== 스파스 픽셀 샘플링 기반 이미지 생성 프로젝트 ===\n")

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
    print("\n2. 신경망 훈련 시작...")
    generator, results = train_image_generator(
        original_image,
        n_iterations=args.iterations,
        n_samples_per_iter=args.samples_per_iter,
        visualize_every=args.visualize_every,
        viz_scale=args.viz_scale,
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
