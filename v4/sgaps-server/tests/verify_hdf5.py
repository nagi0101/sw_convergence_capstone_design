"""
SGAPS-MAE HDF5 Data Verification Script

This script analyzes and verifies the integrity of collected frame data
stored in HDF5 format.

Usage:
    python verify_hdf5.py [filepath] [--session SESSION_ID] [--detailed]

Examples:
    python verify_hdf5.py
    python verify_hdf5.py data/fps_game_test.h5
    python verify_hdf5.py data/fps_game_test.h5 --session client_xxx --detailed
"""

import h5py
import numpy as np
import sys
import argparse
from pathlib import Path


def analyze_state_vector_changes(frames_group, frame_ids, num_samples=10):
    """프레임 간 State Vector 변화를 분석합니다."""

    if len(frame_ids) < 2:
        return

    # 샘플 프레임 선택 (균등 간격)
    step = max(1, len(frame_ids) // num_samples)
    sample_ids = frame_ids[::step][:num_samples]

    print(f"\n{'='*70}")
    print(f"State Vector Changes Analysis ({len(sample_ids)} samples)")
    print(f"{'='*70}")

    states = []
    for fid in sample_ids:
        frame = frames_group[fid]
        state = frame['state_vector'][:]
        states.append(state)

    states = np.array(states)

    # 각 인덱스별 통계
    state_labels = [
        "Pos X", "Pos Y", "Pos Z",
        "Vel X", "Vel Y", "Vel Z",
        "Health", "Grounded", "Crouch", "Dead", "Cam Angle"
    ]

    print(f"\n{'Index':<6} {'Name':<12} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
    print(f"{'-'*70}")

    for i in range(min(11, states.shape[1])):
        values = states[:, i]
        label = state_labels[i] if i < len(state_labels) else f"State[{i}]"

        print(f"{i:<6} {label:<12} {values.min():10.3f} {values.max():10.3f} "
              f"{values.mean():10.3f} {values.std():10.3f}")


def analyze_pixel_statistics(frames_group, frame_ids, num_samples=5):
    """픽셀 데이터 통계를 분석합니다."""

    print(f"\n{'='*70}")
    print(f"Pixel Data Statistics ({len(frame_ids)} frames)")
    print(f"{'='*70}")

    # 전체 픽셀 값 수집 (샘플링)
    step = max(1, len(frame_ids) // num_samples)
    sample_ids = frame_ids[::step][:num_samples]

    all_pixel_values = []
    all_u_coords = []
    all_v_coords = []

    for fid in sample_ids:
        frame = frames_group[fid]
        pixels = frame['pixels'][:]
        all_u_coords.extend(pixels[:, 0])
        all_v_coords.extend(pixels[:, 1])
        all_pixel_values.extend(pixels[:, 2])

    all_pixel_values = np.array(all_pixel_values)
    all_u_coords = np.array(all_u_coords)
    all_v_coords = np.array(all_v_coords)

    print(f"\nPixel Values:")
    print(f"  Min: {all_pixel_values.min():.2f}")
    print(f"  Max: {all_pixel_values.max():.2f}")
    print(f"  Mean: {all_pixel_values.mean():.2f}")
    print(f"  Std: {all_pixel_values.std():.2f}")

    print(f"\nUV Coordinates:")
    print(f"  U range: [{all_u_coords.min():.4f}, {all_u_coords.max():.4f}]")
    print(f"  V range: [{all_v_coords.min():.4f}, {all_v_coords.max():.4f}]")

    # 픽셀 분포 히스토그램 (간단한 텍스트 버전)
    print(f"\nPixel Value Distribution (0-255):")
    hist, bins = np.histogram(all_pixel_values, bins=10, range=(0, 255))
    max_bar_len = 50
    max_count = hist.max()

    for i, count in enumerate(hist):
        bar_len = int((count / max_count) * max_bar_len) if max_count > 0 else 0
        bar = '█' * bar_len
        print(f"  {bins[i]:6.1f}-{bins[i+1]:6.1f}: {bar} ({count})")


def analyze_frame_timing(frames_group, frame_ids):
    """프레임 타이밍 분석."""

    if len(frame_ids) < 2:
        return

    print(f"\n{'='*70}")
    print(f"Frame Timing Analysis")
    print(f"{'='*70}")

    timestamps = []
    for fid in frame_ids:
        frame = frames_group[fid]
        ts = frame.attrs.get('timestamp', 0)
        timestamps.append(ts)

    timestamps = np.array(timestamps)

    if len(timestamps) > 1:
        intervals = np.diff(timestamps)

        print(f"\nCapture Timestamps:")
        print(f"  First frame: {timestamps[0]:.3f}")
        print(f"  Last frame: {timestamps[-1]:.3f}")
        print(f"  Duration: {timestamps[-1] - timestamps[0]:.3f}s")

        print(f"\nFrame Intervals:")
        print(f"  Min: {intervals.min():.4f}s ({1/intervals.min():.1f} FPS)")
        print(f"  Max: {intervals.max():.4f}s ({1/intervals.max():.1f} FPS)")
        print(f"  Mean: {intervals.mean():.4f}s ({1/intervals.mean():.1f} FPS)")
        print(f"  Std: {intervals.std():.4f}s")


def print_frame_details(frame, frame_id, detailed=False):
    """프레임 상세 정보 출력."""

    print(f"\n{'─'*70}")
    print(f"Frame {frame_id}")
    print(f"{'─'*70}")

    # 픽셀 데이터
    pixels = frame['pixels'][:]
    print(f"\nPixels:")
    print(f"  Shape: {pixels.shape} (expected: N x 3)")
    print(f"  Count: {len(pixels)}")

    if detailed and len(pixels) > 0:
        print(f"  Sample (first 5):")
        for i in range(min(5, len(pixels))):
            u, v, value = pixels[i]
            print(f"    [{i}] u={u:.4f}, v={v:.4f}, value={value:.0f}")

    # State vector
    state_vector = frame['state_vector'][:]
    print(f"\nState Vector:")
    print(f"  Length: {len(state_vector)} (stored without sentinel padding)")

    if len(state_vector) >= 11:
        print(f"  FPS Game State:")
        print(f"    Position: ({state_vector[0]:.2f}, {state_vector[1]:.2f}, {state_vector[2]:.2f})")
        print(f"    Velocity: ({state_vector[3]:.2f}, {state_vector[4]:.2f}, {state_vector[5]:.2f})")
        print(f"    Speed: {np.linalg.norm(state_vector[3:6]):.2f}")
        print(f"    Health: {state_vector[6]:.3f} ({state_vector[6]*100:.1f}%)")
        print(f"    Grounded: {state_vector[7]:.0f}")
        print(f"    Crouching: {state_vector[8]:.3f}")
        print(f"    Dead: {state_vector[9]:.0f}")
        print(f"    Camera Angle: {state_vector[10]:.3f} ({state_vector[10]*89:.1f}°)")

    # 메타데이터
    resolution = frame.attrs.get('resolution', 'N/A')
    timestamp = frame.attrs.get('timestamp', 'N/A')

    print(f"\nMetadata:")
    print(f"  Resolution: {list(resolution) if hasattr(resolution, '__iter__') else resolution}")
    print(f"  Timestamp: {timestamp}")


def verify_session_data(filepath='data/fps_game_test.h5', session_id=None, detailed=False):
    """HDF5 파일의 세션 데이터를 검증하고 분석합니다."""

    filepath = Path(filepath)

    if not filepath.exists():
        print(f"❌ Error: File not found: {filepath}")
        return False

    print(f"{'='*70}")
    print(f"SGAPS-MAE HDF5 Data Verification")
    print(f"{'='*70}")
    print(f"File: {filepath}")
    print(f"Size: {filepath.stat().st_size / 1024 / 1024:.2f} MB")

    try:
        with h5py.File(filepath, 'r') as f:
            # 세션 목록
            sessions = list(f.keys())
            print(f"\nTotal sessions: {len(sessions)}")

            if len(sessions) == 0:
                print("❌ No sessions found!")
                return False

            # 세션 선택
            if session_id is not None:
                if session_id not in sessions:
                    print(f"❌ Session '{session_id}' not found!")
                    print(f"Available sessions: {sessions}")
                    return False
                target_session = session_id
            else:
                # 가장 최근 세션 (마지막)
                target_session = sessions[-1]

            print(f"\n{'='*70}")
            print(f"Analyzing Session: {target_session}")
            print(f"{'='*70}")

            session_group = f[target_session]

            # 메타데이터
            if 'metadata' in session_group:
                meta = session_group['metadata']
                created_at = meta.attrs.get('created_at', 'N/A')
                frame_count = meta.attrs.get('frame_count', 'N/A')
                print(f"\nSession Metadata:")
                print(f"  Created: {created_at}")
                print(f"  Frame count: {frame_count}")

            # 프레임 분석
            frames_group = session_group['frames']
            frame_ids = sorted(frames_group.keys(), key=lambda x: int(x))

            print(f"\nFrames: {len(frame_ids)}")

            if len(frame_ids) == 0:
                print("❌ No frames found!")
                return False

            # 첫 프레임 상세 정보
            print_frame_details(frames_group[frame_ids[0]], frame_ids[0], detailed=detailed)

            # 중간 프레임 (간략)
            if len(frame_ids) > 2:
                mid_idx = len(frame_ids) // 2
                print_frame_details(frames_group[frame_ids[mid_idx]], frame_ids[mid_idx], detailed=False)

            # 마지막 프레임 (간략)
            if len(frame_ids) > 1:
                print_frame_details(frames_group[frame_ids[-1]], frame_ids[-1], detailed=False)

            # 통계 분석
            analyze_pixel_statistics(frames_group, frame_ids)
            analyze_state_vector_changes(frames_group, frame_ids)
            analyze_frame_timing(frames_group, frame_ids)

            print(f"\n{'='*70}")
            print(f"✓ Verification Complete")
            print(f"{'='*70}")

            return True

    except Exception as e:
        print(f"\n❌ Error analyzing HDF5 file: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Verify and analyze SGAPS-MAE HDF5 data files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python verify_hdf5.py
  python verify_hdf5.py data/fps_game_test.h5
  python verify_hdf5.py data/fps_game_test.h5 --session client_123 --detailed
        """
    )

    parser.add_argument(
        'filepath',
        nargs='?',
        default='data/sgaps-mae-fps.h5',
        help='Path to HDF5 file (default: data/sgaps-mae-fps.h5)'
    )

    parser.add_argument(
        '--session',
        type=str,
        help='Specific session ID to analyze (default: most recent)'
    )

    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed frame information'
    )

    args = parser.parse_args()

    success = verify_session_data(
        filepath=args.filepath,
        session_id=args.session,
        detailed=args.detailed
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()