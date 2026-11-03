"""
구현 순서 1번(RTSP 상시 연결) 확인용 실행 파일.

강의실 카메라를 모두 상시 연결하고, 몇 초마다 카메라별 상태를 출력한다.
Ctrl+C로 종료한다.

실행 (v2 폴더에서):
    python check_stream.py
    python check_stream.py --room 301 --interval 5
"""

import argparse
import logging
import time

from stream.camera_info import load_cameras
from stream.camera_manager import CameraManager


def print_status(status_list: list[dict]) -> None:
    print("-" * 50)
    for status in status_list:
        state = "연결됨" if status["connected"] else "끊김"
        if status["frame_age_sec"] is None:
            age = "프레임 없음"
        else:
            age = f"마지막 프레임 {status['frame_age_sec']:.1f}초 전"
        print(f"{status['name']:<8} {state:<4} FPS {status['fps']:5.1f}  {age}")


def main() -> None:
    parser = argparse.ArgumentParser(description="카메라 RTSP 상시 연결 확인")
    parser.add_argument("--room", default="301", help="강의실 번호 (기본: 301)")
    parser.add_argument("--interval", type=float, default=5, help="상태 출력 간격(초)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    cameras = load_cameras(args.room)
    print(f"{args.room}호 카메라 {len(cameras)}대 연결 시작: {[cam.name for cam in cameras]}")

    manager = CameraManager(cameras)
    manager.start_all()
    try:
        while True:
            time.sleep(args.interval)
            print_status(manager.get_status())
    except KeyboardInterrupt:
        print("\n종료합니다...")
    finally:
        manager.stop_all()


if __name__ == "__main__":
    main()
