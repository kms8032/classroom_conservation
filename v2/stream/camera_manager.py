"""
여러 카메라의 RtspReader를 한 번에 관리한다.

다음 단계(좌석 판단)의 추론 루프는 이 클래스 하나만 알면 된다.
카메라가 몇 대인지, 각각 어떻게 연결되는지는 몰라도 get_latest_frames()로 프레임을 받는다.
여러 카메라 프레임을 한꺼번에 받아 YOLO에 묶어서(batch) 넣기 위함이다.
"""

import time

from .camera_info import CameraInfo
from .rtsp_reader import RtspReader, TIMEOUT_MS

# 이보다 오래된 프레임은 "지금 화면"으로 보지 않는다.
# 카메라가 멈췄는데 마지막 화면으로 계속 판단하는 것을 막기 위함이다.
MAX_FRAME_AGE_SEC = 2.0


class CameraManager:
    """카메라 여러 대를 시작·정지하고, 최신 프레임과 상태를 모아서 돌려준다."""

    def __init__(self, cameras: list[CameraInfo]):
        # 카메라 이름 → RtspReader
        self._readers = {
            cam.name: RtspReader(cam.name, cam.rtsp_url) for cam in cameras
        }

    def start_all(self) -> None:
        for reader in self._readers.values():
            reader.start()

    def stop_all(self) -> None:
        # 모두에게 먼저 멈추라고 알린 뒤 기다린다.
        # 한 대씩 멈추고 기다리면 카메라 수만큼 시간이 늘어나기 때문이다.
        for reader in self._readers.values():
            reader.stop()

        # 기다리는 시간도 카메라마다 따로 주지 않고 "전체 마감 시각" 하나로 묶는다.
        # 연결 중인 카메라가 여러 대면 종료가 카메라 수 × 타임아웃만큼 늦어지기 때문이다.
        # 마감까지 안 끝난 스레드는 daemon 스레드라 프로그램이 끝날 때 같이 정리된다.
        deadline = time.time() + TIMEOUT_MS / 1000 + 1
        for reader in self._readers.values():
            reader.join(timeout=max(0, deadline - time.time()))

    def get_latest_frames(self, max_age_sec: float = MAX_FRAME_AGE_SEC) -> dict:
        """{카메라 이름: 최신 프레임}을 돌려준다.

        아직 프레임이 없거나 max_age_sec 보다 오래된 카메라는 결과에서 뺀다.
        """
        now = time.time()
        frames = {}
        for name, reader in self._readers.items():
            frame, frame_time = reader.get_latest()
            if frame is None:
                continue
            if now - frame_time > max_age_sec:
                continue
            frames[name] = frame
        return frames

    def get_status(self) -> list[dict]:
        """카메라별 연결 상태 목록."""
        return [reader.get_status() for reader in self._readers.values()]
