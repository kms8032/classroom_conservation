"""
카메라 한 대를 상시 연결해 두고, 가장 최신 프레임 한 장만 들고 있는 스레드.

왜 상시 연결 + 최신 프레임만?
- 매번 새로 연결하면 연결에만 몇 초가 걸려 실시간 판단을 할 수 없다.
- 연결을 열어 두고 읽지 않으면 카메라 버퍼에 옛날 화면이 쌓인다.
  그래서 스레드가 쉬지 않고 읽어서 버퍼를 비우고, 마지막 한 장만 남겨 둔다.
- YOLO 추론은 이 스레드에서 하지 않는다. 읽기만 담당해야 추론이 느려도 버퍼가 쌓이지 않는다.
"""

import os

# RTSP를 TCP로 고정한다. 기본값 UDP는 패킷이 빠지면 화면이 회색으로 깨진다.
# 반드시 import cv2 "전에" 설정해야 적용된다.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import logging
import threading
import time

import cv2

logger = logging.getLogger(__name__)

TIMEOUT_MS = 5000      # 연결·읽기가 5초 넘게 멈추면 실패로 보고 다시 연결한다
RETRY_MIN_SEC = 1      # 재연결 대기 시간: 1초에서 시작해
RETRY_MAX_SEC = 30     # 실패할 때마다 두 배로 늘리되 최대 30초까지만 기다린다
FPS_WINDOW_SEC = 5     # 최근 5초 동안 받은 프레임 수로 FPS를 계산한다


class RtspReader(threading.Thread):
    """카메라 한 대 = 스레드 하나. 끊기면 스스로 다시 연결한다.

    source 에는 RTSP 주소 대신 동영상 파일 경로를 넣어도 된다.
    카메라 없이 녹화 파일로 끊김·재연결을 시험해 볼 때 쓴다.
    """

    def __init__(self, camera_name: str, source: str):
        # daemon=True: 메인 프로그램이 끝나면 이 스레드도 같이 끝나게 한다.
        # 그렇지 않으면 Ctrl+C 후에도 스레드가 남아 프로그램이 꺼지지 않는다.
        super().__init__(name=f"RtspReader-{camera_name}", daemon=True)
        self.camera_name = camera_name
        self._source = source  # 비밀번호가 들어 있으므로 로그에 찍지 않는다

        # 이 스레드는 프레임을 쓰고, 다른 스레드(추론 루프)는 읽는다.
        # 동시에 접근해도 값이 섞이지 않도록 lock으로 보호한다.
        self._lock = threading.Lock()
        self._frame = None
        self._frame_time = 0.0  # 마지막 프레임을 받은 시각 (time.time())
        self._connected = False
        self._fps = 0.0

        self._stop_event = threading.Event()

    # ──────────────────────────────────────────────
    # 다른 스레드에서 부르는 메서드
    # ──────────────────────────────────────────────
    def get_latest(self):
        """(최신 프레임, 받은 시각)을 돌려준다. 아직 받은 프레임이 없으면 (None, 0.0).

        프레임을 복사하지 않고 그대로 넘긴다. 이 스레드는 프레임을 고치지 않고
        새 프레임으로 바꿔 끼우기만 하므로 안전하다.
        단, 받은 쪽에서 프레임 위에 그림을 그리려면 frame.copy()를 먼저 해야 한다.
        """
        with self._lock:
            return self._frame, self._frame_time

    def get_status(self) -> dict:
        """연결 상태를 돌려준다. 화면 출력·점검용."""
        with self._lock:
            if self._frame is None:
                frame_age = None
            else:
                frame_age = time.time() - self._frame_time
            return {
                "name": self.camera_name,
                "connected": self._connected,
                "fps": self._fps,
                "frame_age_sec": frame_age,  # 마지막 프레임이 몇 초 전에 왔는지
            }

    def stop(self) -> None:
        """스레드에 멈추라고 알린다. 실제로 멈출 때까지는 최대 TIMEOUT_MS가 걸릴 수 있다."""
        self._stop_event.set()

    # ──────────────────────────────────────────────
    # 스레드 안에서 도는 코드
    # ──────────────────────────────────────────────
    def run(self) -> None:
        retry_wait = RETRY_MIN_SEC
        while not self._stop_event.is_set():
            cap = self._open()
            if cap is None:
                logger.warning("%s 연결 실패, %d초 후 다시 시도", self.camera_name, retry_wait)
            else:
                logger.info("%s 연결됨", self.camera_name)
                got_frame = self._read_until_fail(cap)
                cap.release()

                with self._lock:
                    self._connected = False
                    self._fps = 0.0
                if self._stop_event.is_set():
                    break

                # 프레임을 한 장이라도 받았으면 정상 연결이었으므로 대기 시간을 처음(1초)으로 되돌린다.
                # 연결만 되고 프레임이 안 오는 경우까지 되돌리면, 1초 간격으로 끝없이 재연결하게 된다.
                if got_frame:
                    retry_wait = RETRY_MIN_SEC
                logger.warning("%s 끊김, %d초 후 다시 연결", self.camera_name, retry_wait)

            # time.sleep 대신 wait를 쓰면 기다리는 중에도 stop()에 바로 반응한다
            self._stop_event.wait(retry_wait)
            retry_wait = min(retry_wait * 2, RETRY_MAX_SEC)

        logger.info("%s 스레드 종료", self.camera_name)

    def _open(self):
        """카메라에 연결한다. 실패하면 None."""
        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, TIMEOUT_MS,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, TIMEOUT_MS,
        ]
        cap = cv2.VideoCapture(self._source, cv2.CAP_FFMPEG, params)
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    def _read_until_fail(self, cap) -> bool:
        """읽기에 실패하거나 stop()이 불릴 때까지 프레임을 계속 읽는다.

        프레임을 한 장이라도 받았으면 True를 돌려준다.
        """
        got_frame = False
        frame_count = 0
        window_start = time.time()

        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                return got_frame

            got_frame = True
            now = time.time()
            with self._lock:
                self._frame = frame
                self._frame_time = now
                self._connected = True

            # FPS는 매 프레임이 아니라 5초마다 한 번 계산한다 (값이 덜 흔들리게)
            frame_count += 1
            elapsed = now - window_start
            if elapsed >= FPS_WINDOW_SEC:
                with self._lock:
                    self._fps = frame_count / elapsed
                frame_count = 0
                window_start = now
        return got_frame
