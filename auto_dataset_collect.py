"""
강의실 데이터 수집용 자동 캡처 스크립트
- 지정한 간격마다 각 카메라에서 사진 1장씩 저장 (정각 기준, 기본 20분 → :00 / :20 / :40)
- 저장 위치: data/<강의실>/<카메라이름>/<날짜>/<시각>.jpg
- 카메라 접속 정보는 .env 파일에서 읽어옴

실행:  python capture_304.py                      (기본: 304호, 20분)
       python capture_304.py --room 301 --interval 10
종료:  Ctrl + C
"""

import os

# [변경] RTSP를 TCP로 고정 (기본값 UDP는 패킷 손실 시 회색으로 번진 깨진 프레임이 저장될 수 있음)
#        반드시 import cv2 "전에" 설정해야 적용됨
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
import time
import argparse
from urllib.parse import quote          # [변경] 비밀번호 특수문자 인코딩용
from datetime import datetime, timedelta
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))  # .env 파일 내용 읽어옴

# .env 파일 확인
REQUIRED = ["CAM_ID", "CAM_PASSWORD", "CAM_PORT", "CAM1_IP", "CAM2_IP", "CAM3_IP", "CAM4_IP"]
missing = [k for k in REQUIRED if not os.getenv(k)]
if missing:
    # [변경] SystemError → SystemExit
    #        SystemError는 "파이썬 내부 오류"용 예외라 의미가 맞지 않음. SystemExit는 메시지만 출력하고 깔끔하게 종료
    raise SystemExit(f".env에 값이 없습니다: {missing}")

# 환경변수 불러오기
CAM_ID = os.getenv("CAM_ID")
# 비밀번호에 @ # : 같은 특수문자가 있어도 RTSP 주소가 깨지지 않도록 인코딩
CAM_PASSWORD = quote(os.getenv("CAM_PASSWORD"), safe="")
CAM_PORT = os.getenv("CAM_PORT")

parser = argparse.ArgumentParser(description="강의실 자동 캡처")
parser.add_argument("--room", default="304", help="강의실 번호")
parser.add_argument("--interval", type=int, default=20, help="캡처 간격(분)")
args = parser.parse_args()

#간격이 0 이하면 다음 캡처 시각 계산에서 0으로 나누기 오류가 나므로 미리 막음
if args.interval <= 0:
    raise SystemExit("--interval은 1 이상이어야 합니다")


# ──────────────────────────────────────────────
# 1. 설정
# ──────────────────────────────────────────────
ROOM = args.room
INTERVAL_MIN = args.interval                     # 캡처 간격(분)
SAVE_ROOT = os.path.join(BASE_DIR, "data", ROOM) # 스크립트 위치 기준 경로
JPEG_QUALITY = 95                                # 학습 데이터용이라 화질 높게
RETRY = 3                                        # 연결 실패 시 재시도 횟수
TIMEOUT_MS = 5000                                # 카메라 연결/읽기 타임아웃(5초)

# 카메라 목록 (접속 정보는 .env에서 읽어옴)
CAMERAS = {
    f"cam{i}": f"rtsp://{CAM_ID}:{CAM_PASSWORD}@{os.getenv(f'CAM{i}_IP')}:{CAM_PORT}/stream1"
    for i in range(1, 5)
}


# ──────────────────────────────────────────────
# 2. 카메라 한 대에서 사진 한 장 가져오기
# ──────────────────────────────────────────────
def open_camera(source):
    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, TIMEOUT_MS,
                  cv2.CAP_PROP_READ_TIMEOUT_MSEC, TIMEOUT_MS]
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)
    return cv2.VideoCapture(source)


def grab_frame(source):
    """
    매번 새로 연결해서 한 장만 읽고 바로 끊는다.
    → 연결을 계속 열어두면 버퍼에 쌓인 '옛날 화면'이 찍힐 수 있고,
      20분 간격이라 계속 열어둘 이유도 없음.
    """
    for attempt in range(1, RETRY + 1):
        cap = open_camera(source)
        if cap.isOpened():
            # 첫 몇 프레임은 깨지거나 검게 나올 수 있어 버리고 읽기
            for _ in range(5):
                cap.read()
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                return frame
        else:
            cap.release()
        print(f"연결/읽기 실패 ({attempt}/{RETRY}), 3초 후 재시도")
        time.sleep(3)
    return None


# ──────────────────────────────────────────────
# 3. 저장
# ──────────────────────────────────────────────
def save_frame(cam_name, frame, now):
    folder = os.path.join(SAVE_ROOT, cam_name, now.strftime("%Y-%m-%d"))
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, now.strftime("%H-%M-%S") + ".jpg")
    # [변경] 저장 성공 여부 확인 (디스크가 가득 차는 등 저장에 실패해도 성공으로 출력되던 문제)
    if not cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
        return None
    return path


# ──────────────────────────────────────────────
# 4. 한 번의 캡처 라운드 (모든 카메라)
# ──────────────────────────────────────────────
def capture_all():
    now = datetime.now()
    print(f"\n[{now:%Y-%m-%d %H:%M:%S}] 캡처 시작")
    for cam_name, source in CAMERAS.items():
        frame = grab_frame(source)
        if frame is None:
            print(f"{cam_name}: 실패")
            continue
        path = save_frame(cam_name, frame, now)
        # 저장 실패 시 별도 메시지 출력
        if path is None:
            print(f"{cam_name}: 저장 실패 (디스크 용량/권한 확인)")
            continue
        h, w = frame.shape[:2]
        print(f"{cam_name}: {path} ({w}x{h})")


# ──────────────────────────────────────────────
# 5. 다음 캡처 시각 계산 (정각 기준으로 맞춤)
# ──────────────────────────────────────────────
def next_capture_time(now):
    base = now.replace(second=0, microsecond=0)
    minutes_to_add = INTERVAL_MIN - (base.minute % INTERVAL_MIN)
    return base + timedelta(minutes=minutes_to_add)


# ──────────────────────────────────────────────
# 6. 메인 루프
# ──────────────────────────────────────────────
def main():
    print(f"{ROOM}호 자동 캡처 시작 — {INTERVAL_MIN}분 간격, 카메라 {len(CAMERAS)}대")
    capture_all()

    while True:
        target = next_capture_time(datetime.now())
        print(f"다음 캡처: {target:%H:%M}")
        time.sleep(max(0, (target - datetime.now()).total_seconds()))
        capture_all()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n캡처 종료")