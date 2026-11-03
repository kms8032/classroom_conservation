"""
카메라 접속 정보를 읽어서 CameraInfo 목록을 만든다.

- 카메라 목록(이름, camera_id, ip_env)은 v2/config/classroom_<강의실>.json 에서 읽는다.
- IP·계정·비밀번호·포트는 저장소 루트의 .env 에서 읽는다.
  .env.example 은 "어떤 키가 필요한지" 보여주는 견본이라 실제 값이 없으므로 읽지 않는다.
- JSON에는 IP를 직접 적지 않고 "ip_env"(.env의 키 이름)만 적는다.
  IP가 Git 기록에 남지 않게 하기 위함이다.
- 제조사마다 RTSP 경로가 달라서, JSON에는 "vendor"(제조사 이름)만 적고
  실제 경로는 아래 VENDOR_PATHS 표 한 곳에서 관리한다.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv

# 이 파일 위치(v2/stream/)를 기준으로 경로를 잡는다.
# 어느 폴더에서 실행하든 같은 파일을 찾게 하기 위함이다.
V2_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = V2_DIR.parent
ENV_PATH = REPO_ROOT / ".env"
CONFIG_DIR = V2_DIR / "config"

# 모든 카메라가 같이 쓰는 .env 키
# (지금은 TP-Link와 Hikvision의 계정·비밀번호·포트가 같다.
#  나중에 제조사마다 달라지면 VENDOR_PATHS처럼 제조사별 표로 나눈다.)
COMMON_ENV_KEYS = ["CAM_ID", "CAM_PASSWORD", "CAM_PORT"]

# 제조사별 RTSP 메인 스트림 경로.
# 메인 스트림을 쓰는 이유: 좌석 ROI 좌표가 메인 스트림 화면 기준으로 찍혀 있기 때문이다.
VENDOR_PATHS = {
    "tplink": "/stream1",                     # 301호 cam1~4 (TP-Link VIGI)
    "hikvision": "/Streaming/Channels/201",   # 301호 cam5, 304호 (Hikvision)
}

# 광각 cam5(교탁·복도 인원 카운트용)는 아직 연결하지 않는다.
# 구현 순서 3번(교탁·복도 카운트)에서 classroom_301.json 의 "cameras"에
# 아래처럼 한 개를 추가하면, 이 코드를 고치지 않아도 cam5가 함께 연결된다.
#   {
#     "camera_id": "<cam5의 camera_id>",
#     "name": "CCTV 5",
#     "ip_env": "CAM_301_IP5",
#     "vendor": "hikvision",
#     ...
#   }


@dataclass(frozen=True)
class CameraInfo:
    """카메라 한 대의 접속 정보. 한 번 만들면 값이 바뀌지 않는다(frozen)."""

    room: str        # 강의실 번호 (예: "301")
    name: str        # 카메라 이름 (예: "CCTV 1") — 로그·화면에는 이 이름만 쓴다
    camera_id: str   # 카메라 고유 ID (JSON에 적힌 값)
    # repr=False: print(camera)나 로그에 객체를 찍어도
    # 비밀번호가 들어 있는 RTSP 주소는 보이지 않게 하기 위함
    rtsp_url: str = field(repr=False)


def load_env() -> None:
    """저장소 루트의 .env 를 읽어 환경변수로 등록한다."""
    if not ENV_PATH.exists():
        raise FileNotFoundError(
            f"{ENV_PATH} 파일이 없습니다. "
            ".env.example 을 복사해 .env 를 만들고 값을 채워 주세요."
        )
    load_dotenv(ENV_PATH)


def _check_env_keys(keys: list[str]) -> None:
    """필요한 키가 .env 에 모두 있는지 확인한다.

    하나씩 확인하다 멈추면 고치고 다시 실행하기를 여러 번 반복해야 하므로,
    빠진 키를 한 번에 모아서 알려준다.
    """
    missing = [key for key in keys if not os.getenv(key)]
    if missing:
        raise KeyError(f".env 에 다음 값이 없습니다: {', '.join(missing)}")


def build_rtsp_url(ip: str, vendor: str) -> str:
    """카메라 IP와 제조사로 RTSP 주소를 만든다."""
    if vendor not in VENDOR_PATHS:
        raise ValueError(
            f"알 수 없는 제조사입니다: {vendor!r} "
            f"(사용 가능: {', '.join(VENDOR_PATHS)})"
        )
    user = os.getenv("CAM_ID")
    # 비밀번호에 @ # : 같은 특수문자가 있으면 주소가 깨지므로 인코딩한다
    password = quote(os.getenv("CAM_PASSWORD"), safe="")
    port = os.getenv("CAM_PORT")
    path = VENDOR_PATHS[vendor]
    return f"rtsp://{user}:{password}@{ip}:{port}{path}"


def load_cameras(room: str) -> list[CameraInfo]:
    """강의실 설정 JSON과 .env 를 읽어 그 강의실의 카메라 목록을 만든다."""
    load_env()

    config_path = CONFIG_DIR / f"classroom_{room}.json"
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    # 설정 파일 하나에는 강의실 하나만 들어 있다
    camera_configs = config["classrooms"][0]["cameras"]

    ip_keys = [cam["ip_env"] for cam in camera_configs]
    _check_env_keys(COMMON_ENV_KEYS + ip_keys)

    cameras = []
    for cam in camera_configs:
        ip = os.getenv(cam["ip_env"])
        cameras.append(
            CameraInfo(
                room=room,
                name=cam["name"],
                camera_id=cam["camera_id"],
                rtsp_url=build_rtsp_url(ip, cam["vendor"]),
            )
        )
    return cameras
