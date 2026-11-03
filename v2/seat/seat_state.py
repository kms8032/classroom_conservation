"""
좌석 상태 세 가지를 정의한다.

상태를 "점유" 같은 문자열로 직접 쓰면 "점유중"처럼 오타가 나도 에러 없이 지나간다.
Enum으로 만들어 두면 SeatState.OCCUPIED 처럼 정해진 값만 쓸 수 있고,
오타는 실행하자마자 AttributeError로 드러난다.
"""

from enum import Enum


class SeatState(Enum):
    """좌석 상태. 값(value)은 화면·API에 그대로 보여줄 한국어 이름이다."""

    EMPTY = "비어 있음"
    OCCUPIED = "점유"
    CHECKING = "확인 중"  # 사람이 안 보이기 시작한 뒤 1분 동안 (가려졌을 수 있어서 기다리는 중)
