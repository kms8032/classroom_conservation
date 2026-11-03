"""
Seat 상태 변화 테스트.

카메라·YOLO 없이 "몇 초에 감지됐는지"만 넣어서 CLAUDE.md의 상태 그림대로 바뀌는지 확인한다.
now를 숫자로 직접 넣기 때문에 1분짜리 조건도 기다리지 않고 바로 시험할 수 있다.

실행 (v2 폴더에서):
    python -m pytest
"""

from seat.seat import ENTER_SEC, LEAVE_SEC, Seat
from seat.seat_state import SeatState

STEP = 0.5  # 추론 간격 후보 (0.5초마다 한 번 update가 불린다고 가정)


def feed(seat: Seat, start: float, end: float, detected: bool) -> float:
    """start초부터 end초까지 STEP 간격으로 같은 감지 결과를 넣는다. 마지막으로 넣은 시각을 돌려준다."""
    now = start
    while now <= end:
        seat.update(detected, now)
        now += STEP
    return now - STEP


def make_occupied_seat() -> Seat:
    """0초부터 5초까지 감지해서 점유가 된 좌석을 만든다. (여러 테스트의 출발점)"""
    seat = Seat("21")
    feed(seat, 0, ENTER_SEC, True)
    assert seat.state == SeatState.OCCUPIED
    return seat


# ──────────────────────────────────────────────
# 비어 있음 → 점유
# ──────────────────────────────────────────────
def test_처음에는_비어_있음():
    seat = Seat("21")
    assert seat.state == SeatState.EMPTY
    assert seat.last_seen is None


def test_5초_연속_감지하면_점유():
    seat = Seat("21")
    feed(seat, 0, ENTER_SEC, True)
    assert seat.state == SeatState.OCCUPIED


def test_5초가_안_됐으면_비어_있음_그대로():
    # 진입 대기 중에는 화면에 "비어 있음"으로 보여야 한다 (지나가는 사람 거르기)
    seat = Seat("21")
    feed(seat, 0, ENTER_SEC - STEP, True)
    assert seat.state == SeatState.EMPTY


def test_지나가는_사람은_점유가_아님():
    # 2초 동안만 보이고 사라진 사람
    seat = Seat("21")
    feed(seat, 0, 2, True)
    feed(seat, 2 + STEP, 10, False)
    assert seat.state == SeatState.EMPTY


def test_1초_이하로_끊긴_것은_계속_감지로_봄():
    # 2초에 한 프레임 놓쳐도 5초 세기는 이어진다
    seat = Seat("21")
    feed(seat, 0, 1.5, True)
    seat.update(False, 2.0)
    feed(seat, 2.5, ENTER_SEC, True)
    assert seat.state == SeatState.OCCUPIED


def test_1초보다_오래_끊기면_처음부터_다시_셈():
    # 0~3초 감지, 3.5~5초 끊김(1.5초), 5.5초부터 다시 감지
    seat = Seat("21")
    feed(seat, 0, 3, True)
    feed(seat, 3.5, 5, False)
    feed(seat, 5.5, 10, True)
    assert seat.state == SeatState.EMPTY  # 5.5초부터 다시 세서 아직 4.5초째

    seat.update(True, 10.5)
    assert seat.state == SeatState.OCCUPIED  # 5.5 + 5 = 10.5초에 점유


# ──────────────────────────────────────────────
# 점유 → 확인 중 → 점유 / 비어 있음
# ──────────────────────────────────────────────
def test_점유에서_감지가_끊기면_확인_중():
    seat = make_occupied_seat()
    seat.update(False, ENTER_SEC + STEP)
    assert seat.state == SeatState.CHECKING


def test_확인_중에_다시_감지되면_바로_점유():
    # 5초를 다시 기다리지 않는다 (가려졌다가 다시 보인 것으로 봄)
    seat = make_occupied_seat()
    seat.update(False, 6)
    seat.update(True, 30)
    assert seat.state == SeatState.OCCUPIED


def test_1분_동안_안_보이면_비어_있음():
    seat = make_occupied_seat()
    last_detected = ENTER_SEC
    feed(seat, last_detected + STEP, last_detected + LEAVE_SEC, False)
    assert seat.state == SeatState.EMPTY
    assert seat.last_seen is None  # 비워지면 기록도 처음 상태로


def test_1분이_안_됐으면_확인_중_그대로():
    seat = make_occupied_seat()
    last_detected = ENTER_SEC
    feed(seat, last_detected + STEP, last_detected + LEAVE_SEC - STEP, False)
    assert seat.state == SeatState.CHECKING


def test_비워진_뒤에는_다시_5초를_기다림():
    seat = make_occupied_seat()
    feed(seat, 5.5, 65, False)
    assert seat.state == SeatState.EMPTY

    seat.update(True, 70)
    assert seat.state == SeatState.EMPTY  # 바로 점유가 되지 않는다


# ──────────────────────────────────────────────
# 그 밖의 기능
# ──────────────────────────────────────────────
def test_last_seen은_마지막_감지_시각():
    seat = make_occupied_seat()
    seat.update(True, 20)
    seat.update(False, 21)
    assert seat.last_seen == 20


def test_reset하면_비어_있음에서_다시_시작():
    # 카메라 재접속 때 쓰는 기능
    seat = make_occupied_seat()
    seat.reset()
    assert seat.state == SeatState.EMPTY
    assert seat.last_seen is None
