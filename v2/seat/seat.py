"""
좌석 하나의 상태와 타이머.

상태는 시간 조건으로만 바뀐다 (CLAUDE.md "2. 좌석 점유"의 그림):

    [비어 있음] ──(5초 이상 연속 감지)──▶ [점유]
         ▲                                 │ (감지 끊김)
         │                                 ▼
         └──(1분 동안 계속 감지 안 됨)── [확인 중] ──(1분 안에 다시 감지)──▶ [점유]

왜 현재 시각(now)을 인자로 받나?
- Seat 안에서 time.time()을 직접 부르면, 테스트할 때 정말로 5초·1분을 기다려야 한다.
- now를 밖에서 넣어 주면 테스트에서는 숫자만 바꿔 "1분 뒤"를 바로 만들 수 있다.
- 실제 실행에서는 추론 루프가 같은 now를 모든 좌석에 넣어 주므로, 좌석끼리 시각이 어긋나지 않는다.

이 클래스는 카메라·YOLO·ROI를 전혀 모른다. "이번에 감지됐는가(True/False)"만 받는다.
"""

from .seat_state import SeatState

ENTER_SEC = 5.0       # 이만큼 연속으로 감지되어야 점유 (지나가는 사람을 거르기 위함)
LEAVE_SEC = 60.0      # 이만큼 계속 안 보여야 비어 있음 (가려져서 잠깐 안 보이는 경우를 거르기 위함)
ENTER_GAP_SEC = 1.0   # 진입 대기 중 이 시간 이하로 끊긴 것은 계속 감지된 것으로 본다
                      # (YOLO가 한두 프레임 놓칠 때마다 5초를 처음부터 다시 세지 않기 위함)


class Seat:
    """좌석 하나. 상태(state)와, 상태를 바꾸는 데 필요한 시각 두 개를 들고 있다."""

    def __init__(self, seat_id: str):
        self.seat_id = seat_id  # 좌석 번호 (예: "21") — JSON의 좌석 번호와 같은 문자열
        self.state = SeatState.EMPTY

        # 연속 감지가 시작된 시각. 비어 있음 상태에서 "5초"를 세는 데 쓴다.
        # 감지가 끊기면 None으로 되돌린다.
        self._first_seen = None
        # 마지막으로 감지된 시각. "끊긴 지 얼마나 됐나"를 세는 데 쓴다.
        # 한 번도 감지된 적이 없으면 None.
        self._last_seen = None

    @property
    def last_seen(self):
        """마지막으로 감지된 시각 (time.time() 형식). 감지된 적이 없으면 None. API 응답용."""
        return self._last_seen

    def update(self, detected: bool, now: float) -> None:
        """이번 추론 결과로 상태를 갱신한다.

        detected: 이번 프레임에서 이 좌석 ROI 안에 사람이 있었는가
                  (두 카메라에 보이는 좌석은 어느 한쪽이라도 감지하면 True로 들어온다)
        now: 이번 추론 시각 (초)
        """
        if self.state == SeatState.EMPTY:
            self._update_empty(detected, now)
        elif self.state == SeatState.OCCUPIED:
            self._update_occupied(detected, now)
        elif self.state == SeatState.CHECKING:
            self._update_checking(detected, now)

    def reset(self) -> None:
        """비어 있음으로 되돌린다.

        카메라가 끊겼다가 다시 연결되었을 때 쓴다. 끊긴 동안의 일을 알 수 없으므로
        서버를 다시 켰을 때처럼 처음부터 시작한다 (5초 안에 실제 상태로 채워진다).
        """
        self.state = SeatState.EMPTY
        self._first_seen = None
        self._last_seen = None

    # ──────────────────────────────────────────────
    # 상태별 처리
    # ──────────────────────────────────────────────
    def _update_empty(self, detected: bool, now: float) -> None:
        if not detected:
            # 짧게 끊긴 것은 봐준다. ENTER_GAP_SEC보다 오래 끊겼을 때만 5초 세기를 취소한다.
            if self._first_seen is not None and now - self._last_seen > ENTER_GAP_SEC:
                self._first_seen = None
            return

        # 처음 감지됐거나, 오래 끊겼다가 다시 감지된 경우 → 5초를 처음부터 센다
        if self._first_seen is None or now - self._last_seen > ENTER_GAP_SEC:
            self._first_seen = now
        self._last_seen = now

        if now - self._first_seen >= ENTER_SEC and self._can_become_occupied():
            self.state = SeatState.OCCUPIED
            self._first_seen = None  # 5초 세기는 끝났으므로 지운다

    def _update_occupied(self, detected: bool, now: float) -> None:
        if detected:
            self._last_seen = now
        else:
            self.state = SeatState.CHECKING

    def _update_checking(self, detected: bool, now: float) -> None:
        if detected:
            # 다시 보이면 5초를 기다리지 않고 바로 점유로 돌아간다.
            # 확인 중은 "원래 앉아 있던 사람이 가려진 것"일 가능성이 높기 때문이다.
            self._last_seen = now
            self.state = SeatState.OCCUPIED
        elif now - self._last_seen >= LEAVE_SEC:
            self.reset()

    def _can_become_occupied(self) -> bool:
        """비어 있음 → 점유로 바꿔도 되는지 마지막으로 확인하는 자리.

        지금은 항상 True다 (YOLO 결과만으로 판단).
        6번(VLM)에서 "적외선 화면이면 VLM에게 한 번 확인"을 붙일 때 여기를 바꾼다.
        움직이지 않는 물건(의자에 걸린 옷 등)은 5초 조건을 그대로 통과하므로,
        점유로 바뀌는 이 순간이 오판을 거를 수 있는 곳이기 때문이다.
        """
        return True
