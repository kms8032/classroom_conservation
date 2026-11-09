"""
점-다각형 ROI 판단 테스트.

카메라·YOLO 없이 좌표 숫자만 넣어서 "안/밖"이 맞게 나오는지 확인한다.

실행 (v2 폴더에서):
    python -m pytest
"""

from geometry.roi import bbox_center, is_point_in_polygon

# 가로·세로 100인 사각형. 꼭짓점을 시계 방향으로 적었다.
SQUARE = [[0, 0], [100, 0], [100, 100], [0, 100]]

# config/classroom_301.json → CCTV 3 → seat_lines "7" (JSON에 적힌 순서 그대로)
#   [0] [1] = 위쪽 선의 왼쪽 끝·오른쪽 끝
#   [2] [3] = 아래쪽 선의 왼쪽 끝·오른쪽 끝
SEAT_7_RAW = [[1245, 466], [1817, 608], [1059, 735], [1808, 949]]

# 테두리를 따라 한 바퀴 돌도록 순서를 바꾼 것: 위 왼쪽 → 위 오른쪽 → 아래 오른쪽 → 아래 왼쪽
# (다음 단계의 좌석 설정 로더가 이 일을 맡는다)
SEAT_7 = [SEAT_7_RAW[0], SEAT_7_RAW[1], SEAT_7_RAW[3], SEAT_7_RAW[2]]


# ──────────────────────────────────────────────
# 사각형
# ──────────────────────────────────────────────
def test_사각형_안():
    assert is_point_in_polygon(50, 50, SQUARE) is True


def test_사각형_밖():
    assert is_point_in_polygon(150, 50, SQUARE) is False


def test_사각형_경계는_안으로_봄():
    # 이웃 좌석과 나눠 쓰는 선 위의 사람을 놓치지 않기 위함
    assert is_point_in_polygon(100, 50, SQUARE) is True


def test_사각형_꼭짓점은_안으로_봄():
    assert is_point_in_polygon(0, 0, SQUARE) is True


def test_실수_좌표도_판단함():
    # YOLO 중심점은 소수점이 있다. 경계 바로 바깥 0.5픽셀은 밖이어야 한다
    assert is_point_in_polygon(99.5, 50.5, SQUARE) is True
    assert is_point_in_polygon(100.5, 50.5, SQUARE) is False


# ──────────────────────────────────────────────
# 기울어진 4점 다각형 (실제 좌석 ROI)
# ──────────────────────────────────────────────
def test_좌석_가운데는_안():
    # 네 꼭짓점의 평균 위치 = 좌석 띠 한가운데
    assert is_point_in_polygon(1482, 690, SEAT_7) is True


def test_좌석_오른쪽_부분도_안():
    # 위쪽 선(y≈640)과 아래쪽 선(y≈880) 사이, 오른쪽 끝(x≈1810) 안쪽
    assert is_point_in_polygon(1700, 750, SEAT_7) is True


def test_좌석_위쪽_선보다_위는_밖():
    # x=1500에서 위쪽 선은 y≈529 — 그보다 위(앞줄 좌석 쪽)
    assert is_point_in_polygon(1500, 400, SEAT_7) is False


def test_좌석_아래쪽_선보다_아래는_밖():
    # x=1500에서 아래쪽 선은 y≈861 — 그보다 아래(뒷줄 좌석 쪽)
    assert is_point_in_polygon(1500, 1000, SEAT_7) is False


def test_JSON_순서_그대로면_좌석_안인데도_밖으로_나옴():
    # 왜 로더에서 순서를 바꿔야 하는지 남겨 두는 테스트.
    # JSON 순서대로 이으면 선이 교차해 ⋈ 모양이 되어, 좌석 안쪽 일부가 "밖"이 된다.
    assert is_point_in_polygon(1700, 750, SEAT_7) is True
    assert is_point_in_polygon(1700, 750, SEAT_7_RAW) is False


# ──────────────────────────────────────────────
# bbox 중심점
# ──────────────────────────────────────────────
def test_bbox_중심점():
    assert bbox_center(10, 20, 30, 60) == (20.0, 40.0)


def test_bbox_중심점은_반올림하지_않음():
    # 폭 11, 높이 7 → 중심이 소수점 .5로 끝난다
    assert bbox_center(0, 0, 11, 7) == (5.5, 3.5)


def test_bbox_중심점으로_바로_판단():
    # 실제로 쓰는 흐름: bbox → 중심점 → ROI 판단
    cx, cy = bbox_center(40, 30, 60, 90)
    assert is_point_in_polygon(cx, cy, SQUARE) is True
