"""
점이 다각형 ROI 안에 있는지 판단하는 함수.

좌석 ROI(2번)와 교탁·복도 ROI(3번)가 같은 방식으로 판단하므로 한 곳에 둔다:

    YOLO bbox ──▶ bbox_center() ──▶ 중심점 (cx, cy) ──▶ is_point_in_polygon() ──▶ True / False

왜 cv2.pointPolygonTest인가? (CLAUDE.md "3. 교탁·복도 인원")
- YOLO는 전체 이미지에서 탐지하고, 다각형 판단은 탐지가 끝난 뒤에 한다.
- 이미지를 다각형 모양으로 잘라(마스킹) 탐지하면 경계에 걸친 사람이 잘려서 탐지가 안 된다.

다각형 점 순서에 주의:
- polygon은 테두리를 따라 한 바퀴 도는 순서(시계 또는 반시계)여야 한다.
- config의 seat_lines는 "위쪽 선 2점 + 아래쪽 선 2점" 순서로 적혀 있어서, 그대로 이으면
  선이 서로 교차하는 나비넥타이(⋈) 모양이 된다. 순서를 맞추는 일은 좌석 설정 로더가 맡는다.
  (이 함수는 받은 순서를 그대로 믿는다 — 교탁·복도처럼 오목한 ROI도 모양 그대로 판단하기 위함)

이 파일은 카메라·YOLO·Seat를 전혀 모른다. 숫자 좌표만 받는다.
"""

import cv2
import numpy as np


def bbox_center(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    """bbox 왼쪽 위 (x1, y1)·오른쪽 아래 (x2, y2) 좌표 → 중심점 (cx, cy).

    반올림하지 않고 실수 그대로 돌려준다.
    YOLO 좌표가 원래 실수이고, 반올림하면 ROI 경계 근처에서 판단이 1픽셀씩 달라질 수 있기 때문이다.
    """
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return cx, cy


def is_point_in_polygon(cx: float, cy: float, polygon: list[list[int]]) -> bool:
    """점 (cx, cy)가 다각형 안에 있으면 True. 경계 위의 점도 True.

    polygon은 config JSON 형식 그대로 받는다: [[x, y], [x, y], ...]
    cv2는 numpy 배열만 받으므로 변환은 여기서 한다 (호출하는 쪽이 numpy를 몰라도 되게).
    """
    # cv2.pointPolygonTest는 다각형을 int32 또는 float32 numpy 배열로 받는다
    contour = np.array(polygon, dtype=np.int32)

    # 세 번째 인자 False: 거리는 필요 없고 안(+1)·경계(0)·밖(-1)만 알면 된다
    # 점은 파이썬 float 튜플로 넘긴다 (numpy 실수나 int를 넘기면 cv2 버전에 따라 오류가 날 수 있음)
    result = cv2.pointPolygonTest(contour, (float(cx), float(cy)), False)

    # 경계 위(0)를 "안"으로 보는 이유:
    # - 이웃한 좌석 ROI는 같은 선을 나눠 쓴다. 경계를 "밖"으로 보면 선 위에 있는 사람이
    #   어느 좌석에서도 감지되지 않는다.
    # - 놓치는 것보다 양쪽 좌석에 다 잡히는 편이 낫다. 진짜로 앉은 좌석이 아니면
    #   5초 연속 감지 조건에서 대부분 걸러진다 (두 카메라 중 한쪽만 감지해도 감지로 보는 규칙과 같은 방향).
    return result >= 0
