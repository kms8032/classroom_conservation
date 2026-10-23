"""
[사람 판정] 사진에서 사람을 찾고, 기존 좌석 영역(ROI)으로 좌석별 점유를 판정한다.

흐름 (같은 시각의 카메라 4대 사진마다):
  1) YOLO-pose로 사람 검출 (imgsz 1280: 640보다 먼 좌석 검출률이 높았음)
  2) 신뢰도로 나누기
       conf >= 0.5         -> 사람으로 확정 (YOLO)
       0.25 <= conf < 0.5  -> 애매함 -> VLM에게 빨간 상자 안이 사람인지 물어봄
       conf < 0.25         -> 버림 (YOLO가 애초에 돌려주지 않음)
     [v2] conf >= 0.5 여도 아래 중 하나에 걸리면 VLM에게 한 번 더 물어봄 (물건 오판 줄이기)
       적외선 : 사진이 흑백(적외선) = 조명이 꺼진 어두운 시간          (밤에 걸린 겉옷)
       고정   : 같은 카메라에서 직전 --fixed-min 분 동안 거의 같은 박스가 계속 있음 (안 움직이는 물건)
       아래끝 : 박스가 사진 맨 아래에 닿아 있고 엉덩이가 안 보임       (카메라 앞 의자 등받이)
                기본은 꺼 둠(--edge-px 10 으로 켬). v1 결과로 미리 따져 보니 의자 등받이 오판은
                전부 밤(적외선)이라 '적외선'으로 이미 잡히고, 낮에는 카메라 앞 실제 사람만 14번 걸렸음.
       -> 버리는 조건이 아니라 VLM에게 넘기는 조건이라, 실제 사람을 놓칠 위험은 낮다.
       -> --no-vlm 이면 판정은 YOLO 그대로 두고 boxes CSV '검사' 열에 이유만 남긴다.
  3) 사람의 기준점이 어느 좌석 영역 안에 있는지로 좌석 배정
       기준점 = 엉덩이 keypoint 2개의 가운데 (안 보이면 박스 중심 = 예전 시스템 방식)
  4) 카메라 합치기: 한 좌석이 두 카메라에 보이면 한 대라도 점유면 점유

좌석 영역: classrooms.json 의 seat_lines (예전 시스템에서 그린 것)
  좌석마다 카메라 사진 좌표 4점 = 앞줄 선 2점 + 뒷줄 선 2점, 네 점을 감싸는 영역이 좌석.
  좌표는 1920x1080 사진 기준. 사진 크기가 다르면 비율대로 맞춰서 쓴다.

앉음/서 있음은 구분하지 않는다. 좌석 영역 안에 사람이 있으면 점유.

준비:
  pip install ultralytics
  ollama pull qwen3-vl:4b-instruct   # [v3] qwen3-vl:4b 는 thinking 버전 (생각을 못 꺼서 한 번에 20~120초)

[v3] VLM 질문 바꿈 (vlm_prompt_test.py 실험으로 결정):
  프롬프트 D: 머리가 보이는지 먼저 묻고, 머리가 없으면 '아님'. 흑백 사진의 옷·의자 주의 안내
  박스 주변을 넓게(1.2배) 잘라 의자까지 보이게 하고, 긴 쪽 512px로 줄여서 보냄
[v3] VLM은 좌석(ROI) 안의 박스에만 물어봄 (YOLO는 그대로 사진 전체에 돌림)
  좌석 밖 + 확정(conf>=0.5) -> 그대로 '사람' (강의실 사람 있음/없음에는 씀)
  좌석 밖 + 애매함          -> 묻지 않고 '아님' (근거: 좌석밖)

사용 예:
  python person_judge.py /Volumes/T9/data 301                    # 매 정시 사진만 (정답지와 같은 기준)
  python person_judge.py /Volumes/T9/data 301 --truth labels/정답지_301.xlsx   # 정답지로 '사람 있음/없음' 채점
  python person_judge.py /Volumes/T9/data 301 --all              # 20분 간격 사진 전부 (사용률 분석용)
  python person_judge.py /Volumes/T9/data 301 --dates 2026-10-16 --draw   # 확인용 그림 저장
  python person_judge.py /Volumes/T9/data 301 --no-vlm           # VLM 없이 YOLO만 (애매한 건 '판정불가')

결과 (results/person_301/):
  seats_301.csv   시각 x 좌석마다 한 줄: 점유/빈자리/판정불가, 근거, 본 카메라
  boxes_301.csv   검출된 사람 박스마다 한 줄: 신뢰도, 기준점, 배정 좌석, VLM 답
                  -> 나중에 YOLO 튜닝용 자동 라벨로도 쓴다
  images/         --draw 를 주면 좌석 영역과 박스를 그린 확인용 사진
"""
import argparse
import base64
import csv
import json
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import date as Date
from pathlib import Path

import cv2
import numpy as np

from common import CAMS, GRAY_SAT, hourly_photos, list_photos, read_truth

ROI_W, ROI_H = 1920, 1080  # seat_lines를 그린 사진 크기
HIP_L, HIP_R = 11, 12      # COCO keypoint 번호: 왼쪽/오른쪽 엉덩이
KPT_MIN = 0.5              # keypoint 신뢰도가 이보다 낮으면 안 보이는 것으로 봄

VLM_PROMPT = (
    "이 사진은 대학교 강의실 CCTV 사진의 일부다.\n"
    "빨간 상자 안에 실제 사람이 있는지만 판단하라. 빨간 상자 바깥은 무시하라.\n"
    "- 의자에 걸린 옷, 가방, 인형, 모니터 화면 속 사람은 사람이 아니다.\n"
    "- 몸의 일부만 보이거나 가려져 있어도 실제 사람이면 사람이다.\n"
    'JSON으로만 답하라: {"person": true 또는 false, "reason": "짧은 이유"}'
)  # v1, v2에서 쓴 프롬프트 (vlm_prompt_test.py 의 A). 흑백 사진의 겉옷을 '사람'이라고 함

# [v3] vlm_prompt_test.py 의 D (넓게완화). instruct 모델로 38개 중 37개, 물건 28/28 '아님', 시간 초과 0
VLM_PROMPT_V3 = (
    "이 사진은 대학교 강의실 CCTV 사진의 일부다. 밤에는 흑백(적외선) 사진일 수 있다.\n"
    "흑백 사진에서는 의자 등받이에 걸린 어두운 옷, 가방, 인형, 의자 자체가 사람처럼 보일 수 있으니 주의하라.\n"
    "빨간 상자 안만 보고 아래 순서로 판단하라. 빨간 상자 바깥은 무시하라.\n"
    "1) 빨간 상자 안에 사람의 머리(얼굴, 뒤통수, 머리카락)가 보이는가?\n"
    "2) 머리 아래로 어깨나 팔 중 하나라도 이어져 보이는가? 책상이나 모니터에 몸 일부가 가려진 것은 괜찮다.\n"
    "옷이나 물건만 있고 머리가 없으면 사람이 아니다.\n"
    'JSON으로만 답하라: {"head": true 또는 false, "person": true 또는 false, "reason": "짧은 이유"}'
)
CROP_MARGIN = 1.2  # [v3] 박스 주변을 넓게 잘라 의자 전체가 보이게 (옷이 의자에 걸린 맥락)


# ---------------------------------------------------------------- 좌석 영역

def load_seats(path, room):
    """classrooms.json -> {cam: {좌석번호: 좌석 영역(다각형)}}
    강의실은 이름에 room(예: '301')이 들어간 것을 찾는다."""
    data = json.load(open(path, encoding="utf-8"))
    rooms = data["classrooms"] if isinstance(data, dict) else data
    found = [c for c in rooms if room in str(c.get("name", ""))]
    if not found:
        names = [c.get("name") for c in rooms]
        raise SystemExit(f"{path}에 '{room}' 강의실 좌석 영역이 없음 (있는 강의실: {names})")

    seats = {}
    for cam in found[0]["cameras"]:
        num = re.findall(r"\d+", cam.get("name", ""))  # 'CCTV 1' -> cam1
        if not num:
            continue
        polys = {}
        for sid, pts in (cam.get("seat_lines") or {}).items():
            if isinstance(pts, dict):
                pts = pts.get("points") or []
            if len(pts) >= 4:
                quad = np.array(pts[:4], dtype=np.float32)
                polys[str(sid)] = cv2.convexHull(quad)  # 네 점을 감싸는 영역
        seats[f"cam{num[0]}"] = polys
    return seats


def scale_polys(polys, w, h):
    """사진 크기가 1920x1080이 아니면 좌석 영역을 비율대로 맞춘다"""
    if (w, h) == (ROI_W, ROI_H):
        return polys
    s = np.array([w / ROI_W, h / ROI_H], dtype=np.float32)
    return {sid: (p * s).astype(np.float32) for sid, p in polys.items()}


def find_seat(polys, x, y):
    """(x, y)가 들어 있는 좌석. 경계에서 두 좌석에 걸리면 영역 안쪽으로 더 깊은 좌석"""
    best, best_d = None, -1.0
    for sid, poly in polys.items():
        d = cv2.pointPolygonTest(poly, (float(x), float(y)), True)  # 안쪽이면 +, 경계 0, 밖 -
        if d >= 0 and d > best_d:
            best, best_d = sid, d
    return best


# ---------------------------------------------------------------- 사람 검출

def detect(model, img, imgsz, conf_low, device=""):
    """YOLO-pose -> [{'box', 'conf', 'point', 'point_type'}, ...]"""
    r = model(img, imgsz=imgsz, conf=conf_low, classes=[0], verbose=False, device=device or None)[0]
    people = []
    if r.boxes is None:
        return people
    kxy = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else None
    kcf = r.keypoints.conf.cpu().numpy() if (r.keypoints is not None and r.keypoints.conf is not None) else None
    for i, b in enumerate(r.boxes):
        x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
        point, kind = ((x1 + x2) / 2, (y1 + y2) / 2), "박스중심"
        if kxy is not None and kcf is not None:
            hips = [kxy[i][k] for k in (HIP_L, HIP_R) if kcf[i][k] >= KPT_MIN]
            if hips:
                point, kind = tuple(np.mean(hips, axis=0).tolist()), "엉덩이"
        people.append({"box": (int(x1), int(y1), int(x2), int(y2)),
                       "conf": round(float(b.conf.item()), 3),
                       "point": (int(point[0]), int(point[1])), "point_type": kind})
    return people


# ---------------------------------------------------------------- [v2] 한 번 더 확인할 조건

def is_infrared(img):
    """흑백(적외선) 사진이면 True = 조명이 꺼져 카메라가 적외선으로 바뀜 (common.is_grayscale과 같은 기준)"""
    small = cv2.resize(img, (320, 180))
    return float(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1].mean()) < GRAY_SAT


def iou(a, b):
    """두 박스가 겹치는 정도 (0~1). 1이면 완전히 같은 박스"""
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def to_abs_min(date, hhmm):
    """('2026-10-17', '21:00') -> 날짜가 바뀌어도 이어지는 분 단위 시각"""
    return Date.fromisoformat(date).toordinal() * 1440 + int(hhmm[:2]) * 60 + int(hhmm[3:5])


def check_reasons(pp, infrared, past, img_h, fixed_iou, edge_px):
    """확정(conf>=0.5) 박스라도 VLM에게 다시 물어볼 이유 목록.
    past: 같은 카메라의 직전 몇 장 [[박스, ...], ...] (--fixed-min 분 안의 사진만)"""
    reasons = []
    if infrared:
        reasons.append("적외선")
    if len(past) >= 2 and all(any(iou(pp["box"], b) >= fixed_iou for b in boxes) for boxes in past):
        reasons.append("고정")  # 직전 사진 모두에 거의 같은 박스 = 몇 시간 동안 안 움직임
    if edge_px and pp["box"][3] >= img_h - edge_px and pp["point_type"] != "엉덩이":
        reasons.append("아래끝")
    return reasons


# ---------------------------------------------------------------- VLM 확인

def crop_with_box(img, box, margin=0.4):
    """박스 주변을 넉넉히 잘라서 원래 박스 자리에 빨간 상자를 그린다"""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    mx, my = int((x2 - x1) * margin) + 20, int((y2 - y1) * margin) + 20
    cx1, cy1, cx2, cy2 = max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)
    crop = img[cy1:cy2, cx1:cx2].copy()
    cv2.rectangle(crop, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), (0, 0, 255), 3)
    long_side = max(crop.shape[:2])
    if long_side > 512:  # 큰 사진은 VLM이 느려져서 긴 쪽을 512로 줄임
        s = 512 / long_side
        crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)))
    return crop


def _post(host, body, timeout):
    req = urllib.request.Request(f"{host}/api/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read()).get("response", "")


def ask_vlm(host, model, crop, timeout):
    """(사람/아님/실패, 이유, 걸린 초)"""
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    body = {"model": model, "prompt": VLM_PROMPT_V3, "images": [base64.b64encode(buf).decode()],
            "stream": False, "format": "json",
            "options": {"temperature": 0},  # 같은 사진엔 같은 답
            "think": False}                 # 생각 모드 끄기 (qwen3 계열 빈 답 방지)
    t0 = time.time()
    try:
        try:
            text = _post(host, body, timeout)
        except urllib.error.HTTPError as e:
            if e.code == 400 and "think" in e.read().decode(errors="ignore").lower():
                body.pop("think")  # 생각 모드 옵션을 모르는 모델이면 빼고 다시
                text = _post(host, body, timeout)
            else:
                raise
    except Exception as e:  # Ollama가 꺼져 있거나 시간 초과
        return "실패", str(e)[:80], round(time.time() - t0, 2)
    sec = round(time.time() - t0, 2)
    try:
        ans = json.loads(text)

        def yes(v):
            return v.strip().lower() in ("true", "yes") if isinstance(v, str) else bool(v)

        p = yes(ans.get("person"))
        if "head" in ans and not yes(ans["head"]):  # [v3] 머리가 없다고 했으면 사람이라고 해도 아님
            p = False
        return ("사람" if p else "아님"), str(ans.get("reason", "")), sec
    except (json.JSONDecodeError, AttributeError):
        return "실패", text[:80] or "(빈 응답)", sec


# ---------------------------------------------------------------- 사진 고르기

def photo_groups(root, room, cams, dates, all_photos, slot_min):
    """{(날짜, 'HH:MM'): {cam: 경로}}  같은 시각에 찍힌 카메라 사진끼리 묶는다"""
    groups = defaultdict(dict)
    if not all_photos:
        for date, hhmm, cam, p in hourly_photos(root, room, cams=cams, dates=dates):
            groups[(date, hhmm)][cam] = p
        return groups
    room_dir = Path(root) / room
    for cam in cams:
        for day in sorted((room_dir / cam).glob("20*-*-*")):
            if dates and day.name not in dates:
                continue
            for minute, p in list_photos(day).items():
                m = int(round(minute / slot_min) * slot_min) % (24 * 60)  # 카메라마다 몇 초씩 다른 걸 같은 칸으로
                groups[(day.name, f"{m // 60:02d}:{m % 60:02d}")].setdefault(cam, p)
    return groups


# ---------------------------------------------------------------- 그림

def draw(img, polys, people, occupied, path):
    out = img.copy()
    for sid, poly in polys.items():
        color = (163, 181, 43) if sid in occupied else (139, 116, 100)  # 점유 / 빈자리
        cv2.polylines(out, [poly.astype(np.int32)], True, color, 2)
        cx, cy = poly.reshape(-1, 2).mean(axis=0).astype(int)
        cv2.putText(out, sid, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    for pp in people:
        color = {"사람": (0, 200, 0), "아님": (0, 0, 255)}.get(pp["final"], (0, 170, 255))
        x1, y1, x2, y2 = pp["box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.circle(out, pp["point"], 6, color, -1)
        cv2.putText(out, f'{pp["conf"]:.2f} {pp["decided_by"]}', (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, 85])


# ---------------------------------------------------------------- 실행

def main():
    ap = argparse.ArgumentParser(description="YOLO-pose + VLM으로 좌석별 점유 판정")
    ap.add_argument("root", help="예: /Volumes/T9/data")
    ap.add_argument("room", help="예: 301")
    ap.add_argument("--device", default="", help="mps(맥 GPU), cpu, 0(서버 GPU). 비우면 자동")
    ap.add_argument("--cams", default=",".join(CAMS), help="쉼표로 구분, 예: cam1,cam2")
    ap.add_argument("--dates", default="", help="쉼표로 구분. 비우면 전체 날짜")
    ap.add_argument("--all", action="store_true", help="정시만이 아니라 모든 사진 (사용률 분석용)")
    ap.add_argument("--slot", type=int, default=5, help="--all일 때 몇 분 단위로 카메라 사진을 묶을지")
    ap.add_argument("--classrooms", default="classrooms.json", help="좌석 영역 파일")
    ap.add_argument("--model", default="yolo26x-pose.pt", help="YOLO-pose 가중치")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf-high", type=float, default=0.5, help="이 이상이면 사람으로 확정")
    ap.add_argument("--conf-low", type=float, default=0.25, help="이 미만은 버림")
    ap.add_argument("--no-vlm", action="store_true", help="VLM을 쓰지 않음 (애매한 건 판정불가)")
    ap.add_argument("--vlm", default="qwen3-vl:4b-instruct",
                    help="[v3] instruct 버전. qwen3-vl:4b 는 thinking 버전이라 think=false 를 무시하고 느림")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--timeout", type=int, default=120, help="VLM 한 번 최대 대기 초")
    ap.add_argument("--no-check", action="store_true", help="[v2] 확정 박스 재확인(적외선/고정/아래끝)을 끔 = v1과 같음")
    ap.add_argument("--fixed-min", type=int, default=120, help="[v2] 이 시간(분) 동안 같은 박스면 '고정'")
    ap.add_argument("--fixed-iou", type=float, default=0.85, help="[v2] '같은 박스'로 볼 겹침 정도")
    ap.add_argument("--edge-px", type=int, default=0,
                    help="[v2] 사진 아래 끝에서 이 픽셀 안이면 '아래끝' (기본 0=끔: 낮에 카메라 앞 실제 사람만 걸려서)")
    ap.add_argument("--truth", help="정답지 엑셀. 주면 '사람 있음/없음'을 채점")
    ap.add_argument("--draw", action="store_true", help="확인용 그림 저장")
    ap.add_argument("-o", "--out", default="", help="기본 results/person_<room>")
    args = ap.parse_args()

    from ultralytics import YOLO  # 무거워서 인자 확인이 끝난 뒤 불러옴

    cams = [c.strip() for c in args.cams.split(",") if c.strip()]
    dates = {d.strip() for d in args.dates.split(",") if d.strip()} or None
    out = Path(args.out or f"results/person_{args.room}")
    out.mkdir(parents=True, exist_ok=True)

    seat_polys = load_seats(args.classrooms, args.room)
    all_seats = sorted({s for c in cams for s in seat_polys.get(c, {})}, key=lambda s: (len(s), s))
    seat_cams = {s: [c for c in cams if s in seat_polys.get(c, {})] for s in all_seats}
    no_roi = [c for c in cams if not seat_polys.get(c)]
    print(f"{args.room}호 좌석 {len(all_seats)}개 (좌석 영역 없는 카메라: {no_roi or '없음'})")

    groups = photo_groups(args.root, args.room, cams, dates, args.all, args.slot)
    print(f"시각 {len(groups)}개 -> YOLO({args.model}, imgsz {args.imgsz})"
          + ("" if args.no_vlm else f" + VLM({args.vlm})"))
    model = YOLO(args.model)

    seat_rows, box_rows, room_pred = [], [], {}
    warned = set()
    n_vlm = vlm_sec = n_skip = 0
    history = defaultdict(list)    # [v2] cam -> [(분 단위 시각, [박스, ...]), ...]
    n_check = defaultdict(int)     # [v2] 재확인 이유별 개수
    for n, (key, photos) in enumerate(sorted(groups.items()), 1):
        date, hhmm = key
        now = to_abs_min(date, hhmm)
        hits = defaultdict(list)  # 좌석 -> [(cam, 사람/판정불가, 근거)]
        anyone = unsure = False
        for cam in cams:
            p = photos.get(cam)
            if p is None:
                continue
            img = cv2.imread(str(p))
            if img is None:
                print(f"  읽기 실패: {p}")
                continue
            h, w = img.shape[:2]
            if (w, h) != (ROI_W, ROI_H) and cam not in warned:
                print(f"  주의: {cam} 사진이 {w}x{h} -> 좌석 영역을 비율로 맞춰서 사용")
                warned.add(cam)
            polys = scale_polys(seat_polys.get(cam, {}), w, h)

            people = detect(model, img, args.imgsz, args.conf_low, args.device)
            infrared = is_infrared(img)
            past = [boxes for t, boxes in history[cam] if now - args.fixed_min <= t < now]
            history[cam] = [(t, b) for t, b in history[cam] if t >= now - args.fixed_min]
            history[cam].append((now, [pp["box"] for pp in people]))
            occupied = set()
            for pp in people:
                pp["vlm"], pp["vlm_reason"], pp["vlm_sec"] = "", "", ""
                pp["check"] = []
                # [v3] 좌석을 먼저 찾고, VLM은 좌석(ROI) 안의 박스에만 물어본다 (v2에서 VLM 질문의 37%가 좌석 밖)
                pp["seat"] = find_seat(polys, *pp["point"]) or ""
                in_seat = bool(pp["seat"])
                if pp["conf"] >= args.conf_high and in_seat and not args.no_check:
                    pp["check"] = check_reasons(pp, infrared, past, h, args.fixed_iou, args.edge_px)
                    for r in pp["check"]:
                        n_check[r] += 1
                if pp["conf"] >= args.conf_high and (not pp["check"] or args.no_vlm):
                    pp["final"], pp["decided_by"] = "사람", "YOLO"  # 좌석 밖이라도 확실하면 '강의실 사람 있음'에 씀
                elif not in_seat:
                    pp["final"], pp["decided_by"] = "아님", "좌석밖"  # 좌석 밖의 애매한 박스는 묻지 않고 버림
                    n_skip += 1
                elif args.no_vlm:
                    pp["final"], pp["decided_by"] = "판정불가", "VLM안씀"
                else:
                    ans, reason, sec = ask_vlm(args.host, args.vlm, crop_with_box(img, pp["box"], CROP_MARGIN),
                                             args.timeout)
                    pp["vlm"], pp["vlm_reason"], pp["vlm_sec"] = ans, reason, sec
                    pp["final"] = {"사람": "사람", "아님": "아님"}.get(ans, "판정불가")
                    pp["decided_by"] = "VLM재확인" if pp["check"] else "VLM"
                    n_vlm += 1
                    vlm_sec += sec
                if pp["final"] == "사람":
                    anyone = True
                    if pp["seat"]:
                        occupied.add(pp["seat"])
                elif pp["final"] == "판정불가":
                    unsure = True
                if pp["seat"] and pp["final"] != "아님":
                    hits[pp["seat"]].append((cam, pp["final"], pp["decided_by"]))
                box_rows.append({"date": date, "time": hhmm, "cam": cam, "conf": pp["conf"],
                                 "box": "%d,%d,%d,%d" % pp["box"], "point": "%d,%d" % pp["point"],
                                 "point_type": pp["point_type"], "seat": pp["seat"],
                                 "판정": pp["final"], "근거": pp["decided_by"],
                                 "검사": "+".join(pp["check"]), "적외선": int(infrared), "vlm_답": pp["vlm"],
                                 "vlm_이유": pp["vlm_reason"], "vlm_초": pp["vlm_sec"], "file": str(p)})
            if args.draw:
                draw(img, polys, people, occupied, out / "images" / f"{date}_{hhmm.replace(':', '')}_{cam}.jpg")

        # 카메라 합치기: 한 대라도 확실히 점유면 점유, 애매한 것만 있으면 판정불가
        n_occ = 0
        for sid in all_seats:
            hs = hits.get(sid, [])
            sure = [h for h in hs if h[1] == "사람"]
            if sure:
                status, basis = "점유", "+".join(sorted({h[2] for h in sure}))
                n_occ += 1
            elif hs:
                status, basis = "판정불가", "+".join(sorted({h[2] for h in hs}))
            else:
                status, basis = "빈자리", ""
            seen = [c for c in seat_cams[sid] if c in photos]
            seat_rows.append({"date": date, "time": hhmm, "seat": sid, "상태": status, "근거": basis,
                              "잡힌_카메라": ",".join(sorted({h[0] for h in (sure or hs)})),
                              "볼수있는_카메라": ",".join(seen)})
        room_pred[key] = "있음" if anyone else ("모름" if unsure else "없음")
        print(f"  [{n}/{len(groups)}] {date} {hhmm} 카메라 {len(photos)}대 -> 점유 {n_occ}/{len(all_seats)}석, "
              f"사람 {room_pred[key]}")

    for name, rows in (("seats", seat_rows), ("boxes", box_rows)):
        path = out / f"{name}_{args.room}.csv"
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["date"])
            w.writeheader()
            w.writerows(rows)
        print(f"CSV: {path} ({len(rows)}줄)")
    if n_check:
        print("[v2] 재확인 조건에 걸린 확정 박스: " + ", ".join(f"{k} {v}개" for k, v in n_check.items())
              + (" (--no-vlm 이라 판정은 YOLO 그대로)" if args.no_vlm else ""))
    if n_skip:
        print(f"[v3] 좌석 밖이라 VLM에게 묻지 않고 버린 애매한 박스 {n_skip}개")
    if n_vlm:
        print(f"VLM에게 물어본 박스 {n_vlm}개, 평균 {vlm_sec / n_vlm:.1f}초")

    if args.truth:
        score_room(read_truth(args.truth, "사람"), room_pred)


def score_room(truth, pred):
    """강의실 단위 '사람 있음/없음' 채점 (정답지의 사람 열은 강의실 전체 기준)"""
    pairs = [(truth[k], pred[k]) for k in truth if k in pred]
    if not pairs:
        print("\n채점할 시각이 없음 (정답지와 사진 시각이 겹치지 않음)")
        return
    judged = [(t, p) for t, p in pairs if p != "모름"]
    right = sum(t == p for t, p in judged)
    print(f"\n== 강의실 사람 있음/없음 채점 ==  정답과 겹치는 시각 {len(pairs)}개")
    if judged:
        print(f"  정확도 {right / len(judged) * 100:.1f}% ({right}/{len(judged)})")
    for t, p, label in (("있음", "없음", "사람 있는데 없다고 함"), ("없음", "있음", "사람 없는데 있다고 함")):
        c = sum(1 for a, b in judged if (a, b) == (t, p))
        if c:
            print(f"  {label}: {c}")
    unsure = len(pairs) - len(judged)
    if unsure:
        print(f"  판정불가(모름): {unsure}")
    wrong = [k for k in truth if k in pred and pred[k] != "모름" and pred[k] != truth[k]]
    if wrong:
        print("  틀린 시각:", ", ".join(f"{d} {t}" for d, t in sorted(wrong)))


if __name__ == "__main__":
    main()