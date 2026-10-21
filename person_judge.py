"""
[사람 판정] 사진에서 사람을 찾고, 기존 좌석 영역(ROI)으로 좌석별 점유를 판정한다.

흐름 (같은 시각의 카메라 4대 사진마다):
  1) YOLO-pose로 사람 검출 (imgsz 1280: 640보다 먼 좌석 검출률이 높았음)
  2) 신뢰도로 나누기
       conf >= 0.5         -> 사람으로 확정 (YOLO)
       0.25 <= conf < 0.5  -> 애매함 -> VLM에게 빨간 상자 안이 사람인지 물어봄
       conf < 0.25         -> 버림 (YOLO가 애초에 돌려주지 않음)
  3) 사람의 기준점이 어느 좌석 영역 안에 있는지로 좌석 배정
       기준점 = 엉덩이 keypoint 2개의 가운데 (안 보이면 박스 중심 = 예전 시스템 방식)
  4) 카메라 합치기: 한 좌석이 두 카메라에 보이면 한 대라도 점유면 점유

좌석 영역: classrooms.json 의 seat_lines (예전 시스템에서 그린 것)
  좌석마다 카메라 사진 좌표 4점 = 앞줄 선 2점 + 뒷줄 선 2점, 네 점을 감싸는 영역이 좌석.
  좌표는 1920x1080 사진 기준. 사진 크기가 다르면 비율대로 맞춰서 쓴다.

앉음/서 있음은 구분하지 않는다. 좌석 영역 안에 사람이 있으면 점유.

준비:
  pip install ultralytics
  ollama pull qwen3-vl:4b

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
from pathlib import Path

import cv2
import numpy as np

from common import CAMS, hourly_photos, list_photos, read_truth

ROI_W, ROI_H = 1920, 1080  # seat_lines를 그린 사진 크기
HIP_L, HIP_R = 11, 12      # COCO keypoint 번호: 왼쪽/오른쪽 엉덩이
KPT_MIN = 0.5              # keypoint 신뢰도가 이보다 낮으면 안 보이는 것으로 봄

VLM_PROMPT = (
    "이 사진은 대학교 강의실 CCTV 사진의 일부다.\n"
    "빨간 상자 안에 실제 사람이 있는지만 판단하라. 빨간 상자 바깥은 무시하라.\n"
    "- 의자에 걸린 옷, 가방, 인형, 모니터 화면 속 사람은 사람이 아니다.\n"
    "- 몸의 일부만 보이거나 가려져 있어도 실제 사람이면 사람이다.\n"
    'JSON으로만 답하라: {"person": true 또는 false, "reason": "짧은 이유"}'
)


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

def detect(model, img, imgsz, conf_low):
    """YOLO-pose -> [{'box', 'conf', 'point', 'point_type'}, ...]"""
    r = model(img, imgsz=imgsz, conf=conf_low, classes=[0], verbose=False)[0]
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


# ---------------------------------------------------------------- VLM 확인

def crop_with_box(img, box, margin=0.4):
    """박스 주변을 넉넉히 잘라서 원래 박스 자리에 빨간 상자를 그린다"""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    mx, my = int((x2 - x1) * margin) + 20, int((y2 - y1) * margin) + 20
    cx1, cy1, cx2, cy2 = max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)
    crop = img[cy1:cy2, cx1:cx2].copy()
    cv2.rectangle(crop, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), (0, 0, 255), 3)
    return crop


def _post(host, body, timeout):
    req = urllib.request.Request(f"{host}/api/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read()).get("response", "")


def ask_vlm(host, model, crop, timeout):
    """(사람/아님/실패, 이유, 걸린 초)"""
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    body = {"model": model, "prompt": VLM_PROMPT, "images": [base64.b64encode(buf).decode()],
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
        p = ans.get("person")
        if isinstance(p, str):
            p = p.strip().lower() in ("true", "yes")
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
    ap.add_argument("--vlm", default="qwen3-vl:4b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--timeout", type=int, default=120, help="VLM 한 번 최대 대기 초")
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
    n_vlm = vlm_sec = 0
    for n, (key, photos) in enumerate(sorted(groups.items()), 1):
        date, hhmm = key
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

            people = detect(model, img, args.imgsz, args.conf_low)
            occupied = set()
            for pp in people:
                pp["vlm"], pp["vlm_reason"], pp["vlm_sec"] = "", "", ""
                if pp["conf"] >= args.conf_high:
                    pp["final"], pp["decided_by"] = "사람", "YOLO"
                elif args.no_vlm:
                    pp["final"], pp["decided_by"] = "판정불가", "VLM안씀"
                else:
                    ans, reason, sec = ask_vlm(args.host, args.vlm, crop_with_box(img, pp["box"]), args.timeout)
                    pp["vlm"], pp["vlm_reason"], pp["vlm_sec"] = ans, reason, sec
                    pp["final"] = {"사람": "사람", "아님": "아님"}.get(ans, "판정불가")
                    pp["decided_by"] = "VLM"
                    n_vlm += 1
                    vlm_sec += sec
                pp["seat"] = find_seat(polys, *pp["point"]) or ""
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
                                 "판정": pp["final"], "근거": pp["decided_by"], "vlm_답": pp["vlm"],
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