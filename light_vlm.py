"""
[조명 판단 방법 B] VLM(Ollama)에게 사진을 보여 주고 강의실 조명이 켜져 있는지 묻는다.

이 강의실은 천장 조명이 화면에 직접 보이지 않으므로,
조명 기구 대신 '바닥·책상이 고르게 밝은지'를 보고 판단하라고 질문한다.
--exclude 로 지정한 영역(예: 복도 불빛)은 검게 가린 뒤 보낸다.

흐름 (정시마다, 고른 카메라마다):
  흑백(적외선) 사진 -> --ask-gray 가 없으면 VLM 없이 '꺼짐'으로 기록 (가정은 evaluate_light.py로 검증)
  컬러 사진         -> VLM에게 질문

준비:
  ollama pull qwen3-vl:4b

사용 예:
  python light_vlm.py /Volumes/T9/data 301
  python light_vlm.py /Volumes/T9/data 301 --cams cam1,cam2 --exclude cam1:0.6,0.3,1,1
  python light_vlm.py /Volumes/T9/data 301 --dates 2026-10-16

결과:
  results/light/vlm_301.csv
"""
import argparse
import base64
import csv
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

from common import hourly_photos, is_grayscale, parse_exclude

PROMPT = (
    "이 사진은 대학교 컴퓨터실(강의실) 안의 CCTV 사진이다.\n"
    "천장 조명 자체는 사진에 보이지 않는다.\n"
    "방이 어떻게 밝혀져 있는지를 보고 천장 조명이 켜져 있는지 판단하라.\n"
    "- 조명이 켜져 있으면 바닥과 책상이 고르고 밝게 비춰지고, 바닥에 반사광이 보일 수 있다.\n"
    "- 조명이 꺼져 있으면 방이 어둡고 칙칙하며, 모니터나 방 밖에서 들어오는 빛으로 일부만 밝다.\n"
    'JSON으로만 답하라: {"lights_on": true 또는 false, "reason": "짧은 이유"}'
)


def _post(host, body, timeout):
    req = urllib.request.Request(f"{host}/api/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read()).get("response", "")


MASK_NOTE = "검은 영역은 일부러 가린 것이니 무시하라.\n"


def ask(host, model, img, timeout, masked=False):
    """(켜짐/꺼짐/해석불가, 이유, 걸린 초). masked=True면 가린 영역 안내를 질문에 붙인다"""
    img = img.convert("RGB")
    img.thumbnail((1024, 1024))  # 크면 느려서 줄임 (방 전체 밝기는 작아도 보임)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    body = {"model": model, "prompt": (MASK_NOTE + PROMPT) if masked else PROMPT, "images": [base64.b64encode(buf.getvalue()).decode()],
            "stream": False, "format": "json",
            "options": {"temperature": 0},  # 같은 사진엔 같은 답
            "think": False}                 # 생각 모드 끄기 (qwen3 계열이 빈 답을 내는 문제 방지)
    t0 = time.time()
    try:
        text = _post(host, body, timeout)
    except urllib.error.HTTPError as e:
        if e.code == 400 and "think" in e.read().decode(errors="ignore").lower():
            body.pop("think")  # 생각 모드 옵션을 모르는 모델이면 빼고 다시
            text = _post(host, body, timeout)
        else:
            raise
    sec = round(time.time() - t0, 2)
    if not text.strip():
        return "해석불가", "(빈 응답)", sec
    try:
        ans = json.loads(text)
        on = ans.get("lights_on")
        if isinstance(on, str):
            on = on.strip().lower() in ("true", "yes")
        return ("켜짐" if on else "꺼짐"), str(ans.get("reason", "")), sec
    except (json.JSONDecodeError, AttributeError):
        return "해석불가", text[:100], sec


def mask(img, rects):
    """제외 영역을 검게 칠한다"""
    if not rects:
        return img
    img = img.convert("RGB")
    d = ImageDraw.Draw(img)
    w, h = img.size
    for x1, y1, x2, y2 in rects:
        d.rectangle([x1 * w, y1 * h, x2 * w, y2 * h], fill=(0, 0, 0))
    return img


def main():
    ap = argparse.ArgumentParser(description="VLM으로 조명 켜짐/꺼짐 판단")
    ap.add_argument("root", help="예: /Volumes/T9/data")
    ap.add_argument("room", help="예: 301")
    ap.add_argument("--cams", default="cam1", help="쉼표로 구분, 예: cam1,cam2")
    ap.add_argument("--dates", default="", help="쉼표로 구분. 비우면 전체 날짜")
    ap.add_argument("--exclude", nargs="*", default=[], help="예: cam1:0.6,0.3,1,1")
    ap.add_argument("--ask-gray", action="store_true", help="흑백 사진도 VLM에게 묻기")
    ap.add_argument("--vlm", default="qwen3-vl:4b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--timeout", type=int, default=180, help="한 장당 최대 대기 초")
    ap.add_argument("-o", "--out", default="results/light")
    args = ap.parse_args()

    cams = [c.strip() for c in args.cams.split(",") if c.strip()]
    dates = {d.strip() for d in args.dates.split(",") if d.strip()} or None
    rects = parse_exclude(args.exclude)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    items = hourly_photos(args.root, args.room, cams=cams, dates=dates)
    print(f"{len(items)}장 -> VLM({args.vlm})")
    rows = []
    for n, (date, hhmm, cam, p) in enumerate(items, 1):
        try:
            img = Image.open(p)
            if is_grayscale(img) and not args.ask_gray:
                ans, reason, sec = "꺼짐", "흑백(적외선) 사진 -> VLM 생략", 0.0
            else:
                ans, reason, sec = ask(args.host, args.vlm, mask(img, rects.get(cam, [])), args.timeout,
                                       masked=bool(rects.get(cam)))
        except Exception as e:  # Ollama가 꺼져 있거나 시간 초과
            ans, reason, sec = "에러", str(e)[:100], ""
        print(f"  [{n}/{len(items)}] {cam} {date} {hhmm} -> {ans} ({sec}s)")
        rows.append({"date": date, "time": hhmm, "cam": cam, "vlm_답": ans,
                     "vlm_이유": reason, "vlm_초": sec, "file": str(p)})

    path = out / f"vlm_{args.room}.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["date"])
        w.writeheader()
        w.writerows(rows)

    asked = [r for r in rows if isinstance(r["vlm_초"], float) and r["vlm_초"] > 0]
    print(f"\n총 {len(rows)}장 / VLM에게 물어본 것 {len(asked)}장")
    if asked:
        print(f"평균 응답 시간: {sum(r['vlm_초'] for r in asked) / len(asked):.1f}초")
    print(f"CSV: {path}")


if __name__ == "__main__":
    main()
