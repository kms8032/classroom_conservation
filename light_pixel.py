"""
[조명 판단 방법 A] 사진의 픽셀 값으로 조명 상태를 판단하기 위한 지표를 계산한다.

이 강의실은 천장 조명이 화면에 직접 보이지 않고 햇빛 영향도 거의 없어서,
'강의실 안이 고르게 밝은지'를 여러 지표로 잰 뒤 정답지로 가장 잘 맞는 지표를 고른다.
(어느 지표·기준값이 좋은지는 evaluate_light.py가 정한다)

지표 (사진마다):
  brightness  평균 밝기 (0~255)
  contrast    밝기의 표준편차 - 조명이 꺼지면 어둡고 칙칙해져 값이 바뀜
  dark_pct    어두운 픽셀(밝기 60 미만) 비율 %
  bright_pct  밝은 픽셀(밝기 200 이상) 비율 %
  saturation  평균 채도 - 흑백(적외선) 판별용
  grayscale   흑백(적외선) 사진이면 1

복도 불빛처럼 강의실 조명과 상관없이 밝아지는 영역은 --exclude 로 빼고 계산할 수 있다.
  좌표는 사진 크기에 대한 비율(0~1): 카메라:왼쪽x,위y,오른쪽x,아래y

사용 예:
  python light_pixel.py /Volumes/T9/data 301
  python light_pixel.py /Volumes/T9/data 301 --exclude cam1:0.6,0.3,1,1

결과:
  results/light/pixel_301.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from common import hourly_photos, keep_mask, parse_exclude, GRAY_SAT

DARK, BRIGHT = 60, 200


def measure(path, rects):
    img = Image.open(path).convert("RGB")
    img.thumbnail((480, 480))  # 속도를 위해 작게 줄여서 계산
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    sat = np.asarray(img.convert("HSV"), dtype=np.float32)[:, :, 1]
    keep = keep_mask(*gray.shape, rects)
    g = gray[keep]
    return {
        "brightness": round(float(g.mean()), 1),
        "contrast": round(float(g.std()), 1),
        "dark_pct": round(float((g < DARK).mean() * 100), 2),
        "bright_pct": round(float((g >= BRIGHT).mean() * 100), 2),
        "saturation": round(float(sat.mean()), 1),  # 흑백 판별은 사진 전체 기준
        "grayscale": int(sat.mean() < GRAY_SAT),
    }


def main():
    ap = argparse.ArgumentParser(description="조명 판단용 픽셀 지표 계산")
    ap.add_argument("root", help="예: /Volumes/T9/data")
    ap.add_argument("room", help="예: 301")
    ap.add_argument("--exclude", nargs="*", default=[], help="예: cam1:0.6,0.3,1,1")
    ap.add_argument("-o", "--out", default="results/light")
    args = ap.parse_args()

    rects = parse_exclude(args.exclude)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    items = hourly_photos(args.root, args.room)
    print(f"사진 {len(items)}장 계산 중...")
    for date, hhmm, cam, p in items:
        try:
            m = measure(p, rects.get(cam, []))
        except Exception as e:  # 깨진 사진 등
            print(f"  건너뜀: {p} ({e})")
            continue
        rows.append({"date": date, "time": hhmm, "cam": cam, **m, "file": str(p)})

    path = out / f"pixel_{args.room}.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # 엑셀에서 한글 안 깨지게
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print("\n== 카메라별 요약 ==")
    for cam in sorted({r["cam"] for r in rows}):
        cr = [r for r in rows if r["cam"] == cam]
        b = [r["brightness"] for r in cr]
        print(f"  {cam}: {len(cr)}장, 흑백 {sum(r['grayscale'] for r in cr)}장, "
              f"밝기 {min(b):.0f}~{max(b):.0f}")
    if rects:
        print(f"  제외 영역: {rects}")
    print(f"\nCSV: {path}")


if __name__ == "__main__":
    main()
