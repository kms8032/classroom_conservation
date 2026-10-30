"""
좌석별 정답지(라벨링 시트)와 좌석 번호를 그린 확인용 이미지를 만드는 스크립트.

정답지_<room>.xlsx 의 '사람' 열은 강의실 전체 기준(있음/없음)이라 좌석별 채점이 안 된다.
이 스크립트는 일부 시각만 골라서 좌석 1~30번마다 점유/빈자리/모름을 적는 시트를 만든다.

시각 고르는 방법 (person_judge.py 결과는 보지 않는다 -> 정답이 시스템 답을 따라가지 않게):
  1) 매 정시에 가장 가까운 사진(5분 이내)이 있는 시각을 모은다
  2) --exclude-dates 날짜는 뺀다 (좌석 영역을 고칠 때 들여다본 날 -> 채점하면 점수가 부풀려짐)
  3) 사람이 적은 정답지(정답지_<room>.xlsx)의 '사람' 열과 시간대로 4묶음으로 나눈다
       낮_있음  09~18시, 사람 있음   -> 좌석 배정이 맞는지
       낮_없음  09~18시, 사람 없음   -> 빈 강의실에서 헛것을 잡는지
       밤_있음  19~08시, 사람 있음   -> 어두울 때도 잡는지
       밤_없음  19~08시, 사람 없음   -> 의자에 걸린 옷, 의자 등받이 오판
  4) 묶음마다 정한 개수만큼 여러 날짜에 고르게 뽑는다 (--seed 가 같으면 항상 같은 시각)

사용 예:
  python labels/make_seat_sheet.py /Volumes/T9/data 301
  python labels/make_seat_sheet.py /Volumes/T9/data 301 --quota 낮_있음:10,낮_없음:4,밤_있음:3,밤_없음:5
  python labels/make_seat_sheet.py /Volumes/T9/data 301 --exclude-dates ""     # 모든 날짜 사용

결과:
  labels/좌석정답지_301.xlsx
  labels/images/seats/301_<날짜>_<HH>00.jpg   (카메라 4대 원본 크기 2x2 + 좌석 영역과 번호)
"""
import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from openpyxl import Workbook
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from common import CAMS, hourly_photos, read_truth  # noqa: E402

TILE_W, TILE_H = 1920, 1080  # 좌석 번호가 잘 보이도록 원본 크기 그대로 붙임
DAY_HOURS = range(9, 19)     # 09:00~18:00 을 '낮'으로 봄
DEFAULT_QUOTA = "낮_있음:8,낮_없음:4,밤_있음:4,밤_없음:4"
DEFAULT_EXCLUDE = "2026-10-13,2026-10-14,2026-10-18"


def load_font(size):
    for f in ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/System/Library/Fonts/Helvetica.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def load_seat_lines(path, room):
    """classrooms.json -> ({cam: {좌석번호: [[x, y] x4]}}, 좌석번호 목록)"""
    data = json.load(open(path, encoding="utf-8"))
    rooms = data["classrooms"] if isinstance(data, dict) else data
    found = [c for c in rooms if room in str(c.get("name", ""))]
    if not found:
        raise SystemExit(f"{path}에 '{room}' 강의실 좌석 영역이 없음")
    lines = {}
    for cam in found[0]["cameras"]:
        num = re.findall(r"\d+", cam.get("name", ""))
        if num:
            lines[f"cam{num[0]}"] = {str(k): v[:4] for k, v in (cam.get("seat_lines") or {}).items()
                                     if len(v) >= 4}
    seats = sorted({s for v in lines.values() for s in v}, key=int)
    return lines, seats


def convex_hull(pts):
    """네 점을 감싸는 다각형 (점 순서가 섞여 있어도 테두리가 꼬이지 않게)"""
    pts = sorted(set(map(tuple, pts)))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def draw_seats(img, seat_lines, font):
    """좌석 영역은 얇은 선, 번호는 검은 바탕 흰 글씨 (사람을 가리지 않게)"""
    w, h = img.size
    sx, sy = w / TILE_W, h / TILE_H
    d = ImageDraw.Draw(img)
    for sid, pts in seat_lines.items():
        poly = [(x * sx, y * sy) for x, y in convex_hull(pts)]
        d.line(poly + [poly[0]], fill=(0, 220, 255), width=2)
        cx = sum(p[0] for p in poly) / len(poly)
        cy = sum(p[1] for p in poly) / len(poly)
        box = d.textbbox((cx, cy), sid, font=font, anchor="mm")
        d.rectangle([box[0] - 6, box[1] - 4, box[2] + 6, box[3] + 4], fill=(0, 0, 0))
        d.text((cx, cy), sid, fill=(255, 255, 255), font=font, anchor="mm")
    return img


def make_grid(paths, seat_lines, title, out_path, font, small):
    grid = Image.new("RGB", (TILE_W * 2, TILE_H * 2), (40, 40, 40))
    d = ImageDraw.Draw(grid)
    for i, cam in enumerate(CAMS):
        x, y = (i % 2) * TILE_W, (i // 2) * TILE_H
        p = paths.get(cam)
        if p:
            tile = Image.open(p).convert("RGB").resize((TILE_W, TILE_H))
            grid.paste(draw_seats(tile, seat_lines.get(cam, {}), font), (x, y))
            label = f"{cam}  {p.stem.replace('-', ':')}"
        else:
            label = f"{cam}  (no image)"
        d.rectangle([x, y, x + 420, y + 56], fill=(0, 0, 0))
        d.text((x + 12, y + 10), label, fill=(255, 255, 0), font=small)
    d.rectangle([TILE_W - 300, TILE_H * 2 - 64, TILE_W + 300, TILE_H * 2], fill=(0, 0, 0))
    d.text((TILE_W, TILE_H * 2 - 32), title, fill=(255, 255, 255), font=small, anchor="mm")
    grid.save(out_path, quality=85)


def pick_times(groups, truth, quota, seed):
    """묶음별로 정한 개수만큼, 여러 날짜에 고르게 시각을 고른다"""
    buckets = defaultdict(list)
    for key in groups:
        part = "낮" if int(key[1][:2]) in DAY_HOURS else "밤"
        buckets[f"{part}_{truth.get(key, '미확인')}"].append(key)

    rng = random.Random(seed)
    chosen = []
    for name, n in quota.items():
        pool = sorted(buckets.get(name, []))
        by_date = defaultdict(list)
        for k in pool:
            by_date[k[0]].append(k)
        for v in by_date.values():
            rng.shuffle(v)
        picked = []
        while len(picked) < n and any(by_date.values()):  # 날짜를 돌아가며 하나씩
            for date in sorted(by_date):
                if by_date[date] and len(picked) < n:
                    picked.append(by_date[date].pop())
        if len(picked) < n:  # 정답지에 없는 시각(미확인)으로 모자란 만큼 채움
            extra = sorted(buckets.get(f"{name.split('_')[0]}_미확인", []))
            rng.shuffle(extra)
            fill = [k for k in extra if k not in chosen][: n - len(picked)]
            picked += fill
            print(f"  주의: {name} 시각이 {len(pool)}개뿐 -> 정답지에 없는 시각 {len(fill)}개로 채움")
        chosen += [(k, name) for k in picked]
        print(f"  {name}: 후보 {len(pool)}개 중 {len(picked)}개")
    return sorted(chosen)


def main():
    ap = argparse.ArgumentParser(description="좌석별 정답지 + 좌석 번호 확인 이미지 만들기")
    ap.add_argument("root", help="예: /Volumes/T9/data")
    ap.add_argument("room", help="예: 301")
    ap.add_argument("--quota", default=DEFAULT_QUOTA, help=f"묶음별 개수 (기본 {DEFAULT_QUOTA})")
    ap.add_argument("--exclude-dates", default=DEFAULT_EXCLUDE, help="뺄 날짜, 쉼표로 구분")
    ap.add_argument("--truth", default="", help="기본 labels/정답지_<room>.xlsx")
    ap.add_argument("--classrooms", default=str(ROOT_DIR / "classrooms.json"))
    ap.add_argument("--seed", type=int, default=0, help="같으면 항상 같은 시각을 고름")
    ap.add_argument("-o", "--out", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()

    out = Path(args.out)
    xlsx = out / f"좌석정답지_{args.room}.xlsx"
    if xlsx.exists():  # 이미 적어 둔 정답을 덮어쓰지 않게
        raise SystemExit(f"{xlsx} 가 이미 있음. 지우거나 이름을 바꾼 뒤 다시 실행 (적어 둔 정답 보호)")
    img_dir = out / "images" / "seats"
    img_dir.mkdir(parents=True, exist_ok=True)
    quota = {k: int(v) for k, v in (q.split(":") for q in args.quota.split(",") if q)}
    excluded = {d.strip() for d in args.exclude_dates.split(",") if d.strip()}
    truth_path = Path(args.truth or out / f"정답지_{args.room}.xlsx")
    truth = read_truth(truth_path, "사람") if truth_path.exists() else {}
    if not truth:
        print(f"주의: {truth_path} 의 '사람' 정답이 없음 -> 모든 시각이 '미확인'으로 분류됨")

    seat_lines, seats = load_seat_lines(args.classrooms, args.room)
    groups = defaultdict(dict)
    for date, hhmm, cam, p in hourly_photos(args.root, args.room):
        if date not in excluded:
            groups[(date, hhmm)][cam] = p
    print(f"좌석 {len(seats)}개, 후보 시각 {len(groups)}개 (뺀 날짜: {sorted(excluded) or '없음'})")

    chosen = pick_times(groups, truth, quota, args.seed)
    font, small = load_font(40), load_font(36)
    rows = []
    for (date, hhmm), bucket in chosen:
        name = f"{args.room}_{date}_{hhmm.replace(':', '')}.jpg"
        make_grid(groups[(date, hhmm)], seat_lines, f"{args.room}  {date}  {hhmm}", img_dir / name, font, small)
        rows.append((date, hhmm, bucket, name))
        print(f"  {date} {hhmm} ({bucket}) -> {name}")

    # ---------------- 엑셀
    wb = Workbook()
    ws = wb.active
    ws.title = "정답"
    head = ["날짜", "시각", "묶음", "사진"] + seats + ["메모"]
    ws.append(head)
    for date, hhmm, bucket, name in rows:
        ws.append([date, hhmm, bucket, name] + [None] * len(seats) + [None])
        c = ws.cell(row=ws.max_row, column=4)
        c.hyperlink = f"images/seats/{name}"  # 클릭하면 사진이 열림
        c.font = Font(color="0563C1", underline="single")

    last = ws.max_row
    first_col, last_col = 5, 4 + len(seats)
    rng_txt = f"{ws.cell(2, first_col).coordinate}:{ws.cell(last, last_col).coordinate}"
    dv = DataValidation(type="list", formula1='"점유,빈자리,모름"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(rng_txt)
    # 입력하면 색이 바뀜: 점유=초록, 모름=노랑, 빈칸=입력 전 표시
    ws.conditional_formatting.add(rng_txt, CellIsRule(operator="equal", formula=['"점유"'],
                                  fill=PatternFill("solid", fgColor="C6EFCE")))
    ws.conditional_formatting.add(rng_txt, CellIsRule(operator="equal", formula=['"모름"'],
                                  fill=PatternFill("solid", fgColor="FFEB9C")))
    blank = PatternFill("solid", fgColor="FFF2CC")
    for r in range(2, last + 1):
        for col in range(first_col, last_col + 1):
            ws.cell(r, col).fill = blank
            ws.cell(r, col).alignment = Alignment(horizontal="center")
    for c in ws[1]:
        c.font = Font(bold=True)
        c.alignment = Alignment(horizontal="center")
    ws.column_dimensions["A"].width = 12
    ws.column_dimensions["B"].width = 7
    ws.column_dimensions["C"].width = 9
    ws.column_dimensions["D"].width = 26
    for col in range(first_col, last_col + 1):
        ws.column_dimensions[ws.cell(1, col).column_letter].width = 7
    ws.column_dimensions[ws.cell(1, last_col + 1).column_letter].width = 30
    ws.freeze_panes = ws.cell(2, first_col)  # 날짜·시각과 좌석 번호가 항상 보이게

    guide = wb.create_sheet("기준")
    for line in [
        ["원칙", "person_judge.py 결과를 보지 말고 사진만 보고 적는다 (시스템 답을 따라 적으면 채점이 의미 없어짐)"],
        ["", "모든 좌석 칸을 채운다. 빈칸은 '아직 안 봄'으로 처리되어 채점에서 빠진다"],
        ["", "빠르게 하려면: 한 줄을 전부 '빈자리'로 채운 뒤 사람이 있는 좌석만 '점유'로 바꾼다"],
        ["점유", "그 좌석 의자에 사람이 앉아 있다 (자는 사람, 기대어 앉은 사람 포함)"],
        ["", "좌석 옆에 서 있는 사람도 그 좌석 영역 안이면 '점유' (지금 시스템은 앉음/서 있음을 구분하지 않음)"],
        ["빈자리", "사람이 없다. 의자에 걸린 옷, 가방, 인형, 책은 사람이 아니므로 '빈자리'"],
        ["모름", "가려지거나 어두워서 확실하지 않을 때. 메모에 좌석 번호와 이유를 적는다"],
        ["카메라", "한 좌석이 두 카메라에 보이면 더 잘 보이는 쪽으로 판단한다"],
        ["", "좌석 번호는 사진 위 하늘색 영역 가운데에 있다. 영역이 안 그려진 카메라에서는 판단 근거로 쓰지 않는다"],
        ["묶음", "시각을 고른 기준 (낮/밤 x 강의실 정답지의 사람 있음/없음). 바꾸지 않는다"],
    ]:
        guide.append(line)
    guide.column_dimensions["A"].width = 8
    guide.column_dimensions["B"].width = 95

    wb.save(xlsx)
    print(f"\n총 {len(rows)}개 시각 x 좌석 {len(seats)}개\n정답지: {xlsx}\n사진: {img_dir}/")


if __name__ == "__main__":
    main()