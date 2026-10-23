"""
VLM 재확인 프롬프트 비교 실험.

person_judge.py v2 에서 VLM에게 물어본 박스만 다시 꺼내서, 프롬프트 여러 개로 같은 박스를 물어보고
어느 프롬프트가 '물건은 아님, 사람은 사람'을 잘 가르는지 비교한다.
(전체를 다시 돌리면 35분 -> 박스만 골라서 10분 안쪽)

비교하는 프롬프트:
  A 기존     : person_judge.py 의 지금 프롬프트 ("일부만 보여도 사람이면 사람")
  B 머리확인 : 흑백 사진 주의 + 머리가 보이는지 먼저 답하게 함. 머리가 없으면 사람 아님으로 처리
  C 넓게보기 : B와 같은 질문, 박스 주변을 더 넓게 잘라 의자 전체가 보이게 함 (옷이 의자에 걸린 맥락)

두 단계로 쓴다:
  1) 처음 실행하면 실험할 박스 목록(cases_<room>.csv)과 잘라낸 사진(crops/)을 만들고 멈춘다.
     정답은 좌석 채점 결과(eval_seats_<room>.csv)에서 아는 것만 미리 채워 둔다.
     빈 '정답' 칸은 crops/ 사진을 보고 사람/아님을 직접 적는다 (모르면 비워 두면 채점에서 빠짐).
  2) 다시 실행하면 프롬프트별로 VLM에게 묻고 채점한다.

사용 예:
  python vlm_prompt_test.py results/person_301_v2/boxes_301.csv            # 1단계: 목록 만들기
  (results/vlm_prompt/cases_301.csv 의 빈 정답 칸 채우기)
  python vlm_prompt_test.py results/person_301_v2/boxes_301.csv            # 2단계: 실험
  python vlm_prompt_test.py results/person_301_v2/boxes_301.csv --prompts A,B --vlm qwen3-vl:2b

결과 (results/vlm_prompt/):
  cases_301.csv    실험할 박스와 정답
  crops/           박스마다 VLM에게 보내는 사진 (프롬프트 A/B 용, C 용)
  answers_301.csv  박스 x 프롬프트마다 VLM 답, 이유, 걸린 초
"""
import argparse
import base64
import csv
import json
import random
import time
import urllib.error
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from person_judge import VLM_PROMPT, _post, crop_with_box

HEAD_PROMPT = (
    "이 사진은 대학교 강의실 CCTV 사진의 일부다. 밤에는 흑백(적외선) 사진일 수 있다.\n"
    "흑백 사진에서는 의자 등받이에 걸린 어두운 옷, 가방, 인형, 의자 자체가 사람처럼 보일 수 있으니 주의하라.\n"
    "빨간 상자 안만 보고 아래 순서로 판단하라. 빨간 상자 바깥은 무시하라.\n"
    "1) 빨간 상자 안에 사람의 머리(얼굴, 뒤통수, 머리카락)가 보이는가?\n"
    "2) 머리와 함께 팔, 어깨, 다리처럼 살아 있는 사람의 몸이 보이는가?\n"
    "옷이나 물건만 있고 머리가 없으면 사람이 아니다.\n"
    'JSON으로만 답하라: {"head": true 또는 false, "person": true 또는 false, "reason": "짧은 이유"}'
)

HEAD_PROMPT_SOFT = (
    "이 사진은 대학교 강의실 CCTV 사진의 일부다. 밤에는 흑백(적외선) 사진일 수 있다.\n"
    "흑백 사진에서는 의자 등받이에 걸린 어두운 옷, 가방, 인형, 의자 자체가 사람처럼 보일 수 있으니 주의하라.\n"
    "빨간 상자 안만 보고 아래 순서로 판단하라. 빨간 상자 바깥은 무시하라.\n"
    "1) 빨간 상자 안에 사람의 머리(얼굴, 뒤통수, 머리카락)가 보이는가?\n"
    "2) 머리 아래로 어깨나 팔 중 하나라도 이어져 보이는가? 책상이나 모니터에 몸 일부가 가려진 것은 괜찮다.\n"
    "옷이나 물건만 있고 머리가 없으면 사람이 아니다.\n"
    'JSON으로만 답하라: {"head": true 또는 false, "person": true 또는 false, "reason": "짧은 이유"}'
)

PROMPTS = {  # 이름: (설명, 질문, 자르는 여백 비율, 머리 확인 여부)
    "A": ("기존", VLM_PROMPT, 0.4, False),
    "B": ("머리확인", HEAD_PROMPT, 0.4, True),
    "C": ("넓게보기", HEAD_PROMPT, 1.2, True),
    "D": ("넓게완화", HEAD_PROMPT_SOFT, 1.2, True),  # C에서 2번 질문 완화 (C가 상체만 보이는 사람을 놓침)
}
MAX_SIDE = 512  # 보내기 전에 긴 쪽을 이 크기로 줄임 (큰 사진일수록 VLM이 느려져 시간 초과가 났음). 0이면 안 줄임


# ---------------------------------------------------------------- 1단계: 실험할 박스 고르기

def load_seat_truth(path):
    """eval_seats_<room>.csv -> {(날짜, 시각, 좌석): 사람/아님}  (좌석 정답지에서 아는 것만)"""
    truth = {}
    if not Path(path).exists():
        return truth
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        if r["정답"] in ("점유", "빈자리"):
            truth[(r["date"], r["time"], r["seat"])] = "사람" if r["정답"] == "점유" else "아님"
    return truth


def make_cases(boxes_csv, out, room, n_day, seed):
    rows = list(csv.DictReader(open(boxes_csv, encoding="utf-8-sig")))
    asked = [r for r in rows if r.get("근거", "").startswith("VLM")]
    night = [r for r in asked if r.get("적외선") == "1"]
    day = [r for r in asked if r.get("적외선") != "1"]
    random.Random(seed).shuffle(day)
    picked = night + day[:n_day]  # 밤 박스는 전부 + 낮 박스 일부(프롬프트를 바꿔도 낮 사람을 계속 잡는지 확인용)

    seat_truth = load_seat_truth(Path(boxes_csv).parent / f"eval_seats_{room}.csv")
    crop_dir = out / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for i, r in enumerate(picked, 1):
        img = cv2.imread(r["file"])
        if img is None:
            print(f"  사진 없음 (외장 SSD 연결 확인): {r['file']}")
            continue
        box = tuple(map(int, r["box"].split(",")))
        cid = f"{i:03d}"
        for margin in sorted({p[2] for p in PROMPTS.values()}):
            cv2.imwrite(str(crop_dir / f"{cid}_m{margin}.jpg"), crop_with_box(img, box, margin))
        cases.append({"id": cid, "date": r["date"], "time": r["time"], "cam": r["cam"], "seat": r["seat"],
                      "conf": r["conf"], "적외선": r.get("적외선", ""), "검사": r.get("검사", ""),
                      "v2_vlm답": r.get("vlm_답", ""),
                      "정답": seat_truth.get((r["date"], r["time"], r["seat"]), ""),
                      "사진": f"crops/{cid}_m0.4.jpg", "box": r["box"], "file": r["file"]})
    if not cases:
        print("실험할 박스가 없음 (VLM에게 물어본 박스가 없거나 사진을 못 찾음). 목록을 만들지 않음.")
        return
    path = out / f"cases_{room}.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(cases[0].keys()))
        w.writeheader()
        w.writerows(cases)
    known = sum(1 for c in cases if c["정답"])
    print(f"\n실험할 박스 {len(cases)}개 (밤 {len(night)}, 낮 {len(cases) - len(night)}) -> {path}")
    print(f"정답을 미리 채운 박스 {known}개 (좌석 정답지에서), 빈 칸 {len(cases) - known}개")
    print(f"빈 '정답' 칸에 crops/ 사진을 보고 '사람' 또는 '아님'을 적은 뒤 다시 실행하세요. 모르면 비워 둠.")


# ---------------------------------------------------------------- 2단계: 프롬프트별로 묻기

def shrink(img, max_side):
    """긴 쪽이 max_side보다 크면 비율을 유지하며 줄임"""
    long_side = max(img.shape[:2])
    if not max_side or long_side <= max_side:
        return img
    s = max_side / long_side
    return cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)


def ask(host, model, prompt, crop, timeout, head_check):
    """-> (사람/아님/실패, 이유, 머리, 초)"""
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    body = {"model": model, "prompt": prompt, "images": [base64.b64encode(buf).decode()],
            "stream": False, "format": "json", "options": {"temperature": 0}, "think": False}
    t0 = time.time()
    try:
        try:
            text = _post(host, body, timeout)
        except urllib.error.HTTPError as e:
            if e.code == 400 and "think" in e.read().decode(errors="ignore").lower():
                body.pop("think")
                text = _post(host, body, timeout)
            else:
                raise
    except Exception as e:
        return "실패", str(e)[:80], "", round(time.time() - t0, 2)
    sec = round(time.time() - t0, 2)
    try:
        ans = json.loads(text)
    except json.JSONDecodeError:
        return "실패", text[:80] or "(빈 응답)", "", sec

    def yes(v):
        return v.strip().lower() in ("true", "yes") if isinstance(v, str) else bool(v)

    person = yes(ans.get("person"))
    head = yes(ans.get("head")) if "head" in ans else ""
    if head_check and head is False:  # 머리가 없다고 했으면 사람이라고 해도 아님으로 처리
        person = False
    return ("사람" if person else "아님"), str(ans.get("reason", "")), head, sec


def run(cases_csv, out, room, names, host, model, timeout, max_side):
    cases = list(csv.DictReader(open(cases_csv, encoding="utf-8-sig")))
    if not cases:
        print(f"{cases_csv} 가 비어 있음. 지우고 1단계부터 다시 실행하세요.")
        return
    print(f"박스 {len(cases)}개 x 프롬프트 {len(names)}개 = VLM 질문 {len(cases) * len(names)}번 "
          f"({model}, 사진 긴 쪽 {max_side or '원본'}px)")
    fields = ["id", "프롬프트", "정답", "답", "머리", "이유", "초", "보낸크기", "적외선", "cam", "seat", "date", "time"]
    path = out / f"answers_{room}.csv"
    f = open(path, "w", newline="", encoding="utf-8-sig")  # 한 줄씩 바로 저장 (중간에 멈춰도 남음)
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    rows = []
    for c in cases:
        c["id"] = c["id"].strip().zfill(3)  # 엑셀로 저장하면 001 -> 1 로 바뀌어서 되돌림
        line = []
        for name in names:
            desc, prompt, margin, head_check = PROMPTS[name]
            crop = shrink(cv2.imread(str(out / "crops" / f"{c['id']}_m{margin}.jpg")), max_side)
            ans, reason, head, sec = ask(host, model, prompt, crop, timeout, head_check)
            row = {"id": c["id"], "프롬프트": f"{name}_{desc}", "정답": c["정답"], "답": ans,
                   "머리": head, "이유": reason, "초": sec, "보낸크기": f"{crop.shape[1]}x{crop.shape[0]}",
                   "적외선": c["적외선"], "cam": c["cam"], "seat": c["seat"], "date": c["date"], "time": c["time"]}
            rows.append(row)
            w.writerow(row)
            f.flush()
            line.append(f"{name}:{ans}({sec:.0f}s)")
        mark = f"정답 {c['정답'] or '?'}"
        print(f"  [{c['id']}] {c['date']} {c['time']} {c['cam']} {c['seat'] or '-':>2}번 "
              f"{'밤' if c['적외선'] == '1' else '낮'} {mark:8s} -> {'  '.join(line)}")
    f.close()

    print("\n== 프롬프트별 채점 (정답이 있는 박스만) ==")
    print("  프롬프트        사람을 사람으로    물건을 아님으로    실패  평균 초")
    for name in names:
        rs = [r for r in rows if r["프롬프트"].startswith(name + "_")]
        k = Counter((r["정답"], r["답"]) for r in rs)
        tp, pos = k[("사람", "사람")], sum(v for (t, _), v in k.items() if t == "사람")
        tn, neg = k[("아님", "아님")], sum(v for (t, _), v in k.items() if t == "아님")
        fails = sum(1 for r in rs if r["답"] == "실패")
        secs = sorted(float(r["초"]) for r in rs if r["답"] != "실패")
        print(f"  {rs[0]['프롬프트']:12s}  {tp:3d}/{pos:<3d}"
              f"{'(' + format(tp / pos * 100, '.0f') + '%)' if pos else '':>8s}  "
              f"{tn:3d}/{neg:<3d}{'(' + format(tn / neg * 100, '.0f') + '%)' if neg else '':>8s}  "
              f"{fails:4d}  {sum(secs) / len(secs) if secs else 0:6.1f}")
    print("  -> '사람을 사람으로'가 떨어지지 않으면서 '물건을 아님으로'가 오르는 프롬프트를 고른다.")

    by_case = defaultdict(dict)
    for r in rows:
        by_case[r["id"]][r["프롬프트"][0]] = r
    wrong = [(cid, d) for cid, d in by_case.items()
             if any(r["정답"] and r["답"] != r["정답"] for r in d.values())]
    if wrong:
        print("\n  틀린 박스 (사진: results/vlm_prompt/crops/<id>_m*.jpg):")
        for cid, d in wrong:
            r0 = next(iter(d.values()))
            print(f"   [{cid}] {r0['date']} {r0['time']} {r0['cam']} {r0['seat'] or '-'}번 정답 {r0['정답']}: "
                  + ", ".join(f"{n}={r['답']}" for n, r in d.items()))
    print(f"\n답 CSV: {path}")


def main():
    ap = argparse.ArgumentParser(description="VLM 재확인 프롬프트 비교")
    ap.add_argument("boxes", help="person_judge.py 결과 boxes_<room>.csv (v2)")
    ap.add_argument("--prompts", default="D", help="비교할 프롬프트, 예: A,C,D (A~C는 첫 실험에서 끝남)")
    ap.add_argument("--max-side", type=int, default=MAX_SIDE, help="보내기 전 사진 긴 쪽 크기, 0이면 원본")
    ap.add_argument("--day", type=int, default=10, help="같이 넣을 낮 박스 수 (낮 사람을 계속 잡는지 확인용)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vlm", default="qwen3-vl:4b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("-o", "--out", default="results/vlm_prompt")
    args = ap.parse_args()

    room = Path(args.boxes).stem.split("_")[-1]
    out = Path(args.out)
    cases_csv = out / f"cases_{room}.csv"
    if not cases_csv.exists():
        make_cases(args.boxes, out, room, args.day, args.seed)
        return
    names = [n.strip() for n in args.prompts.split(",") if n.strip() in PROMPTS]
    run(cases_csv, out, room, names, args.host, args.vlm, args.timeout, args.max_side)


if __name__ == "__main__":
    main()