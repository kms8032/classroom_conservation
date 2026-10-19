"""
조명 판단 방법들을 정답지로 채점한다.

비교하는 방법:
  0. 시간 규칙 (기준선): 정해진 시간대(기본 09~20시)면 켜짐
     -> 이 정도는 이겨야 영상으로 판단하는 의미가 있음
  A. 픽셀 지표: light_pixel.py 결과의 지표 x 카메라 조합마다
     기준값을 정답지로 정해서 채점
  B. VLM: light_vlm.py 결과

기준값을 정답으로 정하고 같은 정답으로 채점하면 점수가 부풀려지므로,
'하루 빼고 검증(leave-one-day-out)'을 쓴다:
  하루를 시험용으로 빼고 -> 나머지 날로 기준값을 정하고 -> 빼 둔 하루를 채점. 모든 날에 반복.

같이 확인하는 가정: '흑백(적외선) 사진이면 꺼짐'이 정답과 맞는가?

사용 예:
  python evaluate_light.py labels/정답지_301.xlsx
  python evaluate_light.py labels/정답지_301.xlsx --pixel results/light/pixel_301.csv --vlm results/light/vlm_301.csv
  # 다른 강의실에서 정한 기준값을 그대로 적용해 보기 (기준값을 새로 정하지 않음)
  python evaluate_light.py labels/정답지_304.xlsx --pixel results/light/pixel_304.csv --fixed bright_pct:13.4

결과:
  화면 출력 + results/light/eval_301.csv (시각마다 정답과 각 방법의 예측)
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

from common import read_truth

METRICS = ["brightness", "contrast", "dark_pct", "bright_pct"]


def fit_threshold(pairs):
    """[(값, 정답)] -> 가장 잘 맞는 (기준값, 방향). 방향 +1: 값 >= 기준값이면 켜짐, -1: 반대"""
    vals = sorted({v for v, _ in pairs})
    cands = [(a + b) / 2 for a, b in zip(vals, vals[1:])] or vals
    best = (-1, None, 1)
    for t in cands:
        for d in (1, -1):
            hit = sum((("켜짐" if (v - t) * d >= 0 else "꺼짐") == g) for v, g in pairs)
            if hit > best[0]:
                best = (hit, t, d)
    return best[1], best[2]


def predict(v, t, d):
    return "켜짐" if (v - t) * d >= 0 else "꺼짐"


def acc(pred, truth):
    keys = [k for k in truth if pred.get(k) in ("켜짐", "꺼짐")]
    hit = sum(pred[k] == truth[k] for k in keys)
    return hit, len(keys)


def fmt(hit, n, total):
    pct = f"{hit / n * 100:5.1f}%" if n else "   - "
    miss = f"  (예측 없음 {total - n})" if total - n else ""
    return f"{pct}  {hit}/{n}{miss}"


def main():
    ap = argparse.ArgumentParser(description="조명 판단 채점")
    ap.add_argument("truth", help="정답지 엑셀 (예: labels/정답지_301.xlsx)")
    ap.add_argument("--pixel", default="", help="light_pixel.py 결과 CSV")
    ap.add_argument("--vlm", default="", help="light_vlm.py 결과 CSV")
    ap.add_argument("--on-hours", default="9-20", help="시간 규칙에서 켜짐으로 볼 시간 (예: 9-20)")
    ap.add_argument("--fixed", default="",
                    help="다른 곳에서 정한 기준값을 그대로 적용. 예: bright_pct:13.4 (값 이상이면 켜짐)")
    ap.add_argument("-o", "--out", default="results/light")
    args = ap.parse_args()

    truth = read_truth(args.truth, "조명")
    if not truth:
        print("정답지에 '켜짐'/'꺼짐'이 하나도 없어요. 정답을 먼저 채워 주세요.")
        return
    room = Path(args.truth).stem.split("_")[-1]
    on = sum(v == "켜짐" for v in truth.values())
    print(f"정답 {len(truth)}개 (켜짐 {on}, 꺼짐 {len(truth) - on}, '모름'·빈칸 제외)")
    print(f"참고: 항상 '{'켜짐' if on > len(truth) - on else '꺼짐'}'이라고만 답해도 {max(on, len(truth) - on) / len(truth) * 100:.1f}%\n")

    table = {k: {"정답": v} for k, v in truth.items()}
    results = []  # (방법 이름, 맞힌 수, 채점된 수)

    # 0. 시간 규칙
    h1, h2 = map(int, args.on_hours.split("-"))
    pred = {k: ("켜짐" if h1 <= int(k[1][:2]) <= h2 else "꺼짐") for k in truth}
    results.append((f"0. 시간 규칙 ({h1}~{h2}시 켜짐)", *acc(pred, truth)))
    for k in truth:
        table[k]["시간규칙"] = pred[k]

    # A. 픽셀 지표
    if args.pixel:
        data = defaultdict(dict)  # (cam, metric) -> {(date, time): 값}
        gray = defaultdict(dict)
        for r in csv.DictReader(open(args.pixel, encoding="utf-8-sig")):
            k = (r["date"], r["time"])
            for m in METRICS:
                data[(r["cam"], m)][k] = float(r[m])
            gray[r["cam"]][k] = r["grayscale"] == "1"

        days = sorted({d for d, _ in truth})
        best_name, best_score = None, -1
        for (cam, m), vals in sorted(data.items()):
            keys = [k for k in truth if k in vals]
            pred = {}
            for day in days:  # 하루 빼고 검증
                train = [(vals[k], truth[k]) for k in keys if k[0] != day]
                test = [k for k in keys if k[0] == day]
                if not train or not test:
                    continue
                t, d = fit_threshold(train)
                for k in test:
                    pred[k] = predict(vals[k], t, d)
            hit, n = acc(pred, truth)
            name = f"A. {cam} {m}"
            results.append((name, hit, n))
            if n and hit / n > best_score:
                best_score, best_name, best_pred = hit / n, name, pred
            # 전체 정답으로 정한 기준값 (실제로 쓸 값, 참고용)
            t, d = fit_threshold([(vals[k], truth[k]) for k in keys])
            results[-1] = (f"{name}  [기준 {'>=' if d > 0 else '<'} {t:.1f}]", hit, n)
        if best_name:
            for k, v in best_pred.items():
                table[k]["픽셀_최고"] = v

        # 기준선 2: 흑백 여부만으로 판단 (흑백=꺼짐, 컬러=켜짐). 픽셀 지표가 이것보다 나아야 의미가 있음
        for cam in sorted(gray):
            pred = {k: ("꺼짐" if g else "켜짐") for k, g in gray[cam].items() if k in truth}
            results.insert(1, (f"0. 흑백 여부만 ({cam}: 흑백=꺼짐, 컬러=켜짐)", *acc(pred, truth)))

        if args.fixed:  # 기준값을 새로 정하지 않고 그대로 적용 (흑백이면 꺼짐 규칙 포함)
            fm, ft = args.fixed.split(":")
            for cam in sorted(gray):
                vals = data.get((cam, fm), {})
                pred = {k: ("꺼짐" if gray[cam].get(k) else predict(vals[k], float(ft), 1))
                        for k in truth if k in vals}
                results.append((f"F. {cam} {fm} >= {ft} 그대로 적용 (+흑백=꺼짐)", *acc(pred, truth)))

        print("== 가정 확인: 흑백(적외선)이면 꺼짐? ==")
        for cam in sorted(gray):
            g = [k for k in truth if gray[cam].get(k)]
            off = sum(truth[k] == "꺼짐" for k in g)
            note = "" if not g or off == len(g) else "  <- 흑백인데 켜짐인 경우 있음!"
            print(f"  {cam}: 흑백 {len(g)}장 중 정답 꺼짐 {off}장{note}")
        print()

    # B. VLM
    if args.vlm:
        votes = defaultdict(list)
        secs = []
        for r in csv.DictReader(open(args.vlm, encoding="utf-8-sig")):
            votes[(r["date"], r["time"])].append(r["vlm_답"])
            try:
                if float(r["vlm_초"]) > 0:
                    secs.append(float(r["vlm_초"]))
            except ValueError:
                pass
        # 카메라가 여러 대면 다수결 (동점이면 켜짐)
        pred = {}
        for k, vs in votes.items():
            ok = [v for v in vs if v in ("켜짐", "꺼짐")]
            if ok:
                pred[k] = "켜짐" if ok.count("켜짐") >= ok.count("꺼짐") else "꺼짐"
        fail = sum(1 for k in truth if k in votes and k not in pred)
        hit, n = acc(pred, truth)
        name = "B. VLM" + (f"  (실패 {fail})" if fail else "")
        if secs:
            name += f"  [평균 {sum(secs) / len(secs):.1f}초]"
        results.append((name, hit, n))
        for k in truth:
            table[k]["VLM"] = pred.get(k, "")

    print("== 정확도 (하루 빼고 검증) ==")
    for name, hit, n in results:
        print(f"  {fmt(hit, n, len(truth))}   {name}")

    # 틀린 시각 (시간 규칙과 최고 픽셀 방법, VLM)
    for col in ["픽셀_최고", "VLM"]:
        wrong = [k for k in sorted(table) if table[k].get(col) in ("켜짐", "꺼짐")
                 and table[k][col] != table[k]["정답"]]
        if wrong:
            label = best_name if col == "픽셀_최고" else "B. VLM"
            print(f"\n{label} 이(가) 틀린 시각 {len(wrong)}개:")
            for k in wrong[:30]:
                print(f"   {k[0]} {k[1]}  정답 {table[k]['정답']}, 예측 {table[k][col]}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"eval_{room}.csv"
    cols = ["date", "time", "정답", "시간규칙", "픽셀_최고", "VLM"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for k in sorted(table):
            w.writerow([k[0], k[1]] + [table[k].get(c, "") for c in cols[2:]])
    print(f"\nCSV: {path}")


if __name__ == "__main__":
    main()
