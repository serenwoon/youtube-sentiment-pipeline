#!/usr/bin/env python3
"""골든셋 평가 하네스 — 표준 라이브러리만 사용.

사용법:
    python3 src/evaluate.py data/sample_comments.csv data/sample_predictions.csv
    python3 src/evaluate.py --bootstrap data/real/golden.csv pred.csv

입력 (CSV, UTF-8, 헤더 필수):
    정답 파일: id,text,label
    예측 파일: id,label
라벨: 긍정 | 부정 | 복합

출력: 클래스별 Precision/Recall/F1, macro/weighted F1, 혼동 행렬.

--bootstrap을 주면 macro F1의 95% 구간과 무작위 추측 대비 우위 확률을 같이 낸다.
점수 하나만 적어두면 그게 실력인지 표본 운인지 구분할 수 없다. 골든셋이 200건
남짓이면 그 차이가 실제로 크다.

골든셋은 평가에만 쓴다. 이 파일을 학습 쪽에서 임포트하지 않는다.
"""
import argparse
import csv
import random
import sys
import unicodedata
from collections import Counter, defaultdict

LABELS = ["긍정", "부정", "복합"]
K = len(LABELS)


def load_labels(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = {row["id"]: row["label"].strip() for row in csv.DictReader(f)}
    unknown = {v for v in rows.values()} - set(LABELS)
    if unknown:
        sys.exit(f"{path}: 정의되지 않은 라벨 {sorted(unknown)}")
    return rows


def macro_f1(counts):
    """counts[정답][예측] 3x3에서 macro F1. 부트스트랩에서 수만 번 호출된다."""
    total = 0.0
    for i in range(K):
        tp = counts[i][i]
        fp = sum(counts[g][i] for g in range(K)) - tp
        fn = sum(counts[i][p] for p in range(K)) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        total += 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return total / K


def bootstrap(gold_idx, pred_idx, rounds, seed):
    """댓글을 복원추출해 macro F1이 표본에 따라 얼마나 흔들리는지 본다.

    같은 리샘플 위에서 무작위 추측도 같이 재고 짝지어 비교한다. 둘이 같은
    댓글을 보므로 표본 운이 상쇄되고, 남는 차이만 방법의 차이다.
    """
    rng = random.Random(seed)
    randrange = rng.randrange
    n = len(gold_idx)
    obs, rnd, wins = [], [], 0
    for _ in range(rounds):
        c = [[0] * K for _ in range(K)]
        r = [[0] * K for _ in range(K)]
        for _ in range(n):
            i = randrange(n)
            g = gold_idx[i]
            c[g][pred_idx[i]] += 1
            r[g][randrange(K)] += 1
        a, b = macro_f1(c), macro_f1(r)
        obs.append(a)
        rnd.append(b)
        wins += a > b
    return obs, rnd, wins / rounds


def interval(values, level=0.95):
    v = sorted(values)
    lo = v[int((1 - level) / 2 * len(v))]
    hi = v[min(int((1 + level) / 2 * len(v)), len(v) - 1)]
    return lo, hi


def report(gold, pred):
    tp, fp, fn = Counter(), Counter(), Counter()
    confusion = defaultdict(Counter)
    for cid, g in gold.items():
        p = pred[cid]
        confusion[g][p] += 1
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1

    n = len(gold)
    print(f"평가 {n}건 — 정확도 {sum(tp.values()) / n:.3f}\n")
    print(f"{'클래스':<4} {'P':>7} {'R':>7} {'F1':>7} {'지지도':>5}")

    f1s, supports = [], []
    for lab in LABELS:
        support = tp[lab] + fn[lab]
        prec = tp[lab] / (tp[lab] + fp[lab]) if tp[lab] + fp[lab] else 0.0
        rec = tp[lab] / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        supports.append(support)
        print(f"{lab:<4} {prec:>7.3f} {rec:>7.3f} {f1:>7.3f} {support:>5}")

    macro = sum(f1s) / len(f1s)
    weighted = sum(f * s for f, s in zip(f1s, supports)) / sum(supports)
    print(f"\nmacro F1    {macro:.3f}")
    print(f"weighted F1 {weighted:.3f}")

    print("\n혼동 행렬 (행=정답, 열=예측)")
    print("      " + "".join(f"{lab:>6}" for lab in LABELS))
    for g in LABELS:
        print(f"{g:<4}" + "".join(f"{confusion[g][p]:>6}" for p in LABELS))


def pad(text, width):
    """한글은 터미널에서 두 칸을 먹는다. 글자 수가 아니라 표시 폭으로 채운다."""
    shown = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return text + " " * max(width - shown, 0)


def report_bootstrap(gold, pred, rounds, seed):
    order = list(gold)
    gold_idx = [LABELS.index(gold[c]) for c in order]
    pred_idx = [LABELS.index(pred[c]) for c in order]

    counts = [[0] * K for _ in range(K)]
    for g, p in zip(gold_idx, pred_idx):
        counts[g][p] += 1
    point = macro_f1(counts)

    obs, rnd, win_rate = bootstrap(gold_idx, pred_idx, rounds, seed)
    obs_lo, obs_hi = interval(obs)
    rnd_lo, rnd_hi = interval(rnd)

    print(f"\n부트스트랩 {rounds:,}회 (시드 {seed})\n")
    print(pad("", 14) + f"{'macro F1':>8}   95% 구간")
    print(pad("관측값", 14) + f"{point:>8.3f}   {obs_lo:.3f} – {obs_hi:.3f}")
    print(pad("무작위 추측", 14)
          + f"{sum(rnd) / len(rnd):>8.3f}   {rnd_lo:.3f} – {rnd_hi:.3f}")

    print(f"\n무작위보다 높을 확률 {win_rate:.1%}  (단측 p = {1 - win_rate:.3f})")
    if win_rate >= 0.95:
        print("→ 무작위 추측보다 높다.")
    elif win_rate <= 0.05:
        print("→ 무작위 추측보다 낮다. 이 방법은 신호가 아니라 편향을 재고 있다.")
    else:
        print("→ 95% 기준으로는 무작위 추측과 구분되지 않는다.")


def main():
    ap = argparse.ArgumentParser(
        description="골든셋 평가 하네스",
        epilog="라벨: " + " | ".join(LABELS),
    )
    ap.add_argument("gold", help="정답 CSV (id,text,label)")
    ap.add_argument("pred", help="예측 CSV (id,label)")
    ap.add_argument("--bootstrap", action="store_true",
                    help="복원추출로 macro F1의 95%% 구간과 무작위 대비 우위 확률")
    ap.add_argument("--rounds", type=int, default=10000, metavar="N",
                    help="부트스트랩 반복 횟수 (기본 10000)")
    ap.add_argument("--seed", type=int, default=20260809)
    args = ap.parse_args()

    gold = load_labels(args.gold)
    pred = load_labels(args.pred)

    missing = set(gold) - set(pred)
    if missing:
        sys.exit(f"예측 누락 {len(missing)}건: {sorted(missing)[:5]} ...")

    report(gold, pred)
    if args.bootstrap:
        report_bootstrap(gold, pred, args.rounds, args.seed)


if __name__ == "__main__":
    main()
