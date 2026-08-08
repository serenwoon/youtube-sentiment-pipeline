#!/usr/bin/env python3
"""골든셋 평가 하네스 — 표준 라이브러리만 사용.

사용법:
    python3 src/evaluate.py data/sample_comments.csv data/sample_predictions.csv

입력 (CSV, UTF-8, 헤더 필수):
    정답 파일: id,text,label
    예측 파일: id,label
라벨: 긍정 | 부정 | 복합

출력: 클래스별 Precision/Recall/F1, macro/weighted F1, 혼동 행렬.

골든셋은 평가에만 쓴다. 이 파일을 학습 쪽에서 임포트하지 않는다.
"""
import csv
import sys
from collections import Counter, defaultdict

LABELS = ["긍정", "부정", "복합"]


def load_labels(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = {row["id"]: row["label"].strip() for row in csv.DictReader(f)}
    unknown = {v for v in rows.values()} - set(LABELS)
    if unknown:
        sys.exit(f"{path}: 정의되지 않은 라벨 {sorted(unknown)}")
    return rows


def main(gold_path, pred_path):
    gold = load_labels(gold_path)
    pred = load_labels(pred_path)

    missing = set(gold) - set(pred)
    if missing:
        sys.exit(f"예측 누락 {len(missing)}건: {sorted(missing)[:5]} ...")

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
    header = "      " + "".join(f"{lab:>6}" for lab in LABELS)
    print(header)
    for g in LABELS:
        print(f"{g:<4}" + "".join(f"{confusion[g][p]:>6}" for p in LABELS))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
