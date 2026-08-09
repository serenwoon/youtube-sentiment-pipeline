#!/usr/bin/env python3
"""베이스라인 0 — 다수 클래스.

전부 한 클래스로 찍는다. 나머지 방법은 최소한 이걸 넘겨야 의미가 있다.
표준 라이브러리만 쓴다.

사용법:
    python3 src/baselines/majority.py data/sample_comments.csv > pred.csv
    python3 src/evaluate.py data/sample_comments.csv pred.csv

    # 학습셋이 따로 있으면 그쪽 분포를 쓴다
    python3 src/baselines/majority.py --train train.csv golden.csv > pred.csv

--train 없이 쓰면 정답 파일의 최빈 클래스를 그대로 쓴다. 정답을 엿보는 셈이라
베이스라인에 유리하게 나오는데, 하한선은 후하게 잡는 편이 낫다.
이것도 못 넘으면 변명의 여지가 없다.
"""
import argparse
import csv
import sys
from collections import Counter


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gold")
    ap.add_argument("--train", help="최빈 클래스를 뽑을 파일 (없으면 gold에서)")
    args = ap.parse_args()

    gold = read(args.gold)
    source = read(args.train) if args.train else gold
    counts = Counter(r["label"].strip() for r in source if r.get("label"))
    if not counts:
        sys.exit("라벨이 없다. label 열을 확인한다.")
    label, n = counts.most_common(1)[0]

    writer = csv.writer(sys.stdout)
    writer.writerow(["id", "label"])
    for row in gold:
        writer.writerow([row["id"], label])

    share = n / sum(counts.values())
    print(f"다수 클래스 '{label}' — {n}/{sum(counts.values())} ({share:.1%})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
