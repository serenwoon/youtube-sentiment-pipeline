#!/usr/bin/env python3
"""학습셋 내부 교차검증으로 설정을 고른다 — 여러 시드로.

왜 시드를 여러 개 쓰나. 학습셋이 154건이면 5-fold의 폴드 경계가 어디 그어지느냐에
따라 macro F1이 0.04씩 흔들린다. 시드 하나로 재고 제일 높은 설정을 고르면
'제일 좋은 설정'이 아니라 '제일 운 좋은 시드'를 고르게 된다.

그래서 시드를 여러 개 돌려 평균과 폭을 같이 본다. 폭이 겹치는 설정끼리는
순위를 매기지 않는다. 몇 번 1위를 했는지도 같이 세는데, 이게 겹침을
가장 알아보기 쉽게 보여준다.

준비:
    pip install scikit-learn

사용법:
    python3 src/tune.py data/real/train.csv
    python3 src/tune.py --seeds 50 data/real/train.csv

골든셋은 여기 넣지 않는다. 설정을 고르는 데 평가셋을 쓰면 그 뒤의 점수는
자기 답안으로 채점한 값이 된다.
"""
import argparse
import csv
import statistics
import unicodedata
import warnings

# evaluate.py에도 같은 함수가 있다. 평가 하네스는 표준 라이브러리만 쓰는 독립
# 파일로 두기로 했으므로(골든셋 오염 방지) 세 줄을 공유하지 않고 복사한다.


def pad(text, width, right=False):
    """한글은 터미널에서 두 칸을 먹는다. 글자 수가 아니라 표시 폭으로 채운다."""
    shown = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    fill = " " * max(width - shown, 0)
    return fill + text if right else text + fill

CONFIGS = [
    ("char_wb(1,3) min_df=2",
     dict(analyzer="char_wb", ngram_range=(1, 3), min_df=2), {}),
    ("char_wb(2,4) min_df=1",
     dict(analyzer="char_wb", ngram_range=(2, 4), min_df=1), {}),
    ("char_wb(2,4) min_df=1 + class_weight",
     dict(analyzer="char_wb", ngram_range=(2, 4), min_df=1),
     dict(class_weight="balanced")),
    ("char_wb(2,5) min_df=1 + class_weight + C=5",
     dict(analyzer="char_wb", ngram_range=(2, 5), min_df=1),
     dict(class_weight="balanced", C=5)),
    ("word(1,2) min_df=1 + class_weight",
     dict(analyzer="word", ngram_range=(1, 2), min_df=1),
     dict(class_weight="balanced")),
]


def main():
    ap = argparse.ArgumentParser(description="TF-IDF 설정 교차검증 (여러 시드)")
    ap.add_argument("train", help="학습셋 CSV (id,text,label)")
    ap.add_argument("--seeds", type=int, default=20, help="시드 개수 (기본 20)")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline

    with open(args.train, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    X = [r["text"] for r in rows]
    y = [r["label"].strip() for r in rows]

    print(f"학습셋 {len(rows)}건 · {args.folds}-fold · 시드 0–{args.seeds - 1}\n")
    print(pad("설정", 44) + pad("평균", 7, right=True)
          + pad("최소", 7, right=True) + pad("최대", 7, right=True)
          + pad("표준편차", 10, right=True))

    scores = {}
    for name, vec, clf in CONFIGS:
        vals = []
        for seed in range(args.seeds):
            model = make_pipeline(
                TfidfVectorizer(**vec),
                LogisticRegression(max_iter=1000, **clf),
            )
            cv = StratifiedKFold(args.folds, shuffle=True, random_state=seed)
            vals.append(cross_val_score(model, X, y, cv=cv,
                                        scoring="f1_macro").mean())
        scores[name] = vals
        print(f"{name:<44}{statistics.mean(vals):>7.3f}{min(vals):>7.3f}"
              f"{max(vals):>7.3f}{statistics.stdev(vals):>10.3f}")

    wins = dict.fromkeys(scores, 0)
    for i in range(args.seeds):
        wins[max(scores, key=lambda k: scores[k][i])] += 1

    print("\n시드별 1위 횟수 — 한 설정이 독식하지 않으면 그 차이는 노이즈다")
    for name, n in sorted(wins.items(), key=lambda kv: -kv[1]):
        if n:
            print(f"  {name:<44}{n:>3}/{args.seeds}")


if __name__ == "__main__":
    main()
