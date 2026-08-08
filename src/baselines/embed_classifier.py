#!/usr/bin/env python3
"""베이스라인 2 — 경량 분류기.

감정분류 정도의 좁은 태스크면 LLM보다 임베딩 + 분류기가
더 싸고 다루기 쉬울 것 같아서 확인해보는 것. 비용과 지연도 같이 잰다.

현재 구현은 TF-IDF + 로지스틱 회귀 (오프라인, 의존성 최소).
TODO: 문장 임베딩(예: 다국어 sentence-transformers)으로 교체해 비교.

준비:
    pip install scikit-learn

사용법 (학습 데이터로 학습 → 골든셋 예측):
    python3 src/baselines/embed_classifier.py train.csv golden.csv > predictions.csv
    python3 src/evaluate.py golden.csv predictions.csv
"""
import csv
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r["id"] for r in rows], [r["text"] for r in rows], [
        r.get("label", "").strip() for r in rows
    ]


def main(train_path: str, eval_path: str) -> None:
    _, train_texts, train_labels = load(train_path)
    eval_ids, eval_texts, _ = load(eval_path)

    # 문자 n-gram — 한국어 교착어 특성상 어절 단위보다 강건하다
    model = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), min_df=2),
        LogisticRegression(max_iter=1000),
    )
    model.fit(train_texts, train_labels)

    writer = csv.writer(sys.stdout)
    writer.writerow(["id", "label"])
    for cid, pred in zip(eval_ids, model.predict(eval_texts)):
        writer.writerow([cid, pred])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
