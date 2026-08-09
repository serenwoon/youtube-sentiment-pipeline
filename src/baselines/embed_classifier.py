#!/usr/bin/env python3
"""베이스라인 2 — TF-IDF + 로지스틱 회귀.

감정분류 정도의 좁은 태스크면 LLM보다 가벼운 분류기가 더 싸고 다루기 쉬울 것
같아서 확인해보는 것. 오프라인이고 의존성은 scikit-learn 하나다.

결과부터 적으면 이 표현으로는 안 됐다. 골든셋 macro F1 0.334로 무작위
추측(0.329)과 구분되지 않는다. 문자 n-gram이 부족하다는 뜻이고, 그래서
표현을 문장 임베딩으로 바꿔본 것이 sbert_classifier.py다.

준비:
    pip install scikit-learn

사용법 (학습 데이터로 학습 → 골든셋 예측):
    python3 src/baselines/embed_classifier.py train.csv golden.csv > predictions.csv
    python3 src/evaluate.py --bootstrap golden.csv predictions.csv
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

    # 설정은 학습셋 내부 5-fold 교차검증으로 골랐다. 골든셋은 보지 않았다.
    # 시드 20개 평균 — python3 src/tune.py data/real/train.csv 로 재현된다.
    #
    #   설정                                    평균   최소–최대    1위
    #   char_wb(1,3) min_df=2                  0.229  0.224–0.247   0/20
    #   char_wb(2,4) min_df=1                  0.229  0.228–0.231   0/20
    #   char_wb(2,4) min_df=1 + class_weight   0.253  0.212–0.282   4/20  ← 채택
    #   char_wb(2,5) min_df=1 + cw + C=5       0.250  0.221–0.283   5/20
    #   word(1,2) min_df=1 + class_weight      0.259  0.222–0.298  11/20
    #
    # 읽는 법. class_weight가 있고 없고는 갈린다(0.229 → 0.25대). 있는 것들끼리는
    # 갈리지 않는다 — 폭이 전부 겹치고 1위가 시드 따라 바뀐다. 채택한 설정이
    # 20번 중 4번만 1위인 게 그 뜻이다. 셋 중 무엇을 써도 같다고 보고, 처음
    # 고른 것을 그대로 둔다. 평균이 제일 높다는 이유로 word(1,2)로 갈아타면
    # 그건 개선이 아니라 노이즈를 따라간 것이다.
    #
    # 문자 n-gram을 쓰는 건 한국어가 교착어라 어절 단위가 잘게 쪼개지기 때문이다.
    # class_weight가 필요한 건 학습셋의 긍정 비중이 51.9%라 가중치 없이는
    # 분류기가 전부 긍정으로 찍는 쪽으로 수렴하기 때문이다.
    model = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
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
