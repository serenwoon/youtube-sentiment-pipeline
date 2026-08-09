#!/usr/bin/env python3
"""베이스라인 3 — 한국어 문장 임베딩 + 로지스틱 회귀.

TF-IDF가 무작위 추측을 못 넘긴 게 이 스크립트를 쓴 이유다. 문자 n-gram은
"기대만큼은 아니지만 나쁘지 않았어요"에서 아무 신호도 못 뽑는다. 문장 임베딩은
대규모 코퍼스에서 배운 의미 표현을 가져오므로, 학습 데이터가 150건뿐이어도
그 위에 얹은 선형 분류기가 동작할 여지가 생긴다.

로컬 모델이라 API 키가 필요 없다. 첫 실행 때 모델을 내려받는다(약 440MB).

준비:
    pip install sentence-transformers scikit-learn

사용법:
    python3 src/baselines/sbert_classifier.py data/real/train.csv data/real/golden.csv > pred.csv
    python3 src/evaluate.py --bootstrap data/real/golden.csv pred.csv

모델과 C는 학습셋 내부 5-fold 교차검증으로 골랐다. 골든셋은 보지 않았다.

C 탐색은 시드 20개로 다시 쟀다 (ko-sroberta 고정, 폴드 시드 0–19):

    설정             평균   최소–최대   시드별 1위
    LogReg C=0.1    0.319  0.257–0.357   1/20
    LogReg C=1      0.331  0.278–0.385   1/20
    LogReg C=3      0.341  0.278–0.406   2/20  ← 채택
    LogReg C=10     0.347  0.308–0.392   1/20
    LogReg C=30     0.359  0.293–0.422  11/20
    LinearSVC C=1   0.342  0.294–0.382   4/20

C=30의 평균이 제일 높지만 여섯 설정의 구간이 전부 겹치고, 한 설정의 시드 간
편차(±0.03)가 설정 사이의 차이보다 크다. 학습 154건에서 이 표로 고를 수 있는
것은 없다는 뜻이다. 처음 고른 C=3을 그대로 둔다 — 평균 순위를 따라 C=30으로
갈아타면 개선이 아니라 노이즈를 좇는 것이다.

아래 모델 비교는 시드 하나로만 쟀다. 모델마다 수백 MB를 내려받아야 해서
C 표처럼 다시 돌리지 않았고, 그래서 이 순위는 그대로 믿을 게 못 된다.
ko-sroberta를 쓰는 근거는 "1위여서"가 아니라 "이 급에서 아무거나 하나"에 가깝다.

    모델 (LogReg C=1, 시드 1개)              CV macro F1
    jhgan/ko-sroberta-multitask                   0.354  ← 채택
    BM-K/KoSimCSE-roberta                         0.335
    intfloat/multilingual-e5-small                0.314
    snunlp/KR-SBERT-V40K-klueNLI-augSTS           0.285
    jhgan/ko-sbert-multitask                      0.285
"""
import csv
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL = "jhgan/ko-sroberta-multitask"
C = 3.0


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows


def main(train_path: str, eval_path: str) -> None:
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    train, evalset = load(train_path), load(eval_path)
    encoder = SentenceTransformer(MODEL)

    def encode(rows):
        return encoder.encode([r["text"] for r in rows], batch_size=32,
                              show_progress_bar=False, normalize_embeddings=True)

    model = LogisticRegression(max_iter=5000, class_weight="balanced", C=C)
    model.fit(encode(train), [r["label"].strip() for r in train])

    writer = csv.writer(sys.stdout)
    writer.writerow(["id", "label"])
    for row, pred in zip(evalset, model.predict(encode(evalset))):
        writer.writerow([row["id"], pred])

    print(f"{MODEL} / LogReg C={C} / 학습 {len(train)}건", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
