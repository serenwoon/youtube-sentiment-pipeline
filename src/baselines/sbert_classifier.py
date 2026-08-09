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
    python3 src/evaluate.py data/real/golden.csv pred.csv

모델과 C는 학습셋 내부 5-fold 교차검증으로 골랐다. 골든셋은 보지 않았다.

    모델 (LogReg C=1)                        CV macro F1
    jhgan/ko-sroberta-multitask                   0.354  ← 채택
    BM-K/KoSimCSE-roberta                         0.335
    intfloat/multilingual-e5-small                0.314
    snunlp/KR-SBERT-V40K-klueNLI-augSTS           0.285
    jhgan/ko-sbert-multitask                      0.285
    (TF-IDF char_wb(2,4) 비교값)                   0.282

    ko-sroberta + C 탐색                     CV macro F1
    C=0.1  0.338 / C=1  0.354 / C=3  0.387 ← 채택
    C=10   0.365 / C=30 0.377
    LinearSVC C=1  0.369

C 값들 사이의 차이는 학습 154건 기준으로 노이즈 범위와 겹친다. C=3을 골랐지만
0.354~0.387 구간은 사실상 구분되지 않는다고 보는 게 맞다.
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
