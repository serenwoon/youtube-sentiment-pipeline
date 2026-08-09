#!/usr/bin/env python3
"""베이스라인 1 — 어휘 규칙.

감정 어휘 목록을 손으로 적고, 긍정어와 부정어가 둘 다 나오면 '복합'으로 본다.
학습하지 않는다. 표준 라이브러리만 쓴다.

사용법:
    python3 src/baselines/lexicon.py data/real/golden.csv > pred.csv
    python3 src/evaluate.py data/real/golden.csv pred.csv

어휘는 자동차 리뷰 도메인 지식으로 적은 것이지 골든셋에서 뽑은 게 아니다.
골든셋을 보고 어휘를 고르면 그 위에서 재는 점수가 무의미해진다.

이 베이스라인의 쓸모는 점수 자체가 아니라 어디서 깨지는지에 있다.
"기대만큼은 아니지만 나쁘지 않았어요" 같은 문장은 어휘로 잡히지 않는다 —
그게 통계 모델이나 LLM이 필요한 이유를 구체적으로 보여준다.
"""
import csv
import re
import sys

POSITIVE = [
    "좋", "만족", "괜찮", "최고", "훌륭", "편하", "편안", "조용", "부드럽",
    "넉넉", "쾌적", "깔끔", "예쁘", "이쁘", "가성비", "혜자", "추천", "굿",
    "낫다", "낮네", "잘나", "잘 나", "감동", "놀랐", "대박", "탁월", "우수",
]

NEGATIVE = [
    "아쉽", "별로", "실망", "불편", "시끄럽", "소음", "답답", "좁", "비싸",
    "구리", "싸구려", "허접", "최악", "심하", "안좋", "안 좋", "개악", "먹통",
    "고장", "결함", "하자", "떨어지", "부족", "무겁", "딱딱", "거슬리", "짜증",
]

# 앞의 서술을 뒤집는 연결어. 있으면 양쪽 감정이 공존할 가능성이 높다
CONTRAST = ["지만", "는데", "however", "however,", "다만", "대신", "반면"]


def classify(text: str) -> str:
    t = text.lower()
    pos = sum(1 for w in POSITIVE if w in t)
    neg = sum(1 for w in NEGATIVE if w in t)
    contrast = any(c in t for c in CONTRAST)

    if pos and neg:
        return "복합"
    # 한쪽 어휘만 있어도 역접이 있으면 반대 감정이 생략된 경우가 많다
    # ("승차감은 좋은데" — 뒤가 잘려도 아쉬움이 함축된다)
    if contrast and (pos or neg):
        return "복합"
    if pos:
        return "긍정"
    if neg:
        return "부정"
    # 어휘가 하나도 안 걸리면 다수 클래스로 넘긴다
    return "긍정"


def main(gold_path: str) -> None:
    with open(gold_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    writer = csv.writer(sys.stdout)
    writer.writerow(["id", "label"])
    unmatched = 0
    for row in rows:
        t = row["text"].lower()
        if not any(w in t for w in POSITIVE + NEGATIVE):
            unmatched += 1
        writer.writerow([row["id"], classify(row["text"])])

    print(f"어휘 미검출 {unmatched}/{len(rows)}건 ({unmatched / len(rows):.1%}) "
          f"— 이들은 다수 클래스로 처리됐다", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
