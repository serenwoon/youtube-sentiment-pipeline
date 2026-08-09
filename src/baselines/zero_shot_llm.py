#!/usr/bin/env python3
"""베이스라인 4 — LLM zero-shot 분류.

학습 없이 프롬프트만으로 어디까지 되는지 본다.
다른 방법과 비교할 때는 같은 골든셋을 써야 한다.

아래 SYSTEM은 docs/labeling-guide.md를 그대로 옮긴 것이다. 사람이 매긴 기준과
모델에게 준 기준이 다르면 그 위에서 잰 점수는 두 방법의 차이가 아니라
지시문의 차이를 재게 된다. 가이드를 고치면 여기도 같이 고친다.

준비:
    pip install anthropic
    export ANTHROPIC_API_KEY=...

사용법:
    python3 src/baselines/zero_shot_llm.py data/real/golden.csv > pred.csv
    python3 src/evaluate.py --bootstrap data/real/golden.csv pred.csv

골든셋 한 건에 호출 한 번이다. 200건이면 200회.
"""
import csv
import json
import sys

import anthropic

LABELS = ["긍정", "부정", "복합"]
MODEL = "claude-opus-5"

SYSTEM = """자동차 리뷰 영상(신차 리뷰·시승기·장기 리뷰)에 달린 댓글의 감정을 분류한다.

평가 대상은 차량이다. 리뷰어·채널·영상에 대한 평가가 아니다.

라벨 정의:
- 긍정: 차량에 대한 평가가 긍정 한 방향 (예: "고속에서도 조용해서 놀랐습니다")
- 부정: 차량에 대한 평가가 부정 한 방향 (예: "풍절음이 심해서 통화가 안 됩니다")
- 복합: 긍정과 부정이 한 댓글 안에 같이 있음 (예: "승차감은 좋은데 실내가 싸구려 같다")

판정 기준:
- 반어는 단어의 사전적 극성이 아니라 문장 전체의 실제 평가를 따른다. "참 잘~한다"는 부정이다.
- 한쪽이 확실히 우세하면 그쪽으로 보낸다. "마감은 좀 아쉽지만 승차감이 정말 좋다"는 긍정이다.
  대등하거나 결론이 없을 때만 복합으로 본다.
- 좋아요 수나 인기와 무관하게 본문만 보고 정한다."""

SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": LABELS}},
    "required": ["label"],
    "additionalProperties": False,
}


def classify(client: anthropic.Anthropic, text: str) -> tuple[str, bool]:
    """(라벨, 실패했는지)를 돌려준다."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        system=SYSTEM,
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": f"댓글: {text}"}],
    )
    if response.stop_reason == "refusal":
        return "복합", True
    text_block = next(b.text for b in response.content if b.type == "text")
    return json.loads(text_block)["label"], False


def main(gold_path: str) -> None:
    client = anthropic.Anthropic()
    writer = csv.writer(sys.stdout)
    writer.writerow(["id", "label"])
    total = failed = 0
    with open(gold_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label, refused = classify(client, row["text"])
            writer.writerow([row["id"], label])
            total += 1
            if refused:
                failed += 1
                print(f"{row['id']} 분류 실패 — '복합'으로 채움", file=sys.stderr)

    # 실패를 조용히 기본값으로 덮으면 그만큼 점수가 우연에 기댄다.
    # 몇 건이 그랬는지 남겨야 그 행을 믿을지 판단할 수 있다.
    print(f"{total}건 완료, 분류 실패 {failed}건 ({failed / total:.1%})",
          file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
