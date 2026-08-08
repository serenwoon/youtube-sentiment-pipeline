#!/usr/bin/env python3
"""베이스라인 ①: LLM zero-shot 분류.

학습 없이 프롬프트만으로 어디까지 되는지 잰다. 다른 방법들과
반드시 같은 골든셋 위에서 비교한다 — 자가 바뀌면 비교가 아니다.

준비:
    pip install anthropic
    export ANTHROPIC_API_KEY=...

사용법:
    python3 src/baselines/zero_shot_llm.py data/sample_comments.csv > predictions.csv
    python3 src/evaluate.py data/sample_comments.csv predictions.csv
"""
import csv
import sys

import anthropic

LABELS = ["긍정", "부정", "복합"]
MODEL = "claude-opus-5"

SYSTEM = """유튜브 댓글의 감정을 분류한다. 라벨 정의:
- 긍정: 영상/채널에 대한 평가가 긍정 단일 방향
- 부정: 평가가 부정 단일 방향
- 복합: 긍정과 부정이 한 댓글 안에 공존 (예: "노래는 좋은데 무대는 아쉽다")"""

SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": LABELS}},
    "required": ["label"],
    "additionalProperties": False,
}


def classify(client: anthropic.Anthropic, text: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        system=SYSTEM,
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": f"댓글: {text}"}],
    )
    if response.stop_reason == "refusal":
        return "복합"  # 분류 불가 시 보수적 기본값
    import json

    text_block = next(b.text for b in response.content if b.type == "text")
    return json.loads(text_block)["label"]


def main(gold_path: str) -> None:
    client = anthropic.Anthropic()
    writer = csv.writer(sys.stdout)
    writer.writerow(["id", "label"])
    with open(gold_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            writer.writerow([row["id"], classify(client, row["text"])])
            print(f"{row['id']} 완료", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
