# 결과

분류 방식별 비교. 전부 같은 골든셋 200건을 `src/evaluate.py`로 평가한 값이다.

| 방법 | macro F1 | weighted F1 | 비용/1K건 | 지연 | 비고 |
| --- | --- | --- | --- | --- | --- |
| 다수 클래스 | | | 0 | — | 하한선 |
| LLM zero-shot | | | | | `src/baselines/zero_shot_llm.py` |
| 임베딩 + 경량 분류기 | | | | | `src/baselines/embed_classifier.py` |
| 오픈모델 + LoRA | | | | | 계획 |

아직 비어 있다. 실제로 돌린 값만 채운다.

## 보는 법

주 지표는 macro F1. 이 데이터는 긍정이 많고 '복합'이 제일 어려워서, weighted F1은 쉬운 클래스 성적에 가려진다.

다수 클래스 행부터 채운다. 전부 '긍정'이라고 답하는 분류기의 점수다. 이게 없으면 나머지 숫자가 좋은 건지 나쁜 건지 판단할 기준이 없다.

```bash
python3 src/baselines/majority.py data/real/golden.csv > /tmp/majority.csv
python3 src/evaluate.py data/real/golden.csv /tmp/majority.csv
```

이 값은 골든셋의 클래스 분포에 따라 크게 달라진다. 긍정이 몰려 있으면 높게 나오고, 고르면 낮게 나온다. 그래서 **점수와 함께 분포도 적어둔다.** 나중에 "왜 이때는 0.4였지" 하고 헷갈리지 않으려면 필요하다.

비용과 지연도 같이 적는다. 정확도만 보면 항상 큰 모델이 이기는데 실제로는 그렇게 고르지 않는다.

## 기록할 것

실행할 때마다 모델명과 버전, 프롬프트나 하이퍼파라미터, 날짜, 골든셋 버전을 남긴다. 몇 달 뒤에 이 숫자가 어떻게 나온 건지 답할 수 없으면 그 행은 지우는 게 낫다.
