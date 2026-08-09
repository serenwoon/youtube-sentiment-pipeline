# youtube-sentiment-pipeline

[![ci](https://github.com/serenwoon/youtube-sentiment-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/serenwoon/youtube-sentiment-pipeline/actions/workflows/ci.yml)

자동차 리뷰 영상 댓글을 수집해 감정을 3분류하고, 그 결과를 골든셋으로 평가하는 파이프라인.

2023년 캡스톤에서 댓글 수집과 Superset 대시보드를 맡았다. 감정분석은 AWS Comprehend에 맡겼고, 결과가 맞는지는 따로 확인하지 않았다. 대시보드에 언어별 막대가 서긴 했는데 그 높이가 맞는지 알 방법이 없었다. 그 부분을 다시 만들면서 평가 단계를 넣었다.

## 실행

평가 하네스는 표준 라이브러리만 쓴다. 설치할 것이 없다.

```bash
python3 src/evaluate.py data/sample_comments.csv data/sample_predictions.csv
```

```
평가 24건 — 정확도 0.833

클래스        P       R      F1   지지도
긍정     0.875   0.875   0.875     8
부정     0.875   0.875   0.875     8
복합     0.750   0.750   0.750     8

macro F1    0.833
weighted F1 0.833

혼동 행렬 (행=정답, 열=예측)
          긍정    부정    복합
긍정       7     0     1
부정       0     7     1
복합       1     1     6
```

수집과 분류기는 각각 API 키와 패키지가 필요하다. 각 스크립트 docstring에 적어뒀다.

## 진행 상황

- [x] 평가 하네스
- [x] 라벨링 기준
- [x] 수집·샘플링·라벨링 도구
- [ ] 골든셋 200건
- [ ] 분류기 비교 (zero-shot / 임베딩 분류기)
- [ ] 대시보드

## 구성

```
data/                        샘플 + 스키마 (원문 비공개)
docs/labeling-guide.md       라벨링 기준
results/                     비교 결과
src/collect_youtube.py       수집 (YouTube Data API)
src/sample_for_labeling.py   라벨링 후보 추출
src/label.py                 라벨링 (터미널)
src/evaluate.py              평가 하네스
src/baselines/               분류기
```

## 라벨

긍정 / 부정 / 복합 3분류.

'복합'은 "승차감은 좋은데 실내가 싸구려 같다" 같은 댓글을 위한 것이다. 차는 평가 항목이 여러 개라(승차감·연비·마감·가격) 한쪽만 좋은 경우가 흔한데, 이걸 긍정이나 부정으로 몰면 라벨에 노이즈가 섞인다.

`NEUTRAL`은 두지 않았다. Comprehend 같은 서비스는 보통 주는데, "1등" 이나 "형 목소리 좋다" 같은 댓글은 감정이 중립인 게 아니라 차에 대한 평가가 없는 것이라 성격이 다르다. 한 라벨로 묶으면 분류기가 애매한 걸 전부 거기 넣어도 점수가 유지된다. 그래서 라벨로 흡수하지 않고 수집 단계에서 걸렀다.

판정 규칙은 [docs/labeling-guide.md](docs/labeling-guide.md)에 있다.

## 골든셋 만들기

```bash
export YOUTUBE_API_KEY=...

# 1. 영상별로 수집
python3 src/collect_youtube.py VIDEO_ID > data/real/raw_VIDEO_ID.csv

# 2. 후보 뽑기 (영상별로 고르게, 시드 고정)
python3 src/sample_for_labeling.py -n 260 data/real/raw_*.csv > data/real/candidates.csv

# 3. 라벨링 — 키 하나씩. 중간에 끊어도 이어서 된다
python3 src/label.py data/real/candidates.csv data/real/golden.csv
```

200건이 목표인데 260건을 뽑는 건 라벨링 중에 규칙 1로 빠지는 게 나오기 때문이다. 제외된 건수는 `excluded.csv`에, 판정이 애매했던 건 `hard_cases.md`에 쌓인다.

## 골든셋을 먼저 만드는 이유

분류기를 만든 다음에 평가셋을 뜨면 기준이 모델 쪽으로 휜다. 잘 맞히는 사례를 무의식적으로 고르게 되기 때문이다. 그래서 라벨링 예산의 일부를 평가 전용으로 먼저 떼어둔다.

한 번 학습이나 프롬프트에 섞이면 되돌릴 수 없다는 점도 있다. 그 뒤에 나오는 점수는 자기 답안으로 채점한 값이 된다.

## 데이터

댓글 원문은 커밋하지 않는다. 저작권과 플랫폼 약관 때문이다. 실데이터는 `data/real/`에 두고 이 경로는 gitignore에 있다. 공개하는 건 스키마와 집계, 평가 결과뿐이다.

`data/sample_*.csv`는 형식 확인용 합성 데이터다. 실제 댓글이 아니다.

## 참고

평가셋을 먼저 고정하는 방식은 이전에 혼자 한 감정분석 프로젝트에서 썼던 것이다. 그때는 GPT-3 파인튜닝에 n8n으로 수집부터 산출까지 묶었다. 도구는 다 바뀌었지만 순서는 같다.
