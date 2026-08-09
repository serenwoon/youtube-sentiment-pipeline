#!/usr/bin/env python3
"""수집한 댓글에서 라벨링 후보를 뽑는다.

영상별로 고르게 뽑아야 특정 영상의 특성을 평가하는 꼴이 안 된다.
시드를 고정하니 같은 입력이면 같은 후보가 나온다.

사용법:
    python3 src/sample_for_labeling.py -n 260 data/real/raw_*.csv > data/real/candidates.csv

    # 학습셋을 따로 뽑을 때 — 골든셋에 이미 쓴 것은 제외한다
    python3 src/sample_for_labeling.py -n 220 \
        --exclude data/real/candidates.csv \
        data/real/raw_*.csv > data/real/train_candidates.csv

입력: id,video_id,text[,like_count,published_at]
출력: id,video_id,text

--exclude에 넘긴 파일들의 id는 후보에서 빠진다. 학습셋과 평가셋이
한 건이라도 겹치면 그 위에서 잰 점수는 의미가 없다.

목표 200건인데 260건쯤 뽑는 이유는 라벨링 중에 규칙 1로 빠지는 게
나오기 때문이다. 제외율을 모를 때는 30% 정도 여유를 둔다.

명백한 쓰레기(빈 문자열, 링크만, 이모지만)는 여기서 먼저 걸러낸다.
몇 건을 걸렀는지는 stderr에 찍히니 제외율 계산할 때 같이 적어둔다.
"""
import argparse
import csv
import random
import re
import sys
import unicodedata
from collections import defaultdict

URL_ONLY = re.compile(r"^\s*(https?://\S+\s*)+$")
MENTION_OR_TAG_ONLY = re.compile(r"^\s*([@#]\S+\s*)+$")


def is_junk(text):
    """라벨러에게 보여줄 가치가 없는 것만 거른다. 판단이 필요한 건 남긴다."""
    t = text.strip()
    if len(t) < 5:
        return True
    if URL_ONLY.match(t) or MENTION_OR_TAG_ONLY.match(t):
        return True
    # 문자·숫자가 하나도 없으면 이모지나 구두점뿐이다
    return not any(unicodedata.category(c)[0] in ("L", "N") for c in t)


def normalize(text):
    return re.sub(r"\s+", " ", text.strip()).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("-n", type=int, default=260, help="뽑을 건수 (기본 260)")
    ap.add_argument("--seed", type=int, default=20260809)
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="이 파일들의 id는 후보에서 뺀다 (학습/평가 분리)")
    args = ap.parse_args()

    excluded_ids = set()
    for path in args.exclude:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("id"):
                    excluded_ids.add(row["id"])

    by_video = defaultdict(list)
    seen_in_video = defaultdict(set)
    total = junk = dup = 0

    skipped = 0
    for path in args.files:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                total += 1
                if row["id"] in excluded_ids:
                    skipped += 1
                    continue
                text = row["text"]
                if is_junk(text):
                    junk += 1
                    continue
                vid = row.get("video_id", "?")
                key = normalize(text)
                # 중복 제거는 영상 안에서만 한다. 전역으로 하면 먼저 읽은
                # 영상이 흔한 표현("연비 좋네요")을 다 가져가고 나머지가 굶는다
                if key in seen_in_video[vid]:
                    dup += 1
                    continue
                seen_in_video[vid].add(key)
                by_video[vid].append(row)

    rng = random.Random(args.seed)
    pool = []
    for vid, rows in sorted(by_video.items()):
        rng.shuffle(rows)
        pool.append(rows)

    # 영상별로 한 건씩 돌아가며 뽑는다. 댓글 수가 적은 영상도 섞이도록.
    # 뽑는 시점에 영상 간 중복을 걸러서 골든셋에 같은 문장이 두 번 안 들어가게 한다
    picked = []
    picked_text = set()
    cross_dup = 0
    idx = 0
    while len(picked) < args.n and any(idx < len(r) for r in pool):
        for rows in pool:
            if idx >= len(rows) or len(picked) >= args.n:
                continue
            row = rows[idx]
            key = normalize(row["text"])
            if key in picked_text:
                cross_dup += 1
                continue
            picked_text.add(key)
            picked.append(row)
        idx += 1
    dup += cross_dup

    writer = csv.DictWriter(sys.stdout, fieldnames=["id", "video_id", "text"])
    writer.writeheader()
    for row in picked:
        writer.writerow({k: row.get(k, "") for k in ("id", "video_id", "text")})

    per_video = defaultdict(int)
    for row in picked:
        per_video[row.get("video_id", "?")] += 1

    ex = f"기사용 {skipped} 제외, " if excluded_ids else ""
    print(f"수집 {total}건 → {ex}쓰레기 {junk} 제외, 중복 {dup} 제외 "
          f"→ 후보 {len(picked)}건", file=sys.stderr)
    print(f"영상별: {dict(per_video)}", file=sys.stderr)
    print(f"시드 {args.seed}", file=sys.stderr)


if __name__ == "__main__":
    main()
