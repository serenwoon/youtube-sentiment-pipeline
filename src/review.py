#!/usr/bin/env python3
"""골든셋에서 규칙 1에 걸릴 만한 것을 다시 본다.

라벨링할 때 규칙 1(평가 대상이 차가 아니면 제외)을 놓치기 쉽다.
광고, 타임스탬프, 리뷰어·채널 얘기처럼 걸릴 법한 것만 골라서 다시 묻는다.

패턴은 후보를 좁히는 용도일 뿐 판정하지 않는다. 판정은 사람이 한다.

사용법:
    python3 src/review.py data/real/golden.csv

키:
    k 그대로 둔다 (차에 대한 평가가 맞다)
    x 제외한다 (규칙 1 - 차 얘기가 아니다)
    q 종료
"""
import csv
import os
import re
import sys
import termios
import tty

PATTERNS = [
    ("광고·링크", r"https?://|견적 무료|광고문의|참여신청|품절 임박|할인 중|이벤트 참여"),
    ("질문·요청", r"\?|궁금|알려주|부탁드|해주시면|가르쳐|어떤가요|인가요|나요\s*$"),
    ("리뷰어·채널", r"형님|모카|우파|구독|영상 잘|잘 ?봤|잘 ?보고|편집|목소리|채널"
                    r"|유튜버|차주분|차주님|리뷰 잘|감사합니다"),
    ("브랜드·국가", r"중국차|중국 전기차|쭝국|짱깨|현기차|현기 |국산차|한국차"),
    ("타임스탬프", r"\d{1,2}:\d{2}"),
    ("인물·정치", r"이명박|윤석열|문재인|대통령|정치"),
]


def read_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main(gold_path):
    if not sys.stdin.isatty():
        sys.exit("키 입력을 받아야 하니 터미널에서 직접 실행한다.")

    out_dir = os.path.dirname(gold_path) or "."
    excl_path = os.path.join(out_dir, "excluded.csv")

    with open(gold_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    flagged = []
    for row in rows:
        for name, pat in PATTERNS:
            if re.search(pat, row["text"]):
                flagged.append((name, row))
                break

    if not flagged:
        print("다시 볼 항목이 없다.")
        return

    print(f"골든셋 {len(rows)}건 중 {len(flagged)}건이 패턴에 걸렸다.")
    print("차에 대한 평가가 맞으면 k, 아니면 x.\n")

    removed = []
    for i, (name, row) in enumerate(flagged, 1):
        print(f"[{i}/{len(flagged)}] ({name}) 현재 라벨: {row['label']}")
        print(f"  {row['text'][:200]}")
        while True:
            key = read_key().lower()
            if key in ("k", "x", "q"):
                break
        print()
        if key == "q":
            break
        if key == "x":
            removed.append(row)

    if not removed:
        print("제외할 것이 없었다. 그대로 둔다.")
        return

    keep_ids = {r["id"] for r in removed}
    with open(gold_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "label"])
        w.writeheader()
        w.writerows(r for r in rows if r["id"] not in keep_ids)

    existing = []
    if os.path.exists(excl_path):
        with open(excl_path, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    with open(excl_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "label"])
        w.writeheader()
        w.writerows(existing)
        w.writerows({"id": r["id"], "label": "대상아님"} for r in removed)

    kept = len(rows) - len(removed)
    total = kept + len(existing) + len(removed)
    print(f"{len(removed)}건을 제외로 옮겼다. 골든셋 {kept}건.")
    print(f"제외율 {(len(existing) + len(removed)) / total:.1%}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
