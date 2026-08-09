#!/usr/bin/env python3
"""골든셋 라벨링 도구.

한 건씩 보여주고 키 하나로 라벨을 매긴다. 중간에 끊어도 이어서 할 수 있다.

사용법:
    python3 src/label.py data/real/candidates.csv data/real/golden.csv

입력 candidates.csv: id,text[,video_id,lang]
출력 golden.csv:     id,text,label
    같은 폴더에 excluded.csv(제외 기록)와 hard_cases.md(애매했던 것)도 쌓인다.

키:
    1 긍정    2 부정    3 복합
    x 제외 — 평가 대상이 아님 (가이드 규칙 1)
    h 애매함 — 제외하고 hard_cases.md에 남긴다
    u 직전 항목 취소
    q 종료 (지금까지 한 것은 저장돼 있다)

판정 기준은 docs/labeling-guide.md. 시작 전에 읽는다.
"""
import csv
import os
import sys
import termios
import tty
from collections import Counter

LABELS = {"1": "긍정", "2": "부정", "3": "복합"}


def read_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_done(path):
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["id"]: r["label"] for r in csv.DictReader(f)}


def append_row(path, fieldnames, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new:
            w.writeheader()
        w.writerow(row)


def rewrite(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main(cand_path, gold_path):
    if not sys.stdin.isatty():
        sys.exit("키 입력을 받아야 하니 터미널에서 직접 실행한다.")
    out_dir = os.path.dirname(gold_path) or "."
    excl_path = os.path.join(out_dir, "excluded.csv")
    hard_path = os.path.join(out_dir, "hard_cases.md")

    rows = load_rows(cand_path)
    done = load_done(gold_path)
    excluded = load_done(excl_path)
    seen = set(done) | set(excluded)
    todo = [r for r in rows if r["id"] not in seen]

    if not todo:
        print(f"남은 항목이 없다. 라벨 {len(done)}건, 제외 {len(excluded)}건.")
        return

    print(f"후보 {len(rows)}건 중 {len(todo)}건 남음. "
          f"(라벨 {len(done)} / 제외 {len(excluded)})")
    print("1 긍정  2 부정  3 복합  x 제외  h 애매  u 취소  q 종료\n")

    history = []
    i = 0
    while i < len(todo):
        row = todo[i]
        counts = Counter(done.values())
        bar = " ".join(f"{k} {counts.get(k, 0)}" for k in LABELS.values())
        print(f"[{i + 1}/{len(todo)}]  {bar}   제외 {len(excluded)}")
        print(f"  {row['text']}")
        if row.get("video_id"):
            print(f"  ({row['video_id']})")

        key = read_key().lower()
        print()

        if key == "q":
            break
        if key == "u":
            if not history:
                print("취소할 게 없다.\n")
                continue
            prev_i, prev_kind, prev_id = history.pop()
            if prev_kind == "label":
                done.pop(prev_id, None)
                rewrite(gold_path, ["id", "text", "label"],
                        [{"id": k, "text": t, "label": v}
                         for k, v, t in [(r["id"], done[r["id"]], r["text"])
                                         for r in rows if r["id"] in done]])
            else:
                excluded.pop(prev_id, None)
                rewrite(excl_path, ["id", "label"],
                        [{"id": k, "label": v} for k, v in excluded.items()])
            i = prev_i
            print("직전 항목을 되돌렸다.\n")
            continue

        if key in LABELS:
            label = LABELS[key]
            done[row["id"]] = label
            append_row(gold_path, ["id", "text", "label"],
                       {"id": row["id"], "text": row["text"], "label": label})
            history.append((i, "label", row["id"]))
        elif key in ("x", "h"):
            reason = "애매" if key == "h" else "대상아님"
            excluded[row["id"]] = reason
            append_row(excl_path, ["id", "label"],
                       {"id": row["id"], "label": reason})
            if key == "h":
                with open(hard_path, "a", encoding="utf-8") as f:
                    f.write(f"- `{row['id']}` {row['text']}\n")
            history.append((i, "exclude", row["id"]))
        else:
            print("모르는 키다. 1/2/3/x/h/u/q\n")
            continue

        i += 1

    counts = Counter(done.values())
    print(f"\n라벨 {len(done)}건 — " +
          ", ".join(f"{k} {counts.get(k, 0)}" for k in LABELS.values()))
    print(f"제외 {len(excluded)}건 "
          f"(제외율 {len(excluded) / max(len(done) + len(excluded), 1):.1%})")
    print(f"저장: {gold_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
