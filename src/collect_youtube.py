#!/usr/bin/env python3
"""유튜브 댓글 수집 뼈대 — YouTube Data API v3 commentThreads.

2023년 캡스톤에서는 Selenium으로 스크롤·정렬을 자동화해 수집했다.
여기서는 공식 Data API를 쓴다 — 플랫폼 약관을 지키고, 페이지 구조 변경에 깨지지 않는다.

사용 전 준비:
    1. Google Cloud Console에서 YouTube Data API v3 키 발급
    2. export YOUTUBE_API_KEY=...

사용법:
    python3 src/collect_youtube.py <VIDEO_ID> > data/real/raw_comments.csv

주의: 수집 원문은 data/real/ (gitignore됨) 밖으로 내보내지 않는다.
"""
import csv
import json
import os
import sys
import urllib.parse
import urllib.request

API_URL = "https://www.googleapis.com/youtube/v3/commentThreads"


def fetch_comments(video_id, api_key, max_pages=5):
    page_token = None
    for _ in range(max_pages):
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": 100,
            "textFormat": "plainText",
            "key": api_key,
        }
        if page_token:
            params["pageToken"] = page_token
        with urllib.request.urlopen(f"{API_URL}?{urllib.parse.urlencode(params)}") as resp:
            data = json.load(resp)
        for item in data.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            yield {
                "id": item["id"],
                "text": snippet["textDisplay"].replace("\n", " "),
                "like_count": snippet["likeCount"],
                "published_at": snippet["publishedAt"],
            }
        page_token = data.get("nextPageToken")
        if not page_token:
            break


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        sys.exit("YOUTUBE_API_KEY 환경변수를 설정하세요")
    writer = csv.DictWriter(
        sys.stdout, fieldnames=["id", "text", "like_count", "published_at"]
    )
    writer.writeheader()
    for row in fetch_comments(sys.argv[1], api_key):
        writer.writerow(row)


if __name__ == "__main__":
    main()
