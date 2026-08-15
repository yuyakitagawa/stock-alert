"""
video/background.py

動画の背景に流す自然映像（縦向き）を Pexels Videos API から取得する。

Pexels はアイキャッチ画像（publish_blog_articles.py）で既に使っている `PEXELS_API_KEY`
をそのまま再利用する。Pexels の動画は無料・商用利用可・クレジット表記不要。
キー未設定・検索失敗・ダウンロード失敗時は None を返し、Remotion 側は従来の
グラデーション背景にフォールバックする（背景のために動画投稿を止めない）。

検索クエリはクジラウォッチ（海）のブランドに合わせた海系の自然映像に限定し、
毎回ランダムに1本選ぶことで「毎日同じ背景」になるのを避ける。
"""
import os
import random

import requests

SEARCH_URL = "https://api.pexels.com/videos/search"

# 海系に限定（サイトのブランドが「クジラ＝海」のため。森や山だと文脈が繋がらない）。
QUERIES = [
    "ocean waves slow motion",
    "underwater ocean",
    "deep blue sea",
    "ocean aerial view",
    "sea surface",
]

# 縦動画の背景として十分な解像度。これ未満の動画ファイルは引き伸ばしでボケるため使わない。
MIN_HEIGHT = 1280
# ダウンロードサイズの安全上限（CIの帯域・時間を食い過ぎないように）。
MAX_BYTES = 80 * 1024 * 1024


def _api_key() -> "str | None":
    return os.getenv("PEXELS_API_KEY") or None


def pick_video_file(videos: list) -> "dict | None":
    """検索結果から背景に使える動画ファイル（縦向き・十分な解像度・サイズ上限内）を
    1つ選ぶ。候補が複数あれば動画単位でランダムに選び、ファイルは
    「MIN_HEIGHT以上で最も小さい」ものを採る（背景用途に4Kは過剰なため）。"""
    candidates = []
    for video in videos:
        files = [
            f for f in video.get("video_files", [])
            if f.get("height") and f.get("width")
            and f["height"] >= MIN_HEIGHT and f["height"] > f["width"]  # 縦向きのみ
        ]
        if not files:
            continue
        files.sort(key=lambda f: f["height"])
        candidates.append({"file": files[0], "duration": video.get("duration") or 0})
    if not candidates:
        return None
    return random.choice(candidates)


def fetch(out_dir: str) -> "dict | None":
    """自然映像を1本ダウンロードし {"filename", "durationSec"} を返す。失敗時 None。"""
    key = _api_key()
    if key is None:
        print("[background] PEXELS_API_KEY 未設定のため背景動画をスキップします")
        return None

    query = random.choice(QUERIES)
    try:
        resp = requests.get(
            SEARCH_URL,
            headers={"Authorization": key},
            params={"query": query, "orientation": "portrait", "per_page": 15},
            timeout=20,
        )
        if not resp.ok:
            print(f"  ⚠ Pexels動画検索失敗 HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        picked = pick_video_file(resp.json().get("videos", []))
        if picked is None:
            print(f"  ⚠ 縦向きの背景候補が見つかりませんでした（query={query}）")
            return None
    except Exception as e:
        print(f"  ⚠ Pexels動画検索例外: {e}")
        return None

    url = picked["file"]["link"]
    filename = "background.mp4"
    path = os.path.join(out_dir, filename)
    try:
        os.makedirs(out_dir, exist_ok=True)
        with requests.get(url, stream=True, timeout=120) as dl:
            if not dl.ok:
                print(f"  ⚠ 背景動画ダウンロード失敗 HTTP {dl.status_code}")
                return None
            written = 0
            with open(path, "wb") as f:
                for chunk in dl.iter_content(chunk_size=1 << 20):
                    written += len(chunk)
                    if written > MAX_BYTES:
                        print("  ⚠ 背景動画がサイズ上限を超えたため中止します")
                        return None
                    f.write(chunk)
    except Exception as e:
        print(f"  ⚠ 背景動画ダウンロード例外: {e}")
        return None

    duration = picked["duration"] or 10
    print(f"[background] 背景動画を取得: {query} ({written / 1024 / 1024:.1f} MB / {duration}s)")
    return {"filename": filename, "durationSec": float(duration)}
