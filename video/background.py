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

# シーンごとにランダムに使う背景プール。自然全般＋人物（オーナー指定）。
# 人物素材はPexelsライセンス上、装飾背景としての商用利用は許可されている
# （映っている人物が当サービスを推奨しているかのような見せ方だけが禁止）。
NATURE_QUERIES = [
    # 海（ブランドの基調）
    "ocean waves slow motion",
    "underwater ocean",
    "sea surface",
    "ocean aerial view",
    # 海以外の自然
    "forest sunlight",
    "mountain aerial",
    "waterfall nature",
    "sunset sky clouds",
    "rain window",
]
PEOPLE_QUERIES = [
    "young japanese woman smiling",
    "beautiful woman portrait",
    "woman city walking",
    "woman using smartphone",
    "woman cafe relaxing",
]

# プール4本のうち人物素材に確保する本数（オーナー指定で半分に引き上げ。2026-08-16）。
PEOPLE_SLOTS = 2
QUERIES = NATURE_QUERIES + PEOPLE_QUERIES

# ループの継ぎ目が目立たない最短尺。3秒素材を12秒のシーンで4周させると安っぽくなる。
MIN_DURATION_SEC = 7

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
        if (video.get("duration") or 0) < MIN_DURATION_SEC:
            continue
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


def _search(key: str, query: str) -> "dict | None":
    """queryで検索し、使える動画ファイルを1つ返す。無ければNone。"""
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
        return pick_video_file(resp.json().get("videos", []))
    except Exception as e:
        print(f"  ⚠ Pexels動画検索例外: {e}")
        return None


def _download(url: str, path: str) -> "int | None":
    """urlをpathへ保存し書き込みバイト数を返す。サイズ超過・失敗時はNone。"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
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
        return written
    except Exception as e:
        print(f"  ⚠ 背景動画ダウンロード例外: {e}")
        return None


def fetch_pool(out_dir: str, count: int = 4) -> list:
    """異なるクエリからcount本を目標に背景動画を集め、
    [{"filename", "durationSec"}, ...] を返す（0本なら空リスト）。
    クエリはシャッフルして順に試し、検索やダウンロードに失敗したものは飛ばす。
    count本に届かなくても取れたぶんだけ返す（シーン側で使い回す）。"""
    key = _api_key()
    if key is None:
        print("[background] PEXELS_API_KEY 未設定のため背景動画をスキップします")
        return []

    pool = []
    # 人物枠をPEOPLE_SLOTS本確保し、残りは自然系。どちらもシャッフルして毎日変える。
    # 人物クエリが全滅した場合は自然系だけでcount本まで埋める（枠は保証ではなく優先）。
    people_queries = random.sample(PEOPLE_QUERIES, len(PEOPLE_QUERIES))
    nature_queries = random.sample(NATURE_QUERIES, len(NATURE_QUERIES))
    plan = [(q, True) for q in people_queries[:PEOPLE_SLOTS]] \
        + [(q, False) for q in nature_queries] \
        + [(q, True) for q in people_queries[PEOPLE_SLOTS:]]
    people_count = 0
    for query, is_people in plan:
        if len(pool) >= count:
            break
        # 人物の取得済み本数が枠に達したら、以降の人物クエリは飛ばして自然系を優先
        if is_people and people_count >= PEOPLE_SLOTS:
            continue
        picked = _search(key, query)
        if picked is None:
            continue
        filename = f"bg_{len(pool)}.mp4"
        written = _download(picked["file"]["link"], os.path.join(out_dir, filename))
        if written is None:
            continue
        duration = float(picked["duration"] or 10)
        kind = "人物" if is_people else "自然"
        print(f"[background] 背景動画を取得[{kind}]: {query} ({written / 1024 / 1024:.1f} MB / {duration:.0f}s)")
        pool.append({"filename": filename, "durationSec": duration, "people": is_people})
        if is_people:
            people_count += 1
    return pool


def assign_backgrounds(scenes: list, pool: list) -> None:
    """各シーンにプールから背景をランダム割当する（その場で書き込み）。
    先頭シーン（hook）は人物素材を優先する（オーナー指定: 冒頭は人物が良い）。
    同じ映像が連続すると切り替わりのカット感が消えるため、プールが2本以上あれば
    直前のシーンと同じものは選ばない。"""
    if not pool:
        return
    prev = None
    for i, scene in enumerate(scenes):
        candidates = [b for b in pool if b is not prev] if len(pool) > 1 else pool
        if i == 0:
            people = [b for b in candidates if b.get("people")]
            if people:
                candidates = people
        chosen = random.choice(candidates)
        scene["backgroundVideo"] = chosen["filename"]
        scene["backgroundVideoDurationSec"] = chosen["durationSec"]
        prev = chosen
