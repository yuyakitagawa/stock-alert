"""
video/background.py

動画の背景に流す自然映像（縦向き）を Pexels Videos API から取得する。

Pexels はアイキャッチ画像（publish_blog_articles.py）で既に使っている `PEXELS_API_KEY`
をそのまま再利用する。Pexels の動画は無料・商用利用可・クレジット表記不要。
キー未設定・検索失敗・ダウンロード失敗時は None を返し、Remotion 側は従来の
グラデーション背景にフォールバックする（背景のために動画投稿を止めない）。

実写を使うのは「文字だけのシーン」（company / filer）に限る。金額・保有比率・株価チャートを
読ませるシーンは Remotion 側のブランドグラデーション背景に固定する。実写の明部に数字が
沈む事故（2026-08-19のインフルエンサーレビューで多数指摘）を、可読性の調整ではなく
構造で無くすため。
"""
import os
import random
import re

import requests

SEARCH_URL = "https://api.pexels.com/videos/search"

# 背景プール。人物が写り込まず、暗めで文字が乗る素材が返りやすいクエリに寄せる。
QUERIES = [
    # 海（ブランドの基調）
    "ocean waves aerial",
    "deep sea surface sunlight",
    "ocean waves slow motion",
    # 抽象・都市（金融の文脈に合い、人物が返りにくい）
    "abstract dark blue particles",
    "city skyline night timelapse",
    "ink in water dark",
    # 自然
    "forest canopy aerial",
    "storm clouds timelapse",
]

# Pexels の動画URLは https://www.pexels.com/video/woman-eating-sushi-12345/ のように
# 被写体がスラッグに入る。自然系のクエリでも人物クリップが返ってくるため、
# スラッグに人物・食事・生活の語を含む素材は使わない（金融解説の画としてノイズになる）。
REJECT_WORDS = (
    "woman", "man", "girl", "boy", "people", "person", "model", "portrait",
    "female", "male", "couple", "dance", "food", "eating", "restaurant",
    "lifestyle", "hand", "face", "child", "family", "wedding", "party",
)

# 実写の背景動画を敷くシーン。数字（hook / deal / change / chart）と締め（cta）は
# 背景をブランドのグラデーションに固定し、文字が実写に負けないようにする。
VIDEO_BG_KINDS = {"company", "filer"}

# ループの継ぎ目が目立たない最短尺。3秒素材を12秒のシーンで4周させると安っぽくなる。
MIN_DURATION_SEC = 7

# 縦動画の背景として十分な解像度。これ未満の動画ファイルは引き伸ばしでボケるため使わない。
MIN_HEIGHT = 1280
# ダウンロードサイズの安全上限（CIの帯域・時間を食い過ぎないように）。
MAX_BYTES = 80 * 1024 * 1024


def _api_key() -> "str | None":
    return os.getenv("PEXELS_API_KEY") or None


def has_rejected_subject(url: str) -> bool:
    """Pexels の動画URLのスラッグに人物・生活系の語が含まれるか。
    部分一致だと "germany" が "man" に引っかかるため、区切りで分割した語で判定する。"""
    words = set(re.split(r"[^a-z]+", (url or "").lower()))
    return bool(words & set(REJECT_WORDS))


def pick_video_file(videos: list) -> "dict | None":
    """検索結果から背景に使える動画ファイル（縦向き・十分な解像度・サイズ上限内・
    人物が写っていない）を1つ選ぶ。候補が複数あれば動画単位でランダムに選び、ファイルは
    「MIN_HEIGHT以上で最も小さい」ものを採る（背景用途に4Kは過剰なため）。"""
    candidates = []
    for video in videos:
        if (video.get("duration") or 0) < MIN_DURATION_SEC:
            continue
        if has_rejected_subject(video.get("url") or ""):
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


def fetch_pool(out_dir: str, count: int = 2) -> list:
    """異なるクエリからcount本を目標に背景動画を集め、
    [{"filename", "durationSec"}, ...] を返す（0本なら空リスト）。
    クエリはシャッフルして順に試し、検索やダウンロードに失敗したものは飛ばす。
    count本に届かなくても取れたぶんだけ返す（シーン側で使い回す）。"""
    key = _api_key()
    if key is None:
        print("[background] PEXELS_API_KEY 未設定のため背景動画をスキップします")
        return []

    pool = []
    for query in random.sample(QUERIES, len(QUERIES)):
        if len(pool) >= count:
            break
        picked = _search(key, query)
        if picked is None:
            continue
        filename = f"bg_{len(pool)}.mp4"
        written = _download(picked["file"]["link"], os.path.join(out_dir, filename))
        if written is None:
            continue
        duration = float(picked["duration"] or 10)
        print(f"[background] 背景動画を取得: {query} ({written / 1024 / 1024:.1f} MB / {duration:.0f}s)")
        pool.append({"filename": filename, "durationSec": duration})
    return pool


def assign_backgrounds(scenes: list, pool: list) -> None:
    """実写背景を使うシーン（VIDEO_BG_KINDS）にだけプールから割当する（その場で書き込み）。
    同じ映像が連続すると切り替わりのカット感が消えるため、プールが2本以上あれば
    直前に使ったものは選ばない。対象外のシーンには何も書かないので、Remotion 側は
    ブランドのグラデーション背景で描く。"""
    if not pool:
        return
    prev = None
    for scene in scenes:
        if scene.get("kind") not in VIDEO_BG_KINDS:
            continue
        candidates = [b for b in pool if b is not prev] if len(pool) > 1 else pool
        chosen = random.choice(candidates)
        scene["backgroundVideo"] = chosen["filename"]
        scene["backgroundVideoDurationSec"] = chosen["durationSec"]
        prev = chosen
