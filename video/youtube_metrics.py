"""
video/youtube_metrics.py

YouTubeチャンネル（Shorts）の再生数・登録者数を日次で取得し、Supabaseの`youtube_videos`
（最新値）・`youtube_video_metrics`（日次スナップショット）・`youtube_channel_stats`
（チャンネル単位）に保存する。

なぜ必要か:
  動画パイプラインは毎営業日アップロードしているのに、動画IDすら保存しておらず
  「何本出して何回見られたか」を後から追えなかった。続けるか止めるかの判断材料が
  ゼロの状態で回り続けていた（2026-08-27に発覚）。実測は総再生4,747回・登録者3人・
  公開7本で、X（60日で34インプレッション）より桁違いに届いている。一方GA4上の
  サイト流入は28日で22セッション＝再生の0.46%で、動画→サイトの導線が細い。

認証:
  公開動画の統計はサービスアカウント（`gcp_key.json`、`GOOGLE_APPLICATION_CREDENTIALS`で
  差し替え可）のトークンで読める。アップロード用のOAuthリフレッシュトークンは
  scopeが`youtube.upload`だけで統計を読めないが、取り直す必要はない。

実行:
  python3 video/youtube_metrics.py            # 取得して保存
  python3 video/youtube_metrics.py --report   # 保存に加えて尺別・投稿日別の要約を表示
"""
import argparse
import os
import re
import sys
from datetime import date, datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/stock-alert/.env"))

from lib import supabase_client as sb  # noqa: E402

API_ROOT = "https://www.googleapis.com/youtube/v3"
SCOPE = "https://www.googleapis.com/auth/youtube.readonly"
# サイトのフッター等に載せているチャンネル（kujira-watch/src/lib/site.ts の YOUTUBE_CHANNEL_URL）。
CHANNEL_HANDLE = "@kujira-watch"
# videos.list の id 上限。
BATCH = 50
# 「短尺」の境目（秒）。実測で39〜45秒の3本が平均1,224回、66〜89秒の4本が平均571回だった
# ため、まずこの線で分けて様子を見る。本数が増えたら見直す。
SHORT_SEC = 60


def credentials_path() -> str:
    return os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gcp_key.json")


def access_token() -> "str | None":
    path = credentials_path()
    if not os.path.exists(path):
        return None
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account

    creds = service_account.Credentials.from_service_account_file(path, scopes=[SCOPE])
    creds.refresh(Request())
    return creds.token


def parse_duration(iso: str) -> int:
    """ISO8601の再生時間(PT1M29S)を秒にする。尺と再生数の関係を見るために必要。"""
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso or "")
    if not m:
        return 0
    h, mi, s = (int(v) if v else 0 for v in m.groups())
    return h * 3600 + mi * 60 + s


def _get(token: str, path: str, params: dict) -> dict:
    resp = requests.get(f"{API_ROOT}/{path}", params=params,
                        headers={"Authorization": f"Bearer {token}"}, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    return resp.json()


def fetch_channel(token: str) -> dict:
    """{'uploads': プレイリストID, 'subscribers':…, 'total_views':…, 'video_count':…}"""
    j = _get(token, "channels", {"part": "statistics,contentDetails", "forHandle": CHANNEL_HANDLE})
    items = j.get("items") or []
    if not items:
        raise RuntimeError(f"チャンネル {CHANNEL_HANDLE} が見つかりません")
    st = items[0].get("statistics", {})
    return {
        "uploads": items[0].get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads", ""),
        "subscribers": int(st.get("subscriberCount") or 0),
        "total_views": int(st.get("viewCount") or 0),
        "video_count": int(st.get("videoCount") or 0),
    }


def record_upload(video_id: str, title: str = "") -> bool:
    """投稿直後に1本ぶんを`youtube_videos`へ記録する（再生数などは後追いで埋まる）。

    なぜ投稿側で書くのか（2026-08-28）:
      統計の収集（main）は手動実行なので、それを待つと当日の成果物ハートビート
      （tools/output_heartbeat.py）からは「今日は動画0本」に見える。実際、Shortsは
      毎営業日出ているのにハートビートは毎日「動画0本」と鳴っていた。
      既存行は上書きしない（insert_ignore）。収集済みの再生数を投稿側が潰さないため。
    """
    if not video_id or not sb.is_configured():
        return False
    sb.insert_ignore("youtube_videos",
                     [{"video_id": video_id,
                       "published_at": datetime.now(timezone.utc).isoformat(),
                       "title": title}],
                     on_conflict="video_id")
    return True


def fetch_videos(token: str, uploads_playlist: str) -> list[dict]:
    """アップロード済み動画の統計。チャンネルのuploadsプレイリストから辿るので、
    投稿側の記録（record_upload）が漏れた過去分も後追いで拾える。"""
    ids, page = [], None
    while uploads_playlist:
        params = {"part": "contentDetails", "playlistId": uploads_playlist, "maxResults": BATCH}
        if page:
            params["pageToken"] = page
        j = _get(token, "playlistItems", params)
        ids += [i["contentDetails"]["videoId"] for i in j.get("items", [])]
        page = j.get("nextPageToken")
        if not page:
            break

    videos = []
    for i in range(0, len(ids), BATCH):
        j = _get(token, "videos", {"part": "statistics,snippet,contentDetails",
                                   "id": ",".join(ids[i:i + BATCH])})
        for v in j.get("items", []):
            st, sn = v.get("statistics", {}), v.get("snippet", {})
            videos.append({
                "video_id": v["id"],
                "published_at": sn.get("publishedAt"),
                "title": sn.get("title"),
                "duration_sec": parse_duration(v.get("contentDetails", {}).get("duration", "")),
                "views": int(st.get("viewCount") or 0),
                "likes": int(st.get("likeCount") or 0),
                "comments": int(st.get("commentCount") or 0),
            })
    videos.sort(key=lambda v: v["published_at"] or "")
    return videos


def save(videos: list[dict], channel: dict) -> None:
    if not sb.is_configured():
        print("[youtube_metrics] Supabase未設定のため保存をスキップします")
        return
    now = datetime.now(timezone.utc).isoformat()
    today = date.today().isoformat()
    sb.upsert("youtube_videos", [{**v, "metrics_updated_at": now} for v in videos],
              on_conflict="video_id")
    sb.upsert("youtube_video_metrics",
              [{"video_id": v["video_id"], "measured_on": today, "views": v["views"],
                "likes": v["likes"], "comments": v["comments"]} for v in videos],
              on_conflict="video_id,measured_on")
    sb.upsert("youtube_channel_stats", [{
        "measured_on": today, "subscribers": channel["subscribers"],
        "total_views": channel["total_views"], "video_count": channel["video_count"],
    }], on_conflict="measured_on")


def summarize(videos: list[dict]) -> dict:
    """尺別の平均再生数。短尺のほうが伸びるという仮説を毎回同じ切り口で検証するために出す。"""
    short = [v for v in videos if 0 < v["duration_sec"] <= SHORT_SEC]
    long_ = [v for v in videos if v["duration_sec"] > SHORT_SEC]
    avg = lambda xs: sum(v["views"] for v in xs) / len(xs) if xs else 0.0  # noqa: E731
    return {"short_n": len(short), "short_avg_views": avg(short),
            "long_n": len(long_), "long_avg_views": avg(long_)}


def report(videos: list[dict], channel: dict) -> None:
    print(f"チャンネル: 登録者{channel['subscribers']}人 / 公開{channel['video_count']}本 "
          f"/ 総再生{channel['total_views']:,}回\n")
    for v in videos:
        print(f"  {(v['published_at'] or '')[:10]}  再生{v['views']:6,}  👍{v['likes']:>3}  "
              f"💬{v['comments']:>3}  {v['duration_sec']:>3}秒  {(v['title'] or '')[:40]}")
    s = summarize(videos)
    print(f"\n  {SHORT_SEC}秒以下: {s['short_n']}本 平均{s['short_avg_views']:,.0f}回")
    print(f"  {SHORT_SEC}秒超  : {s['long_n']}本 平均{s['long_avg_views']:,.0f}回")


def main() -> int:
    p = argparse.ArgumentParser(description="YouTubeの再生数・登録者数を記録する")
    p.add_argument("--report", action="store_true", help="尺別の要約も表示する")
    a = p.parse_args()

    token = access_token()
    if not token:
        print(f"[youtube_metrics] サービスアカウント鍵が見つかりません: {credentials_path()}")
        return 1
    try:
        channel = fetch_channel(token)
        videos = fetch_videos(token, channel["uploads"])
    except Exception as e:
        print(f"[youtube_metrics] 取得に失敗: {e}")
        return 1

    save(videos, channel)
    print(f"[youtube_metrics] {len(videos)}本を記録しました"
          f"（登録者{channel['subscribers']}人 / 総再生{channel['total_views']:,}回）")
    if a.report:
        print()
        report(videos, channel)
    # 1本も取れないのは公開失敗か認証の異常。緑のまま流すと何日でも気付けない。
    return 0 if videos else 1


if __name__ == "__main__":
    sys.exit(main())
