"""
video/youtube_analytics.py

YouTube Analytics API v2 から「何秒で離脱されたか」を取る。

なぜ必要か（2026-08-30）:
  これまで記録できていたのは再生数・高評価・コメント数だけで（video/youtube_metrics.py）、
  それは Data API の公開統計だから読めていた。しかし動画の中身を直す判断に要るのは
  「hookで何割が消えたか」であって再生数ではない。実測は9本・総再生6,909回・登録者5人で、
  再生は届いているのに登録もサイト流入も伸びていない。どのシーンで落ちているかが
  分からないまま演出を変えても、良くなったか悪くなったかを言えない。

認証:
  サービスアカウントでは読めない（チャンネルの所有者本人のOAuthが要る）。
  video/youtube_client.py が投稿に使っているリフレッシュトークンを流用するが、
  そのトークンの scope に `yt-analytics.readonly` が含まれている必要がある。
  古いトークン（upload のみ）では403になるので、`python video/youtube_auth.py` で
  取り直す。取り直しても upload は含まれるので投稿側の動作は変わらない。

実行:
  取得は video/youtube_metrics.py 側から呼ぶ（単体では動かさない）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests  # noqa: E402

from video import youtube_client  # noqa: E402

REPORTS_URL = "https://youtubeanalytics.googleapis.com/v2/reports"
# チャンネル開設日より前。開始日を動画ごとに変えると本ごとの集計期間が揃わない。
START_DATE = "2026-08-01"
# 維持率カーブの分解能。APIは elapsedVideoTimeRatio を 0.00〜1.00 の101点で返す。
CURVE_POINTS = 101


def _query(token: str, params: dict) -> "dict | None":
    try:
        res = requests.get(REPORTS_URL, params=params,
                           headers={"Authorization": f"Bearer {token}"}, timeout=30)
    except Exception as e:
        print(f"  ⚠ Analytics取得例外: {e}")
        return None
    if not res.ok:
        # 403は原因が2つあり、対処がまったく違う。本文で見分けないと
        # 「APIが無効」なのに「トークンを取り直せ」と案内して空振りする（2026-08-30に発生）。
        hint = ""
        body = res.text
        if res.status_code in (401, 403):
            if "has not been used in project" in body or "SERVICE_DISABLED" in body:
                hint = ("　→ GCPプロジェクトで YouTube Analytics API が有効化されていません。"
                        "`gcloud services enable youtubeanalytics.googleapis.com` か"
                        "Cloud Consoleで有効化してください")
            else:
                hint = ("　→ リフレッシュトークンのscopeに yt-analytics.readonly がありません。"
                        "`python video/youtube_auth.py` で取り直してください")
        print(f"  ⚠ Analytics取得失敗 HTTP {res.status_code}: {body[:200]}{hint}")
        return None
    return res.json()


def _rows(payload: "dict | None") -> list:
    return (payload or {}).get("rows") or []


def video_stats(token: str, end_date: str) -> dict:
    """動画ID -> {'avg_view_pct':…, 'avg_view_sec':…, 'subscribers_gained':…}。

    再生数は Data API 側で取れているのでここでは取らない。欲しいのは
    「最後まで見られた割合」と「その動画が何人の登録に繋がったか」の2つ。"""
    payload = _query(token, {
        "ids": "channel==MINE",
        "startDate": START_DATE,
        "endDate": end_date,
        "dimensions": "video",
        # video次元のレポートは views か estimatedMinutesWatched を含めて
        # そのどちらかで降順ソートしないと 400 "The query is not supported" になる。
        # viewsそのものは Data API 側で取れているので、ここでは並べ替えのためだけに要る。
        "metrics": "views,averageViewPercentage,averageViewDuration,subscribersGained",
        "sort": "-views",
        "maxResults": 200,
    })
    out = {}
    for row in _rows(payload):
        if len(row) < 5:
            continue
        out[row[0]] = {
            # Shortsはループ再生ぶんが積み上がるので100%を超える（実測169.6%の回がある）。
            # 丸めるだけで頭打ちにはしない。切り捨てると「何周されたか」が消える。
            "avg_view_pct": round(float(row[2] or 0), 1),
            "avg_view_sec": int(row[3] or 0),
            "subscribers_gained": int(row[4] or 0),
        }
    return out


def retention_curve(token: str, video_id: str, end_date: str) -> list:
    """[(経過割合0〜1, その時点で見ている人の割合), …]。公開直後や再生が少ない動画は空。"""
    payload = _query(token, {
        "ids": "channel==MINE",
        "startDate": START_DATE,
        "endDate": end_date,
        "dimensions": "elapsedVideoTimeRatio",
        "metrics": "audienceWatchRatio",
        "filters": f"video=={video_id}",
    })
    curve = [(float(r[0]), float(r[1])) for r in _rows(payload) if len(r) >= 2]
    curve.sort()
    return curve


def survival_at(curve: list, duration_sec: int, seconds: float) -> "float | None":
    """指定秒時点で、**冒頭を1.0としたときに**残っている割合。

    2つ割り戻しが要る:
      1. カーブは秒ではなく尺に対する経過割合で返るので、尺の違う動画を並べるには秒→割合に直す。
      2. audienceWatchRatio はループ再生ぶんが乗って1.0を超える（実測で冒頭1.40の回がある）。
         生値のままだと「冒頭1.40→3秒1.30（93%残）」の動画が「冒頭1.05→3秒0.95（90%残）」より
         悪く見える。hookの出来を比べたいので、カーブの先頭で正規化する。
    """
    if not curve or duration_sec <= 0 or seconds < 0:
        return None
    target = seconds / duration_sec
    if target > 1:
        return None
    head = curve[0][1]
    if not head:
        return None
    # 目標割合以上で最も手前の点。カーブは離散なので内挿はせず直近の実測点を採る。
    for ratio, watch in curve:
        if ratio >= target:
            return round(watch / head, 3)
    return round(curve[-1][1] / head, 3)
