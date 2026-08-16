"""
video/youtube_client.py

生成した縦動画を YouTube Shorts としてアップロードする。

認証は OAuth 2.0（インストール済みアプリ）のリフレッシュトークン方式。
Google Cloud Console で YouTube Data API v3 を有効化し、「デスクトップアプリ」の
OAuth クライアントを作ったうえで、ローカルで `python video/youtube_auth.py` を1回実行して
リフレッシュトークンを取得する。以下3つが揃っていない場合はアップロードをスキップする
（他のプラットフォームへの投稿は止めない）:
  YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET, YOUTUBE_REFRESH_TOKEN

YouTube 側は「縦長（9:16）かつ3分以内」の動画を自動的に Shorts として扱うため、
アップロード時に Shorts 専用のフラグを立てる必要はない（#Shorts は保険として説明文に入れる）。
"""
import os

import requests

TOKEN_URL = "https://oauth2.googleapis.com/token"
UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
SITE_URL = "https://kujira-watch.com"
# 説明文の登録導線用。?sub_confirmation=1 で登録確認ダイアログ付きで開く。
# ハンドルを変更したらここも更新すること（kujira-watch側 src/lib/site.ts と対）。
CHANNEL_URL = "https://www.youtube.com/@kujira-watch"

# YouTube のタイトル上限は100字。日本語でも文字数カウントは同じ。
TITLE_MAX_CHARS = 90
# 「ニュースと政治」。大量保有報告書の解説はここが最も近い。
CATEGORY_ID = "25"


def _access_token() -> "str | None":
    client_id = os.getenv("YOUTUBE_CLIENT_ID")
    client_secret = os.getenv("YOUTUBE_CLIENT_SECRET")
    refresh_token = os.getenv("YOUTUBE_REFRESH_TOKEN")
    if not all([client_id, client_secret, refresh_token]):
        return None
    try:
        resp = requests.post(
            TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
            timeout=20,
        )
        if not resp.ok:
            print(f"  ⚠ YouTubeトークン更新失敗 HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        return resp.json().get("access_token")
    except Exception as e:
        print(f"  ⚠ YouTubeトークン更新例外: {e}")
        return None


def build_title(props: dict) -> str:
    direction = "売却" if props.get("direction") == "sell" else "取得"
    suffix = " #Shorts"
    # filerNameは古い記事だと未設定。空のまま連結すると「【銘柄】が…取得」という
    # 主語のねじれた文になるため（実際に初回投稿で発生）、汎用の主語に置き換える。
    filer = props.get("filerName") or "大口投資家"
    title = f"【{props.get('stockName', '')}】{filer}が推定{props.get('dealAmountOku')}億円を{direction}{suffix}"
    if len(title) > TITLE_MAX_CHARS:
        # 切り詰めても Shorts 判定用のタグは必ず残す（末尾の「…」ぶんも差し引く）
        head_limit = TITLE_MAX_CHARS - len(suffix) - 1
        title = title[:head_limit].rstrip() + "…" + suffix
    return title


def build_description(props: dict) -> str:
    """説明文。記事URLにUTMを付けて、GA4でShorts経由の流入を識別できるようにする。"""
    article_id = props.get("articleId")
    url = (
        f"{SITE_URL}/articles/{article_id}?utm_source=youtube&utm_medium=social&utm_campaign=auto_video"
        if article_id
        else SITE_URL
    )
    scenes = props.get("scenes", [])
    hook = scenes[0]["caption"] if scenes else ""
    # 中間シーン（hookとcta以外）の字幕を要点の箇条書きとして載せる
    points = "\n".join(
        f"・{s['caption']}" for s in scenes[1:-1] if s.get("caption")
    )
    return (
        f"{hook}\n\n"
        f"{points}\n\n"
        f"▼記事で詳しく読む\n{url}\n\n"
        f"▼チャンネル登録で、毎日の大口売買を1分でチェック\n{CHANNEL_URL}?sub_confirmation=1\n\n"
        f"EDINETの大量保有報告書から、大口投資家（クジラ）の売買を毎日追いかけています。\n"
        f"その日の全開示・投資家別の履歴はサイトで:\n"
        f"{SITE_URL}?utm_source=youtube&utm_medium=social&utm_campaign=auto_video\n\n"
        f"音声: VOICEVOX:ずんだもん\n"
        f"※本動画は公開情報の要約であり、投資勧誘・投資助言ではありません。投資判断はご自身の責任でお願いします。\n\n"
        f"#Shorts #EDINET #大量保有報告書 #日本株 #{props.get('stockName', '')}"
    )


def upload(video_path: str, props: dict, privacy_status: str = "public") -> "str | None":
    """動画をアップロードし、YouTubeの動画IDを返す。失敗・未設定時はNone。"""
    token = _access_token()
    if token is None:
        print("[youtube_client] YOUTUBE_CLIENT_ID等が未設定のためアップロードをスキップします")
        return None

    file_size = os.path.getsize(video_path)
    metadata = {
        "snippet": {
            "title": build_title(props),
            "description": build_description(props),
            "tags": ["EDINET", "大量保有報告書", "日本株", "株式投資", props.get("stockName", "")],
            "categoryId": CATEGORY_ID,
        },
        "status": {
            "privacyStatus": privacy_status,
            "selfDeclaredMadeForKids": False,
        },
    }

    try:
        init = requests.post(
            UPLOAD_URL,
            params={"uploadType": "resumable", "part": "snippet,status"},
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=UTF-8",
                "X-Upload-Content-Length": str(file_size),
                "X-Upload-Content-Type": "video/mp4",
            },
            json=metadata,
            timeout=30,
        )
        if not init.ok:
            print(f"  ⚠ YouTubeアップロード開始失敗 HTTP {init.status_code}: {init.text[:300]}")
            return None
        session_url = init.headers.get("Location")
        if not session_url:
            print("  ⚠ YouTubeアップロードのセッションURLが返りませんでした")
            return None

        with open(video_path, "rb") as f:
            put = requests.put(
                session_url,
                headers={"Content-Type": "video/mp4", "Content-Length": str(file_size)},
                data=f,
                timeout=600,
            )
        if not put.ok:
            print(f"  ⚠ YouTubeアップロード失敗 HTTP {put.status_code}: {put.text[:300]}")
            return None

        video_id = put.json().get("id")
        print(f"  ▶️ YouTube投稿: https://youtube.com/shorts/{video_id}")
        return video_id
    except Exception as e:
        print(f"  ⚠ YouTubeアップロード例外: {e}")
        return None
