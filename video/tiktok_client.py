"""
video/tiktok_client.py

生成した縦動画を TikTok にアップロードする（Content Posting API v2）。

■ 重要な制約
TikTok の Content Posting API は、アプリが審査（audit）を通るまで一般公開の投稿ができない。
未審査アプリで直接投稿すると privacy_level は SELF_ONLY（自分のみ閲覧可）に限定される。
そのため既定では **inbox（下書き）へのアップロード** を使う。アップロードすると TikTok アプリに
通知が届き、オーナーがアプリ上で内容を確認して手動で公開する運用になる。
審査通過後に環境変数 TIKTOK_DIRECT_POST=1 を設定すると、直接公開（PUBLIC_TO_EVERYONE）に切り替わる。

認証は OAuth 2.0 のリフレッシュトークン方式。TikTok for Developers でアプリを作り、
scope に video.upload（直接投稿する場合は video.publish も）を付けたうえで、ローカルで
`python video/tiktok_auth.py` を1回実行してリフレッシュトークンを取得する。
以下3つが揃っていない場合はアップロードをスキップする（他のプラットフォームは止めない）:
  TIKTOK_CLIENT_KEY, TIKTOK_CLIENT_SECRET, TIKTOK_REFRESH_TOKEN
"""
import os

import requests

API_BASE = "https://open.tiktokapis.com/v2"
TOKEN_URL = f"{API_BASE}/oauth/token/"
INBOX_INIT_URL = f"{API_BASE}/post/publish/inbox/video/init/"
DIRECT_INIT_URL = f"{API_BASE}/post/publish/video/init/"
CREATOR_INFO_URL = f"{API_BASE}/post/publish/creator_info/query/"
SITE_URL = "https://kujira-watch.com"

# TikTokのキャプション上限は2200字だが、表示上は冒頭しか読まれないため短く保つ。
CAPTION_MAX_CHARS = 150


def _access_token() -> "str | None":
    client_key = os.getenv("TIKTOK_CLIENT_KEY")
    client_secret = os.getenv("TIKTOK_CLIENT_SECRET")
    refresh_token = os.getenv("TIKTOK_REFRESH_TOKEN")
    if not all([client_key, client_secret, refresh_token]):
        return None
    try:
        resp = requests.post(
            TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "client_key": client_key,
                "client_secret": client_secret,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            },
            timeout=20,
        )
        if not resp.ok:
            print(f"  ⚠ TikTokトークン更新失敗 HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        return resp.json().get("access_token")
    except Exception as e:
        print(f"  ⚠ TikTokトークン更新例外: {e}")
        return None


def build_caption(props: dict) -> str:
    direction = "売却" if props.get("direction") == "sell" else "取得"
    head = f"{props.get('stockName', '')}｜{props.get('filerName', '')}が推定{props.get('dealAmountOku')}億円を{direction}"
    if len(head) > CAPTION_MAX_CHARS:
        head = head[: CAPTION_MAX_CHARS - 1] + "…"
    # TikTokはキャプション内のURLがリンクにならないため、プロフィール誘導の文言にする。
    return f"{head}\n詳しくはプロフィールのリンク（kujira-watch.com）から\n#日本株 #株式投資 #EDINET #大量保有報告書"


def _upload_bytes(upload_url: str, video_path: str) -> bool:
    size = os.path.getsize(video_path)
    try:
        with open(video_path, "rb") as f:
            resp = requests.put(
                upload_url,
                headers={
                    "Content-Type": "video/mp4",
                    "Content-Length": str(size),
                    "Content-Range": f"bytes 0-{size - 1}/{size}",
                },
                data=f,
                timeout=600,
            )
        if not resp.ok:
            print(f"  ⚠ TikTok動画転送失敗 HTTP {resp.status_code}: {resp.text[:300]}")
            return False
        return True
    except Exception as e:
        print(f"  ⚠ TikTok動画転送例外: {e}")
        return False


def _allowed_privacy_levels(token: str) -> list:
    """直接投稿の前に必須の creator_info クエリ。アカウントが許可している公開範囲を返す。"""
    try:
        resp = requests.post(
            CREATOR_INFO_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=UTF-8",
            },
            timeout=20,
        )
        if not resp.ok:
            print(f"  ⚠ TikTok creator_info取得失敗 HTTP {resp.status_code}: {resp.text[:200]}")
            return []
        return resp.json().get("data", {}).get("privacy_level_options", [])
    except Exception as e:
        print(f"  ⚠ TikTok creator_info例外: {e}")
        return []


def upload(video_path: str, props: dict) -> "str | None":
    """動画をアップロードし、publish_id を返す。失敗・未設定時はNone。
    TIKTOK_DIRECT_POST=1 なら直接公開、それ以外は inbox（下書き）へ送る。"""
    token = _access_token()
    if token is None:
        print("[tiktok_client] TIKTOK_CLIENT_KEY等が未設定のためアップロードをスキップします")
        return None

    size = os.path.getsize(video_path)
    source_info = {
        "source": "FILE_UPLOAD",
        "video_size": size,
        # 64MB未満は1チャンクで送れる。この動画は20秒・数MBなので常に1チャンク。
        "chunk_size": size,
        "total_chunk_count": 1,
    }

    direct = os.getenv("TIKTOK_DIRECT_POST") == "1"
    if direct:
        allowed = _allowed_privacy_levels(token)
        privacy = "PUBLIC_TO_EVERYONE" if "PUBLIC_TO_EVERYONE" in allowed else "SELF_ONLY"
        if privacy != "PUBLIC_TO_EVERYONE":
            print("  ⚠ アカウントが一般公開を許可していないため SELF_ONLY で投稿します")
        init_url = DIRECT_INIT_URL
        payload = {
            "post_info": {
                "title": build_caption(props),
                "privacy_level": privacy,
                "disable_duet": False,
                "disable_comment": False,
                "disable_stitch": False,
            },
            "source_info": source_info,
        }
    else:
        init_url = INBOX_INIT_URL
        payload = {"source_info": source_info}

    try:
        init = requests.post(
            init_url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=UTF-8",
            },
            json=payload,
            timeout=30,
        )
        if not init.ok:
            print(f"  ⚠ TikTokアップロード開始失敗 HTTP {init.status_code}: {init.text[:300]}")
            return None
        data = init.json().get("data", {})
        upload_url = data.get("upload_url")
        publish_id = data.get("publish_id")
        if not upload_url or not publish_id:
            print(f"  ⚠ TikTokアップロードURLが返りませんでした: {init.text[:300]}")
            return None
    except Exception as e:
        print(f"  ⚠ TikTokアップロード開始例外: {e}")
        return None

    if not _upload_bytes(upload_url, video_path):
        return None

    if direct:
        print(f"  🎵 TikTok投稿: publish_id={publish_id}")
    else:
        print(f"  🎵 TikTok下書き送信: publish_id={publish_id}（TikTokアプリの通知から公開してください）")
    return publish_id
