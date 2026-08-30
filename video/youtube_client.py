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

from video.post_text import SITE_NAME, SITE_URL, article_url, hashtag

TOKEN_URL = "https://oauth2.googleapis.com/token"
UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
THUMBNAIL_URL = "https://www.googleapis.com/upload/youtube/v3/thumbnails/set"
COMMENT_URL = "https://www.googleapis.com/youtube/v3/commentThreads"
# 説明文の登録導線用。?sub_confirmation=1 で登録確認ダイアログ付きで開く。
# ハンドルを変更したらここも更新すること（kujira-watch側 src/lib/site.ts と対）。
CHANNEL_URL = "https://www.youtube.com/@kujira-watch"

# YouTube のタイトル上限は100字。日本語でも文字数カウントは同じ。
TITLE_MAX_CHARS = 90
# 「ニュースと政治」。大量保有報告書の解説はここが最も近い。
CATEGORY_ID = "25"


def access_token() -> "str | None":
    client_id = os.getenv("YOUTUBE_CLIENT_ID")
    client_secret = os.getenv("YOUTUBE_CLIENT_SECRET")
    refresh_token = os.getenv("YOUTUBE_REFRESH_TOKEN")
    if not all([client_id, client_secret, refresh_token]):
        print("[youtube_client] YOUTUBE_CLIENT_ID等が未設定のためアップロードをスキップします")
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
            if "invalid_grant" in resp.text:
                # OAuth同意画面が「テスト」状態だとリフレッシュトークンは7日で失効する。
                # 恒久対策は同意画面を「本番」に公開すること（README参照）。
                print("  → リフレッシュトークンが失効しています。"
                      "ローカルで `python video/youtube_auth.py` を実行し、"
                      "`gh secret set YOUTUBE_REFRESH_TOKEN` で再登録してください")
            return None
        return resp.json().get("access_token")
    except Exception as e:
        print(f"  ⚠ YouTubeトークン更新例外: {e}")
        return None


def is_configured() -> bool:
    return all(os.getenv(k) for k in
               ("YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET", "YOUTUBE_REFRESH_TOKEN"))


def check_auth() -> bool:
    """アップロードできる状態かを、レンダリング前に1リクエストで確かめる。
    リフレッシュトークンが失効していると230秒かけて書き出した動画の行き先が消えるため
    （2026-08-25に実際に発生）、重い処理に入る前にここで落とす。"""
    return access_token() is not None


def build_title(props: dict) -> str:
    direction = "売却" if props.get("direction") == "sell" else "取得"
    suffix = " #Shorts"
    # filerNameは古い記事だと未設定。空のまま連結すると「【銘柄】が…取得」という
    # 主語のねじれた文になるため（実際に初回投稿で発生）、汎用の主語に置き換える。
    filer = props.get("filerName") or "大口投資家"
    # 保有比率まで入れると「何%になったのか」が一覧で分かり、クリックの理由が増える。
    ratio = props.get("holdingRatio") or 0
    ratio_part = f"、保有比率{ratio}%へ" if ratio else ""
    title = (
        f"【{props.get('stockName', '')}】{filer}が推定{props.get('dealAmountOku')}億円を"
        f"{direction}{ratio_part}{suffix}"
    )
    if len(title) > TITLE_MAX_CHARS:
        # 切り詰めても Shorts 判定用のタグは必ず残す（末尾の「…」ぶんも差し引く）
        head_limit = TITLE_MAX_CHARS - len(suffix) - 1
        title = title[:head_limit].rstrip() + "…" + suffix
    return title


def build_description(props: dict) -> str:
    """説明文。Shortsは畳まれた状態だと**先頭1行しか見えない**ため、1行目を
    「何が見られるか」を書いたリンクの見出しにし、2行目に記事URLを置く。
    以前は1行目が動画内の字幕と同じhookで、画面に出ている情報の繰り返しに1行使っていた。
    記事URLにはUTMを付けて、GA4でShorts経由の流入を識別できるようにする。"""
    url = article_url(props.get("articleId") or "", "youtube")
    scenes = props.get("scenes", [])
    hook = scenes[0]["caption"] if scenes else ""
    stock = props.get("stockName") or "この銘柄"
    # 中間シーン（hookとcta以外）の字幕を要点の箇条書きとして載せる
    points = "\n".join(
        f"・{s['caption']}" for s in scenes[1:-1] if s.get("caption")
    )
    stock_tag = hashtag(props.get("stockName", ""))
    # 説明文の先頭3つのハッシュタグはタイトル上部に表示される枠なので、
    # #Shorts のような機能タグではなく実際に検索される語を先に置く。
    tags = " ".join(t for t in ["#日本株", "#大量保有報告書", stock_tag,
                                "#EDINET", "#株式投資", "#Shorts"] if t)
    return (
        f"▼{stock}の保有推移・提出者の全開示はこちら（{SITE_NAME}）\n{url}\n\n"
        f"{hook}\n"
        f"{points}\n\n"
        f"EDINETの大量保有報告書から、大口投資家の売買を毎日追いかけています。\n"
        f"その日の全開示・投資家別の売買履歴も{SITE_NAME}で公開中:\n"
        f"{SITE_URL}?utm_source=youtube&utm_medium=social&utm_campaign=auto_video\n\n"
        f"▼チャンネル登録で、毎日の大口売買を1分でチェック\n{CHANNEL_URL}?sub_confirmation=1\n\n"
        f"音声: VOICEVOX:ずんだもん\n"
        f"※本動画は公開情報の要約であり、投資勧誘・投資助言ではありません。投資判断はご自身の責任でお願いします。\n\n"
        f"{tags}"
    )


def upload(video_path: str, props: dict, privacy_status: str = "public") -> "str | None":
    """動画をアップロードし、YouTubeの動画IDを返す。失敗・未設定時はNone。"""
    token = access_token()
    if token is None:
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


def build_comment(props: dict) -> str:
    """投稿直後に自分のチャンネルから残すコメント。

    なぜコメントか（2026-08-30）:
      Shortsは説明文が畳まれていて、視聴者が能動的に開かないとURLに触れられない。
      実測でも再生5,800回に対しサイト流入は28セッション（約0.5%）しかなく、
      細いのは配信ではなく動画→サイトの導線だった。コメント欄は1タップで開くうえ
      リンクがそのまま出るので、説明文より導線として短い。
      ※固定（ピン留め）はYouTube Data APIに機能が無いため、必要ならStudioで手動。
    """
    url = article_url(props.get("articleId") or "", "youtube_comment")
    stock = props.get("stockName") or "この銘柄"
    return (f"{stock}の保有比率の推移と、提出者の他の保有銘柄はこちらにまとめています。\n{url}")


def post_comment(video_id: str, text: str) -> bool:
    """動画に自チャンネルのコメントを1件投稿する。

    scope `youtube.force-ssl` が要る。リフレッシュトークンが upload だけで発行された
    古いものだと403になるが、その場合も動画は公開済みなのでFalseを返すだけにする。"""
    token = access_token()
    if token is None or not video_id or not text:
        return False
    try:
        res = requests.post(
            COMMENT_URL,
            params={"part": "snippet"},
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type": "application/json; charset=UTF-8"},
            json={"snippet": {"videoId": video_id,
                              "topLevelComment": {"snippet": {"textOriginal": text}}}},
            timeout=30,
        )
        if not res.ok:
            hint = ""
            if res.status_code in (401, 403):
                hint = ("　→ リフレッシュトークンのscopeに youtube.force-ssl がありません。"
                        "`python video/youtube_auth.py` で取り直してください")
            print(f"  ⚠ コメント投稿失敗 HTTP {res.status_code}: {res.text[:200]}{hint}")
            return False
        print("  💬 記事リンクのコメントを投稿しました")
        return True
    except Exception as e:
        print(f"  ⚠ コメント投稿例外: {e}")
        return False


def set_thumbnail(video_id: str, image_path: str) -> bool:
    """カスタムサムネイル（PNG/JPG, 2MB以下）を設定する。失敗しても投稿自体は完了しているのでFalseを返すだけ。"""
    token = access_token()
    if token is None or not video_id or not os.path.exists(image_path):
        return False
    try:
        with open(image_path, "rb") as f:
            res = requests.post(
                THUMBNAIL_URL,
                params={"videoId": video_id, "uploadType": "media"},
                headers={"Authorization": f"Bearer {token}", "Content-Type": "image/png"},
                data=f,
                timeout=60,
            )
        if not res.ok:
            print(f"  ⚠ サムネイル設定失敗 HTTP {res.status_code}: {res.text[:300]}")
            return False
        print("  🖼 カスタムサムネイルを設定しました")
        return True
    except Exception as e:
        print(f"  ⚠ サムネイル設定例外: {e}")
        return False
