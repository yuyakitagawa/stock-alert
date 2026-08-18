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

# アップロードのチャンク制約（Content Posting API）。1チャンクで丸ごと送れるのは
# 64MB未満のファイルだけで、それ以上を1チャンクで init すると
# 「The chunk size is invalid」で400になる（2026-08-18: 85MBの動画で実際に失敗し、
# YouTubeには上がったのにTikTokだけ来ない状態になった）。
CHUNK_MIN_BYTES = 5 * 1024 * 1024
CHUNK_MAX_BYTES = 64 * 1024 * 1024
MAX_CHUNK_COUNT = 1000
# 分割時の1チャンク。5MB以上64MB以下ならどの値でもよく、リトライ単位が大きくなりすぎない10MBを採る。
DEFAULT_CHUNK_BYTES = 10 * 1024 * 1024


def _chunk_plan(size: int) -> "tuple[int, int]":
    """(chunk_size, total_chunk_count) を返す。
    64MB未満は丸ごと1チャンク。それ以上は10MB単位に割り、割り切れない端数は最終チャンクへ
    まとめて載せる（TikTokは total_chunk_count = video_size // chunk_size を期待する）。"""
    if size < CHUNK_MAX_BYTES:
        return size, 1
    chunk = DEFAULT_CHUNK_BYTES
    if size // chunk > MAX_CHUNK_COUNT:
        # 10GB超。チャンク数の上限に収まるようチャンクを大きくする
        chunk = max(CHUNK_MIN_BYTES, min(CHUNK_MAX_BYTES, -(-size // MAX_CHUNK_COUNT)))
    return chunk, size // chunk


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
    """公開時に貼るキャプション。定型の誘導文ではなく取引の詳細で埋める
    （2026-08-17オーナー指示。VOICEVOXクレジットは動画内の締め画面にあるため
    キャプションには入れない）。"""
    direction = "売却" if props.get("direction") == "sell" else "取得"
    # filerNameは古い記事だと未設定。空のまま連結すると「銘柄｜が…取得」という
    # 主語のねじれた文になるため（youtube_client.build_titleと同じ問題）、汎用の主語に置き換える。
    filer = props.get("filerName") or "大口投資家"
    stock = props.get("stockName", "")
    code = props.get("stockCode", "")
    head = f"【{stock}（{code}）】{filer}が推定{props.get('dealAmountOku')}億円を{direction}"
    if len(head) > CAPTION_MAX_CHARS:
        head = head[: CAPTION_MAX_CHARS - 1] + "…"

    lines = [head]

    # 2行目: 保有比率・開示日・投資家分類という取引のファクト
    facts = []
    ratio = props.get("holdingRatio")
    if ratio:
        facts.append(f"保有比率{ratio}%")
    disc = props.get("discDate") or ""
    if len(disc) == 10:  # YYYY-MM-DD
        facts.append(f"{int(disc[5:7])}/{int(disc[8:10])}開示")
    label = props.get("dealTypeLabel")
    if label and label != "大量保有報告書":
        facts.append(label)
    if facts:
        lines.append("・".join(facts) + "の大量保有報告書")

    # 3行目以降: 台本の要点字幕から詳細を最大3つ（hookとctaは除く）
    detail_kinds = ("company", "change", "outlook", "chart")
    details = [
        s["caption"] for s in props.get("scenes", [])
        if s.get("kind") in detail_kinds and s.get("caption")
    ][:3]
    if details:
        lines.append("")
        lines.extend(f"・{d}" for d in details)

    stock_tag = f" #{stock}" if stock else ""
    lines.append("")
    lines.append(f"#日本株 #株式投資 #EDINET #大量保有報告書{stock_tag}")
    return "\n".join(lines)


def _upload_bytes(upload_url: str, video_path: str, chunk_size: int, total_chunks: int) -> bool:
    """init で申告したチャンク割りのとおりに PUT する。1チャンクなら丸ごと1回。"""
    size = os.path.getsize(video_path)
    try:
        with open(video_path, "rb") as f:
            for i in range(total_chunks):
                start = i * chunk_size
                # 端数は最終チャンクにまとめて載せる（init の申告と一致させる）
                end = size - 1 if i == total_chunks - 1 else start + chunk_size - 1
                f.seek(start)
                data = f.read(end - start + 1)
                resp = requests.put(
                    upload_url,
                    headers={
                        "Content-Type": "video/mp4",
                        "Content-Length": str(len(data)),
                        "Content-Range": f"bytes {start}-{end}/{size}",
                    },
                    data=data,
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
    chunk_size, total_chunks = _chunk_plan(size)
    source_info = {
        "source": "FILE_UPLOAD",
        "video_size": size,
        "chunk_size": chunk_size,
        "total_chunk_count": total_chunks,
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

    if not _upload_bytes(upload_url, video_path, chunk_size, total_chunks):
        return None

    if direct:
        print(f"  🎵 TikTok投稿: publish_id={publish_id}")
    else:
        print(f"  🎵 TikTok下書き送信: publish_id={publish_id}（TikTokアプリの通知から公開してください）")
    return publish_id
