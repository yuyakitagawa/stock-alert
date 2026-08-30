"""
web/x_client.py
publish_blog_articles.py が投稿した新着記事のうち、ホームページの「注目」枠
（kujira-watch側 getFeaturedArticles()、直近プールをdealAmount降順に並べ直した
上位3件）に入っている記事だけをX(Twitter)へ自動投稿する。サイトで目立っていない
小粒な開示が「その日一番大きい」というだけでXに投稿される事態を避けるため、
「その日新規公開した記事」×「現在サイトで注目表示されている記事」の積集合を対象にする
（該当が無い地味な日は0件のこともある）。訂正報告書だけは既報の前提を覆すため
「注目」枠に関係なく全件投稿する。

投稿の型は docs/x_post_improvement_1000.md（インフルエンサー1000人コンサル）の
収束10施策に沿う:
- 1行目は記事タイトルの流用ではなく「誰が・どの銘柄を・どうした」のフック
- 銘柄は `社名(証券コード)` を素で本文に書く（株クラはこの表記で検索する）。
  ハッシュタグは母数のある `#日本株 #大量保有報告書` の2つだけ
- 添付画像の1枚目は保有比率と金額を大きく出す数字カード（x_card_image）、
  2枚目が株価チャート。両方にalt（代替テキスト）を付ける
- URLは一切入れない。X APIの従量課金はリンク入り投稿が$0.20/本（リンク無し$0.015の13倍）で、
  投稿コストのほぼ全部がこの加算だった（2026-08-22、14本で$2.2）。誘導は末尾の
  PROFILE_CTA「詳細はプロフィールのリンクから」に統一する（自己リプライにURLを置いても
  そのリプライが$0.20になるので回避にならない）
- 投稿したtweet_idはSupabaseの`x_posts`に記録し、web/x_metrics.pyが日次で
  インプレッション等を追記する（どの型が伸びたかを検証できるようにするため）

認証はOAuth 1.0a User Context（X API v2 POST /2/tweetsの投稿、および画像アップロード用の
v1.1 media/uploadエンドポイントの両方に必須）。
X Developer Portalで「Read and Write」権限のAppを作成し、以下4つを取得する:
  X_API_KEY, X_API_KEY_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET
いずれか未設定の場合は投稿をスキップする（他のステップには影響しない）。
"""
import os
from datetime import datetime, timedelta, timezone

import requests
from requests_oauthlib import OAuth1

from web.x_post_format import clean_name, label, weighted_len

# 本文にURLを入れない代わりの誘導行。記事・日次サマリー・週次・答え合わせで共通。
PROFILE_CTA = "詳細はプロフィールのリンクから"
API_URL = "https://api.x.com/2/tweets"
MEDIA_UPLOAD_URL = "https://upload.twitter.com/1.1/media/upload.json"
MEDIA_METADATA_URL = "https://api.twitter.com/1.1/media/metadata/create.json"

# 1回の実行（＝毎時バッチ1回）で投稿する通常記事の上限。以前は3件で、同一時刻に
# 同じ型の投稿が3連続でタイムラインに並んでいた（ミュートの引き金になる）。
# 毎時バッチなので1件ずつでも大きい順に自然と分散する。
ARTICLES_PER_RUN = 1

# 投稿してよいJSTの時間帯（この外では自動投稿しない）。手動実行やCIの遅延で
# 深夜に投稿が飛ぶのを防ぐ。
POST_HOURS_JST = range(8, 23)

# 日次サマリーを投稿するUTC時（0時UTC=9時JST=edinet_blog.ymlの初便）。
# 以前は21時JSTだったが、同ジャンル9アカウント879投稿の実測（2026-08-30、tools/x_benchmark.py）で
# 朝7-11時JSTの中央値インプレッションが2,080、夕16-19時が710、夜20-23時が1,094だったため朝に移した。
# 毎時バッチのうちこの1回だけ投稿することで、外部ストレージ無しで1日1回の重複ガードにする。
# 朝の便では当日の開示がまだ出ていないので、対象は「前営業日」になる（summary_target_date）。
DAILY_SUMMARY_UTC_HOUR = 0

# ハッシュタグ。#EDINET は検索母数が小さく、記号を除いた `#社名` は実際には検索されない
# （株クラは「社名 コード」で検索する）ため、母数のある2つだけに絞り本文側にコードを書く。
TAGS = "#日本株 #大量保有報告書"

# 1行目に載せる社名・ファンド名の表示上限（Xのカウント単位。全角=2・半角=1）。
STOCK_LABEL_MAX_UNITS = 26
FILER_LABEL_MAX_UNITS = 26


def _auth() -> "OAuth1 | None":
    key = os.getenv("X_API_KEY")
    key_secret = os.getenv("X_API_KEY_SECRET")
    token = os.getenv("X_ACCESS_TOKEN")
    token_secret = os.getenv("X_ACCESS_TOKEN_SECRET")
    if not all([key, key_secret, token, token_secret]):
        return None
    return OAuth1(key, key_secret, token, token_secret)


# OAuth 1.0aのアクセストークンは「発行された時点」のスコープを保持する。App権限をRead onlyから
# Read and Writeに変えても、変更前に発行したトークンは読み取り専用のままで、投稿は403
# `You are not permitted to perform this action.` になる（2026-08-18の日次便で発生）。
# v1.1のレスポンスヘッダ x-access-level がトークンの実権限（read / read-write）を返すので、
# これで「Portalの設定」ではなく「実際に使っているトークン」の権限を確認する。
VERIFY_CREDENTIALS_URL = "https://api.twitter.com/1.1/account/verify_credentials.json"


def verify_auth() -> dict:
    """アクセストークンの実権限を確認する。{ok, access_level, screen_name, reason} を返す。"""
    auth = _auth()
    if auth is None:
        return {"ok": False, "access_level": "", "screen_name": "", "reason": "X_API_KEY等が未設定"}
    try:
        resp = requests.get(VERIFY_CREDENTIALS_URL, auth=auth, timeout=20)
    except Exception as e:
        return {"ok": False, "access_level": "", "screen_name": "", "reason": f"通信例外: {e}"}
    access_level = resp.headers.get("x-access-level", "")
    if resp.status_code != 200:
        return {"ok": False, "access_level": access_level, "screen_name": "",
                "reason": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    screen_name = ""
    try:
        screen_name = resp.json().get("screen_name", "")
    except Exception:
        pass
    if "write" not in access_level:
        return {"ok": False, "access_level": access_level, "screen_name": screen_name,
                "reason": "トークンが読み取り専用。X Developer PortalでApp権限をRead and Writeに"
                          "した上で、アクセストークンを再発行して X_ACCESS_TOKEN / "
                          "X_ACCESS_TOKEN_SECRET を更新してください（権限変更前に発行した"
                          "トークンは読み取り専用のままです）"}
    return {"ok": True, "access_level": access_level, "screen_name": screen_name, "reason": ""}


# ---------------------------------------------------------------- 投稿ログ

def log_post(tweet_id: str, kind: str, body: str, **fields) -> None:
    """投稿をSupabaseの`x_posts`に記録する。どの型が伸びたかを後から検証するための土台で、
    これが無いとフォーマット変更の効果を判定できない（施策1）。記録の失敗で投稿自体を
    止めたくないため、例外は握りつぶす。"""
    try:
        from lib import supabase_client as sb

        if not sb.is_configured():
            return
        row = {
            "tweet_id": tweet_id,
            "posted_at": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "body": body,
            "body_weighted_len": weighted_len(body),
            "variant": fields.get("variant") or "no_link",
            "has_media": bool(fields.get("has_media")),
            "link_in_reply": False,
            "stock_code": fields.get("stock_code") or None,
            "article_id": fields.get("article_id") or None,
            "reply_to_tweet_id": fields.get("reply_to") or None,
        }
        sb.upsert("x_posts", [row], on_conflict="tweet_id")
    except Exception as e:
        print(f"  ⚠ x_postsへの記録に失敗（投稿は成功）: {e}")


def find_prior_tweet(stock_code: str) -> "str | None":
    """同じ銘柄について直近に投稿したtweet_id。訂正報告書を既報のスレッドに
    ぶら下げるために使う（施策7）。見つからなければNone。"""
    if not stock_code:
        return None
    try:
        from lib import supabase_client as sb

        row = sb.select_one(
            "x_posts",
            f"stock_code=eq.{stock_code}&kind=in.(article,correction)"
            "&order=posted_at.desc&select=tweet_id",
        )
        return (row or {}).get("tweet_id")
    except Exception:
        return None


# ---------------------------------------------------------------- 投稿API

def upload_media(image_bytes: bytes, alt_text: str = "") -> "str | None":
    """画像(PNG)をv1.1 media/uploadへアップロードし、post_tweetのmedia_idsに使う
    media_id_stringを返す。alt_textを渡すと代替テキストも設定する（画像内の数字が
    読み上げ環境に伝わらない問題への対応。施策4）。失敗時はNone。"""
    auth = _auth()
    if auth is None:
        return None
    try:
        resp = requests.post(
            MEDIA_UPLOAD_URL, auth=auth,
            files={"media": ("card.png", image_bytes, "image/png")}, timeout=30,
        )
        if not resp.ok:
            print(f"  ⚠ X画像アップロード失敗 HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        media_id = resp.json().get("media_id_string")
    except Exception as e:
        print(f"  ⚠ X画像アップロード例外: {e}")
        return None

    if media_id and alt_text:
        try:
            # alt設定の失敗は投稿を止めない（画像自体は既に上がっている）
            requests.post(
                MEDIA_METADATA_URL, auth=auth, timeout=20,
                json={"media_id": media_id, "alt_text": {"text": alt_text[:1000]}},
            )
        except Exception as e:
            print(f"  ⚠ alt設定に失敗（画像は添付されます）: {e}")
    return media_id


def post_tweet(text: str, media_ids: "list | None" = None, reply_to: "str | None" = None,
               kind: str = "other", **log_fields) -> "str | None":
    """投稿し、成功したらtweet_idを返す（失敗時None）。tweet_idはx_postsに記録する。
    以前はboolを返していたが、それだとどの投稿が伸びたかを後から追えなかった（施策1）。"""
    auth = _auth()
    if auth is None:
        return None
    payload: dict = {"text": text}
    if media_ids:
        payload["media"] = {"media_ids": list(media_ids)}
    if reply_to:
        payload["reply"] = {"in_reply_to_tweet_id": reply_to}
    try:
        resp = requests.post(API_URL, auth=auth, json=payload, timeout=20)
        if resp.status_code not in (200, 201):
            print(f"  ⚠ X投稿失敗 HTTP {resp.status_code}: {resp.text[:200]}")
            # 401/403は設定不備が原因なので、ログだけ見て原因が分かるよう実権限を添える
            if resp.status_code in (401, 403):
                v = verify_auth()
                print(f"    → トークンの実権限: {v['access_level'] or '不明'}"
                      + (f" / {v['reason']}" if v["reason"] else ""))
            return None
        tweet_id = (resp.json().get("data") or {}).get("id") or ""
    except Exception as e:
        print(f"  ⚠ X投稿例外: {e}")
        return None

    if tweet_id:
        log_post(tweet_id, kind, text, has_media=bool(media_ids), reply_to=reply_to, **log_fields)
    return tweet_id or None


# ---------------------------------------------------------------- 本文の組み立て

def _format_oku(amount_oku: "float | None") -> str:
    """金額を読みやすい丸めにする。「推定取得金額: 120.5億円」のような機械的な表記より
    「約120億円」のほうが1行目から視線が滑らない（施策2）。"""
    v = float(amount_oku or 0)
    if v >= 10000:
        return f"約{v / 10000:.1f}兆円"
    if v >= 10:
        return f"約{v:,.0f}億円"
    if v > 0:
        return f"約{v:.1f}億円"
    return ""


def _ratio_phrase(ratio: "float | None", prior: "float | None", change_pt: "float | None") -> str:
    """保有比率の変化。前後が分かるときは `3.21%→5.02%`、分からないときはpt差だけ出す。"""
    if ratio is not None and prior is not None:
        return f"保有比率 {prior:.2f}%→{ratio:.2f}%"
    if ratio is not None:
        return f"保有比率 {ratio:.2f}%"
    if change_pt is not None:
        return f"保有比率 {change_pt:+.2f}pt"
    return ""


def article_facts(article: dict) -> dict:
    """記事dictから投稿・画像・altで共通して使う事実を取り出す。"""
    tags = article.get("tags") or ""
    ratio = article.get("holdingRatio")
    change = article.get("ratioChangePct")
    prior = None
    if ratio is not None and change is not None:
        prior = round(float(ratio) - float(change), 2)
    is_correction = "訂正" in tags
    is_sell = "売り" in tags
    if is_correction:
        badge, kind = "訂正", "correction"
    elif is_sell:
        badge, kind = "売却", "sell"
    elif prior is not None and prior <= 0.01:
        badge, kind = "新規取得", "buy"
    else:
        badge, kind = "買い増し", "buy"
    return {
        "stock_name": clean_name(article.get("stockName") or ""),
        "stock_code": article.get("stockCode") or "",
        "filer_name": clean_name(article.get("filerName") or "") or "大口投資家",
        "ratio": None if ratio is None else float(ratio),
        "prior": prior,
        "change_pt": None if change is None else float(change),
        "amount_label": "" if is_correction else _format_oku(article.get("dealAmount")),
        "badge": badge,
        "kind": kind,
        "is_correction": is_correction,
    }


def build_tweet_text(article: dict, insight: str = "") -> str:
    """記事1件分の投稿本文。1行目は「誰が・どの銘柄を・どうした」のフックにし、
    2行目に金額と保有比率の変化を置く（施策2）。insightは提出者の文脈1行（施策9）で、
    文字数に収まらなければ落とす。URLは入れず、末尾の PROFILE_CTA で誘導する。"""
    f = article_facts(article)
    stock = label(f["stock_name"], STOCK_LABEL_MAX_UNITS)
    stock_label = f"{stock}({f['stock_code']})" if f["stock_code"] else stock
    filer = label(f["filer_name"], FILER_LABEL_MAX_UNITS)

    if f["is_correction"]:
        head = f"⚠️ 訂正｜{filer}が{stock_label}の届出保有比率を訂正"
        second = _ratio_phrase(f["ratio"], f["prior"], f["change_pt"])
        if f["change_pt"] is not None:
            second = f"{second}（{f['change_pt']:+.2f}pt）" if second else f"{f['change_pt']:+.2f}pt"
    else:
        mark = "🔴" if f["kind"] == "sell" else "🟢"
        verb = {"売却": "を売却", "新規取得": "を新規取得", "買い増し": "を買い増し"}[f["badge"]]
        head = f"{mark} {filer}が{stock_label}{verb}"
        parts = [p for p in (f["amount_label"], _ratio_phrase(f["ratio"], f["prior"], f["change_pt"])) if p]
        second = "・".join(parts)

    def assemble(with_insight: bool) -> str:
        lines = [head]
        if second:
            lines.append(second)
        if with_insight and insight:
            lines.append(insight)
        lines += [PROFILE_CTA, TAGS]
        return "\n".join(lines)

    for with_insight in (True, False):
        text = assemble(with_insight)
        if weighted_len(text) <= 270:
            return text
    return assemble(False)


def build_article_alt(article: dict) -> str:
    """数字カード画像の代替テキスト（施策4）。画像でしか出ていない数字を文章で持たせる。"""
    f = article_facts(article)
    stock = f"{f['stock_name']}（{f['stock_code']}）" if f["stock_code"] else f["stock_name"]
    ratio = _ratio_phrase(f["ratio"], f["prior"], f["change_pt"]) or "保有比率の記載なし"
    amount = f"、推定{f['amount_label']}" if f["amount_label"] else ""
    return f"{f['badge']}。{f['filer_name']}が{stock}。{ratio}{amount}。出典はEDINET提出書類。"


def build_article_media(article: dict) -> list:
    """記事投稿の添付画像を作ってアップロードする。**添付は必ず1枚だけ**。

    かつては1枚目=数字カード、2枚目=株価チャートの2枚組にしていたが、Xは複数画像を
    左右に並べて両方とも切り落とすため、タイムラインでは銘柄名も数字もチャートも読めない
    （2026-08-19、オーナー指摘）。投稿の主張を運んでいるのは数字カードの方なので、
    カードだけを1枚で出し、チャートはリンク先の記事に任せる。
    カードが作れなかったとき（フォント欠如等）だけチャートを代替として添える。"""
    from web.x_card_image import build_deal_card

    f = article_facts(article)
    date_label = (article.get("dealDate") or "")[:10].replace("-", "/")
    card = build_deal_card(
        f["stock_name"], f["stock_code"], f["filer_name"], f["badge"],
        f["ratio"], f["prior"], f["amount_label"] or "―", date_label, kind=f["kind"],
    )
    if card:
        media_id = upload_media(card, alt_text=build_article_alt(article))
        if media_id:
            return [media_id]

    from web.publish_blog_articles import generate_price_chart_image

    chart = generate_price_chart_image(article.get("stockCode") or "", f["stock_name"])
    if chart:
        media_id = upload_media(
            chart, alt_text=f"{f['stock_name']}（{f['stock_code']}）の直近3ヶ月の終値推移チャート。",
        )
        if media_id:
            return [media_id]
    return []


def _summary_line(article: dict, sign: str) -> str:
    """サマリーの「最大買い増し/最大売却」1行。提出者名が取れた記事は括弧書きで添える。"""
    filer = article.get("filerName") or ""
    filer_part = f"（{filer}）" if filer else ""
    return f"{article['stockName']} {sign}{article['dealAmount']}億円{filer_part}"


def build_daily_summary_text(articles: list, date_str: str, total_count: "int | None" = None) -> "str | None":
    """その日(dealDate)の記事一覧から日次サマリー本文を組み立てる。
    0件の日はNone（投稿しない）。articlesはmicroCMSのdealAmount降順取得を前提とする。
    total_countはmicroCMSのtotalCount（100件超の日にarticlesが先頭100件に切れていても
    件数表示を正確にするため）。URLは入れず、全件一覧へはプロフィールのリンクで誘導する。"""
    if not articles:
        return None
    count = total_count if total_count is not None else len(articles)
    total_oku = round(sum(a.get("dealAmount") or 0 for a in articles), 1)
    buys = [a for a in articles if "売り" not in (a.get("tags") or "")]
    sells = [a for a in articles if "売り" in (a.get("tags") or "")]

    month_day = f"{int(date_str[5:7])}/{int(date_str[8:10])}"
    lines = [f"🐋 {month_day}のクジラ｜大量保有報告書", f"{count}件・合計{total_oku}億円", ""]
    if buys:
        lines += ["🟢 最大買い増し", _summary_line(buys[0], "+"), ""]
    if sells:
        lines += ["🔴 最大売却", _summary_line(sells[0], "-"), ""]
    lines += [f"この日の全{count}件は{PROFILE_CTA[3:]}", TAGS]
    return "\n".join(lines)


def build_daily_summary_media(articles: list, date_str: str, total_count: int) -> list:
    """日次サマリーの一覧カード。これまでテキストのみで、画像が無い投稿はタイムラインで
    埋もれていた（施策4）。"""
    from web.x_card_image import build_list_card

    month_day = f"{int(date_str[5:7])}/{int(date_str[8:10])}"
    total_oku = round(sum(a.get("dealAmount") or 0 for a in articles), 1)
    rows = []
    for a in articles[:5]:
        is_sell = "売り" in (a.get("tags") or "")
        code = a.get("stockCode") or ""
        left = f"{a.get('stockName') or ''}（{code}）" if code else (a.get("stockName") or "")
        rows.append((left, f"{'-' if is_sell else '+'}{a.get('dealAmount') or 0}億円",
                     "sell" if is_sell else "buy"))
    card = build_list_card(
        f"{month_day}のクジラ｜大量保有報告書",
        f"{total_count}件・合計{total_oku}億円",
        rows,
        f"全{total_count}件は kujira-watch.com/date/{date_str}",
    )
    if not card:
        return []
    alt = (f"{month_day}の大量保有報告書は{total_count}件、合計約{total_oku}億円。"
           + "。".join(f"{left} {right}" for left, right, _ in rows))
    media_id = upload_media(card, alt_text=alt)
    return [media_id] if media_id else []


def build_video_tweet_text(props: dict, youtube_id: str) -> str:
    """YouTube Shorts公開時のクロス投稿本文。propsはvideo/build_script.pyのbuild()が
    返す動画props（stockName / filerName / direction / dealAmountOku）。"""
    direction = "売却" if props.get("direction") == "sell" else "取得"
    # 動画タイトル(youtube_client.build_title)と同じく、提出者名が無い記事は汎用の主語にする
    filer = clean_name(props.get("filerName") or "") or "大口投資家"
    stock = clean_name(props.get("stockName") or "")
    body = (f"🎬 1分解説｜{label(filer, FILER_LABEL_MAX_UNITS)}が"
            f"{label(stock, STOCK_LABEL_MAX_UNITS)}を{_format_oku(props.get('dealAmountOku'))}{direction}")
    # YouTubeのURLも「リンク入り投稿」として$0.20課金されるため入れない（youtube_idは
    # 投稿ログの突き合わせ用に残す）。動画への誘導はプロフィール経由に統一する。
    return f"{body}\n動画は{PROFILE_CTA[3:]}\n#Shorts {TAGS}"


# ---------------------------------------------------------------- 投稿の実行

def within_posting_hours(now_utc=None) -> bool:
    """自動投稿してよい時間帯（JST 8〜22時）か。深夜の自動投稿を防ぐ（施策5）。"""
    now = now_utc or datetime.now(timezone.utc)
    return (now + timedelta(hours=9)).hour in POST_HOURS_JST


def fetch_articles_by_deal_date(date_str: str) -> "tuple[list, int]":
    """microCMSからdealDateが指定日の記事をdealAmount降順で取得する（最大100件）。
    (記事リスト, totalCount)を返す。失敗時は([], 0)。"""
    from web.publish_blog_articles import _microcms_base_url, _microcms_headers

    try:
        resp = requests.get(
            _microcms_base_url(),
            headers=_microcms_headers(),
            params={
                "filters": f"dealDate[equals]{date_str}T00:00:00.000Z",
                "orders": "-dealAmount",
                "limit": 100,
                "fields": "id,stockName,stockCode,dealAmount,tags,filerName",
            },
            timeout=20,
        )
        if not resp.ok:
            print(f"  ⚠ 日次サマリー用の記事取得失敗 HTTP {resp.status_code}")
            return [], 0
        data = resp.json()
        return data.get("contents", []), data.get("totalCount", 0)
    except Exception as e:
        print(f"  ⚠ 日次サマリー用の記事取得例外: {e}")
        return [], 0


def summary_target_date(now_utc) -> str:
    """日次サマリーが対象にする日（JST）。朝9時の便では当日の開示はまだ無いので前営業日を返す。
    土日はEDINETの提出が無いため金曜まで遡る（祝日は0件でスキップされる）。"""
    d = (now_utc + timedelta(hours=9)).date() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def post_daily_summary(now_utc=None, force: bool = False) -> bool:
    """前営業日ぶんの日次サマリーを1日1回(9時JST=0時UTCの初便のみ)Xへ投稿する。
    forceで時刻ガードを無視できる（手動実行用）。該当時刻以外・0件・認証未設定はFalse。"""
    if _auth() is None:
        return False
    now = now_utc or datetime.now(timezone.utc)
    if not force and now.hour != DAILY_SUMMARY_UTC_HOUR:
        return False
    target = summary_target_date(now)
    articles, total_count = fetch_articles_by_deal_date(target)
    text = build_daily_summary_text(articles, target, total_count)
    if text is None:
        print("  [x_client] 本日の開示記事が無いため日次サマリーをスキップします")
        return False
    media_ids = build_daily_summary_media(articles, target, total_count)
    tweet_id = post_tweet(text, media_ids=media_ids, kind="daily")
    if tweet_id:
        print(f"  🐦 X日次サマリー投稿: {target} 全{total_count}件")
        return True
    return False


def post_top_articles(published: list, featured_ids: set, top_n: int = ARTICLES_PER_RUN,
                      now_utc=None) -> int:
    """publish_blog_articles.build_and_publish()が返すpublishedリスト（今日新規公開した
    記事）のうち、featured_ids（現在ホームページで「注目」表示されている記事id）にも
    含まれるものを金額規模順に最大top_n件、加えて訂正報告書の記事を全件Xへ投稿する。

    訂正記事はdealAmount=0のため「注目」枠(dealAmount上位)には決して入らないが、既に公開済みの
    記事の前提を覆す開示なので件数制限を設けない（実例: 2026-08-18、太陽誘電6976の
    15.22%→4.41%訂正で株価-11.5%。既報の「16.61%で大株主化」だけがXに流れたまま残った）。
    同じ銘柄で過去に投稿していれば、その投稿への自己リプライとしてぶら下げる（施策7）。"""
    if _auth() is None:
        print("[x_client] X_API_KEY等が未設定のため投稿をスキップします")
        return 0
    if not within_posting_hours(now_utc):
        print("[x_client] 投稿時間帯(JST 8〜22時)外のため投稿をスキップします")
        return 0

    from web.x_insight import build_insight_line, fetch_filer_context

    featured = [a for a in published if a.get("id") and a["id"] in featured_ids
                and "訂正" not in (a.get("tags") or "")]
    featured.sort(key=lambda a: a.get("dealAmount", 0), reverse=True)
    corrections = [a for a in published if a.get("id") and "訂正" in (a.get("tags") or "")]
    corrections.sort(key=lambda a: abs(a.get("ratioChangePct") or 0), reverse=True)
    candidates = corrections + featured[:top_n]

    posted = 0
    for article in candidates:
        is_correction = "訂正" in (article.get("tags") or "")
        context = fetch_filer_context(article.get("filerName") or "", article.get("stockCode") or "")
        insight = build_insight_line(clean_name(article.get("stockName") or ""), context)
        text = build_tweet_text(article, insight=insight)
        media_ids = build_article_media(article)
        reply_to = find_prior_tweet(article.get("stockCode") or "") if is_correction else None

        tweet_id = post_tweet(text, media_ids=media_ids,
                              kind="correction" if is_correction else "article",
                              reply_to=reply_to,
                              stock_code=article.get("stockCode"), article_id=article.get("id"))
        if tweet_id:
            print(f"  🐦 X投稿: {text.splitlines()[0]}"
                  + (f"（画像{len(media_ids)}枚）" if media_ids else "")
                  + ("（既報へのリプ）" if reply_to else ""))
            posted += 1
    return posted


def post_video_tweet(props: dict, youtube_id: str) -> bool:
    """動画公開のXクロス投稿。X認証未設定ならスキップ（動画投稿自体は止めない）。"""
    if _auth() is None:
        print("[x_client] X_API_KEY等が未設定のため動画クロス投稿をスキップします")
        return False
    return post_tweet(build_video_tweet_text(props, youtube_id), kind="video") is not None


if __name__ == "__main__":
    # 手動確認用: python -m web.x_client --verify（アクセストークンの実権限を表示する）
    import sys

    if "--verify" in sys.argv:
        result = verify_auth()
        mark = "✅" if result["ok"] else "❌"
        print(f"{mark} X認証: access_level={result['access_level'] or '不明'} "
              f"account={result['screen_name'] or '-'}")
        if result["reason"]:
            print(f"   {result['reason']}")
        sys.exit(0 if result["ok"] else 1)
    print("使い方: python -m web.x_client --verify")
