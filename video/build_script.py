"""
video/build_script.py

microCMSに公開済みのブログ記事から、縦動画（YouTube Shorts / TikTok）1本ぶんの
台本＝Remotionに渡すprops JSONを組み立てる。

対象記事の選び方は web/x_client.py と同じ思想:
「直近に新規公開された記事」×「サイトのホームで注目枠に入っている記事」の積集合の先頭1件。
サイト上で目立っていない小粒な開示だけが動画になる事態を防ぐため。該当が無い日は None を返し、
その日は投稿しない（無理に毎日出さない）。

台本の文言は Claude が記事本文から生成するが、記事に書かれていない事実は足させない
（記事本文自体が publish_blog_articles.py で事実のみから生成されているため、
ここで新しい情報源を混ぜると事実確認の連鎖が切れる）。
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone

import requests

sys.path.insert(0, os.path.expanduser("~/stock-alert"))

from web.publish_blog_articles import (  # noqa: E402
    CLAUDE_MODEL,
    _microcms_base_url,
    _microcms_headers,
    get_featured_article_ids,
)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# 「直近に公開された記事」とみなす時間幅。edinet_blog.yml は平日9-21時JSTに毎時回るので、
# 1日1回の動画バッチから見て当日ぶんを取りこぼさないよう余裕をもって36時間にする。
RECENT_HOURS = 36

# 動画1本あたりの候補プール。この中から featured との積集合を取る。
CANDIDATE_LIMIT = 20


def fetch_recent_articles(hours: int = RECENT_HOURS) -> list:
    """直近hours時間にmicroCMSへ公開された記事を、金額規模の大きい順で返す。"""
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    try:
        resp = requests.get(
            _microcms_base_url(),
            headers=_microcms_headers(),
            params={
                "orders": "-dealDate,-dealAmount",
                "limit": CANDIDATE_LIMIT,
                "filters": f"publishedAt[greater_than]{since}",
                "fields": "id,title,body,stockName,stockCode,dealType,dealDate,dealAmount,tags,filerName,attentionScore",
            },
            timeout=20,
        )
        if resp.status_code != 200:
            print(f"  ⚠ 記事取得失敗 HTTP {resp.status_code}: {resp.text[:200]}")
            return []
        return resp.json().get("contents", [])
    except Exception as e:
        print(f"  ⚠ 記事取得例外: {e}")
        return []


def pick_article(articles: list, featured_ids: set) -> "dict | None":
    """注目枠に入っている記事のうち、金額規模が最大のものを1件選ぶ。"""
    candidates = [a for a in articles if a.get("id") in featured_ids]
    if not candidates:
        return None
    candidates.sort(key=lambda a: a.get("dealAmount") or 0, reverse=True)
    return candidates[0]


def _strip_html(html: str) -> str:
    text = re.sub(r"<[^>]+>", "", html or "")
    return re.sub(r"\s+", " ", text).strip()


def _trim(text: str, limit: int) -> str:
    """作り直しても字数上限を超えた場合の最後の砦。動画のレイアウトが崩れるくらいなら
    末尾を落とす（ここに到達するのは稀）。"""
    return text if len(text) <= limit else text[: limit - 1] + "…"


# 縦動画の1行は40字を超えると3行に折り返して読み切れなくなるため、この長さを上限とする。
# 生成が長すぎた場合は一度だけ作り直し、それでも超えるものは末尾を詰める。
BULLET_MAX_CHARS = 40
HOOK_MAX_CHARS = 30


def generate_script(article: dict, _retry: bool = True) -> "dict | None":
    """記事本文から縦動画用の台本（hook / bullets / closing）を生成する。
    パース失敗時は None（呼び出し側は動画を作らない）。"""
    import anthropic

    if not ANTHROPIC_API_KEY:
        print("  ⚠ ANTHROPIC_API_KEY 未設定のため台本を生成できません")
        return None

    body_text = _strip_html(article.get("body", ""))
    is_sell = "売り" in (article.get("tags") or "")
    direction_label = "売却（保有比率の減少）" if is_sell else "取得（保有比率の増加）"

    prompt = f"""以下は、日本株の大量保有報告書（EDINET開示）を解説した公開済みブログ記事です。
この記事に書かれている事実だけを使って、縦型ショート動画（20秒）の台本を作ってください。
記事に無い数字・意図・背景は絶対に足さないでください。

記事タイトル: {article.get('title', '')}
対象銘柄: {article.get('stockName', '')}（{article.get('stockCode', '')}）
提出者: {article.get('filerName', '')}
取引の向き: {direction_label}
推定金額: {article.get('dealAmount')}億円

記事本文:
{body_text[:2000]}

次の3つを作ってください。**字数は動画のレイアウト上の制約なので厳守してください。**
1. hook: 冒頭1.5秒で指を止めさせる一文（20〜{HOOK_MAX_CHARS}字。{HOOK_MAX_CHARS}字を超えたら不合格）。
   誇張や煽りは禁止。記事にある具体的な数字を1つ含めること。体言止めか言い切りで、句点は付けない。
2. bullets: 要点3行（各25〜{BULLET_MAX_CHARS}字。{BULLET_MAX_CHARS}字を超えたら不合格）。
   1行目は「何が起きたか」、2行目は「数字の意味・前回からの変化」、3行目は「読者にとっての着眼点」。
   会社名や提出者名は動画の別の場面で表示済みなので、bulletsでは繰り返さず要点だけを短く書くこと。
   記事の※推測部分を使う場合は「〜の可能性」と明示して断定しない。
3. closing: 締めの一文（12〜18字）。サイトへ誘導する自然な言い回しにする（例: 「続きはクジラウォッチで」）。

出力はJSON形式のみとし、コードフェンスや他のテキストは含めないでください:
{{"hook": "...", "bullets": ["...", "...", "..."], "closing": "..."}}
"""
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[4:] if text.lower().startswith("json") else text
        data = json.loads(text)
        bullets = data.get("bullets") or []
        if not data.get("hook") or len(bullets) < 3:
            print("  ⚠ 台本の形式が不正（hook欠落 or bullets不足）")
            return None

        bullets = bullets[:3]
        too_long = [b for b in bullets if len(b) > BULLET_MAX_CHARS] or (
            [data["hook"]] if len(data["hook"]) > HOOK_MAX_CHARS else []
        )
        if too_long and _retry:
            print(f"  ↻ 台本が長すぎるため作り直します（最長{max(len(t) for t in too_long)}字）")
            return generate_script(article, _retry=False)

        return {
            "hook": _trim(data["hook"], HOOK_MAX_CHARS),
            "bullets": [_trim(b, BULLET_MAX_CHARS) for b in bullets],
            "closing": data.get("closing") or "続きはクジラウォッチで",
        }
    except Exception as e:
        print(f"  ⚠ 台本生成失敗: {e}")
        return None


def deal_type_label(article: dict) -> str:
    """microCMSのdealTypeはセレクト型なので配列で返る（例: ["国内アセットマネジメント"]）。
    動画には1行のラベルとして出すため先頭要素だけを使う。未設定なら汎用ラベル。"""
    value = article.get("dealType")
    if isinstance(value, list):
        return value[0] if value else "大量保有報告書"
    return value or "大量保有報告書"


def build_props(article: dict, script: dict) -> dict:
    """Remotion の ArticleShort コンポジションに渡す props（video/remotion/src/types.ts の
    ShortProps と同じ形）を組み立てる。"""
    is_sell = "売り" in (article.get("tags") or "")
    deal_date = (article.get("dealDate") or "")[:10]
    return {
        "stockName": article.get("stockName") or "",
        "stockCode": str(article.get("stockCode") or ""),
        # filerName は古い記事だと未設定のことがある（microCMSは空フィールドを返さない）。
        # 動画側は空文字なら提出者名の行を出さない。
        "filerName": article.get("filerName") or "",
        "dealTypeLabel": deal_type_label(article),
        "direction": "sell" if is_sell else "buy",
        "dealAmountOku": float(article.get("dealAmount") or 0),
        "holdingRatio": extract_holding_ratio(article),
        "discDate": deal_date,
        "hook": script["hook"],
        "bullets": script["bullets"],
        "closing": script["closing"],
    }


def extract_holding_ratio(article: dict) -> float:
    """記事本文から今回の保有比率を拾う。microCMSのarticlesスキーマには保有比率の
    専用フィールドが無く（publish_blog_articles.pyのpayload参照）、本文中の
    「5.21%」という表記が唯一の出どころのため、最後に現れる◯.◯◯%を採用する
    （本文は「前回◯%→今回◯%」の順で書かれるため、末尾側が今回の比率になる）。
    拾えない場合は0.0を返し、動画側は比率タイルを0.00%と表示する。"""
    text = _strip_html(article.get("body", ""))
    matches = re.findall(r"(\d{1,2}\.\d{1,2})\s*%", text)
    if not matches:
        return 0.0
    return float(matches[-1])


def build(dry_run: bool = False) -> "dict | None":
    """動画1本ぶんの props を返す。対象記事が無ければ None。"""
    articles = fetch_recent_articles()
    if not articles:
        print("[build_script] 直近に新規公開された記事がありません")
        return None

    featured_ids = get_featured_article_ids()
    article = pick_article(articles, featured_ids)
    if article is None:
        print(f"[build_script] 新着{len(articles)}件のうち注目枠に入る記事が無いため動画は作りません")
        return None

    print(f"[build_script] 対象記事: {article['title']}（{article['stockName']} / id={article['id']}）")
    script = generate_script(article)
    if script is None:
        return None

    props = build_props(article, script)
    props["articleId"] = article["id"]
    props["articleTitle"] = article["title"]
    if dry_run:
        print(json.dumps(props, ensure_ascii=False, indent=2))
    return props


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="", help="props JSONの出力先パス（未指定なら標準出力）")
    p.add_argument("--dry-run", action="store_true", help="生成結果を表示するのみ")
    args = p.parse_args()

    props = build(dry_run=args.dry_run)
    if props is None:
        sys.exit(1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(props, f, ensure_ascii=False, indent=2)
        print(f"[build_script] props を書き出しました: {args.out}")
    elif not args.dry_run:
        print(json.dumps(props, ensure_ascii=False))


if __name__ == "__main__":
    main()
