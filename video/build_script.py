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
    get_company_description,
    get_featured_article_ids,
    get_filer_profile,
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


def _trim_narration(text: str, limit: int) -> str:
    """ナレーション用の切り詰め。「…こ…」のような文の途中切りはそのまま読み上げられて
    しまうため、上限内の最後の句点で切り落とす（句点が見つからない場合のみ_trimに落ちる）。"""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    last_period = cut.rfind("。")
    if last_period >= 20:  # 先頭近くの句点で切ると1文も残らないため下限を設ける
        return cut[: last_period + 1]
    return _trim(text, limit)


# 画面に出す字幕(caption)の上限。縦画面で1行に収まり、読み上げと同時に目で追える長さ。
CAPTION_MAX_CHARS = 26
# 1シーンの読み上げ文の上限。長すぎるとシーンが間延びして飽きられる。
NARRATION_MAX_CHARS = 90

# 動画の構成。kind は Remotion 側の見せ方（video/remotion/src/scenes/）に対応する。
# 「記事の内容をほぼ読む」構成にするため、事実の羅列だけでなく銘柄と提出者の説明を挟む。
SECTION_SPEC = [
    ("company", "この会社が何をしている会社か。事業内容を初見の視聴者にわかるように"),
    ("deal", "誰がいくら分をどう動かしたのか。金額と保有比率という数字の事実"),
    ("filer", "その提出者がどんな投資家か。運用方針や性格がわかるように"),
    ("change", "前回の開示からどう変わったのか。数字の推移が持つ意味"),
    ("outlook", "この取引が今後どんな意味を持ちうるか。記事の※推測部分に対応する"),
]


def generate_script(article: dict, company_description: str = "", filer_profile: str = "",
                    _retry: bool = True) -> "dict | None":
    """記事本文から縦動画用の台本を生成する。各シーンは
    `narration`（読み上げ文）と `caption`（画面に出す短い字幕）の対で構成する。
    パース失敗時は None（呼び出し側は動画を作らない）。"""
    import anthropic

    if not ANTHROPIC_API_KEY:
        print("  ⚠ ANTHROPIC_API_KEY 未設定のため台本を生成できません")
        return None

    body_text = _strip_html(article.get("body", ""))
    is_sell = "売り" in (article.get("tags") or "")
    direction_label = "売却（保有比率の減少）" if is_sell else "取得（保有比率の増加）"

    # 銘柄・提出者の説明は記事本文に入り切っていないことがあるため、Supabaseに
    # キャッシュ済みの事実を別途渡す（どちらも publish_blog_articles.py が生成・保存したもの）。
    company_line = f"\n対象銘柄の事業内容: {company_description}" if company_description else ""
    filer_line = f"\n提出者のプロフィール: {filer_profile[:800]}" if filer_profile else ""

    sections_spec = "\n".join(
        f'  - kind="{kind}": {desc}' for kind, desc in SECTION_SPEC
    )

    prompt = f"""以下は、日本株の大量保有報告書（EDINET開示）を解説した公開済みブログ記事です。
この記事と補足事実だけを使って、**ナレーション付きの縦型ショート動画**の台本を作ってください。
記事・補足事実に無い数字・意図・背景は絶対に足さないでください。

記事タイトル: {article.get('title', '')}
対象銘柄: {article.get('stockName', '')}（{article.get('stockCode', '')}）
提出者: {article.get('filerName', '')}
取引の向き: {direction_label}
推定金額: {article.get('dealAmount')}億円{company_line}{filer_line}

記事本文:
{body_text[:2500]}

この動画は音声ナレーションで記事の内容をほぼ読み上げ、画面には要点だけを大きく出します。
そのため各シーンについて **narration（読み上げ文）** と **caption（画面に出す字幕）** の
2つを作ってください。captionはnarrationの要約であって、同じ文をそのまま入れないこと。

1. hook: 冒頭で指を止めさせる部分。
   - narration: 40〜60字。記事にある具体的な数字を含め、視聴者が「続きを聞きたい」と思う入り。
   - caption: {CAPTION_MAX_CHARS}字以内。体言止めか言い切りで句点は付けない。
2. sections: 以下の5つを**この順で**作る（kindは指定どおりの文字列にすること）。
{sections_spec}
   各セクション:
   - narration: 50〜{NARRATION_MAX_CHARS}字。話し言葉として自然に繋がるようにし、
     前のセクションと同じ言い回しを繰り返さない。数字は読み上げやすい表記にする
     （例: 「8.77%」は「8.77パーセント」）。
   - caption: {CAPTION_MAX_CHARS}字以内。そのシーンで一番伝えたい一点だけ。
   kind="outlook" は断定を避け「〜の可能性があります」等の推測の言い回しにすること。
   補足事実が無くて書けないkindがある場合でも、記事本文から言える範囲で必ず埋めること。
3. closing: 締め。
   - narration: 20〜35字。サイトで続きが読めることを伝える。
   - caption: 12〜18字。

**字数は動画のレイアウトと尺の制約なので厳守してください。**

出力はJSON形式のみとし、コードフェンスや他のテキストは含めないでください:
{{"hook": {{"narration": "...", "caption": "..."}},
  "sections": [{{"kind": "company", "narration": "...", "caption": "..."}}, ...5件...],
  "closing": {{"narration": "...", "caption": "..."}}}}
"""
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[4:] if text.lower().startswith("json") else text
        data = json.loads(text)
    except Exception as e:
        print(f"  ⚠ 台本生成失敗: {e}")
        return None

    scenes = _flatten_scenes(data)
    if scenes is None:
        print("  ⚠ 台本の形式が不正（hook / sections / closing の欠落）")
        return None

    too_long = [s["caption"] for s in scenes if len(s["caption"]) > CAPTION_MAX_CHARS]
    too_long += [s["narration"] for s in scenes if len(s["narration"]) > NARRATION_MAX_CHARS]
    if too_long and _retry:
        print(f"  ↻ 台本が長すぎるため作り直します（最長{max(len(t) for t in too_long)}字）")
        return generate_script(article, company_description, filer_profile, _retry=False)

    for scene in scenes:
        scene["caption"] = _trim(scene["caption"], CAPTION_MAX_CHARS)
        scene["narration"] = _trim_narration(scene["narration"], NARRATION_MAX_CHARS)
    return {"scenes": scenes}


def _flatten_scenes(data: dict) -> "list | None":
    """Claudeの出力（hook / sections / closing）を、Remotionが順に再生する
    フラットなシーン列に変換する。kindが見せ方の分岐になる。"""
    hook = data.get("hook") or {}
    closing = data.get("closing") or {}
    sections = data.get("sections") or []
    if not hook.get("narration") or not closing.get("narration") or len(sections) < len(SECTION_SPEC):
        return None

    scenes = [{"kind": "hook", "caption": hook.get("caption", ""), "narration": hook["narration"]}]
    expected_kinds = [kind for kind, _ in SECTION_SPEC]
    for kind, section in zip(expected_kinds, sections):
        if not section.get("narration"):
            return None
        # kindはClaudeの出力に頼らず期待順で上書きする（見せ方の分岐がずれると崩れるため）
        scenes.append({
            "kind": kind,
            "caption": section.get("caption", ""),
            "narration": section["narration"],
        })
    scenes.append({"kind": "cta", "caption": closing.get("caption", ""), "narration": closing["narration"]})
    return scenes


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
        "scenes": script["scenes"],
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

    # 銘柄の事業内容と提出者のプロフィールは publish_blog_articles.py が既に生成して
    # Supabaseにキャッシュ済み。動画で「どんな会社か・どんな投資家か」を語るために再利用する
    # （キャッシュが無い場合はClaudeが生成して保存するので、ここが初出になることもある）。
    company_description = get_company_description(
        str(article.get("stockCode") or ""), article.get("stockName") or ""
    )
    filer_name = article.get("filerName") or ""
    filer_profile = (
        get_filer_profile(filer_name, deal_type_label(article)) if filer_name else ""
    )

    script = generate_script(article, company_description, filer_profile)
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
