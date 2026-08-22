"""
video/build_script.py

microCMSに公開済みのブログ記事から、縦動画（YouTube Shorts）1本ぶんの
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
import unicodedata
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

# 台本の組み立てに必要な記事フィールド（一覧取得・ID指定取得で共通）。
ARTICLE_FIELDS = "id,title,body,stockName,stockCode,dealType,dealDate,dealAmount,tags,filerName"

# 記事URL（https://kujira-watch.com/articles/xxxx）から記事IDを取り出す。
ARTICLE_URL_RE = re.compile(r"/articles/([A-Za-z0-9_-]+)")


def fetch_recent_articles(hours: int = RECENT_HOURS) -> list:
    """直近hours時間にmicroCMSへ公開された記事を、金額規模の大きい順で返す。"""
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    filters = f"publishedAt[greater_than]{since}"
    try:
        resp = requests.get(
            _microcms_base_url(),
            headers=_microcms_headers(),
            params={
                "orders": "-dealDate,-dealAmount",
                "limit": CANDIDATE_LIMIT,
                "filters": filters,
                "fields": ARTICLE_FIELDS,
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


def parse_article_id(value: str) -> str:
    """記事IDを正規化する。記事URLを丸ごと渡してもID部分だけを取り出す
    （オーナーがブラウザからURLをコピペして手動実行できるようにするため）。"""
    value = (value or "").strip()
    m = ARTICLE_URL_RE.search(value)
    if m:
        return m.group(1)
    return value.strip("/")


def fetch_article_by_id(article_id: str) -> "dict | None":
    """記事ID指定で1件取得する。公開時刻も注目枠も問わないので、
    気に入った記事を後からいつでも動画にできる。"""
    try:
        resp = requests.get(
            f"{_microcms_base_url()}/{article_id}",
            headers=_microcms_headers(),
            params={"fields": ARTICLE_FIELDS},
            timeout=20,
        )
        if resp.status_code != 200:
            print(f"  ⚠ 記事{article_id}の取得失敗 HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        return resp.json()
    except Exception as e:
        print(f"  ⚠ 記事取得例外: {e}")
        return None


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
    if last_period >= 12:  # 先頭近くの句点で切ると1文も残らないため下限を設ける
        return cut[: last_period + 1]
    return _trim(text, limit)


# 画面に出す字幕(caption)の上限。縦画面で1行に収まり、読み上げと同時に目で追える長さ。
CAPTION_MAX_CHARS = 26
# 1シーンの読み上げ文の上限。90字（約12秒）だと1シーンが持たず、総尺も89秒に膨らんでいた。
# 55字＝約7秒に落として総尺45〜50秒を狙う（2026-08-19のインフルエンサーレビュー）。
NARRATION_MAX_CHARS = 55

# 動画の構成。kind は Remotion 側の見せ方（video/remotion/src/scenes/）に対応する。
# 「記事の内容をほぼ読む」構成にするため、事実の羅列だけでなく銘柄と提出者の説明を挟む。
# outlook（推測シーン）は 2026-08-19 に廃止した。中央ビジュアルが無く10.9秒を
# 「〜の可能性があります」だけで使っており、尺の浪費と投資助言リスクを同時に抱えていたため。
SECTION_SPEC = [
    ("company", "この会社が何をしている会社か。事業内容を初見の視聴者にわかるように"),
    ("deal", "誰がいくら分をどう動かしたのか。金額と保有比率という数字の事実"),
    ("filer", "その提出者がどんな投資家か。運用方針や性格がわかるように"),
    ("change", "前回の開示からどう変わったのか。数字の推移が持つ意味"),
]


# 作り直しでも直らなかったシーンを、記事の事実だけで組み直すための定型文。
# 数値はすべて記事・開示由来で、ここで新しい事実を作らない（build_price_scene と同じ考え方）。
# company / filer は事実の言い換えが必要で定型化できないため、壊れていたらシーンごと落とす。
DROPPABLE_KINDS = ("company", "filer")


def _facts(article: dict) -> dict:
    """定型文の組み立てに使う、記事から確定できる事実。"""
    is_sell = "売り" in (article.get("tags") or "")
    filer = article.get("filerName") or ""
    return {
        "stock": article.get("stockName") or "この銘柄",
        # 提出者名は英語の正式名だと読み上げが長くなるため、20字を超えたら分類ラベルで代える
        "filer": filer if 0 < len(filer) <= 20 else deal_type_label(article),
        "amount": article.get("dealAmount") or 0,
        "ratio": extract_holding_ratio(article),
        "prev": extract_prev_holding_ratio(article),
        "verb": "売却しました" if is_sell else "取得しました",
    }


def _template_scene(kind: str, f: dict) -> "dict | None":
    """壊れたシーンを事実だけで組み直す。組めない場合は None（呼び出し側が落とす）。"""
    if kind == "hook":
        return {"kind": "hook",
                "caption": f"{f['stock']}に{f['amount']}億円",
                "narration": f"{f['filer']}が{f['stock']}を{f['amount']}億円{f['verb']}。"}
    if kind == "deal":
        return {"kind": "deal",
                "caption": f"推定{f['amount']}億円・保有{f['ratio']}%",
                "narration": f"推定の金額は{f['amount']}億円。"
                             f"保有比率は{f['ratio']}パーセントです。"}
    if kind == "change" and f["prev"] is not None:
        diff = abs(f["ratio"] - f["prev"])
        move = "減らしました" if f["prev"] > f["ratio"] else "積み増しました"
        return {"kind": "change",
                "caption": f"{f['prev']}%から{diff:.2f}ポイント",
                "narration": f"前回の{f['prev']}パーセントから"
                             f"{diff:.2f}ポイント{move}。"}
    if kind == "cta":
        return {"kind": "cta",
                "caption": "続きはブログで公開中",
                "narration": "詳しくは大口投資家の監視ブログで。"}
    return None


def salvage_scenes(scenes: list, article: dict) -> "list | None":
    """作り直しでも読み上げ文が直らなかったシーンを、落とすか定型文に差し替える。

    1シーンの生成に失敗しただけで動画を丸ごと諦めると、その日の投稿が飛ぶ
    （2026-08-19・20と2日続けて0件になった）。壊れた文を読み上げないことと、
    毎日1本出すことを両立させるため、事実だけで組み直せるシーンは組み直し、
    組み直せないシーン（会社説明・投資家説明）は落とす。"""
    facts = _facts(article)
    salvaged = []
    for scene in scenes:
        if not is_broken_narration(scene["narration"]):
            salvaged.append(scene)
            continue
        kind = scene["kind"]
        if kind in DROPPABLE_KINDS:
            print(f"  ↷ {kind}シーンの読み上げ文が直らないためシーンごと落とします")
            continue
        replacement = _template_scene(kind, facts)
        if replacement is None:
            print(f"  ↷ {kind}シーンを組み直せないため落とします")
            continue
        print(f"  ↷ {kind}シーンの読み上げ文を事実ベースの定型文に差し替えました")
        salvaged.append(replacement)

    kinds = [s["kind"] for s in salvaged]
    if "hook" not in kinds or "cta" not in kinds:
        print("  ⚠ hook / cta を組み直せないため動画を作りません")
        return None
    return salvaged


def is_broken_narration(text: str) -> bool:
    """読み上げ文が文の途中で切れているか。切り詰めの「…」がそのまま画面に出て
    VOICEVOXにも読み上げられていた実害（2026-08-19に指摘）を検知するために使う。"""
    text = (text or "").rstrip()
    return "…" in text or not text.endswith(("。", "！", "？"))


def generate_script(article: dict, company_description: str = "", filer_profile: str = "",
                    _retries: int = 3, _feedback: str = "") -> "dict | None":
    """記事本文から縦動画用の台本を生成する。各シーンは
    `narration`（読み上げ文）と `caption`（画面に出す短い字幕）の対で構成する。
    パース失敗時・文が途中で切れている場合は None（呼び出し側は動画を作らない。
    途中で切れた文を読み上げる動画を出すくらいなら、その日は投稿しない）。

    字数超過は _retries 回まで作り直す。作り直しでは「何字の文が長すぎたか」を
    プロンプトに足して伝える（同じ指示をそのまま投げ直しても同じ長さが返るため。
    2026-08-19に日本製鉄の回が2回とも86〜93字で返って投稿0件になった実障害への対策）。"""
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

この動画は**40秒前後のショート動画**です。記事を読み上げるのではなく、記事の要点だけを
テンポよく短く言い切ります。各シーンについて **narration（読み上げ文）** と
**caption（画面に出す字幕）** の2つを作ってください。captionはnarrationの要約であって、
同じ文をそのまま入れないこと。

**1文は40字以内**にしてください。説明が入り切らない場合は情報を削ってください
（2文に分けるのではなく、そのシーンで一番大事な一点だけを残す）。

大量保有報告書は株式の取得・売却の開示であって企業買収ではありません。
「買収」「経営権を握る」「TOB」など、開示内容を超える語は使わないでください。

取引の言い方は開示の事実に合わせてください。**今回が新規保有なら「買い増し」とは書かず
「新規に取得」「新たに保有」**と書きます（逆に、前回の開示がある場合だけ「買い増し」が使えます）。

1. hook: 冒頭で指を止めさせる部分。最初の1秒で固有名詞か金額が耳に入ることが最優先。
   - narration: **22〜30字**。語順は〈誰が〉→〈何を〉→〈いくら〉に固定する。
     「今日は」「みなさん」「ご存知ですか」などの前置きで始めるのは禁止。
     型の例（記事の事実に合うものを選ぶ）:
     ・「香港のモノ言う株主が、調剤薬局大手を24.5億円。」
     ・「アクティビストが〈銘柄名〉を〈金額〉億円買いました。」
   - caption: {CAPTION_MAX_CHARS}字以内。体言止めか言い切りで句点は付けない。銘柄名か金額を含める。
2. sections: 以下の4つを**この順で**作る（kindは指定どおりの文字列にすること）。
{sections_spec}
   各セクション:
   - narration: **35〜{NARRATION_MAX_CHARS}字**。話し言葉として自然に繋がるようにし、
     前のセクションと同じ言い回しを繰り返さない。数字は読み上げやすい表記にする
     （例: 「8.77%」は「8.77パーセント」）。
   - caption: {CAPTION_MAX_CHARS}字以内。そのシーンで一番伝えたい一点だけ。数字を1つ含める。
   補足事実が無くて書けないkindがある場合でも、記事本文から言える範囲で必ず埋めること。
3. closing: 締め。
   - narration: **14〜20字**。「大口投資家の監視ブログ」でサイトの続きが読めることを伝える。
     URLは読み上げない（英字URLは音声で聞き取れないため）。検索を促す言い方はしない
     （サイト名で検索上位を取れていないため、言っても辿り着けない）。
   - caption: 10〜18字。

**すべての narration は必ず「。」で終わる完結した文にしてください。**
文の途中で終わる出力は使えません。**字数は動画のレイアウトと尺の制約なので厳守してください。**

{_feedback}出力はJSON形式のみとし、コードフェンスや他のテキストは含めないでください:
{{"hook": {{"narration": "...", "caption": "..."}},
  "sections": [{{"kind": "company", "narration": "...", "caption": "..."}}, ...4件...],
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

    over_caption = [s["caption"] for s in scenes if len(s["caption"]) > CAPTION_MAX_CHARS]
    over_narration = [s["narration"] for s in scenes if len(s["narration"]) > NARRATION_MAX_CHARS]
    if (over_caption or over_narration) and _retries > 0:
        # captionとnarrationは上限が別（26字 / 55字）。どちらが超えたかを言い分けないと、
        # 29字のcaption超過を「narrationが長い」と伝えて見当違いの直しをさせることになる
        # （2026-08-20の博報堂DYの回が実際にこれで3回とも外し、投稿0件になった）。
        problems = []
        if over_caption:
            worst = max(over_caption, key=len)
            problems.append(
                f"caption「{worst[:40]}」が{len(worst)}字ありました（上限{CAPTION_MAX_CHARS}字）"
            )
        if over_narration:
            worst = max(over_narration, key=len)
            problems.append(
                f"narration「{worst[:60]}」が{len(worst)}字ありました（上限{NARRATION_MAX_CHARS}字）"
            )
        print(f"  ↻ 台本が長すぎるため作り直します（{' / '.join(problems)}）")
        feedback = (
            "【前回の出力の問題】" + "。".join(problems) + "。\n"
            f"captionは{CAPTION_MAX_CHARS}字以内、narrationは1文40字以内かつ"
            f"{NARRATION_MAX_CHARS}字以内が絶対条件です。字数を数えてから出力してください。\n\n"
        )
        return generate_script(article, company_description, filer_profile,
                               _retries - 1, feedback)

    for scene in scenes:
        scene["caption"] = _trim(scene["caption"], CAPTION_MAX_CHARS)
        scene["narration"] = _trim_narration(scene["narration"], NARRATION_MAX_CHARS)

    broken = [s["narration"] for s in scenes if is_broken_narration(s["narration"])]
    if broken and _retries > 0:
        print(f"  ↻ 読み上げ文が文の途中で切れているため作り直します: {broken[0][-24:]}")
        feedback = (
            f"【前回の出力の問題】narration「{broken[0][:60]}」が長すぎて途中で切れました。\n"
            f"narrationは1文40字以内・1シーン{NARRATION_MAX_CHARS}字以内で、"
            "必ず「。」で終わる完結した文にしてください。\n\n"
        )
        return generate_script(article, company_description, filer_profile,
                               _retries - 1, feedback)

    scenes = salvage_scenes(scenes, article)
    if scenes is None:
        return None
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


# 法人格の表記ゆれ（「株式会社」の有無など）で照合を落とさないために除く語。
_FILER_SUFFIX = re.compile(r"(株式会社|合同会社|有限会社|一般社団法人|公益財団法人|財団法人)")


def _normalize_name(text: str) -> str:
    """提出者名の照合用に正規化する。全角半角・空白・中黒・法人格の差を吸収する
    （開示データは「久世　良太」、本文は「久世良太氏」のように書かれるため）。"""
    text = unicodedata.normalize("NFKC", text or "")
    text = _FILER_SUFFIX.sub("", text)
    return re.sub(r"[\s　・.,．，]", "", text).lower()


def resolve_filer_name(article: dict) -> str:
    """記事の提出者名。microCMSのfilerNameが空の旧記事（2026-08-16以前）は、
    同じ銘柄・同じ開示日の大量保有報告書の提出者を候補に挙げ、**記事本文に名前が
    書かれているもの**を選ぶ。

    同一銘柄・同一開示日には複数の提出者がいることが普通なので、開示データだけでは
    一意に決まらない。本文との突き合わせを条件にすることで、誤った提出者名を
    動画のタイトルに載せないようにする。一意に決まらなければ空文字を返し、
    呼び出し側は「大口投資家」という総称にフォールバックする。"""
    name = article.get("filerName") or ""
    if name:
        return name
    code = str(article.get("stockCode") or "")
    disc_date = (article.get("dealDate") or "")[:10]
    body = article.get("body") or ""
    if not (code and disc_date and body):
        return ""
    try:
        from lib.db import sb

        rows = sb.select(
            "edinet_large_holdings",
            f"issuer_code=eq.{code}&disc_date=eq.{disc_date}&select=filer_name",
        )
    except Exception as e:
        print(f"  ⚠ 提出者名の照会に失敗しました（総称で続行）: {e}")
        return ""

    normalized_body = _normalize_name(_strip_html(body))
    hits = {
        r["filer_name"] for r in rows
        if _normalize_name(r.get("filer_name")) and _normalize_name(r["filer_name"]) in normalized_body
    }
    if len(hits) == 1:
        found = hits.pop()
        print(f"  ↷ 記事にfilerNameが無いため開示データから特定しました: {found}")
        return found
    return ""


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
        # filerName は古い記事だと未設定。開示データと本文の突き合わせで特定を試み、
        # それでも決まらなければ空文字（動画側は提出者名の行を出さない）。
        "filerName": resolve_filer_name(article),
        "dealTypeLabel": deal_type_label(article),
        "direction": "sell" if is_sell else "buy",
        "dealAmountOku": float(article.get("dealAmount") or 0),
        "holdingRatio": extract_holding_ratio(article),
        "prevHoldingRatio": extract_prev_holding_ratio(article),
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


def extract_prev_holding_ratio(article: dict) -> "float | None":
    """記事本文から前回開示の保有比率を拾う。本文は「前回◯%→今回◯%」の順で書かれるため
    末尾から2番目の◯.◯◯%を前回とみなす。ただしこれは書式に依存した推定なので、
    向き（買いなら増加・売りなら減少）と矛盾する場合は None を返す。
    None の回は change シーンごと出さない（誤った数字を出すくらいならシーンを削る）。"""
    text = _strip_html(article.get("body", ""))
    matches = re.findall(r"(\d{1,2}\.\d{1,2})\s*%", text)
    if len(matches) < 2:
        return None
    prev, current = float(matches[-2]), float(matches[-1])
    if prev == current:
        return None
    is_sell = "売り" in (article.get("tags") or "")
    if is_sell and prev < current:
        return None
    if not is_sell and prev > current:
        return None
    return prev


def build_price_scene(code: str, disc_date: str = "") -> "dict | None":
    """直近3ヶ月の株価推移シーン（kind="chart"）を作る。データは yahoo_price_cache
    （lib.utils.get_prices、記事の埋め込みチャートと同じソース）。ナレーションは
    Claudeではなくテンプレートで組み立てる（実数値の読み上げに創作の余地を作らないため）。
    株価が取得できない銘柄は None（チャートシーン無しで動画は成立する）。

    disc_date を渡すと、その日に対応する位置（discIndex）を返し、動画側が
    「開示はここ」の縦線を引く。チャートの範囲外なら None のままで縦線は出ない。"""
    try:
        from lib.utils import get_prices

        prices = get_prices(code, days=100)
        if prices is None or len(prices) < 20:
            return None
        closes = [round(float(c), 1) for c in prices["Close"].values[-63:]]
        dates = [str(d)[:10] for d in prices.index[-63:]]
    except Exception as e:
        print(f"  ⚠ チャート用株価取得失敗: {e}")
        return None

    disc_index = None
    if disc_date:
        earlier = [i for i, d in enumerate(dates) if d <= disc_date]
        disc_index = earlier[-1] if earlier else None

    latest = closes[-1]
    change_pct = (latest / closes[0] - 1) * 100
    if change_pct >= 0:
        trend = f"3ヶ月でおよそ{abs(change_pct):.0f}パーセントの上昇"
        caption = f"株価は3ヶ月で+{abs(change_pct):.0f}%"
    else:
        trend = f"3ヶ月でおよそ{abs(change_pct):.0f}パーセントの下落"
        caption = f"株価は3ヶ月で−{abs(change_pct):.0f}%"
    narration = (
        f"株価の推移も見てみましょう。直近の終値は{latest:,.0f}円で、{trend}となっています。"
    )
    scene = {
        "kind": "chart",
        "caption": caption,
        "narration": narration,
        "closes": closes,
        "dates": dates,
    }
    if disc_index is not None:
        scene["discIndex"] = disc_index
    return scene


def build(dry_run: bool = False, article_id: str = "") -> "dict | None":
    """動画1本ぶんの props を返す。対象記事が無ければ None。
    article_id指定時は「直近36h×注目枠」の通常選定を使わず、その記事をそのまま動画にする。"""
    if article_id:
        article_id = parse_article_id(article_id)
        article = fetch_article_by_id(article_id)
        if article is None:
            print(f"[build_script] 記事{article_id}が見つかりません")
            return None
    else:
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

    # 前回の保有比率が本文から確定できない回は change シーンを丸ごと落とす。
    # 比較対象の無い1本バーを「保有比率の推移」と称して見せないため。
    if props.get("prevHoldingRatio") is None:
        script["scenes"] = [s for s in script["scenes"] if s["kind"] != "change"]
        print("  ↷ 前回の保有比率が特定できないため change シーンを外します")

    # 株価推移シーンを締めの直前に挿す（「最後に株価」というオーナー指定の位置。
    # ctaの後だと締めの後にまた本編が来て不自然なため）。
    price_scene = build_price_scene(
        str(article.get("stockCode") or ""), props.get("discDate") or ""
    )
    if price_scene is not None:
        script["scenes"].insert(len(script["scenes"]) - 1, price_scene)
    props["scenes"] = script["scenes"]
    props["articleId"] = article["id"]
    props["articleTitle"] = article["title"]
    if dry_run:
        print(json.dumps(props, ensure_ascii=False, indent=2))
    return props


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="", help="props JSONの出力先パス（未指定なら標準出力）")
    p.add_argument("--dry-run", action="store_true", help="生成結果を表示するのみ")
    p.add_argument("--article-id", default="",
                   help="記事ID指定（通常選定を使わずこの記事を動画にする。記事URLを丸ごと渡してもよい）")
    args = p.parse_args()

    props = build(dry_run=args.dry_run, article_id=args.article_id)
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
