"""
web/x_weekly_trending.py

週次「大口投資家の取引急増ランキング」のX自動投稿（x_post.yml、土曜18:00 JST）。
平日の記事投稿・日次サマリーが無い週末のタイムラインを埋め、/trending への導線を作る。
日曜18:00の「今週のアクティビストの動き」（x_weekly_activists.py）と対になる週末2本立て。

集計ロジックは kujira-watch の /trending ページ（src/lib/trendingStats.ts）のPython移植で、
直近期間とその前の同じ長さの期間の大量保有・変更報告書の開示件数を Supabase
`edinet_large_holdings` から数え、開示が増えた（delta>0）銘柄・投資家を推定売買金額の
大きい順に出す。比較窓は /trending と同じ7日（前週比）。
EDINET開示そのものに取引金額は無いため、金額はマテリアライズドビュー
`edinet_holding_amounts`（保有比率の変化幅×発行済株式数×開示日終値の概算）から doc_id で
引く。訂正報告書など金額を推定できない開示は0億円扱いで、件数には入るが金額には入らない。

Xの文字数は全角2単位換算の280単位制限があるため、本文を組んだあと
`weighted_len()` で実測し、超過する場合は投資家→銘柄の順で末尾の行を落とす。
"""
import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.expanduser("~/stock-alert"))

from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/stock-alert/.env"))

from lib import supabase_client as sb  # noqa: E402
from web.x_client import PROFILE_CTA, TAGS, post_tweet, upload_media  # noqa: E402
from web.x_post_format import POST_MAX_WEIGHTED, clean_name, label, weighted_len  # noqa: E402

# 比較窓。「直近7日 vs その前7日」の前週比。週次投稿で30日窓を使うと隣り合う日曜の投稿で
# データが23日分重複してほぼ同じランキングが並んでしまうため7日にしていたもので、
# 2026-08-27に/trendingページ側も7日窓へ揃えたため現在は両者同じ窓。
WINDOW_DAYS = 7

# 投稿に載せる件数。文字数制限が厳しいため銘柄3・投資家2に絞る（全量は/trendingで見せる）。
ISSUER_LIMIT = 3
FILER_LIMIT = 2

# ラベルの表示上限（Xのカウント単位。全角=2・半角=1）。EDINETの正式名称は長い
# （「〇〇ホールディングス株式会社」等）ため、行が2行に折り返す前に切り詰める。
# 銘柄は末尾に証券コード（+12単位）が付き、投資家は半角の海外ファンド名が多いため
# 投資家側を長めに取る。2026-08-27に右側の数値を「+N件」（約5単位）から
# 「推定1,274億円」（約13単位）へ変えたぶん、行の総単位数が変わらないよう上限を下げた。
LABEL_MAX_UNITS = 22
FILER_LABEL_MAX_UNITS = 42


def fetch_holdings(range_from: str, range_to: str) -> list:
    """期間内の開示行を返す。issuer_code / disc_date / filer_name が欠けた行は
    集計に使えないため除外する（kujira-watch側 getHoldingsInRange() と同じ規律）。"""
    q = (
        f"disc_date=gte.{range_from}&disc_date=lte.{range_to}"
        "&select=doc_id,issuer_code,issuer_name,disc_date,filer_name"
    )
    rows = sb.select("edinet_large_holdings", q)
    return [r for r in rows if r.get("issuer_code") and r.get("disc_date") and r.get("filer_name")]


def fetch_amounts(range_from: str, range_to: str) -> dict:
    """期間内の開示1件ごとの推定売買金額（億円）を doc_id 引きで返す。
    kujira-watch側 getHoldingAmountsInRange() と同じ集計元。"""
    q = (
        f"disc_date=gte.{range_from}&disc_date=lte.{range_to}"
        "&select=doc_id,deal_amount_oku"
    )
    rows = sb.select("edinet_holding_amounts", q)
    return {r["doc_id"]: float(r["deal_amount_oku"]) for r in rows if r.get("deal_amount_oku") is not None}


def amount_label(oku: float) -> str:
    """金額の表示。1兆円（1万億円）以上は兆円へ繰り上げる（kujira-watch側 formatAmountParts と同じ規律）。"""
    if oku >= 10000:
        return f"推定{oku / 10000:,.1f}兆円"
    return f"推定{oku:,.0f}億円"


def build_trending(rows: list, current_from: str, key_of, label_of, limit: int,
                   amounts: "dict | None" = None) -> list:
    """trendingStats.ts の buildTrendingIssuers()＋selectDirection() と同一ロジック。
    delta（直近WINDOW_DAYS日の件数 - その前の同じ日数の件数）が正のものを、
    直近WINDOW_DAYS日の推定売買金額の降順 → delta降順 → 件数降順で返す。"""
    amounts = amounts or {}
    entries: dict = {}
    for row in rows:
        key = key_of(row)
        if not key:
            continue
        entry = entries.setdefault(
            key, {"label": label_of(row), "count": 0, "prev_count": 0, "amount": 0.0}
        )
        if row["disc_date"] >= current_from:
            entry["count"] += 1
            entry["amount"] += amounts.get(row.get("doc_id"), 0.0)
        else:
            entry["prev_count"] += 1

    result = [
        {
            "key": key,
            "label": e["label"],
            "count": e["count"],
            "prev_count": e["prev_count"],
            "delta": e["count"] - e["prev_count"],
            # 億円未満は表示しないので丸めて浮動小数の誤差を持ち回さない（TS側 toCounts と同じ）。
            "amount": round(e["amount"]),
        }
        for key, e in entries.items()
    ]
    result = [e for e in result if e["delta"] > 0]
    result.sort(key=lambda e: (-e["amount"], -e["delta"], -e["count"]))
    return result[:limit]


def entry_metric(entry: dict) -> str:
    """ランキング行の右側に出す数値。並べ替えの軸である推定売買金額を出し、
    金額を推定できない開示（訂正報告書・株価や発行済株式数が取れない銘柄）だけ
    増加件数にフォールバックする（「推定0億円」は誤解を招くため出さない）。"""
    amount = entry.get("amount") or 0
    return amount_label(amount) if amount > 0 else f"+{entry['delta']}件"


def build_weekly_trending_text(issuers: list, filers: list) -> "str | None":
    """投稿本文を組み立てる。急増銘柄が1件も無い週はNone（投稿しない）。
    280単位制限に収まらない場合は投資家→銘柄の順で末尾の行から落とす。"""
    if not issuers:
        return None

    def render(issuer_n: int, filer_n: int) -> str:
        lines = ["🐋 大口投資家の取引急増ランキング（前週比）", "", "📈 銘柄"]
        for i, e in enumerate(issuers[:issuer_n], 1):
            lines.append(f"{i}. {label(e['label'], LABEL_MAX_UNITS)} {entry_metric(e)}")
        if filers[:filer_n]:
            lines += ["", "👤 投資家"]
            for i, e in enumerate(filers[:filer_n], 1):
                lines.append(f"{i}. {label(e['label'], FILER_LABEL_MAX_UNITS)} {entry_metric(e)}")
        # ハッシュタグは母数のある2つだけ（`#社名` は実際には検索されないため付けない。
        # 銘柄は本文のランキング行に `社名（コード）` として素で載っている）
        # URLは入れない（リンク入り投稿は$0.20課金）。全ランキングへはプロフィール経由で誘導する
        lines += ["", f"全ランキングは{PROFILE_CTA[3:]}", TAGS]
        return "\n".join(lines)

    # 収まるまで行を減らす（最低でも銘柄1件は残す）。
    for issuer_n, filer_n in [
        (ISSUER_LIMIT, FILER_LIMIT),
        (ISSUER_LIMIT, 1),
        (ISSUER_LIMIT, 0),
        (2, 0),
        (1, 0),
    ]:
        text = render(issuer_n, filer_n)
        if weighted_len(text) <= POST_MAX_WEIGHTED:
            return text
    return text  # 銘柄1件でも超える場合はそのまま出す（ラベル切り詰め済みで実際には起きない想定）


def build_trending_media(issuers: list, filers: list) -> list:
    """ランキングの一覧カードを作ってアップロードする（施策4。失敗時は空リスト＝画像なし）。"""
    from web.x_card_image import build_list_card

    rows = [(e["label"], entry_metric(e), "buy") for e in issuers[:3]]
    rows += [(e["label"], entry_metric(e), "none") for e in filers[:2]]
    card = build_list_card("大口投資家の取引急増ランキング", "前週比（直近7日 vs その前7日）・推定売買金額順",
                           rows, "全ランキングは kujira-watch.com/trending")
    if not card:
        return []
    alt = "大量保有報告書の提出が前週比で増えた銘柄と投資家を推定売買金額の大きい順に並べたランキング。" + "。".join(
        f"{left} {right}" for left, right, _ in rows
    )
    media_id = upload_media(card, alt_text=alt)
    return [media_id] if media_id else []


def run(dry_run: bool = False) -> int:
    # CIランナーはUTCで動くため、date.today()だと集計窓がJST基準から1日ずれる。
    # サイトの/trendingや投稿文言と揃えるためJSTの今日を使う。
    today = (datetime.now(timezone.utc) + timedelta(hours=9)).date()
    current_from = (today - timedelta(days=WINDOW_DAYS - 1)).isoformat()
    range_from = (today - timedelta(days=WINDOW_DAYS * 2 - 1)).isoformat()
    range_to = today.isoformat()

    rows = fetch_holdings(range_from, range_to)
    if not rows:
        print("[x_weekly_trending] 開示データが取得できないため投稿しません")
        return 0

    # 金額が取れなくても投稿自体は成立させる（全件0億円＝増加件数順にフォールバックする）。
    amounts = fetch_amounts(current_from, range_to)

    issuers = build_trending(
        rows, current_from,
        key_of=lambda r: r["issuer_code"],
        label_of=lambda r: f"{clean_name(r.get('issuer_name')) or r['issuer_code']}（{r['issuer_code']}）",
        limit=ISSUER_LIMIT,
        amounts=amounts,
    )
    filers = build_trending(
        rows, current_from,
        key_of=lambda r: r["filer_name"],
        label_of=lambda r: clean_name(r["filer_name"]) or r["filer_name"],
        limit=FILER_LIMIT,
        amounts=amounts,
    )

    text = build_weekly_trending_text(issuers, filers)
    if text is None:
        print("[x_weekly_trending] 急増銘柄が無い週のため投稿しません")
        return 0

    if dry_run:
        print("[x_weekly_trending] --dry-run のため投稿しません。本文:")
        print(text)
        return 0

    if post_tweet(text, media_ids=build_trending_media(issuers, filers), kind="weekly_trending"):
        print("[x_weekly_trending] 🐦 週次急増ランキングを投稿しました")
        return 0
    print("[x_weekly_trending] 投稿に失敗しました")
    return 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="本文の生成・表示のみで投稿しない")
    args = p.parse_args()
    sys.exit(run(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
