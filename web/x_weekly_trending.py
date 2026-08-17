"""
web/x_weekly_trending.py

週次「クジラ急増ランキング」のX自動投稿（x_weekly_post.yml、日曜19:00 JST）。
平日の記事投稿・日次サマリーが無い週末のタイムラインを埋め、/trending への導線を作る。

集計は kujira-watch の /trending ページ（src/lib/trendingStats.ts）のPython移植:
直近30日と、その前30日の大量保有・変更報告書の開示件数を Supabase
`edinet_large_holdings` から数え、増加件数（delta）の多い順に銘柄・投資家を出す。
開示データには推定取引金額が無いため、比較の軸は金額ではなく開示件数（/trendingと同じ理由）。

Xの文字数は全角2単位換算の280単位制限があるため、本文を組んだあと
`weighted_len()` で実測し、超過する場合は投資家→銘柄の順で末尾の行を落とす。
"""
import argparse
import os
import re
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.expanduser("~/stock-alert"))

from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/stock-alert/.env"))

from lib import supabase_client as sb  # noqa: E402
from web.x_client import SITE_URL, _stock_hashtag, post_tweet  # noqa: E402

# /trending（kujira-watch/src/app/(ja)/trending/page.tsx）と同じ30日固定窓。
WINDOW_DAYS = 30

# 投稿に載せる件数。文字数制限が厳しいため銘柄3・投資家2に絞る（全量は/trendingで見せる）。
ISSUER_LIMIT = 3
FILER_LIMIT = 2

# ラベルの表示上限。EDINETの正式名称は長い（「〇〇ホールディングス株式会社」等）ため
# 行が2行に折り返して読みにくくなる前に切り詰める。
LABEL_MAX_CHARS = 14

# Xの投稿上限は280単位（全角=2単位、URLは一律23単位）。URL・改行・絵文字の
# 揺らぎを考慮した安全マージンとして、この単位数以下に収める。
POST_MAX_WEIGHTED = 270
URL_WEIGHTED_UNITS = 23


def fetch_holdings(range_from: str, range_to: str) -> list:
    """期間内の開示行を返す。issuer_code / disc_date / filer_name が欠けた行は
    集計に使えないため除外する（kujira-watch側 getHoldingsInRange() と同じ規律）。"""
    q = (
        f"disc_date=gte.{range_from}&disc_date=lte.{range_to}"
        "&select=issuer_code,issuer_name,disc_date,filer_name"
    )
    rows = sb.select("edinet_large_holdings", q)
    return [r for r in rows if r.get("issuer_code") and r.get("disc_date") and r.get("filer_name")]


def build_trending(rows: list, current_from: str, key_of, label_of, limit: int) -> list:
    """trendingStats.ts の buildTrending() と同一ロジック。
    delta（直近30日の件数 - その前30日の件数）が正のものを delta降順 → 件数降順で返す。"""
    entries: dict = {}
    for row in rows:
        key = key_of(row)
        if not key:
            continue
        entry = entries.setdefault(key, {"label": label_of(row), "count": 0, "prev_count": 0})
        if row["disc_date"] >= current_from:
            entry["count"] += 1
        else:
            entry["prev_count"] += 1

    result = [
        {
            "key": key,
            "label": e["label"],
            "count": e["count"],
            "prev_count": e["prev_count"],
            "delta": e["count"] - e["prev_count"],
        }
        for key, e in entries.items()
    ]
    result = [e for e in result if e["delta"] > 0]
    result.sort(key=lambda e: (-e["delta"], -e["count"]))
    return result[:limit]


def _clean_name(name: str) -> str:
    """EDINET正式名称から「株式会社」等の法人格・空白を落として表示用に短くする。
    280単位制限下では法人格の4〜5文字が投資家セクション1行ぶんに相当するため。"""
    return re.sub(r"株式会社|（株）|\(株\)|\s+|　+", "", name or "")


def _label(text: str) -> str:
    return text if len(text) <= LABEL_MAX_CHARS else text[: LABEL_MAX_CHARS - 1] + "…"


def weighted_len(text: str) -> int:
    """Xの文字数カウントの近似。ASCII=1単位・それ以外（全角・絵文字）=2単位。
    URLは呼び出し側で23単位として別途加算する。"""
    return sum(1 if ord(c) < 128 else 2 for c in text)


def build_weekly_trending_text(issuers: list, filers: list) -> "str | None":
    """投稿本文を組み立てる。急増銘柄が1件も無い週はNone（投稿しない）。
    280単位制限に収まらない場合は投資家→銘柄の順で末尾の行から落とす。"""
    if not issuers:
        return None

    url = f"{SITE_URL}/trending?utm_source=x&utm_medium=social&utm_campaign=weekly_trending"

    def render(issuer_n: int, filer_n: int) -> str:
        lines = [f"🐋 クジラ出没が急増中（直近{WINDOW_DAYS}日）", "", "📈 銘柄"]
        for i, e in enumerate(issuers[:issuer_n], 1):
            lines.append(f"{i}. {_label(e['label'])} +{e['delta']}件")
        if filers[:filer_n]:
            lines += ["", "👤 投資家"]
            for i, e in enumerate(filers[:filer_n], 1):
                lines.append(f"{i}. {_label(e['label'])} +{e['delta']}件")
        # ハッシュタグは銘柄名のみ（証券コード付きの「#〇〇1234」は検索で使われないため外す）
        top_tag = _stock_hashtag(re.sub(r"（[0-9A-Z]+）$", "", issuers[0]["label"]))
        tag_line = "#大量保有報告書" + (f" {top_tag}" if top_tag else "")
        lines += ["", "↓ 全ランキング", url, tag_line]
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
        if weighted_len(text.replace(url, "")) + URL_WEIGHTED_UNITS <= POST_MAX_WEIGHTED:
            return text
    return text  # 銘柄1件でも超える場合はそのまま出す（ラベル切り詰め済みで実際には起きない想定）


def run(dry_run: bool = False) -> int:
    today = date.today()
    current_from = (today - timedelta(days=WINDOW_DAYS - 1)).isoformat()
    range_from = (today - timedelta(days=WINDOW_DAYS * 2 - 1)).isoformat()
    range_to = today.isoformat()

    rows = fetch_holdings(range_from, range_to)
    if not rows:
        print("[x_weekly_trending] 開示データが取得できないため投稿しません")
        return 0

    issuers = build_trending(
        rows, current_from,
        key_of=lambda r: r["issuer_code"],
        label_of=lambda r: f"{_clean_name(r.get('issuer_name')) or r['issuer_code']}（{r['issuer_code']}）",
        limit=ISSUER_LIMIT,
    )
    filers = build_trending(
        rows, current_from,
        key_of=lambda r: r["filer_name"],
        label_of=lambda r: _clean_name(r["filer_name"]) or r["filer_name"],
        limit=FILER_LIMIT,
    )

    text = build_weekly_trending_text(issuers, filers)
    if text is None:
        print("[x_weekly_trending] 急増銘柄が無い週のため投稿しません")
        return 0

    if dry_run:
        print("[x_weekly_trending] --dry-run のため投稿しません。本文:")
        print(text)
        return 0

    if post_tweet(text):
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
