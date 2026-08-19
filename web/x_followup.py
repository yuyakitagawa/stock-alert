"""
web/x_followup.py

週次の「答え合わせ」投稿。3ヶ月前に大量保有報告書が出た銘柄が、その後どうなったかを
実績で出す（docs/x_post_improvement_1000.md 施策6）。

事実を流すだけの速報アカウントは信用が積み上がらない。開示後の株価は
`yahoo_price_cache`（記事ページの「開示後の株価推移」と同じデータ）で既に持っているので、
勝った話だけでなく平均・上昇銘柄数・最も下げた銘柄まで含めて出す。

株式分割・併合で終値が不連続な銘柄（キャッシュが調整前の値を持っている場合がある）は
リターンが実態とかけ離れるため、日次で±40%を超える変動がある銘柄は除外する。

実行:
  python3 web/x_followup.py --dry-run   # 本文の生成・表示のみ
  python3 web/x_followup.py             # Xへ投稿
"""
import argparse
import os
import statistics
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.expanduser("~/stock-alert"))

from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/stock-alert/.env"))

from lib import supabase_client as sb  # noqa: E402
from web.x_client import SITE_URL, TAGS, link_in_reply, publish, upload_media  # noqa: E402
from web.x_post_format import clean_name, label  # noqa: E402

# 何日前の開示を振り返るか（63営業日≒3ヶ月。記事ページの+3ヶ月表示と同じ考え方）。
LOOKBACK_DAYS = 91
# 対象日に開示が無い場合に遡って探す日数（土日祝・提出ゼロの日をまたぐため）。
SEARCH_BACK_DAYS = 7
# 新規・変更報告書のみ（訂正報告書は実際の持分変動ではないため除外）
DOC_TYPE_CODES = ("350", "360")
# 開示日から基準終値までに許容する暦日ギャップ（priceReturns.tsのMAX_BASE_GAP_DAYSと同じ）
MAX_BASE_GAP_DAYS = 7
# 株式分割等で終値が不連続になっている銘柄を弾く日次変動のしきい値
SPLIT_GUARD_PCT = 40.0
STOCK_LABEL_MAX_UNITS = 22


def find_target_date(today) -> "tuple[str, list]":
    """3ヶ月前付近で開示があった日を1日選び、(日付, 開示リスト)を返す。無ければ("", [])。"""
    for back in range(SEARCH_BACK_DAYS):
        target = (today - timedelta(days=LOOKBACK_DAYS + back)).isoformat()
        rows = sb.select(
            "edinet_large_holdings",
            f"disc_date=eq.{target}&doc_type_code=in.({','.join(DOC_TYPE_CODES)})"
            "&select=issuer_code,issuer_name,holding_ratio,filer_name",
        )
        rows = [r for r in rows if r.get("issuer_code")]
        if len(rows) >= 5:
            return target, rows
    return "", []


def fetch_returns(codes: list, disc_date: str) -> dict:
    """{code: 騰落率%}。開示日終値→直近終値。基準が取れない・分割等で不連続な銘柄は除く。"""
    out: dict = {}
    for i in range(0, len(codes), 50):
        chunk = codes[i:i + 50]
        rows = sb.select(
            "yahoo_price_cache",
            f"code=in.({','.join(chunk)})&date=gte.{disc_date}"
            "&select=code,date,close&order=code.asc,date.asc",
        )
        series: dict = {}
        for r in rows:
            if r.get("close") is not None:
                series.setdefault(r["code"], []).append((r["date"], float(r["close"])))
        for code, points in series.items():
            if len(points) < 2:
                continue
            base_date, base_close = points[0]
            gap = (datetime.fromisoformat(base_date) - datetime.fromisoformat(disc_date)).days
            if gap > MAX_BASE_GAP_DAYS or base_close <= 0:
                continue
            closes = [c for _, c in points]
            if any(abs(closes[j] / closes[j - 1] - 1) * 100 >= SPLIT_GUARD_PCT
                   for j in range(1, len(closes)) if closes[j - 1] > 0):
                continue  # 分割・併合で不連続な銘柄
            out[code] = (closes[-1] / base_close - 1) * 100
    return out


def build_followup_text(disc_date: str, stats: dict, link_url: str = "") -> "str | None":
    """答え合わせ本文。勝った銘柄だけでなく平均・上昇銘柄数・最下位まで出す。"""
    if not stats or stats["n"] < 5:
        return None
    month_day = f"{int(disc_date[5:7])}/{int(disc_date[8:10])}"
    best, worst = stats["best"], stats["worst"]
    lines = [
        f"🐋 答え合わせ｜{month_day}に大量保有報告書が出た{stats['n']}銘柄のその後",
        f"開示日の終値→直近: 平均 {stats['mean']:+.1f}%・中央値 {stats['median']:+.1f}%",
        f"上がったのは {stats['n_up']}/{stats['n']}銘柄",
        "",
        f"📈 最大 {label(best['name'], STOCK_LABEL_MAX_UNITS)}({best['code']}) {best['ret']:+.1f}%",
        f"📉 最小 {label(worst['name'], STOCK_LABEL_MAX_UNITS)}({worst['code']}) {worst['ret']:+.1f}%",
    ]
    if link_url:
        lines += ["", link_url]
    lines.append(TAGS)
    return "\n".join(lines)


def build_stats(holdings: list, returns: dict) -> dict:
    names = {}
    for h in holdings:
        names.setdefault(h["issuer_code"], clean_name(h.get("issuer_name")) or h["issuer_code"])
    items = [{"code": c, "name": names.get(c, c), "ret": r} for c, r in returns.items()]
    if not items:
        return {}
    items.sort(key=lambda x: x["ret"], reverse=True)
    values = [i["ret"] for i in items]
    return {
        "n": len(items),
        "n_up": sum(1 for v in values if v > 0),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "best": items[0],
        "worst": items[-1],
        "items": items,
    }


def build_media(disc_date: str, stats: dict) -> list:
    from web.x_card_image import build_list_card

    month_day = f"{int(disc_date[5:7])}/{int(disc_date[8:10])}"
    rows = [(f"{i['name']}（{i['code']}）", f"{i['ret']:+.1f}%", "buy" if i["ret"] > 0 else "sell")
            for i in (stats["items"][:2] + stats["items"][-2:])]
    card = build_list_card(
        f"答え合わせ｜{month_day}の大量保有報告書 {stats['n']}銘柄",
        f"開示日終値→直近: 平均 {stats['mean']:+.1f}%・上昇 {stats['n_up']}/{stats['n']}銘柄",
        rows, f"開示の一覧は kujira-watch.com/date/{disc_date}",
    )
    if not card:
        return []
    alt = (f"{month_day}に大量保有報告書が出た{stats['n']}銘柄の開示日終値から直近終値までの騰落率。"
           f"平均{stats['mean']:+.1f}%、上昇は{stats['n_up']}銘柄。")
    media_id = upload_media(card, alt_text=alt)
    return [media_id] if media_id else []


def run(dry_run: bool = False) -> int:
    today = (datetime.now(timezone.utc) + timedelta(hours=9)).date()
    disc_date, holdings = find_target_date(today)
    if not disc_date:
        print("[x_followup] 3ヶ月前付近に対象の開示が見つからないため投稿しません")
        return 0

    codes = sorted({h["issuer_code"] for h in holdings})
    stats = build_stats(holdings, fetch_returns(codes, disc_date))
    url = f"{SITE_URL}/date/{disc_date}?utm_source=x&utm_medium=social&utm_campaign=followup"
    text = build_followup_text(disc_date, stats, link_url="" if link_in_reply() else url)
    if text is None:
        print(f"[x_followup] {disc_date}は株価が取れた銘柄が少ないため投稿しません")
        return 0

    if dry_run:
        print("[x_followup] --dry-run のため投稿しません。本文:")
        print(text)
        return 0

    if publish(text, link_line=f"↓ この日の開示一覧\n{url}" if link_in_reply() else "",
               media_ids=build_media(disc_date, stats), kind="followup"):
        print(f"[x_followup] 🐦 {disc_date}の答え合わせを投稿しました")
        return 0
    print("[x_followup] 投稿に失敗しました")
    return 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="本文の生成・表示のみで投稿しない")
    args = p.parse_args()
    sys.exit(run(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
