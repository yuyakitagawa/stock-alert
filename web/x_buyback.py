"""
web/x_buyback.py

平日夕方「本日の自社株買い決定」のX自動投稿（x_post.yml、平日19:00 JST）。

TDnetの自社株買い「決定」開示（取得枠の新設・ToSTNeT-3買付）のうち当日分を、上限金額の大きい順に
並べて出す。対象は tdnet_buybacks（lib/buyback.py が原文PDFから抽出した取得枠）で、月次の
取得状況報告（進捗）は載せない。自社株買いは15:30〜17:00の引け後に集中して開示されるため、
16:00 JSTの daily_alert.yml だけでは当日分を取りこぼす。このスクリプト自身が投稿前に
TDnet取得（fetch）と取得枠抽出（enrich）を回してから当日分を集計する。

ext_tdnet_disclosures.disclosed_at はTDnetの公表時刻（JST壁時計）をそのまま入れているため、
「当日」はJSTの日付文字列で絞る。
"""
import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.expanduser("~/stock-alert"))

from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/stock-alert/.env"))

from lib import supabase_client as sb  # noqa: E402
from web.x_client import PROFILE_CTA, post_tweet, upload_media  # noqa: E402
from web.x_post_format import POST_MAX_WEIGHTED, clean_name, label, weighted_len  # noqa: E402

# これ未満の取得枠は載せない（ToSTNeT-3の役員持株会向け数千万円の買付等が混じるため）。
MIN_AMOUNT_YEN = 100_000_000

# 本文に載せる件数の候補。1行40〜50単位なので280単位では3〜4件が上限。
LADDER = [5, 4, 3, 2, 1]
STOCK_LABEL_MAX_UNITS = 18


def refresh_today(days: int = 1) -> None:
    """投稿前に当日のTDnet開示を取り込み、決定開示の取得枠を抽出する。失敗しても投稿は続ける
    （16:00便で取り込んだ分だけで出す）。"""
    try:
        from lib.tdnet import scan_disclosures

        scan_disclosures(days_back=days, only_categorized=True)
    except Exception as e:
        print(f"[x_buyback] TDnet取得に失敗（既存データで続行）: {e}")
    try:
        from lib.buyback import enrich

        enrich(days=days + 1)
    except Exception as e:
        print(f"[x_buyback] 取得枠抽出に失敗（既存データで続行）: {e}")


def fetch_decisions(date_str: str) -> list:
    """date_str（JST、YYYY-MM-DD）に開示された決定で上限金額が取れているもの。"""
    rows = sb.select(
        "tdnet_buybacks",
        f"disclosed_at=gte.{date_str}T00:00:00&disclosed_at=lt.{date_str}T23:59:59"
        "&extract_ok=is.true&select=code,disclosed_at,title,max_amount_yen,ratio_pct,method,will_cancel"
        "&order=max_amount_yen.desc.nullslast",
    )
    rows = [r for r in rows if (r.get("max_amount_yen") or 0) >= MIN_AMOUNT_YEN]
    if not rows:
        return []
    codes = ",".join(sorted({r["code"] for r in rows}))
    names = {
        n["code"]: n["name"]
        for n in sb.select("jpx_stock_list", f"code=in.({codes})&select=code,name")
        if n.get("name")
    }
    out = []
    for r in rows:
        out.append({
            "code": r["code"],
            "stock": clean_name(names.get(r["code"], "")) or r["code"],
            "amount_yen": int(r["max_amount_yen"]),
            "ratio": float(r["ratio_pct"]) if r.get("ratio_pct") is not None else None,
            "method": r.get("method") or "",
            "will_cancel": bool(r.get("will_cancel")),
        })
    return out


def format_oku(yen: int) -> str:
    oku = yen / 1e8
    if oku >= 100:
        return f"{oku:,.0f}億円"
    if oku >= 10:
        return f"{oku:.0f}億円"
    return f"{oku:.1f}億円"


def _line(d: dict) -> str:
    stock = label(d["stock"], STOCK_LABEL_MAX_UNITS)
    ratio = f" 発行済{d['ratio']:.1f}%" if d["ratio"] is not None else ""
    cancel = " 消却" if d["will_cancel"] else ""
    return f"・{stock}({d['code']}) 上限{format_oku(d['amount_yen'])}{ratio}{cancel}"


def build_buyback_text(decisions: list, date_str: str) -> "str | None":
    """投稿本文。対象が無い日はNone（投稿しない）。"""
    if not decisions:
        return None
    total = sum(d["amount_yen"] for d in decisions)
    md = f"{int(date_str[5:7])}/{int(date_str[8:10])}"

    def render(n: int) -> str:
        lines = [f"🏦 本日の自社株買い決定（{md}）"]
        lines += [_line(d) for d in decisions[:n]]
        if len(decisions) > n:
            lines.append(f"ほか{len(decisions) - n}件")
        # 末尾は最小限にして銘柄行に単位を回す（1行40〜46単位。解説文を1行足すと載る銘柄が1件減る）。
        # URLは入れない（リンク入り投稿は$0.20課金）。
        lines += [
            "",
            f"計{len(decisions)}件・上限合計{format_oku(total)}",
            PROFILE_CTA,
            "#日本株 #自社株買い",
        ]
        return "\n".join(lines)

    for n in LADDER:
        text = render(n)
        if weighted_len(text) <= POST_MAX_WEIGHTED:
            return text
    return text


def build_buyback_media(decisions: list, date_str: str) -> list:
    """上限金額順の一覧カード（失敗時は空リスト＝画像なし）。"""
    from web.x_card_image import build_list_card

    rows = [
        (f"{d['stock']}（{d['code']}）",
         f"上限{format_oku(d['amount_yen'])}" + (f" / {d['ratio']:.1f}%" if d["ratio"] is not None else ""),
         "buy")
        for d in decisions[:6]
    ]
    total = sum(d["amount_yen"] for d in decisions)
    card = build_list_card(f"本日の自社株買い決定 {date_str[5:].replace('-', '/')}",
                           f"{len(decisions)}件・上限合計{format_oku(total)}",
                           rows, "取得枠の履歴は kujira-watch.com/stocks/[コード]")
    if not card:
        return []
    alt = "本日決定された自社株買いの取得枠（上限金額順）。" + "。".join(f"{l} {r}" for l, r, _ in rows)
    media_id = upload_media(card, alt_text=alt)
    return [media_id] if media_id else []


def run(dry_run: bool = False, date_str: "str | None" = None, no_refresh: bool = False) -> int:
    today = date_str or (datetime.now(timezone.utc) + timedelta(hours=9)).date().isoformat()
    if not no_refresh and not dry_run:
        refresh_today()

    decisions = fetch_decisions(today)
    text = build_buyback_text(decisions, today)
    if text is None:
        print(f"[x_buyback] {today} は上限{MIN_AMOUNT_YEN/1e8:.0f}億円以上の自社株買い決定が無いため投稿しません")
        return 0

    if dry_run:
        print("[x_buyback] --dry-run のため投稿しません。本文:")
        print(text)
        return 0

    if post_tweet(text, media_ids=build_buyback_media(decisions, today), kind="buyback_daily"):
        print(f"[x_buyback] 🐦 本日の自社株買い決定（{len(decisions)}件）を投稿しました")
        return 0
    print("[x_buyback] 投稿に失敗しました")
    return 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="本文の生成・表示のみで投稿しない")
    p.add_argument("--date", help="集計日（JST、YYYY-MM-DD。既定は今日）")
    p.add_argument("--no-refresh", action="store_true", help="投稿前のTDnet取得・抽出を省く")
    args = p.parse_args()
    sys.exit(run(dry_run=args.dry_run, date_str=args.date, no_refresh=args.no_refresh))


if __name__ == "__main__":
    main()
