"""
web/x_weekly_activists.py

週次「今週のアクティビストの動きまとめ」のX自動投稿（x_post.yml、日曜18:00 JST）。
土曜の急増ランキング（x_weekly_trending.py）と対になる週末2本立ての片方で、
こちらは「誰が・どの銘柄で・何%→何%動いたか」という中身を見せて /activists へ誘導する。

対象は `edinet_filer_classification.category = 'アクティビスト'` に分類済みの提出者
（kujira-watchの/activistsページと同じ母集団）。直近7日(JST)の大量保有・変更報告書から、
提出者×銘柄ごとに「週の開始時点の保有比率 → 週末時点の保有比率」を集計し、
変化幅(pt)の大きい順に買い増し・売却を出す。訂正報告書は実際の持分変動ではないため除外する。

同じ提出者・銘柄で週内に複数回開示があるケース（実データで発生）は、最初の開示の
直前保有割合と最後の開示の保有割合を突き合わせ、週を通した正味の変化として1行にまとめる。
"""
import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.expanduser("~/stock-alert"))

from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/stock-alert/.env"))

from lib import supabase_client as sb  # noqa: E402
from lib.edinet import disclosure_kind_label  # noqa: E402
from web.x_client import SITE_URL, post_tweet, upload_media  # noqa: E402
from web.x_post_format import clean_name, fits, label  # noqa: E402

WINDOW_DAYS = 7

# 載せる (買い件数, 売り件数) の候補。1行あたり40〜50単位使うため、280単位制限では
# 実際に載るのは3件前後になる。買い増しのほうが「アクティビストが仕掛けている」
# ニュース価値が高いため、収まらないときは売りから先に削る。
LADDER = [(2, 2), (2, 1), (1, 1), (1, 0)]

# 1行の見た目を揃える表示上限（Xのカウント単位。全角=2・半角=1）。
STOCK_LABEL_MAX_UNITS = 18
FUND_LABEL_MAX_UNITS = 18

# 保有比率の変化がこれ未満（pt）の開示は「動き」として載せない。
# 変更報告書は1%以上の増減で提出義務が生じるが、共同保有者の入れ替え等で
# ごく小さい変化の開示も混じるため、まとめの見栄えを保つ下限を設ける。
MIN_DELTA_PT = 0.5


def fetch_activist_filers() -> set:
    rows = sb.select(
        "edinet_filer_classification", "category=eq.アクティビスト&select=filer_name"
    )
    return {r["filer_name"] for r in rows if r.get("filer_name")}


def fetch_holdings(range_from: str, range_to: str) -> list:
    q = (
        f"disc_date=gte.{range_from}&disc_date=lte.{range_to}"
        "&select=filer_name,issuer_code,issuer_name,disc_date,submit_date,"
        "holding_ratio,holding_ratio_prior,doc_type_code,doc_description"
    )
    rows = sb.select("edinet_large_holdings", q)
    return [r for r in rows if r.get("issuer_code") and r.get("filer_name") and r.get("disc_date")]


def build_moves(rows: list, activists: set) -> list:
    """提出者×銘柄ごとに週を通した保有比率の変化を集計して返す。
    戻り値の各要素: {filer, stock, code, start, end, delta, is_new}"""
    by_pair: dict = {}
    for row in rows:
        if row["filer_name"] not in activists:
            continue
        if disclosure_kind_label(row.get("doc_description"), row.get("doc_type_code")) == "訂正":
            continue
        if row.get("holding_ratio") is None:
            continue
        by_pair.setdefault((row["filer_name"], row["issuer_code"]), []).append(row)

    moves = []
    for (filer, code), group in by_pair.items():
        # 同日に複数開示があるため submit_date まで見て時系列に並べる
        group.sort(key=lambda r: (r["disc_date"], r.get("submit_date") or ""))
        first, last = group[0], group[-1]
        prior = first.get("holding_ratio_prior")
        first_kind = disclosure_kind_label(first.get("doc_description"), first.get("doc_type_code"))
        if first_kind == "新規" or prior == 0:
            is_new, start = True, 0.0
        elif prior is None:
            # 変更報告書なのに直前保有割合が取れていない＝週初の水準が不明。
            # 「新規」と誤って書かないため、この提出者×銘柄は載せない。
            continue
        else:
            is_new, start = False, float(prior)
        end = float(last["holding_ratio"])
        delta = end - start
        if abs(delta) < MIN_DELTA_PT:
            continue
        moves.append({
            "filer": clean_name(filer) or filer,
            "stock": clean_name(last.get("issuer_name")) or code,
            "code": code,
            "start": start,
            "end": end,
            "delta": delta,
            "is_new": is_new,
        })
    return moves


def _move_line(move: dict) -> str:
    stock = label(move["stock"], STOCK_LABEL_MAX_UNITS)
    fund = label(move["filer"], FUND_LABEL_MAX_UNITS)
    ratio = (
        f"新規{move['end']:.1f}%"
        if move["is_new"]
        else f"{move['start']:.1f}→{move['end']:.1f}%"
    )
    return f"・{stock}({move['code']}) {ratio} {fund}"


def build_activist_text(moves: list) -> "str | None":
    """投稿本文を組み立てる。対象の動きが1件も無い週はNone（投稿しない）。
    全体の件数は末尾のサマリー行で示し、個別の行は変化幅の大きいものだけ載せる。"""
    buys = sorted([m for m in moves if m["delta"] > 0], key=lambda m: -m["delta"])
    sells = sorted([m for m in moves if m["delta"] < 0], key=lambda m: m["delta"])
    if not buys and not sells:
        return None

    url = f"{SITE_URL}/activists?utm_source=x&utm_medium=social&utm_campaign=weekly_activists"
    fund_count = len({m["filer"] for m in moves})

    def render(buy_n: int, sell_n: int) -> str:
        lines = ["🦈 今週のアクティビストの動き"]
        if buys[:buy_n]:
            lines += ["", "📈 買い増し・新規"]
            lines += [_move_line(m) for m in buys[:buy_n]]
        if sells[:sell_n]:
            lines += ["", "📉 売却"]
            lines += [_move_line(m) for m in sells[:sell_n]]
        lines += [
            "",
            f"今週の動き{len(moves)}件・{fund_count}ファンド",
            url,
            "#日本株 #アクティビスト",
        ]
        return "\n".join(lines)

    for buy_n, sell_n in LADDER:
        text = render(buy_n, sell_n)
        if fits(text, url):
            return text
    return text


def build_activist_media(moves: list) -> list:
    """今週の動きの一覧カードを作ってアップロードする（施策4。失敗時は空リスト＝画像なし）。"""
    from web.x_card_image import build_list_card

    ordered = sorted(moves, key=lambda m: -abs(m["delta"]))[:5]
    rows = [
        (f"{m['stock']}（{m['code']}）",
         f"{m['delta']:+.2f}pt", "buy" if m["delta"] > 0 else "sell")
        for m in ordered
    ]
    card = build_list_card("今週のアクティビストの動き",
                           f"{len(moves)}件・{len({m['filer'] for m in moves})}ファンド",
                           rows, "全一覧は kujira-watch.com/activists")
    if not card:
        return []
    alt = "今週のアクティビストによる保有比率の変化。" + "。".join(
        f"{left} {right}" for left, right, _ in rows
    )
    media_id = upload_media(card, alt_text=alt)
    return [media_id] if media_id else []


def run(dry_run: bool = False) -> int:
    # CIランナーはUTCで動くため、集計窓は明示的にJSTの今日を基準にする。
    today = (datetime.now(timezone.utc) + timedelta(hours=9)).date()
    range_from = today - timedelta(days=WINDOW_DAYS - 1)

    activists = fetch_activist_filers()
    if not activists:
        print("[x_weekly_activists] アクティビスト分類の提出者が取得できないため投稿しません")
        return 0

    rows = fetch_holdings(range_from.isoformat(), today.isoformat())
    moves = build_moves(rows, activists)
    text = build_activist_text(moves)
    if text is None:
        print("[x_weekly_activists] 今週はアクティビストの動きが無いため投稿しません")
        return 0

    if dry_run:
        print("[x_weekly_activists] --dry-run のため投稿しません。本文:")
        print(text)
        return 0

    if post_tweet(text, media_ids=build_activist_media(moves), kind="weekly_activists"):
        print("[x_weekly_activists] 🐦 今週のアクティビストの動きを投稿しました")
        return 0
    print("[x_weekly_activists] 投稿に失敗しました")
    return 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="本文の生成・表示のみで投稿しない")
    args = p.parse_args()
    sys.exit(run(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
