"""
web/x_filer_record.py

「この投資家が入った銘柄はその後どうなるか」を実績で出す投稿（週次）。

既存の `web/x_followup.py` は**開示日ごと**の答え合わせ（8/13に出た12銘柄のその後）で、
これはEDINETを見ている人なら誰でも同じ集計ができる。同ジャンル9アカウント879投稿の実測
（2026-08-30、tools/x_benchmark.py）では、開示を流すだけの5アカウントが全て中央値
エンゲージメント0〜2に張り付いていた。同じ材料で戦う限りこの天井は超えない。

こちらだけが持っているのは**提出者ごとの累積実績**。同じ投資家の過去の新規報告書を全部
集めて「その後3ヶ月でどうなったか」を出せるのは、開示を1年半ぶん貯めているからで、
速報botには書けない。実測(2024-12〜)では例えば:
  Evo Fund        n=39  平均-19.9%  勝率20.5%
  Oasis           n=21  平均+18.1%  勝率85.7%
  全開示の平均     n=2657 平均 +5.5%  勝率54.6%

**必ず全開示平均と並べて出す**こと。地合いの良い期間なら全体が上がるので、単体の
「+18%」は投資家の実力を意味しない。差分で語る。

除外する提出者: 証券会社・信託銀行の自己名義。これらの大量保有報告書は自己売買や
担保・貸株の建玉であって「その投資家がその銘柄を選んだ」という意味を持たないため、
実績として並べると読者を誤らせる。

コスト: Anthropic APIは使わない（Supabaseの既存テーブルを読むだけ）。

実行:
  python3 web/x_filer_record.py --dry-run           # 本文の生成・表示のみ
  python3 web/x_filer_record.py --filer "Ｅｖｏ　Ｆｕｎｄ" --dry-run
  python3 web/x_filer_record.py                     # Xへ投稿
"""
import argparse
import os
import statistics
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.expanduser("~/stock-alert"))

from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/stock-alert/.env"))

from lib import supabase_client as sb  # noqa: E402
from web.x_client import PROFILE_CTA, TAGS, post_tweet  # noqa: E402
from web.x_post_format import clean_name, label  # noqa: E402

# 開示から何日後の株価と比べるか（x_followup.py と同じ3ヶ月）
HORIZON_DAYS = 91
# 集計対象の期間。HORIZON_DAYS ぶんの先の株価が要るので、直近はまだ答えが出ていない。
LOOKBACK_DAYS = 700
# この件数に満たない提出者は投稿しない。少数の当たり外れで平均が動くため。
MIN_EVENTS = 10
# 開示日・判定日それぞれで、終値を探しにいく許容日数（休場をまたぐため）
MAX_PRICE_GAP_DAYS = 7
# 株式分割等で終値が不連続な銘柄を弾く（x_followup.py の SPLIT_GUARD_PCT と同じ考え方）
SPLIT_GUARD_PCT = 40.0
STOCK_LABEL_MAX_UNITS = 20
FILER_LABEL_MAX_UNITS = 24

# 自己名義の建玉であって投資判断ではない提出者。名前にこれを含むものを実績集計から外す。
# （野村證券の267件のような自己売買が混じると「投資家の成績」ではなくなる）
# 「證券」は野村證券のように旧字体で登記されている社名があり、「証券」だけでは漏れる
# （実際に野村證券の255件が集計に混じっていた）。
EXCLUDED_KEYWORDS = ("証券", "證券", "信託銀行", "銀行", "フィナンシャル・グループ",
                     "フィナンシャルグループ", "ホールディングス")


def _has_excluded_keyword(filer_name: str) -> bool:
    return any(k in (filer_name or "") for k in EXCLUDED_KEYWORDS)


def fetch_events(today: date = None) -> list:
    """新規の大量保有報告書（変更・訂正を除く）を、集計期間ぶん取る。"""
    today = today or date.today()
    since = (today - timedelta(days=LOOKBACK_DAYS)).isoformat()
    until = (today - timedelta(days=HORIZON_DAYS)).isoformat()
    rows = sb.select(
        "edinet_large_holdings",
        # limit はここで指定しない。supabase_client.select() が limit/offset を付けて
        # ページングするため、クエリ側にも limit を書くと1つのURLに limit が2つ並び、
        # 1ページ目が1000行を超えて終了条件に当たらず、同じ行を何度も取り込んでしまう
        # （実測: 同じ日に3回走らせて全開示平均が +4.9% / +6.3% / +6.7% とブレていた）。
        "select=filer_name,issuer_code,issuer_name,disc_date,doc_description"
        f"&disc_date=gte.{since}&disc_date=lte.{until}"
        "&issuer_code=not.is.null",
    )
    return [r for r in rows or [] if "変更報告書" not in (r.get("doc_description") or "")
            and "訂正" not in (r.get("doc_description") or "")]


def fetch_prices(codes: list) -> dict:
    """銘柄ごとの (日付, 終値) の並びを返す。"""
    prices = {}
    for i in range(0, len(codes), 100):
        chunk = codes[i:i + 100]
        rows = sb.select(
            "yahoo_price_cache",
            "select=code,date,close&code=in.(" + ",".join(chunk) + ")"
            "&order=date.asc",
        )
        for r in rows or []:
            if r.get("close"):
                prices.setdefault(r["code"], []).append((r["date"], float(r["close"])))
    return prices


def _close_on_or_after(series: list, target: str) -> "float | None":
    """target以降で最初の終値。休場を挟んでも MAX_PRICE_GAP_DAYS までは許す。"""
    limit = (date.fromisoformat(target) + timedelta(days=MAX_PRICE_GAP_DAYS)).isoformat()
    for d, close in series:
        if d >= target:
            return close if d <= limit else None
    return None


def _is_discontinuous(series: list) -> bool:
    """株式分割・併合で終値が飛んでいる銘柄か。リターンが実態とかけ離れるので使わない。"""
    for (d1, c1), (d2, c2) in zip(series, series[1:]):
        if c1 > 0 and abs(100.0 * (c2 - c1) / c1) > SPLIT_GUARD_PCT:
            return True
    return False


def compute_returns(events: list, prices: dict) -> list:
    """各開示に「開示日の終値→3ヶ月後の終値」のリターンを付ける。取れないものは落とす。"""
    out = []
    for e in events:
        series = prices.get(e["issuer_code"])
        if not series or _is_discontinuous(series):
            continue
        d0 = e["disc_date"][:10]
        d1 = (date.fromisoformat(d0) + timedelta(days=HORIZON_DAYS)).isoformat()
        p0 = _close_on_or_after(series, d0)
        p1 = _close_on_or_after(series, d1)
        if not p0 or not p1 or p0 <= 0:
            continue
        out.append({**e, "ret": 100.0 * (p1 - p0) / p0})
    return out


def summarize(rows: list) -> dict:
    values = [r["ret"] for r in rows]
    if not values:
        return {}
    ranked = sorted(rows, key=lambda r: -r["ret"])
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "win_rate": 100.0 * sum(1 for v in values if v > 0) / len(values),
        "best": ranked[0],
        "worst": ranked[-1],
    }


def rank_filers(rows: list) -> list:
    """提出者ごとの実績。件数が足りないものと自己名義の提出者は落とす。"""
    by_filer = {}
    for r in rows:
        if _has_excluded_keyword(r.get("filer_name")):
            continue
        by_filer.setdefault(r["filer_name"], []).append(r)
    out = []
    for name, items in by_filer.items():
        if len(items) < MIN_EVENTS:
            continue
        out.append({"filer_name": name, **summarize(items)})
    return sorted(out, key=lambda f: -abs(f["mean"]))


def pick_weekly(ranked: list, today: date) -> "dict | None":
    """週ごとに別の提出者を選ぶ。毎週1位を出すと同じ投稿が並ぶため、ISO週番号で順に回す。"""
    if not ranked:
        return None
    return ranked[today.isocalendar()[1] % len(ranked)]


def build_text(rec: dict, overall: dict) -> "str | None":
    """本文。全開示平均を必ず併記する（地合いを実力と誤認させないため）。

    実測ベンチマークでは140字超の投稿が中央値インプレッション562に対し59字以下は2,759で、
    短いほど伸びていた。事実2行＋比較1行＋注記に抑える。
    """
    if not rec or not overall:
        return None
    name = label(clean_name(rec["filer_name"]), FILER_LABEL_MAX_UNITS)
    direction = "上" if rec["mean"] > overall["mean"] else "下"
    lines = [
        f"🐋 {name}が新規で5%を届け出た{rec['n']}銘柄、3ヶ月後は平均{rec['mean']:+.1f}%（勝率{rec['win_rate']:.0f}%）",
        f"同期間の全開示平均は{overall['mean']:+.1f}%なので、明確に{direction}。",
        f"最大 {label(clean_name(rec['best'].get('issuer_name')), STOCK_LABEL_MAX_UNITS)}"
        f"({rec['best']['issuer_code']}) {rec['best']['ret']:+.1f}%",
        "",
        PROFILE_CTA,
        TAGS,
    ]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="提出者ごとの3ヶ月後リターン実績を投稿する")
    p.add_argument("--dry-run", action="store_true", help="投稿せず本文を表示するだけ")
    p.add_argument("--filer", help="この提出者について投稿する（既定は実績の振れ幅が大きい順の1位）")
    p.add_argument("--list", action="store_true", help="集計できた提出者を一覧表示する")
    args = p.parse_args()

    events = fetch_events()
    if not events:
        print("[x_filer_record] 対象の開示が0件でした")
        return 1
    codes = sorted({e["issuer_code"] for e in events})
    print(f"[x_filer_record] 開示{len(events)}件 / 銘柄{len(codes)}件の株価を取得します")
    rows = compute_returns(events, fetch_prices(codes))
    if not rows:
        print("[x_filer_record] リターンを計算できた開示が0件でした")
        return 1
    overall = summarize(rows)
    print(f"[x_filer_record] 全体 n={overall['n']} 平均{overall['mean']:+.1f}% "
          f"中央値{overall['median']:+.1f}% 勝率{overall['win_rate']:.1f}%")

    ranked = rank_filers(rows)
    if args.list:
        print(f"\n{'提出者':<44}{'n':>5}{'平均':>9}{'勝率':>8}")
        for f in ranked:
            print(f"{f['filer_name'][:42]:<44}{f['n']:>5}{f['mean']:>8.1f}%{f['win_rate']:>7.0f}%")
        return 0

    if args.filer:
        rec = next((f for f in ranked if args.filer in f["filer_name"]), None)
        if rec is None:
            print(f"[x_filer_record] {args.filer} は対象外です（{MIN_EVENTS}件未満か自己名義の提出者）")
            return 1
    else:
        rec = pick_weekly(ranked, date.today())
    text = build_text(rec, overall)
    if not text:
        print("[x_filer_record] 本文を作れませんでした")
        return 1
    print("\n" + text)
    if args.dry_run:
        return 0
    return 0 if post_tweet(text) else 1


if __name__ == "__main__":
    sys.exit(main())
