"""
web/x_disclosure_facts.py

開示原文にしか書かれていない事実だけを投稿する枠（不定期・週1便で候補があるときだけ）。

同ジャンル9アカウント879投稿の実測（2026-08-30、tools/x_benchmark.py）では、開示を
そのまま流すアカウント5つが全て中央値エンゲージメント0〜2に張り付いていた。「誰が何%
取得した」は誰でも同じものを出せるので差がつかない。

一方で、EDINETの大量保有報告書には**本表(XBRL)まで開かないと分からない事実**があり、
速報botはここを拾っていない。この枠が出すのは次の2種類:

  ①全額借入   取得資金の全額が借入で自己資金ゼロ（funding_borrowings>0 かつ funding_own=0）
  ②遅延開示   報告義務の発生日から30日を超えて提出（obligation_date と disc_date の差）

株価を一切使わないため、`web/x_followup.py` のような騰落率の不確実性が構造的に発生しない。
数字は開示に書いてある値そのもので、後から変わらない。

**個人名義の提出者は除外する**。「◯◯氏が498日遅れて提出」は事実だが、私人を名指しで
問題視する投稿になる。法人・ファンドに限る（publish_blog_articles.py が分類「個人」に
description を持たせないのと同じ理由）。

コスト: Anthropic APIは使わない（Supabaseの既存テーブルを読むだけ）。

実行:
  python3 web/x_disclosure_facts.py --dry-run          # 本文の生成・表示のみ
  python3 web/x_disclosure_facts.py --days 30 --dry-run # 窓を広げて候補を見る
  python3 web/x_disclosure_facts.py                    # Xへ投稿
"""
import argparse
import os
import sys
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, os.path.expanduser("~/stock-alert"))

from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/stock-alert/.env"))

from lib import supabase_client as sb  # noqa: E402
from web.x_client import PROFILE_CTA, TAGS, post_tweet  # noqa: E402
from web.x_post_format import clean_name, label  # noqa: E402

KIND = "disclosure_fact"
# 候補を探す窓（日）。週1便なので既定は8日（便の遅延で1日空くのを許容する）。
DEFAULT_DAYS = 8
# 報告義務の発生から提出までがこの日数を超えたら「遅延」として扱う。
LATE_THRESHOLD_DAYS = 30
# 同じ銘柄を短期間に繰り返さないための除外期間（日）。
RECENT_STOCK_DAYS = 30
STOCK_LABEL_MAX_UNITS = 24
FILER_LABEL_MAX_UNITS = 24


def fetch_candidates(days: int) -> list:
    """窓の中の開示を、判定に必要な列だけ取る。"""
    since = (date.today() - timedelta(days=days)).isoformat()
    rows = sb.select(
        "edinet_large_holdings",
        "select=doc_id,filer_name,issuer_code,issuer_name,disc_date,holding_ratio,holding_ratio_prior,"
        "funding_total,funding_own,funding_borrowings,obligation_date,doc_description"
        f"&disc_date=gte.{since}&issuer_code=not.is.null",
    )
    return [r for r in rows or [] if "訂正" not in (r.get("doc_description") or "")]


# 法人であることが名前から確実に分かる語。これを含まない提出者は個人とみなして除外する。
CORPORATE_TOKENS = ("株式会社", "有限会社", "合同会社", "合資会社", "財団", "社団", "法人",
                    "組合", "基金", "銀行", "信託", "証券", "證券", "保険", "ファンド",
                    "Ｌｔｄ", "Ltd", "ＬＬＣ", "LLC", "ＬＬＰ", "LLP", "Ｉｎｃ", "Inc",
                    "Ｃｏｒｐ", "Corp", "Ｃｏｍｐａｎｙ", "Company", "Ｆｕｎｄ", "Fund",
                    "Ｐａｒｔｎｅｒｓ", "Partners", "Ｃａｐｉｔａｌ", "Capital",
                    "Ｍａｎａｇｅｍｅｎｔ", "Management", "Ｉｎｖｅｓｔｍｅｎｔ", "Investment",
                    "Ｇｒｏｕｐ", "Group", "Ｈｏｌｄｉｎｇｓ", "Holdings", "Ｐｔｅ", "Pte",
                    "Ｓ．Ａ", "Ｎ．Ｖ", "ＧｍｂＨ", "GmbH", "Ａ／Ｓ")


def looks_like_individual(filer_name: str) -> bool:
    """名前から法人と判別できない提出者は個人として扱う。

    分類マスター（edinet_filer_classification）に載っていない提出者が実際にいるため、
    マスターだけに頼ると「蕪木 登」のような個人名がそのまま投稿に載る。私人を名指しで
    問題視する投稿になるので、判定は保守側（怪しければ個人）に倒す。
    """
    name = filer_name or ""
    return not any(t in name for t in CORPORATE_TOKENS)


def fetch_individual_filers() -> set:
    """分類が「個人」の提出者名。私人を名指しする投稿を避けるために使う。"""
    rows = sb.select("edinet_filer_classification", "select=filer_name&category=eq.個人")
    return {r["filer_name"] for r in rows or []}


def is_amendment(row: dict) -> bool:
    """変更報告書か。既に5%以上を持っている届出なので「X%取得」とは書けない
    （比率が5%未満まで下がった変更報告書もあり、「3%取得」という誤った本文になる）。"""
    return "変更報告書" in (row.get("doc_description") or "")


def is_fully_borrowed(row: dict) -> bool:
    """取得資金の全額が借入か。自己資金の欄が空の開示は判定しない（未取得と区別できない）。"""
    borrowings = row.get("funding_borrowings")
    own = row.get("funding_own")
    return bool(borrowings and borrowings > 0 and own is not None and own == 0)


def late_days(row: dict) -> "int | None":
    """報告義務の発生から提出までの日数。しきい値以下・欠損はNone。"""
    obligation, disc = row.get("obligation_date"), row.get("disc_date")
    if not obligation or not disc:
        return None
    try:
        gap = (date.fromisoformat(disc[:10]) - date.fromisoformat(obligation[:10])).days
    except ValueError:
        return None
    return gap if gap > LATE_THRESHOLD_DAYS else None


def recently_posted_codes() -> set:
    """直近に同じ枠で投稿した銘柄。同じ会社が続けて出るのを避ける。"""
    # PostgRESTのクエリ文字列で "+00:00" の + が空白として解釈され 400 になるため Z を使う。
    since = (datetime.now(timezone.utc) - timedelta(days=RECENT_STOCK_DAYS)) \
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = sb.select("x_posts", f"select=stock_code&kind=eq.{KIND}&posted_at=gte.{since}")
    return {r["stock_code"] for r in rows or [] if r.get("stock_code")}


def pick(rows: list, individuals: set, posted: set) -> "tuple[str, dict] | tuple[None, None]":
    """投稿する1件を選ぶ。全額借入を優先し（金額が大きいものから）、無ければ遅延（長い順）。"""
    usable = [r for r in rows
              if r.get("filer_name") not in individuals
              and not looks_like_individual(r.get("filer_name"))
              and r.get("issuer_code") not in posted]
    # 全額借入は**新規報告書に限る**。変更報告書の取得資金欄は保有分全体の資金であって
    # その回の買い増し分ではないため、「買い増し」と「取得資金14億円」を並べると、その
    # 0.11ポイントの買い増しに14億円かかったように読める（実測でそう出た）。新規報告書なら
    # 届け出た保有分と取得資金が一対一で対応する。
    borrowed = sorted([r for r in usable if is_fully_borrowed(r) and not is_amendment(r)],
                      key=lambda r: -(r.get("funding_borrowings") or 0))
    if borrowed:
        return "borrowed", borrowed[0]
    late = sorted([r for r in usable if late_days(r) is not None],
                  key=lambda r: -late_days(r))
    if late:
        return "late", late[0]
    return None, None


def _oku(yen: float) -> str:
    return f"{yen / 100_000_000:.0f}億円" if yen >= 100_000_000 else f"{yen / 100_000_000:.1f}億円"


def build_text(kind: str, row: dict) -> "str | None":
    """本文。実測で140字超が中央値インプレッション562・59字以下が2,759だったので短く保つ。

    **「今年N件」のような希少性は書かない**。取得資金の内訳も報告義務発生日も本表(XBRL)を
    開かないと取れず、その解析が済んでいるのは2026年の開示の約2%しかない。母数を「全開示」
    として件数を出すと、実際には「解析済みの中で」でしかない数字を全体の数字として提示する
    ことになる（実測: 全額借入を「今年4件」と書きかけたが、直近30日だけで5件あった）。
    """
    if not row:
        return None
    stock = label(clean_name(row.get("issuer_name")), STOCK_LABEL_MAX_UNITS)
    filer = label(clean_name(row.get("filer_name")), FILER_LABEL_MAX_UNITS)
    ratio = row.get("holding_ratio")
    if kind == "borrowed":
        # pick() が新規報告書に限っているので「X%取得」と書ける。
        lines = [
            f"🐋 {filer}が{stock}({row['issuer_code']})を{ratio}%取得",
            f"取得資金{_oku(row['funding_borrowings'])}は全額が借入。自己資金はゼロ。",
        ]
    else:
        days = late_days(row)
        if days is None:
            return None
        # 変更報告書も混じるため「取得」「買い増し」とは書かず、現在の保有比率だけを言う。
        lines = [
            f"🐋 {stock}({row['issuer_code']})の{ratio}%を持つ{filer}、",
            f"報告義務の発生から{days}日たってから提出。",
            "",
            "大量保有報告書は原則5営業日以内に出すもの。",
        ]
    lines += ["", PROFILE_CTA, TAGS]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="開示原文の事実だけを投稿する")
    p.add_argument("--dry-run", action="store_true", help="投稿せず本文を表示するだけ")
    p.add_argument("--days", type=int, default=DEFAULT_DAYS, help="候補を探す窓（日）")
    p.add_argument("--list", action="store_true", help="窓の中の候補を一覧表示する")
    args = p.parse_args()

    rows = fetch_candidates(args.days)
    individuals = fetch_individual_filers()
    posted = set() if args.dry_run else recently_posted_codes()
    print(f"[x_disclosure_facts] 直近{args.days}日の開示{len(rows)}件から候補を探します")

    if args.list:
        for r in rows:
            if r.get("filer_name") in individuals or looks_like_individual(r.get("filer_name")):
                continue
            if is_fully_borrowed(r):
                print(f"  全額借入 {r['disc_date'][:10]} {r['issuer_name']}({r['issuer_code']}) "
                      f"{r['filer_name']} {_oku(r['funding_borrowings'])}")
            elif late_days(r) is not None:
                print(f"  遅延{late_days(r):>4}日 {r['disc_date'][:10]} {r['issuer_name']}"
                      f"({r['issuer_code']}) {r['filer_name']}")
        return 0

    kind, row = pick(rows, individuals, posted)
    if kind is None:
        print("[x_disclosure_facts] 該当する開示が無いため投稿しません")
        return 0
    text = build_text(kind, row)
    if not text:
        print("[x_disclosure_facts] 本文を作れませんでした")
        return 1
    print("\n" + text)
    if args.dry_run:
        return 0
    tweet_id = post_tweet(text, kind=KIND, stock_code=row["issuer_code"])
    return 0 if tweet_id else 1


if __name__ == "__main__":
    sys.exit(main())
