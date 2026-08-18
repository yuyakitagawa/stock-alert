"""既存記事の filerName フィールドをEDINET開示から逆引きしてバックフィルする。

filerName は2026-08-15にmicroCMSへ追加したフィールドで、それ以前に生成された記事には
入っていない（2026-08-19時点で791本中326本が未設定）。

このフィールドが無いと `kujira-watch/src/lib/articleIndexability.ts` の
`supersededArticleIds()`（同一「銘柄×提出者」で最新1本だけをindexするカニバリゼーション対策）
が判定できず、同じ投資家×同じ銘柄の記事が何本も検索結果に並んだままになる。

逆引きは stockCode + dealDate で `edinet_large_holdings` を引く。同日に複数の提出者が
同じ銘柄へ報告書を出しているケースは、記事タイトルに提出者名が含まれているかで絞り込み、
それでも一意にならないものはスキップする（誤った提出者を書き込む方が害が大きい）。

使い方:
    python3 tools/backfill_article_filer_name.py            # dry-run（既定）
    python3 tools/backfill_article_filer_name.py --apply    # 実際に書き込む
"""
import argparse
import os
import re
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

import lib.supabase_client as sb
from tools.reclassify_blog_articles import fetch_all_articles
from web.publish_blog_articles import update_article

load_dotenv()

# 法人格の表記ゆれ。EDINETの提出者名は「株式会社ストラテジックキャピタル」だが記事タイトルは
# 「ストラテジックキャピタル」と略すため、突合前に落とす。
_LEGAL_SUFFIXES = [
    "株式会社", "合同会社", "有限会社", "合資会社", "合名会社",
    "一般社団法人", "一般財団法人", "公益社団法人", "公益財団法人",
    "投資事業有限責任組合", "相互会社",
]
# 長音符(ー)は「アドバイザーズ」のような語の一部なので落とさない。落とすとカタカナ社名が壊れる。
_NOISE_RE = re.compile(r"[\s・．，,.\-‐−―（）()「」【】]")

# 短すぎる語での部分一致は誤爆する（例:「ＭＣ」が無関係な記事タイトルに含まれる）。
MIN_MATCH_LEN = 4


def normalize(text: str) -> str:
    """全角/半角・記号・法人格を落として突合用のキーにする。"""
    s = unicodedata.normalize("NFKC", text or "")
    for suffix in _LEGAL_SUFFIXES:
        s = s.replace(suffix, "")
    s = _NOISE_RE.sub("", s)
    return s.lower()


def pick_filer(title: str, candidates: list[str]) -> tuple[str | None, str]:
    """同日・同銘柄の提出者候補から記事の提出者を1つ選ぶ。(選んだ名前, 理由) を返す。"""
    uniq = sorted({c for c in candidates if c})
    if not uniq:
        return None, "候補なし"
    if len(uniq) == 1:
        return uniq[0], "候補1件"
    norm_title = normalize(title)
    matched = [c for c in uniq if len(normalize(c)) >= MIN_MATCH_LEN and normalize(c) in norm_title]
    if len(matched) == 1:
        return matched[0], f"候補{len(uniq)}件→タイトル一致で特定"
    return None, f"候補{len(uniq)}件・タイトル一致{len(matched)}件で一意にならず"


def load_disclosures(stock_codes: set[str]) -> dict[tuple[str, str], list[str]]:
    """対象銘柄の開示を1クエリでまとめて取得し (銘柄コード, 開示日) -> 提出者名リスト にする。"""
    if not stock_codes:
        return {}
    codes = ",".join(sorted(stock_codes))
    # PostgRESTはorderを指定しないと返る行の順序を保証しない。sb.select()は1000行ずつ
    # offsetでページングするため、順序が不定だと行の重複・取りこぼしが起きる
    # （実測: 順序未指定だと「候補なし」が73件出たが、order指定で3件まで減った）。
    rows = sb.select(
        "edinet_large_holdings",
        f"issuer_code=in.({codes})&select=issuer_code,disc_date,filer_name&order=doc_id",
    )
    table: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        key = (row.get("issuer_code"), row.get("disc_date"))
        if not key[0] or not key[1]:
            continue
        table.setdefault(key, []).append(row.get("filer_name"))
    return table


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="実際にmicroCMSへ書き込む（既定はdry-run）")
    parser.add_argument("--limit", type=int, default=0, help="処理する記事数の上限（動作確認用）")
    args = parser.parse_args()

    articles = fetch_all_articles()
    targets = [a for a in articles if not a.get("filerName") and a.get("stockCode") and a.get("dealDate")]
    print(f"全記事 {len(articles)}件 / filerName未設定 {len(targets)}件")
    if args.limit:
        targets = targets[: args.limit]

    table = load_disclosures({a["stockCode"] for a in targets})
    print(f"EDINET開示 {len(table)}組（銘柄×開示日）を取得")

    resolved, skipped, failed = 0, 0, 0
    skip_reasons: dict[str, int] = {}
    for article in targets:
        disc_date = article["dealDate"][:10]
        candidates = table.get((article["stockCode"], disc_date), [])
        filer, reason = pick_filer(article.get("title", ""), candidates)
        if not filer:
            skipped += 1
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            continue
        resolved += 1
        if not args.apply:
            print(f"  [dry-run] {article['id']} {article['stockCode']} {disc_date} -> {filer}（{reason}）")
            continue
        if not update_article(article["id"], {"filerName": filer}):
            failed += 1
            print(f"  [失敗] {article['id']} -> {filer}")

    print(f"\n特定できた: {resolved}件 / スキップ: {skipped}件" + (f" / 書き込み失敗: {failed}件" if args.apply else ""))
    for reason, count in sorted(skip_reasons.items(), key=lambda kv: -kv[1]):
        print(f"  スキップ理由: {reason} — {count}件")
    if not args.apply:
        print("\n--apply を付けると実際に書き込みます。")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
