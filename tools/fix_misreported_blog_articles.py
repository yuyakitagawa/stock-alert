#!/usr/bin/env python3
"""
tools/fix_misreported_blog_articles.py
「変更報告書なのに新規保有として公開された記事」をEDINETの実データで是正する。

背景（2026-08-19の監査）: EDINETはメタデータ公開とXBRL本文の可用性にラグがあり、提出直後の便では
holding_ratio_prior が取れないことがある。その状態で記事化すると publish_blog_articles の
ratio_change_pct() が「今回比率の全量＝新規取得」とみなし、
  - タイトルが「X%を新規保有」（実際は変更報告書＝前回比率あり）
  - ratioChangePct が比率全量（実例: セイコーグループ8050 −10.57pt、実際は −0.15pt）
  - estimate_deal_amount_oku() の推定金額が同じ倍率で過大（同 966.1億円 → 実際は十数億円）
という誤りが公開される。直近14日の照合可能56件中13件（23%）が該当していた。
生成側は publish_blog_articles.should_wait_for_prior_ratio() で止めたので、本スクリプトは
既に公開済みの記事を回収する。

是正内容（EDINET開示 = 正）:
  ratioChangePct → holding_ratio - holding_ratio_prior（売りは負値）
  dealAmount     → 正しい変化幅で再概算（訂正記事は0のまま）
  title/titleEn  → 決定的テンプレで組み直し（「新規保有」→「引き上げ/引き下げ」）
  tags           → 方向（売り）を付け直す
  body           → 誤った変化幅・金額・「実質的な新規保有」という記述を含むため既定で再生成
                   （--keep-body で据え置き。本文末尾の株価チャート<figure>は引き継ぐ）

是正後に is_worth_publishing() の基準（推定3億円以上 または 変化1pt以上）を割る記事は、
そもそも記事化すべきでなかったもの。--delete 指定時のみ、全フィールドをlogsへバックアップして
削除する（指定が無ければ一覧を表示するだけ）。

実行:
    python3 tools/fix_misreported_blog_articles.py                  # dry-run（対象の表示のみ）
    python3 tools/fix_misreported_blog_articles.py --limit 5        # 先頭5件だけ確認
    python3 tools/fix_misreported_blog_articles.py --apply          # microCMSを更新
    python3 tools/fix_misreported_blog_articles.py --apply --delete # 基準割れ記事の削除も行う
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

import lib.supabase_client as sb
from lib.edinet import disclosure_doc_label
from tools.cleanup_duplicate_blog_articles import delete_article
from tools.reclassify_blog_articles import fetch_all_articles
from tools.scan_large_holdings import is_correction_report, is_sell_disclosure
from web.publish_blog_articles import (
    MICROCMS_DOMAIN, MICROCMS_KEY, MicroCMSPermissionError,
    build_article_titles, classify_filer, dp_level_label, estimate_deal_amount_oku,
    disclosure_close_price, generate_article_body_checked, get_company_description,
    get_pit_ranking_snapshot, is_worth_publishing, update_article,
)

load_dotenv()

_FIGURE_RE = re.compile(r"<figure>.*?</figure>", re.S)
# 英語タイトルのテンプレから提出者名・銘柄名のローマ字表記を回収する
# （再生成時に英語名を作り直さず、既存記事の表記を引き継ぐため）。
_TITLE_EN_RE = re.compile(
    r"^(?P<filer>.+?) (?:Takes .+? Stake in|Cuts Stake in|Raises Stake in|Corrects Reported Stake in) "
    r"(?P<stock>.+?) \(\d+\)"
)
# 変化幅の食い違いをこのポイント数以上で「誤り」とみなす（丸め差は無視する）。
MISMATCH_TOLERANCE_PT = 0.01


def normalize_filer(name: str) -> str:
    return (name or "").replace("　", "").replace(" ", "").strip()


def fetch_disclosures(codes: list) -> dict:
    """(銘柄コード, 開示日, 正規化した提出者名) → EDINET開示 の辞書を返す。"""
    rows = []
    for i in range(0, len(codes), 100):
        chunk = ",".join(codes[i:i + 100])
        rows += sb.select(
            "edinet_large_holdings",
            f"issuer_code=in.({chunk})&select=issuer_code,disc_date,filer_name,holding_ratio,"
            "holding_ratio_prior,doc_type_code,doc_description",
        )
    out = {}
    for r in rows:
        key = (str(r.get("issuer_code")), r.get("disc_date"), normalize_filer(r.get("filer_name")))
        out.setdefault(key, []).append(r)
    return out


def find_disclosure(article: dict, by_key: dict) -> "dict | None":
    """記事に対応するEDINET開示を1件に特定する。提出者名が保存されていない旧記事
    （2026-08-15以前）は、同一銘柄・同一開示日の開示が1件のときだけ特定できたとみなす。"""
    code = str(article.get("stockCode") or "")
    disc_date = (article.get("dealDate") or "")[:10]
    if not code or not disc_date:
        return None
    filer = normalize_filer(article.get("filerName"))
    if filer:
        rows = by_key.get((code, disc_date, filer)) or []
    else:
        rows = [r for (c, d, _), rs in by_key.items() if c == code and d == disc_date for r in rs]
    return rows[0] if len(rows) == 1 else None


def corrected_values(article: dict, row: dict) -> "dict | None":
    """EDINET開示から、記事が持つべき変化幅・方向・金額を組み立てる。
    直前保有割合が取れない開示（新規の大量保有報告書など）は検証できないのでNoneを返す。"""
    ratio = row.get("holding_ratio")
    prior = row.get("holding_ratio_prior")
    if ratio is None or prior is None:
        return None
    desc = row.get("doc_description") or ""
    is_correction = is_correction_report(desc)
    is_sell = is_sell_disclosure(desc, ratio, prior)
    change = round(abs(ratio - prior), 2)
    signed_change = round(-change if is_sell else change, 2)
    old_change = article.get("ratioChangePct")
    # 対象は「誤りが実際に画面へ出ている記事」だけに絞る。
    #  (a) ratioChangePctが入っていて実データとズレている（2026-08-15以降の記事）
    #  (b) 前回比率が0より大きいのにタイトルが「新規保有」（変化幅を比率全量とみなした痕跡。
    #      ratioChangePctフィールドが無い時代の記事もこの一点で検出できる）
    # ratioChangePctが未設定なだけの旧記事（タイトルは正しい）は、当時のタイトルがLLM生成で
    # 現在のテンプレと形が違うため作り直すと不必要な書き換えになる。対象にしない。
    field_mismatch = old_change is not None and abs(old_change - signed_change) >= MISMATCH_TOLERANCE_PT
    title_mismatch = prior > 0 and "新規保有" in (article.get("title") or "")
    if not field_mismatch and not title_mismatch:
        return None
    if is_correction:
        deal_amount = 0.0
    else:
        deal_amount = estimate_deal_amount_oku(
            str(article["stockCode"]), change, (article.get("dealDate") or "")[:10]
        )
    return {
        "row": row,
        "holding_ratio": ratio,
        "prior_ratio": prior,
        "change": change,
        "signed_change": signed_change,
        "is_sell": is_sell,
        "is_correction": is_correction,
        "deal_amount": deal_amount,
    }


def english_names(title_en: str) -> tuple:
    match = _TITLE_EN_RE.match(title_en or "")
    return (match.group("stock"), match.group("filer")) if match else ("", "")


def build_fact_sheet(article: dict, fix: dict, filer_name: str) -> dict:
    code = str(article["stockCode"])
    name = article.get("stockName") or code
    disc_date = (article.get("dealDate") or "")[:10]
    snapshot = get_pit_ranking_snapshot(code, disc_date)
    context_dp = snapshot.get("drop_prob") if snapshot else None
    filer_info = classify_filer(filer_name)
    return {
        "stock_name": name,
        "stock_code": code,
        "filer_name": filer_name,
        "doc_type_label": disclosure_doc_label(
            fix["row"].get("doc_description"), fix["row"].get("doc_type_code", "")
        ),
        "holding_ratio": fix["holding_ratio"],
        "disc_date": disc_date,
        "deal_amount_oku": fix["deal_amount"],
        "direction": "sell" if fix["is_sell"] else "buy",
        "deal_amount_label": "推定売却金額" if fix["is_sell"] else "推定取得金額",
        # 本文の株価は金額の概算に使った開示日終値と同じ値（サイトの「基準終値」と同源）。
        "context_close": disclosure_close_price(code, disc_date),
        "context_dp_level": dp_level_label(context_dp) if context_dp is not None else None,
        "filer_description": filer_info.get("description") or "",
        "company_description": get_company_description(code, name),
        "ratio_change_pct": fix["change"],
        "prior_ratio": fix["prior_ratio"],
        "is_correction": fix["is_correction"],
    }


def build_tags(article: dict, fix: dict) -> str:
    tag_list = ["EDINET", "自動生成"]
    if fix["is_correction"]:
        tag_list.append("訂正")
    if fix["is_sell"]:
        tag_list.append("売り")
    return ",".join(tag_list)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true", help="実際にmicroCMSを更新する（無指定はdry-run）")
    p.add_argument("--limit", type=int, default=None, help="処理件数の上限（動作確認用）")
    p.add_argument("--keep-body", action="store_true", help="本文を再生成しない（構造化フィールドのみ是正）")
    p.add_argument("--delete", action="store_true", help="是正後に基準未満となる記事を削除する")
    p.add_argument("--backup", default=None, help="削除時のバックアップJSONの出力先")
    args = p.parse_args()

    if args.apply and (not MICROCMS_DOMAIN or not MICROCMS_KEY):
        print("MICROCMS_SERVICE_DOMAIN / MICROCMS_API_KEY が未設定です。")
        sys.exit(1)

    articles = fetch_all_articles()
    codes = sorted({str(a.get("stockCode")) for a in articles if a.get("stockCode")})
    by_key = fetch_disclosures(codes)
    print(f"記事総数: {len(articles)}件 / EDINET開示: {sum(len(v) for v in by_key.values())}件")

    targets = []
    for a in articles:
        row = find_disclosure(a, by_key)
        if row is None:
            continue
        fix = corrected_values(a, row)
        if fix is not None:
            targets.append((a, fix, row.get("filer_name") or a.get("filerName") or ""))

    print(f"EDINETと変化幅が食い違う記事: {len(targets)}件")
    if args.limit is not None:
        targets = targets[: args.limit]

    updated, deleted, failed = 0, 0, []
    to_delete = []
    for a, fix, filer_name in targets:
        code = str(a["stockCode"])
        name = a.get("stockName") or code
        if fix["deal_amount"] is None:
            print(f"  ⚠ {a['id']}: {name}({code}) 株価・株式数が取れず金額を再概算できないためスキップ")
            failed.append(a["id"])
            continue

        if not is_worth_publishing(fix["deal_amount"], fix["signed_change"]):
            print(f"  🗑 {a['id']}: {name}({code}) 是正後 {fix['deal_amount']}億円 / "
                  f"{fix['signed_change']}pt で基準未満 — 記事化すべきでなかった開示")
            to_delete.append(a)
            continue

        fact_sheet = build_fact_sheet(a, fix, filer_name)
        stock_en, filer_en = english_names(a.get("titleEn") or "")
        titles = build_article_titles(fact_sheet, stock_name_en=stock_en, filer_name_en=filer_en)
        payload = {
            "title": titles["title"],
            "dealAmount": fix["deal_amount"],
            "ratioChangePct": fix["signed_change"],
            "tags": build_tags(a, fix),
        }
        if a.get("titleEn"):
            payload["titleEn"] = titles["titleEn"]

        print(f"  {a['id']}: {name}({code}) {a.get('ratioChangePct')}pt→{fix['signed_change']}pt / "
              f"{a.get('dealAmount')}億円→{fix['deal_amount']}億円")
        print(f"      {a.get('title')}\n   →  {titles['title']}")

        if not args.keep_body:
            generated = generate_article_body_checked(fact_sheet) if args.apply else None
            if args.apply and generated is None:
                print(f"      ⚠ 本文の再生成に失敗したため構造化フィールドのみ更新します")
            elif generated is not None:
                new_body = generated["body"]
                figure = _FIGURE_RE.search(a.get("body") or "")
                if figure:
                    new_body += figure.group(0)
                payload["body"] = new_body
                if generated.get("bodyEn"):
                    payload["bodyEn"] = generated["bodyEn"]

        if not args.apply:
            continue
        try:
            if update_article(a["id"], payload):
                updated += 1
            else:
                failed.append(a["id"])
        except MicroCMSPermissionError as e:
            print(f"      ⚠ 権限エラーのため中断: {e}")
            break

    if to_delete:
        print(f"\n是正後に基準未満となる記事: {len(to_delete)}件")
        if args.apply and args.delete:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = args.backup or os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "logs", f"deleted_misreported_articles_{stamp}.json",
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(to_delete, f, ensure_ascii=False, indent=2)
            print(f"バックアップ: {path}")
            for a in to_delete:
                if delete_article(a["id"]):
                    deleted += 1
        else:
            print("（--apply --delete を付けると、バックアップを取ってから削除します）")

    print(f"\n完了: 更新 {updated}件 / 削除 {deleted}件 / 失敗 {len(failed)}件")
    if failed:
        print(f"失敗した記事id: {failed}")


if __name__ == "__main__":
    main()
