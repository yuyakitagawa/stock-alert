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
  title          → 決定的テンプレで組み直し（「新規保有」→「引き上げ/引き下げ」）
  tags           → 方向（売り）を付け直す
  body           → 誤った変化幅・金額・「実質的な新規保有」という記述を含むため既定で再生成
                   （--keep-body で据え置き。本文末尾の株価チャート<figure>は引き継ぐ）

是正後に is_indexable_article() の基準（推定3億円以上 または 変化1pt以上＝表示側のindex基準）を割る記事は、
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
from lib.edinet import disclosure_doc_label, disclosure_kind_label, resolve_filer
from tools.cleanup_duplicate_blog_articles import delete_article
from tools.reclassify_blog_articles import fetch_all_articles
from tools.scan_large_holdings import is_correction_report, is_sell_disclosure
from web.publish_blog_articles import (
    MICROCMS_DOMAIN, MICROCMS_KEY, MicroCMSPermissionError,
    build_article_titles, build_context_facts, classify_filer, estimate_deal_amount_oku,
    disclosure_close_price, generate_article_body_checked, get_company_description,
    is_indexable_article, update_article,
)

load_dotenv()

_FIGURE_RE = re.compile(r"<figure>.*?</figure>", re.S)
# 変化幅の食い違いをこのポイント数以上で「誤り」とみなす（丸め差は無視する）。
MISMATCH_TOLERANCE_PT = 0.01

# (銘柄コード, 正規化した提出者名) → [(開示日, 保有比率)] 。fetch_disclosures()が組み立てる。
HISTORY: dict = {}


def normalize_filer(name: str) -> str:
    return (name or "").replace("　", "").replace(" ", "").strip()


def fetch_disclosures(codes: list) -> dict:
    """(銘柄コード, 開示日, 正規化した提出者名) → EDINET開示 の辞書を返す。"""
    rows = []
    for i in range(0, len(codes), 100):
        chunk = ",".join(codes[i:i + 100])
        # order=doc_id は必須。PostgRESTのlimit/offsetページングはORDER BYが無いと
        # ページ間で行が重複・欠落しうる（同じ開示が2行に見えると「一意に特定できない」
        # としてスキップされ、実行ごとに対象件数が変わる。2026-08-19に実測）。
        rows += sb.select(
            "edinet_large_holdings",
            f"issuer_code=in.({chunk})&order=doc_id&select=doc_id,issuer_code,disc_date,filer_name,"
            "holding_ratio,holding_ratio_prior,doc_type_code,doc_description",
        )
    seen_docs = set()
    out = {}
    HISTORY.clear()
    for r in rows:
        if r.get("doc_id") in seen_docs:
            continue
        seen_docs.add(r.get("doc_id"))
        key = (str(r.get("issuer_code")), r.get("disc_date"), normalize_filer(r.get("filer_name")))
        out.setdefault(key, []).append(r)
        if r.get("holding_ratio") is not None:
            HISTORY.setdefault(
                (str(r.get("issuer_code")), normalize_filer(r.get("filer_name"))), []
            ).append((r.get("disc_date"), float(r["holding_ratio"])))
    for entries in HISTORY.values():
        entries.sort()
    return out


def previous_ratio(code: str, filer_name: str, disc_date: str) -> "float | None":
    """直前保有割合を持たない開示のために、同一銘柄・同一提出者の1つ前の開示比率を返す。
    fetch_disclosures()が作った履歴を引くだけなので追加クエリは発生しない。"""
    entries = HISTORY.get((code, normalize_filer(filer_name))) or []
    past = [ratio for d, ratio in entries if d and d[:10] < disc_date]
    return past[-1] if past else None


def find_disclosure(article: dict, by_key: dict) -> "dict | None":
    """記事に対応するEDINET開示を1件に特定する。

    提出者名が保存されている記事はそれで引く。引けない場合（filerName未保存の旧記事、
    表記ゆれ、同一銘柄・同一開示日に複数の提出者がいる開示）は、事実カード書き出しと
    同じ resolve_filer() でタイトル等から一意化を試みる。ここを素通りさせると、金額が
    保有総額のまま公開されている記事が是正されずに残る（2026-08-27の実測で、
    一意化できなかった100本を標本検査したところ29%が過大なままだった）。
    誤った提出者で是正すると別の投資家の取引として公開されるため、決まらなければNone。
    """
    code = str(article.get("stockCode") or "")
    disc_date = (article.get("dealDate") or "")[:10]
    if not code or not disc_date:
        return None
    filer = normalize_filer(article.get("filerName"))
    if filer:
        rows = by_key.get((code, disc_date, filer)) or []
        if len(rows) == 1:
            return rows[0]
    candidates = [r for (c, d, _), rs in by_key.items() if c == code and d == disc_date for r in rs]
    if len(candidates) == 1:
        return candidates[0]
    name = resolve_filer(article, candidates)
    if not name:
        return None
    rows = [r for r in candidates if r.get("filer_name") == name]
    return rows[0] if len(rows) == 1 else None


def corrected_values(article: dict, row: dict) -> "dict | None":
    """EDINET開示から、記事が持つべき変化幅・方向・金額を組み立てる。
    直前保有割合が取れない開示（新規の大量保有報告書など）は検証できないのでNoneを返す。"""
    ratio = row.get("holding_ratio")
    if ratio is None:
        return None
    desc = row.get("doc_description") or ""
    prior = row.get("holding_ratio_prior")
    inferred_prior = False
    if prior is None:
        # 直前保有割合を持たない開示（特例報告に多い）。過去開示から前回比率を引く。
        # これを引かずに記事化していた時期があり、その場合「今回比率の全量＝今回動いた分」と
        # みなされて変化幅も推定金額も実態の数倍〜数十倍に膨らんでいる
        # （実例: セリア2782の買い増し1.31ptが「推定取得金額300.9億円」＝保有総額として公開）。
        prior = previous_ratio(
            str(article.get("stockCode") or ""), row.get("filer_name") or "",
            (article.get("dealDate") or "")[:10],
        )
        if prior is None:
            # 過去開示も無い＝新規の大量保有報告書。「前回0%」として扱い、保有比率そのものが
            # ズレていた記事を拾えるようにする。共同保有の合算対応（2026-08-30、lib.edinet の
            # _aggregate_ratio）で holding_ratio が「筆頭保有者の1枠」から「提出者＋共同保有者の
            # 合算」に変わったため、新規開示の記事でも見出しの比率・推定金額がズレている
            # （実測: 2026-08-20以降の開示430件のうち212件＝49%で比率が変わった）。
            # 前回0%として組み直しても、比率が変わっていない記事は変化幅が一致するので対象外に
            # なる（下の field_mismatch 判定）。
            # 変更報告書で前回比率が取れない場合は0%とみなすと全量が動いたことになり実態と
            # かけ離れるため、従来どおり是正しない。
            if disclosure_kind_label(desc, str(row.get("doc_type_code") or "")) != "新規":
                return None
            prior = 0.0
        else:
            inferred_prior = True
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
    # (c) 前回比率を過去開示から補ったうえで、記事が保有比率の全量を変化幅として使った痕跡が
    #     あるもの。ratioChangePctを保存していなかった時代の記事はこの経路でしか検出できない。
    #     再計算した変化幅が保有比率と一致する場合は「全量が動いた」で正しいので対象外。
    #     inferred_prior（前回比率をこちらで補ったか）は条件にしない。DBに前回比率が
    #     後から埋まっていても、記事が公開された当時は取れておらず全量を使っていた例が
    #     多数ある（実例: セリア2782の買い増し1.31ptが「推定取得金額300.9億円」＝保有総額。
    #     DBには前回比率13.93%が入っている）。判定はDBの状態ではなく記事側の痕跡で行う。
    full_ratio_mismatch = (
        old_change is None and abs(change - float(ratio)) >= MISMATCH_TOLERANCE_PT
    )
    # (d) 保有比率が前回と同一（変化幅0pt）なのに、記事が推定取得/売却金額を出しているもの。
    #     担保契約や共同保有者の変更などで再提出された変更報告書で、売買は起きていない。
    #     実例: 「Jトラスト、KeyHolderの約30%を取得—41.4億円」「日本レイが東テクの14.88%を
    #     取得、240億円超を投下」。実在企業について起きていない取引を見出しで断定しており、
    #     薄いどころか誤報にあたる（2026-08-27の全件照合で40本）。
    no_move_mismatch = change == 0 and float(article.get("dealAmount") or 0) > 0
    if not field_mismatch and not title_mismatch and not full_ratio_mismatch and not no_move_mismatch:
        return None
    if is_correction:
        deal_amount = 0.0
    else:
        deal_amount = estimate_deal_amount_oku(
            str(article["stockCode"]), change, (article.get("dealDate") or "")[:10]
        )
        # 上場廃止・コード変更などでyfinanceが株価を返さない銘柄は再概算できない。
        # 変化幅が元から正しい記事（誤っていたのはタイトルだけ）なら、既存の推定金額は
        # その変化幅で計算されたものなのでそのまま使い、タイトルだけ直す。
        if deal_amount is None and not field_mismatch and article.get("dealAmount") is not None:
            deal_amount = article["dealAmount"]
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


def build_fact_sheet(article: dict, fix: dict, filer_name: str) -> dict:
    code = str(article["stockCode"])
    name = article.get("stockName") or code
    disc_date = (article.get("dealDate") or "")[:10]
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
        "filer_description": filer_info.get("description") or "",
        "company_description": get_company_description(code, name),
        "ratio_change_pct": fix["change"],
        "prior_ratio": fix["prior_ratio"],
        # 是正のために本文を作り直すのだから、同じ回で厚みも取る。開示を横断して初めて書ける
        # 周辺事実（保有の積み上げ履歴・その投資家の他の保有銘柄・その銘柄の他の大株主・
        # 開示日時点の指標）を渡さないと、正しい数字の薄い記事に置き換わるだけになる。
        "context_facts": build_context_facts(code, filer_name, disc_date),
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
        if fix["change"] == 0:
            print(f"  🗑 {a['id']}: {name}({code}) 前回開示から保有比率が動いていない"
                  f"（{fix['holding_ratio']}% → {fix['holding_ratio']}%）— 売買を伴わない開示")
            to_delete.append(a)
            continue

        if fix["deal_amount"] is None:
            print(f"  ⚠ {a['id']}: {name}({code}) 株価・株式数が取れず金額を再概算できないためスキップ")
            failed.append(a["id"])
            continue

        if not is_indexable_article(fix["deal_amount"], fix["signed_change"]):
            print(f"  🗑 {a['id']}: {name}({code}) 是正後 {fix['deal_amount']}億円 / "
                  f"{fix['signed_change']}pt で基準未満 — 記事化すべきでなかった開示")
            to_delete.append(a)
            continue

        fact_sheet = build_fact_sheet(a, fix, filer_name)
        titles = build_article_titles(fact_sheet)
        payload = {
            "title": titles["title"],
            "dealAmount": fix["deal_amount"],
            "ratioChangePct": fix["signed_change"],
            "tags": build_tags(a, fix),
        }

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
