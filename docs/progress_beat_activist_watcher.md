# 競合「アクティビストウォッチャー」対抗施策（2026-08-30 開始）

競合: https://obaonline.net/apps/activist-watcher/ja/ （iOS/Androidアプリ、System Design Office OBA）
- データ範囲 2025-01-01〜。保有比率推移グラフ・平均取得単価・変更点ハイライト・
  ウォッチ無制限・ランキング100位・過去分PDFは **有料**。
- 株価データを一切持たない＝「買った後どうなったか」は出せない。ここがうちの勝ち筋。

## ① EDINET開示のバックフィル（2024-12-03〜2025-06-17）
現状 `edinet_large_holdings` は 2025-06-18〜（20,351件）。価格キャッシュ `yahoo_price_cache` が
2024-12-03からあるので、そこまで遡れば追加分も全件リターン計算が付き、投資家の勝率統計の
サンプル数が増える。EDINET APIは無料（Anthropic課金なし）。

- [x] `scan_large_holdings()` に `end_date` を追加（現状 start_date→当日で既存範囲を再走査してしまう）
- [x] `tools/scan_large_holdings.py` に `--end` を追加
- [~] バックフィル実行（2024-12-03 〜 2025-06-15。2025-06-16/17は動作確認で投入済み）
      → バックグラウンド実行中 `logs/edinet_backfill_20241203_20250615.log`
- [ ] 件数・期間をSQLで確認
- [ ] `investor_returns_3m` 系マテビューのリフレッシュ

## ② リターン実績の露出強化
- [x] `/investors` 一覧のカードに「開示3ヶ月後 平均±X%・勝率Y%（N件）」（`getFilerReturnMap()`を新設）
- [x] TOPに「{開示日}の買い開示は、その後どうなったか」枠（`TopReturnPreview.tsx`＋`getLatestReturnCohort()`）。
      上位3件だけでなく母数の平均・勝率を必ず併記する
- [x] README更新（`/`と`/investors`の行）
- [x] tsc / lint / build / `next start`で実データ表示確認（TOP=200・/investors=200、
      2026/05/28の買い開示10件・平均+16.8%・勝率70%・上位3件が実データで描画）
- [ ] コミット & push・本番確認（※pr284がorigin側と8対8で分岐しておりpush不可。オーナー判断待ち）

## ③ 保有目的の保存と変更ハイライト
- [x] XBRLから保有目的を抽出できるか確認（`jplvh_cor:PurposeOfHolding` / `ActOfMakingImportantProposalEtc` /
      `TotalNumberOfStocksEtcHeld` / `TotalAmountOfFundingForAcquisition` が取得可。
      contextRef が `FilingDateInstant`(接尾辞なし)＝共同保有者を含む合算、`...HolderNMember`＝保有者別）
- [x] `edinet_large_holdings` に列追加（purpose_of_holding / important_proposal / shares_held /
      shares_outstanding / funding_total / funding_own / funding_borrowings / obligation_date）
- [ ] 取得・表示・差分ハイライト


## ⚠️ 並行セッションとの競合（2026-08-30 11:14 検知）
同一作業ツリーで別セッションが 11:12〜11:14 に lib/edinet.py・lib/db.py を編集し、
`parse_holding_details()` / `classify_purpose()` / `average_acquisition_price()` /
`HOLDING_DETAIL_COLUMNS` を追加した（＝③と同じ範囲）。本セッションの `end_date` 追加の上に
乗っているので、後発はあちら。列名が食い違ったため、**DDLをあちらのコードに合わせた**
（`funding_amount` を削除し `funding_total/funding_own/funding_borrowings/shares_outstanding/obligation_date` を追加）。
stock-alert-dd / stock-alert-db に担当範囲を照会中。返答があるまで lib/ 配下は触らない。


## ④ holding_ratio の共同保有バグ修正（2026-08-30・オーナー「直して」で着手）
XBRLは保有割合を「保有者ごとのcontext」と「メンバー無しの合算context(FilingDateInstant)」の
両方に持ち、報告書の見出し数値は後者。従来は正規表現で先に見つかった値＝筆頭保有者の1枠を
拾っていた。実測1,101件中656件（60%）がズレ。

- [x] `_aggregate_ratio()` / `_normalize_ratio()` を追加（`_aggregate_or_sum`は`int(float())`で
      0.1975が0に落ちるため流用不可。別関数にした）
- [x] `holding_ratio_prior` も同じ扱いに。`jplvh_cor:`で取れない開示向けに旧正規表現をフォールバックで残す
- [x] tests/test_holding_details.py 12件（うち6件はstock-alert-dbが追加）・test_scan_large_holdings.py 13件 PASS
- [x] `tools/backfill_holding_details.py` 新設（蓄積済み行のXBRL引き直し。`--only-missing`で再開、
      `--dry-run`で差分だけ数える）
- [x] dry-run実測: 2026-08-20以降430件で比率変更212件（49%）・取得失敗0件。
      例 0.39%→5.12% / 0.66%→8.05% / 23.25%→56.26%
- [x] コミット `4929ffc3`
- [~] 全行スイープ実行中（バックフィル完走を待って自動起動する形で予約。
      `logs/edinet_detail_sweep_20260830.log`）
- [ ] スイープ後に `investor_returns_3m` / `investor_return_positions_3m` をリフレッシュ
- [ ] 公開済み記事の数字の是正（`ratioChangePct`・`dealAmount`・タイトル）
      → stock-alert-db が担当。`tools/fix_misreported_blog_articles.py` を拡張する方針。
        **Anthropic APIは使わない**（現行ツールは決定的テンプレートで本文再生成。LLM呼び出し無し）

## ⑤ git分岐（B・オーナー「相談して」）
- [x] 未コミット5本の持ち主を全セッションに照会 → 終了済みセッションの取りこぼしと判明。
      stock-alert-dd が内容確認のうえコミット（`c4e6ff08`）
- [x] こちらの担当分をコミット（`7f75e601` / `4929ffc3`）。作業ツリーをクリアにした
- [ ] stock-alert-dd が `git merge origin/pr284` → push（担当合意済み。着手OKを通知済み）
- [ ] `git branch -u origin/pr284`（upstreamが別ブランチを向いている）
- 履歴の書き換え（rebase/drop/force push）はしない方針で合意（CLAUDE.md §8・2026-07-16の消失事故の前例）
