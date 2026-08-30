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
