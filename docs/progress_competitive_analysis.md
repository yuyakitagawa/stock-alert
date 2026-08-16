# kujira-watch 競合分析と機能ギャップ解消（2026-08-16）

## 競合分析サマリー

| 機能 | IRBANK | M&A Online | 青山乃木坂 | アクティビストメディア | kujira-watch |
|---|---|---|---|---|---|
| 全開示の速報一覧テーブル | ○ | ○ | ○ | △ | **×（最大のギャップ）** |
| 新規/変更/訂正の種別表示 | ○ | ○ | ○ | ○ | × |
| EDINET原文（PDF）リンク | △ | ○ | ○ | △ | × |
| 保有比率の増減表示 | ○ | ○ | ○ | ○ | ○（投資家/銘柄ページ） |
| 投資家軸の横断トラッキング | △ | × | ○ | △ | ○（差別化点） |
| 投資家の勝率/リターン実績 | × | × | × | × | ○（独自） |
| AI解説記事・注目度スコア | × | 記事あり | × | 解説あり | ○（独自） |
| ウォッチリスト/アラート | × | メルマガ | 会員限定 | × | ×（認証はスコープ外） |

- IRBANK: https://irbank.net/share （報告義務発生日別の全件一覧・増減表示）
- M&A Online: https://maonline.jp/db/shareholding_reports （提出日/業種/保有者フィルタ・PDF取得・速報メルマガ）
- 青山乃木坂パートナーズ: https://aoyama-nogizaka.com/activist-dashboard （注目投資家フィルタ・トレンドグラフ・会員向けアラート/CSV）
- アクティビストメディア: https://activist-media.net/fund-holdings （アクティビスト保有一覧）
- 株探: 5%ルール速報の記事配信

## 結論（取り入れるもの）

**最大のギャップ = 「記事化された開示しか見えない」こと。** 競合3社は全開示の
速報一覧を持つが、kujira-watchはAI記事化された一部の開示しか表示していない。
Supabase `edinet_large_holdings` に全開示19,799件（2025-06-18〜、doc_id付き）が
既にあるため、フロント追加のみで実装可能。

採用: **`/disclosures` 開示速報ページ新設**（+ EDINET原文リンク・種別バッジ）
- 提出日ごとにグループ化した全開示一覧（IRBANK方式）
- 種別（新規=350/変更=360/訂正=doc_descriptionで判定）フィルタ（M&A Online方式）
- 保有比率 前回→今回 + 増減pt（既存 `RatioTransition` を再利用）
- EDINET原文PDFへの直リンク `https://disclosure2dl.edinet-fsa.go.jp/searchdocument/pdf/{doc_id}.pdf`（200確認済み）
- 銘柄（記事がある銘柄のみリンク、/trendingと同じ規律）・投資家への内部リンク
- `?page=` ページング（100件/ページ、/investorsと同じ素のリンク方式）

見送り（理由つき）:
- ウォッチリスト・アラート・CSV: 認証/会員機能はREADMEでスコープ外と明記済み
- アクティビスト現在保有の横断一覧: `/investors?category=アクティビスト` で部分カバー。次候補
- 月別トレンドグラフ: `/trending` の期間比較で部分カバー

## 実装チェックリスト

- [x] 競合リサーチ（IRBANK / M&A Online / 青山乃木坂 / アクティビストメディア）
- [x] Supabaseデータ確認（edinet_large_holdings 19,799件・doc_id・EDINET PDF直リンク200確認）
- [x] `src/lib/disclosures.ts` 新設（ページング取得+件数、unstable_cache）
- [x] `src/app/(ja)/disclosures/page.tsx` 新設（種別フィルタは new/change/correction の3種）
- [x] Header ナビに「開示速報」追加
- [x] sitemap.ts / llms.txt に /disclosures 追加
- [x] kujira-watch/README.md 更新（同一コミット）
- [x] `npm run build` + tsc + lint + Pythonテスト全PASS + `next start`でローカル実表示確認
      （すべて19,799/新規2,743/変更13,927/訂正3,129件・PDF リンク・ページング動作確認済み）
- [x] 【追加修正】既存バグ: `docTypeLabel`が変更報告書を「大量保有報告書」、訂正報告書を
      「変更報告書」と誤表示（`doc_type_code` 350は新規・変更両方、360は訂正だった）。
      `/investors/[filer]`・`/stocks/[code]`をdoc_description判定の`disclosureDocLabel`に差し替え。
      Python側（web/publish_blog_articles.py等）の同根バグは別タスクとしてチップ化済み
- [x] コミット & push（4f9807da。並行セッションのmain更新をrebaseで取り込み済み）
- [x] 本番デプロイ確認: https://kujira-watch.com/disclosures がHTTP 200、
      件数（すべて19,799/新規2,743/変更13,927/訂正3,129）・EDINET PDFリンク・
      ?type=correction フィルタの動作を本番で確認（2026-08-16）

## 第2弾: アクティビスト保有銘柄一覧（2026-08-16 続き）

- [x] `/activists` 新設: アクティビスト（51ファンド）のEDINET最新開示を「現在の保有」として横断集計
      （41ファンド・283銘柄・296件。比率5%未満は報告義務外のため除外）。
      「複数のアクティビストが保有する銘柄」＋「ファンド別の保有銘柄」の2段構成。
      `src/lib/activists.ts`（.in()15件分割クエリ、unstable_cache 1時間）
- [x] ヘッダー「アクティビスト保有」・sitemap・llms.txt・README・`/ranking/activist`からの相互リンク
- [x] tsc / lint / build / next startでの実データ表示確認
- [x] コミット & push、本番 https://kujira-watch.com/activists のHTTP 200確認

## 第3弾: 月別開示件数トレンドグラフ（2026-08-16 続き）

- [x] `/trending`に「月別の開示件数トレンド」棒グラフを追加（青山乃木坂のトレンドグラフに対応）。
      インラインSVG自前描画・単一系列ネイビー1色・当月は薄色で集計中表示・
      `<details>`の表で全数値も閲覧可（`MonthlyDisclosureTrend.tsx`＋`getMonthlyDisclosureCounts()`）
- [x] README更新・build・実表示確認・コミット & push・本番確認

**→ 競合分析で採用した3項目（開示速報・アクティビスト保有一覧・月別トレンドグラフ）すべて完了。**
