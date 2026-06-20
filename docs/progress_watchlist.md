# 進捗: 値上げ力ウォッチリスト（toC独占×インフレ耐性）

オーナー方針: 「値上げできる会社＝シェア独占toC銘柄」を将来の買い候補として保存し、Webにも掲載する。

## 対象11銘柄（オーナー選定）
カゴメ(2811)/カルビー(2229)/日清食品HD(2897)/アサヒGHD(2502)/ユニ・チャーム(8113)/
花王(4452)/資生堂(4911)/ロート製薬(4527)/久光製薬(4530)/ピジョン(7956)/シマノ(7309)

## 設計
- 静的なテーマ別キュレーションリスト → Supabaseテーブル新設はせず、CSV(正本) + フロントTS定数で持つ。
- Webページはライブの `web_rankings`（最新net/シグナル/終値）と銘柄コードで突き合わせて表示。
- シェア率・海外比率は概算（公知ベース）。ページ上に「概算・要検証」を明示。

## ステップ
- [x] `data/pricing_power_watchlist.csv` 正本作成
- [x] `frontend/lib/watchlist.ts` 型付き定数
- [x] `frontend/app/watchlist/page.tsx` 新ページ（ライブnet突き合わせ）
- [x] `frontend/components/Navbar.tsx` ナビ追加 + i18n
- [x] README 更新
- [x] ビルド/プレビュー確認（tsc OK、`/watchlist` HTTP200・11銘柄・当日ランキング突き合わせ確認）
- [x] commit

## 追加: お得度（高値からの下落率）＋ PER/PBR（オーナー要望）
- お得度 = 52週高値からの下落率（−30%↓=🔥大お得 / −20%↓=お得 / −10%↓=やや安 / それ以外=高値圏）。
- PER/PBR = Yahoo `summaryDetail.trailingPE` / `defaultKeyStatistics.priceToBook`。
- `frontend/lib/data.ts`: `fetchWatchMetricsMap`（認証1回共有 + 全fetchにAbortSignalタイムアウト）。
- chart APIは `range=5d` の `meta.fiftyTwoWeekHigh` を使いペイロード最小化。
- 検証: Yahoo直算出とページのバッジ分布が完全一致（お得4/やや安5/高値圏2）。
- 注意: サンドボックスは fc.yahoo.com 404 で PER/PBR が `—`。本番Vercelは既存 /stocks が同じ認証を使えているため要確認。

