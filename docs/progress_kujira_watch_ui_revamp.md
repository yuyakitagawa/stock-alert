# kujira-watch UI改善 進捗

最終更新: 2026-08-14
着手: 2026-08-14〜

## 対象
`kujira-watch/`（本番: kujira-watch.com）のUI改善。2026-08-11にユーザーから一括で要望を受領。

---

## TOPページ
- [x] 見出し「今日の注目取引」を追加（sr-onlyだったh1を可視化）
- [x] ヘッダーの絞り込み（フィルター）の位置を修正: Header.tsx(全ページ共通sticky)から`CategoryFilterDetails`コンポーネントに切り出し、見出し直下・日付一覧の直前に移動

## 今週の動き
- [x] 注目の取引を掲載（金額規模上位3件を`FeaturedArticleCard`で表示。あわせて「今週のポイント」で買い/売り件数・金額、投資家分類別・銘柄別の上位内訳を集計表示）
- [x] 各取引の下に「この日の記事を見る」リンクを追加（全記事カード表示は廃止し、取引日ごとに件数・金額付きで`/date/[date]`へのリンクに集約）

## 記事ページ
- [x] 銘柄｜取引日｜金額規模｜[取引企業] という形式で「取引企業」列を新規追加
  - microCMS「articles」APIに`filerName`フィールド追加済み（ユーザー対応）。
  - `web/publish_blog_articles.py`のpayloadに`filerName: filer_name`を追加、`Article`型に`filerName?: string`を追加、articles/[id]/page.tsxのdlに「取引企業」列（`/investors/[filer]`へのリンク）を追加。`filerName`が無い記事（既存公開済み分）は列自体を出さず3列のまま表示される（実機確認済み、崩れなし）。
  - 次回の日次自動投稿（`daily_alert.yml`）分から新規記事に反映される。既存記事には遡って入らない。

## 個別銘柄ページ
- [x] 大量保有報告書の提出投資家を1件ずつ改行して表示（ulをflex-wrapからspace-y-2に変更）

## investorsページ（投資家個別ページ）
- [x] 新見出し「主な保有銘柄」セクションを新設（holdingsをissuerCodeで重複排除、保有比率付きで表示）
- [x] 取引履歴セクションの見出しを「最近の取引」に変更

## 投資家一覧ページ
- [x] フィルターを見出しの下に配置（カテゴリ別・件数付き、searchParams`?category=`でSSRフィルタ）
- [x] 見出しに件数を `(N件)` 形式で表示

## 銘柄一覧ページ
- [x] 独自のフィルターを追加（業種/セクターで絞り込み。`jpx_stock_list.sector`をSupabaseから取得し件数付きで表示。searchParams`?sector=`）

---

## 未解決・引き継ぎ事項
- 記事ページの「取引企業」列: 上記の通りmicroCMSダッシュボードでの手動フィールド追加待ち。
- 作業中、別セッション（詳細不明）が同一リポジトリ内で`web/publish_blog_articles.py`・`lib/jpx_market_data.py`・テスト等を並行編集していた形跡あり。今回の変更範囲とは重ならないが、次回作業時は`git status`で衝突がないか確認すること。
