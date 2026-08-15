# Discover/SEO改善計画（kujira-watch）

前提: ChatGPTの助言2本（Discover戦略・SEO構造戦略）をもとに、現状のkujira-watch実装を確認し、
「すでにある機能」と「本当のギャップ」を切り分けた上での実行計画。

## 0. 現状確認結果（advice記事の提案のうち実装済みのもの）

助言の多くはすでに実装済み。重複作業を避けるため明記する。

- 銘柄ページ `(ja)/stocks/[code]`、投資家ページ `(ja)/investors/[filer]`
- ランキング `(ja)/ranking`、週次まとめ `(ja)/weekly`、カテゴリ `(ja)/category/[category]`
- 日付アーカイブ `(ja)/date/[date]`（「◯月◯日の大口投資家の動き」＝実質「今日のクジラ」速報ページ）
- 構造化データ（BlogPosting等、`(ja)`配下ほぼ全ページ）、OGP、RSS `feed.xml`
- 運営者情報・データソース・免責事項ページ `(ja)/about`
- 英語版一式 `(en)/en/*`

## 1. ギャップ（優先度順）

- [ ] **robots meta に `max-image-preview: large` が未設定**
  - `kujira-watch/src/app/(ja)/layout.tsx` と `(en)/en/layout.tsx` の `metadata.robots` に `googleBot.max-image-preview: 'large'` を追加するだけ。低コスト・即効性あり。
- [x] ~~記事化の重要度スコアリングが「保有比率の絶対値」のみ」~~ → **見送りに変更**（2026-08-15再検討）
  - 調査の結果、`build_and_publish()`は`max_articles`未指定（CI実運用時のデフォルト）だと該当する開示を**全件**記事化する仕様と判明。これは`docs/progress_stocks_thin_content.md`が前提とする「/stocks/[code]全2978件」のSEOデータベース戦略（advice記事2本目が推奨する「1,000銘柄×1ページ」路線）と整合しており、意図的な設計。
  - 記事化自体を閾値でブロックすると、この既定路線のページ数・データ網羅性を減らす方向に働き矛盾するため見送り。
  - 一方「どれを目立たせるか」は`getFeaturedArticles()`（date優先→金額降順、homepage/weeklyで使用）で既に実装済みだったため、対応不要と判断。
- [x] **アイキャッチ画像がタイトルテキスト＋ストック写真のみ** — 2026-08-15対応
  - `generate_eyecatch_image()`（`web/publish_blog_articles.py`）をニュースカード型（投資家名／銘柄名+保有比率／売買方向バッジ+開示日）に変更。`build_eyecatch_for_article()`のシグネチャも`(category, card)`に変更し、呼び出し元・テスト（`tests/test_publish_blog_articles.py`）を追随修正。テスト63件通過確認済み。
- [x] **`(ja)/date/[date]` に編集的ハイライトがない** — 2026-08-15対応
  - `getArticlesByDealDate()`が既に`-dealAmount`順で返す点を利用し、`contents[0]`をホームページ/週次まとめと同じ`FeaturedArticleCard`でページ冒頭にハイライト、残りを既存グリッドに表示するよう変更。`tsc --noEmit`/`eslint`エラー無し、ローカル(http://localhost:3002/date/2026-08-14)で目視確認済み（英語版`/en`には対応する日付アーカイブページが存在しないため対応不要）。

## 2. 見送り（advice記事の提案のうち優先度を下げるもの）

- 速報→翌日分析→特集の3段階展開: `investors/[filer]`ページがすでに投資家ごとの保有推移・過去の売買を集約しており、特集相当を兼ねている。専用の「特集記事」を別途量産する優先度は低い。
- 銘柄×投資家の関係ページ: `investors/[filer]`および`stocks/[code]`双方向リンクで概ねカバー済み。専用ページ新設は現状不要。

## 3. 進捗

- [x] 現状調査（実装済み機能の棚卸し）— 2026-08-15
- [x] robots meta修正（(ja)/(en) layout.tsx）— 2026-08-15、googlebot metaに max-image-preview:large / max-snippet:-1 / max-video-preview:-1 を追加。ローカルでレンダリング確認済み
- [x] 記事化スコアリングの設計・検証 — 2026-08-15、SEOデータベース戦略と矛盾するため見送りと結論（上記参照）
- [x] アイキャッチのニュースカード化 — 2026-08-15
- [x] date/[date]の編集ハイライト追加 — 2026-08-15
- [ ] 本番デプロイ・反映確認（push後）
