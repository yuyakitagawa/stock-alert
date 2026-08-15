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
- [ ] **記事化の重要度スコアリングが「保有比率の絶対値」のみ**
  - `web/publish_blog_articles.py` の `build_and_publish()` は `abs(holding_ratio)` でソートし `max_articles` で頭打ちしているだけ（前回比増減・取得金額・投資家カテゴリ・過去登場回数は未加味）。
  - 複合スコア（前回比増減／取得金額／`classify_filer()`のカテゴリ／filer_nameの過去記事本数）を追加し、閾値未満は記事化せずデータ掲載のみに留める案。
  - CLAUDE.md 7節に従い、A/Bまたはバックテスト的な検証（記事数減で流入・CTR悪化しないか）をしてから採用。不採用なら分岐ごと削除。
- [ ] **アイキャッチ画像がタイトルテキスト＋ストック写真のみ**
  - `generate_eyecatch_image()`（`web/publish_blog_articles.py`）は黒帯にタイトル文字を焼き込むだけで、投資家名・保有比率・売買方向・日付などの構造化情報は載っていない。
  - 「ニュースカード型」（例: 🐋 投資家名 / 銘柄名 保有比率X% / 買い増し・新規取得 / 日付）に強化するとDiscoverのカード面での視認性が上がる見込み。
- [ ] **`(ja)/date/[date]` に編集的ハイライトがない**
  - 現状は開示の一覧表示のみ。「本日最も注目すべき1件」をページ冒頭にハイライトすると、advice記事の「今日のクジラ速報」フォーマットに近づく。

## 2. 見送り（advice記事の提案のうち優先度を下げるもの）

- 速報→翌日分析→特集の3段階展開: `investors/[filer]`ページがすでに投資家ごとの保有推移・過去の売買を集約しており、特集相当を兼ねている。専用の「特集記事」を別途量産する優先度は低い。
- 銘柄×投資家の関係ページ: `investors/[filer]`および`stocks/[code]`双方向リンクで概ねカバー済み。専用ページ新設は現状不要。

## 3. 進捗

- [x] 現状調査（実装済み機能の棚卸し）— 2026-08-15
- [x] robots meta修正（(ja)/(en) layout.tsx）— 2026-08-15、googlebot metaに max-image-preview:large / max-snippet:-1 / max-video-preview:-1 を追加。ローカルでレンダリング確認済み
- [ ] 記事化スコアリングの設計・検証
- [ ] アイキャッチのニュースカード化
- [ ] date/[date]の編集ハイライト追加
