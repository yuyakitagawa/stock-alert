# SEO対策: Google Search Console インデックス未登録の解消

対象サイト: kujira-watch.com（正式ソースは `stock-alert/kujira-watch/`）

## 背景（2026-08-12にユーザーから共有されたGSCカバレッジ）

### 1. 検出 - インデックス未登録（149件）
- 状態: Googleがurlの存在は認知しているが、まだクロールしていない。
- 主な原因: サイト開設直後/急激なページ増加でクロールが追いつかない、内部リンクが少なく巡回優先度が低い。
- 対策: サイトマップ(sitemap.xml)をGoogleに送信・更新 / 登録したい重要ページへの内部リンクを他ページから張る。

### 2. クロール済み - インデックス未登録（104件）
- 状態: Googleが巡回したが「インデックス登録しない」と判断。
- 主な原因: コンテンツ量が少ない(薄い)、他ページと内容が酷似、検索意図を満たす品質に未達。
- 対策: 該当URLを特定し記事の質・独自性を高める（加筆・リライト）。価値が低いページ（タグ・検索結果ページ等）なら放置も可。
- 例: https://kujira-watch.com/articles/5f2o0ulbo , https://kujira-watch.com/articles/1f72d8k-mm

## 実行タイミング
ユーザーのClaude/AI利用クレジットが木曜日にリセットされるため、着手はユーザーからの声掛け後（[[feedback_seo_indexing_wait]] 参照）。それまでは作業しない。

## ユーザー指定タスク（2026-08-12追加）

### 1. XMLサイトマップの確認と更新
- 新着記事が生成された際、自動で `sitemap.xml` に追加・更新される仕組みを構築する。
- Google Search Console の「サイトマップ」メニューから送信・自動取得されているか確認する。

### 2. 内部リンク構造の強化
- 「注目の記事」「最新の報告書一覧」など重要ページへのリンクを、トップページ／サイドバーなど浅い階層に配置する。

### 3. 「クロール済み - インデックス未登録」ページの精査
- Search Console上で該当URL一覧を抽出する。
- 各URLが以下に該当しないか確認する。
  - 情報量が極端に少ない記事
  - 定型文が大半を占める記事

## TODO（着手後の順番）
- [x] sitemap.xml の現状仕組みを確認 → `src/app/sitemap.ts`は`export const dynamic = "force-dynamic"`でmicroCMSから毎リクエスト時に生成しており、新着記事は自動的にsitemapへ反映される。追加実装は不要と判断。
  - 検証(2026-08-14): 本番`https://kujira-watch.com/sitemap.xml`を直接curlで取得し確認。全1713URL中、記事URLは359件、ユーザー提示の2例（`/articles/5f2o0ulbo`, `/articles/1f72d8k-mm`）とも含まれていることを`<loc>`タグで確認済み（WebFetchツールの要約では見落とされていたが、生XMLでは実在）。sitemap自体は正常に機能している。
- [x] トップページ／サイドバー等、浅い階層への重要ページリンク配置を確認 → `Header.tsx`で「今週の動き」「大口投資家一覧」「株式銘柄一覧」を全ページ共通ヘッダーに、カテゴリ13種も折りたたみnavで既に配置済み。記事詳細ページも「関連記事(同カテゴリ)」「銘柄ページへのリンク」「投資家名の自動リンク化」「パンくず」を実装済み。追加実装は不要と判断。
- [x] 「クロール済み-未登録」104件の主因を特定 → ユーザー提示の例 https://kujira-watch.com/articles/1f72d8k-mm を分析した結果、`web/publish_blog_articles.py`の記事生成プロンプトが本文500〜700字固定・EDINET開示の限られた事実のみを根拠にしており、定型文（下落リスク水準の言及・投資家分類の説明等）の比率が高く同工異曲の記事になりやすい構成だった。
- [x] 対策として、既に計算済みだが本文生成に使っていなかった「保有比率の変化幅(ratio_change_pct)」を新たな事実としてプロンプトに追加投入し、本文目標を650〜900字に微増（[web/publish_blog_articles.py](../web/publish_blog_articles.py)、テスト2件追加、README更新、[[feedback_abbreviations]]遵守）。字数を機械的に増やすのではなく、既存データで裏付けられる事実を1つ追加する形にして創作リスクを回避。2026-08-14実装、テスト63件全成功（同ファイルを並行編集していた別セッションの英訳(titleEn/bodyEn)生成・重複判定強化・PATCH更新化と合流済み）。今後生成される新規記事から反映（過去公開済みの104件は未対応）。
- [ ] Google Search Console「サイトマップ」メニューで送信状況・自動取得状況を確認（ユーザー側の手動作業。Search Console UIの操作はエージェントから実行不可）
- [x] 既存公開済み記事のリライト用スクリプト作成 → [tools/rewrite_thin_blog_articles.py](../tools/rewrite_thin_blog_articles.py)。stockCode+dealDateからedinet_large_holdingsを逆引きしてfact_sheetを再構築し、現行の`generate_article_body()`（650〜900字・保有比率変化幅つき）で本文のみ再生成、既存タイトル・株価チャートは据え置く。`--dry-run`で確認可能。
- [x] ユーザー判断: 104件全部のCSVエクスポートは「無理」とのことで、ユーザーがSearch Consoleから手動で貼り付けた9件（URL末尾が途中で切れていた1件を除く）に絞って実行することで合意。
- [x] 2026-08-14実行: `tools/rewrite_thin_blog_articles.py --ids ...`で9件を処理。
  - 成功（本文リライト・microCMS反映済み）: `5f2o0ulbo`, `8hdil4kba2u`, `1f72d8k-mm`, `c5mjetpi6v34`, `bu_95xpig3j`, `5ptvefs-wqr`, `84j0pf0zdpu`（7件）
  - スキップ（複数提出者で一意特定不能、手動確認が必要）: `a1ny77_tlsu`（4569, 2026-08-05, 候補: 荻原年/荻原明/荻原万里子）
  - スキップ（`edinet_large_holdings`に該当データなし）: `5z7hnve3_z`
  - 実行中にmicroCMS APIキーの権限変更を発見: PUTが`Content is already exists. If you want update, please use PATCH request.`で拒否されるようになっていた（2026-08-14時点）。`web/publish_blog_articles.py`の`update_article()`を`_put_once`→`_patch_once`（PUT→PATCH）に切替、`tools/reclassify_blog_articles.py`もこの共有関数を使うため恩恵を受ける。テスト2件更新、README更新。
  - 事故と復旧: 動作確認のため`5f2o0ulbo`にPATCHで一時テスト文字列を書き込んでしまい、直後に元の本文へ復元済み（実害なし、公開サイトに一時的にテスト文言が出た可能性は数分未満）。
- [x] ユーザー判断（2026-08-14）: 残り95件は**そのまま放置**。今後生成される新規記事だけ改善プロンプト（650〜900字・保有比率変化幅つき）の恩恵を受ければよく、既存104件の追加バックフィルは行わない。`tools/rewrite_thin_blog_articles.py`は将来気が変わった場合のために残置（`--ids`で個別指定して再開可能）。
- [x] 本タスクはここで区切り。対応後の検証は[[feedback_seo_indexing_wait]]の通りインデックス反映確認は数日〜数週間待つ前提（急かさない）。

## 2026-08-18 追記: GSC検証失敗通知とsitemap分割

- GSC「インデックス登録エラーを完全に修正できませんでした」通知を調査。対象URL（/stocks/・/en/stocks/・/date/・/investors/）は全て200・index,follow・canonical正常で、技術的エラーなし。Googleの品質判断による未登録であり、検証失敗自体は実害なし（再検証も不要）。
- 調査中に判明した実問題2つ:
  1. sitemap.xml の応答が毎回9〜11秒（1回は20秒タイムアウト）。原因は`getTranslatedArticlesForSitemap`が英語本文bodyEn全文を毎リクエスト全件取得していたこと。
  2. 2026-08-15の投資家全件掲載修正（PostgREST 1000行上限回避）でURL数が1,713→6,196に急増。壊れてはいないが「インデックス未登録」件数は今後増えて見える（母数増のため異常ではない）。
- [x] 対応（2026-08-18）: `/sitemap.xml`をsitemapindex化し、`generateSitemaps`で6分割（pages/stocks/dates/investors/articles/articles-en）。データ取得は`unstable_cache`1時間で応答高速化。`app/sitemap.xml/route.ts`はmetadata予約名と衝突するため実体は`sitemap-index.xml`+rewrite。ローカル検証で合計6,196URL（分割前と一致）・未知ID404・hreflang出力を確認。
- GSC側の追加作業は不要（/sitemap.xmlのURLは不変のため再送信不要。Googleはsitemapindexを自動で辿る）。
- 形式の変遷（同日内）: 一度「従来の単一urlset形式が良い」で分割を撤回→最終的に「sitemapindex形式で良い。ただしインデックスも子と同じ1タグ1行のXML記述にする」で確定。sitemapindexのエントリを`<sitemap>`/`<loc>`/`</sitemap>`各行分割に整形して再デプロイ。
