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

## 2026-08-18 追記2: 「検出 - インデックス未登録」の再調査と対策実装

ユーザーからGSCのスクリーンショット（/articles/配下10件が「検出 - インデックス未登録」）を受けて再調査。

### 実測した現状（全911記事をmicroCMS APIから取得して計測）
| 項目 | 実測値 |
|---|---|
| sitemap総URL | 6,196（8/14は1,713 → 4日で3.6倍） |
| 記事数 | 911（8/14は359。8/16だけで416本作成） |
| 本文の実文字数 | 中央値445字・最大648字・**911本すべて800字未満**（プロンプトの指示は650〜900字） |
| 推定取得金額1億円未満の記事 | 78本（0億円が6本） |
| 開示1件だけの投資家ページ | 2,950件中995件 |
| 記事のTTFB | 1.8〜2.8秒（`revalidate = 60`でクローラーがほぼ毎回再生成に当たる） |
| 全記事のupdatedAt | 8/14〜8/18に集中（attentionScore削除等の一括バッチで書き換わりlastmodが無意味化） |

サンプル4URLはいずれもHTTP 200・`index, follow`・canonical正常・sitemap収録済みで技術的エラーは無し。
原因は「URLの急増＋テンプレートの低品質判定によるクロール枠の枯渇」と判断。

### 実装した対策（2026-08-18）
- [x] **lastmodを取引日(EDINET開示日)基準に統一** — `src/app/sitemap.ts`の記事・銘柄・日別・月別・
      カテゴリ・ハブの`<lastmod>`を`updatedAt`から`dealDate`へ変更。記事ページのJSON-LD
      `datePublished`/`dateModified`とOGP`publishedTime`/`modifiedTime`も`dealDate`に一本化し、
      一括バッチで日付が動かないようにした（ja/en両方）。
- [x] **記事化の足切り** — `lib/articleIndexability.ts`（表示側）と`web/publish_blog_articles.py`の
      `is_worth_publishing()`（生成側）に「推定金額3億円以上 **または** 保有比率の変化1pt以上」を実装。
      しきい値は両者で必ず揃える（ずれると「sitemapに載っているのにnoindex」になる）。
      既存記事もこの条件で描画時に`noindex, follow`＋sitemap除外（記事URL 911→782）。
      生成側の足切りはClaude呼び出しの前に置いたのでAPI費用も減る。
- [x] **薄い投資家ページをsitemapから除外** — 開示1件だけの投資家を除外（2,950→1,952）。
      ページ自体は残し、内部リンクからは辿れる。
- [x] **本文の文字数を実測して不足なら再生成** — `generate_article_body_checked()`を追加
      （下限650字、1回だけ再生成、2回目も不足なら長い方を採用）。指示だけでは守られていなかったため。
- [x] **記事ページの定型文を削減** — 「分類の説明文」「提出期限5営業日のズレ＋免責の長文(約150字)」
      「出典ボックスの免責の重複」「投資家ブロックの定型説明」を削除・圧縮し、`/faq/basics`と`/about`への
      リンクに寄せた。可視文字数 2,279字 → 1,957字（定型文の割合を約320字ぶん削減）。
- [x] **revalidateを60秒→86400秒** — 記事本文は公開後ほぼ不変。TTFB悪化の主因を解消。
- [x] **titleテンプレートを「記事名｜サイト名」に反転** — 全記事の`<title>`が同じ11文字で始まる状態を解消（ja/en）。

### 未対応（判断待ち・API制限待ち）
- 英語版911本の扱い（機械翻訳の全記事横展開を続けるか、価値の高い記事に絞るか）はユーザー判断待ち。
- 既存記事の本文リライト（450字→650字以上）は`ANTHROPIC_API_KEY`が2026-09-01まで停止中のため実行不可。
  復帰後に`tools/rewrite_thin_blog_articles.py`で実行する。
- 同一銘柄の連投記事の統合（402Aに17本、8783に10本など12銘柄）は既存URLの整理を伴うため未着手。
- `kujira-watch/README.md`への反映は、並行セッションが同ファイルを編集中のため保留。

### 追加で判明した最大のボトルネック: 全ページが動的レンダリングだった（同日中に修正）
デプロイ後にレスポンスヘッダーを測ったところ、`revalidate`を延ばしても記事ページは
`x-vercel-cache: MISS` / `cache-control: no-store` のままで、毎リクエストがサーバー実行だった。
原因は2つ:
1. supabase-jsのfetchにキャッシュ指定が無く、Next.js 15以降のfetch既定(no-store)により
   Supabaseを読むページ全体が動的扱いになっていた（microCMS側は`next: { revalidate }`済みで対象外）。
   → `lib/investors.ts`・`lib/priceReturns.ts`の読み取りを`unstable_cache`(1時間)に載せた。
2. Next 16では`generateStaticParams`の無い動的セグメントがISRにならずリクエスト毎のSSRになる
   （`generateStaticParams`を持つ`/category`・`/faq`だけがPRERENDERだった）。
   → `/articles/[id]`(直近200)・`/stocks/[code]`(上位100)・`/date/[date]`(直近60)・
   `/investors/[filer]`(上位100)に`generateStaticParams`を追加。一部でも事前生成すると
   ルート全体がISR扱いになり、事前生成外のパラメータも2回目以降はCDNから返る。

本番実測（対策前 → 対策後）:
| ページ | TTFB前 | TTFB後 | キャッシュ |
|---|---|---|---|
| /articles/[id] | 1.8〜2.8秒 | 0.17〜0.68秒 | HIT |
| /stocks/[code] | 1.5秒 | 0.37秒 | HIT |
| /date/[date] | 1.4秒 | 0.17秒 | HIT/PRERENDER |

注意: 事前生成のパスはそのままファイル名になるため、全角の長い提出者名で
ENAMETOOLONGとなり本番ビルドが3回失敗した（並行セッションが`MAX_PRERENDER_SEGMENT_LEN`で修正済み）。

## 2026-08-18 追記3: 英語版の絞り込みと低価値記事の削除

- GSC実測（直近3か月・ページ`/en/`）: 表示33回・クリック0・平均掲載順位23.3。需要は極小だが
  8/14頃から表示が立ち上がっており完全にゼロではない。ユーザー判断で「全記事の機械翻訳展開は
  やめ、価値の高い記事だけ英訳する」方針を採用。
- [x] 英語版の基準を`isIndexableEnArticle()`として実装（`lib/articleIndexability.ts`）:
  **アクティビスト分類 or 推定金額100億円以上 or (保有比率変化5pt以上=新規5%取得相当 かつ 金額20億円以上)**。
  英語サイトマップと`/en/articles/[id]`のnoindex、ja側hreflangの有無をこの基準で揃えた
  （noindexページを代替言語として宣言しないため）。対象は911→約335本。
- [x] 基準未満の既存記事129本をmicroCMSから削除（ユーザー判断で noindex 据え置きではなく削除）。
  `tools/delete_low_value_blog_articles.py`（`--delete`指定時のみ実行、対象の全フィールドを
  `logs/deleted_low_value_articles_<日時>.json`へバックアップしてから削除）。記事913→786本。
