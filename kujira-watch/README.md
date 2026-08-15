# 大口投資家の監視ブログ

EDINET大量保有報告書などの公開情報をもとに、機関投資家・インサイダー・自社株買いなど
「クジラ」（相場を動かすほどの資金力を持つ大口投資家の俗称）の動きを監視・解説するブログ。
ブランド名は「大口投資家の監視ブログ」、ドメインは`kujira-watch.com`（クジラのイメージで確保）。
SEO/AIO（AI Overview・LLM引用）対策済み。

デプロイ先: https://kujira-watch.com/ （旧URL: https://stock-alert-lyart.vercel.app/ 。
進捗はリポジトリルートの `docs/progress_blog_seo_aio.md` を参照）

## スタック

- Next.js 16 (App Router) + TypeScript
- MUI (Material UI) v9 + Emotion（`@mui/material-nextjs`でApp Router用SSR配線。テーマは`src/theme.ts`、ブランドカラー[紺/青/金]を反映）
- Tailwind CSS v4（ページレイアウト・グリッド・`@tailwindcss/typography`でのリッチテキスト本文装飾を担当。コンポーネント単位のスタイルはMUI側）
- microCMS（`microcms-js-sdk`）
- Supabase（`@supabase/supabase-js`。フッターの累計訪問者カウンター用。トレーディングシステム側と同じプロジェクトの`blog_visit_counter`テーブル+`increment_blog_visit_counter` RPC。加えて`/stocks/[code]`の会社情報カードが同プロジェクトの`jpx_stock_list`・`gen_rankings`テーブルを、`/investors`・`/investors/[filer]`が`edinet_large_holdings`・`edinet_filer_classification`・集計ビュー`edinet_filer_summary`を、`/ranking`が`filer_win_rate`テーブルを参照）
- Vercel想定（ISR: `revalidate = 60`、`@vercel/analytics`でアクセス計測、`@vercel/speed-insights`でCore Web Vitals計測）

## セットアップ

```bash
npm install
cp .env.local.example .env.local
```

`.env.local` に microCMS サービスの値と、SEO用のサイトURL/サイト名を設定する。

```
MICROCMS_SERVICE_DOMAIN=xxxx
MICROCMS_API_KEY=xxxx
NEXT_PUBLIC_SITE_URL=https://kujira-watch.com
NEXT_PUBLIC_SITE_NAME=大口投資家の監視ブログ
```

`NEXT_PUBLIC_SITE_URL` / `NEXT_PUBLIC_SITE_NAME` は独自ドメイン・ブランド名が決まった際に
値を差し替えるだけで、metadata・OGP・構造化データ・サイトマップの全ページに反映される
（未設定時は現行のVercelドメイン・現行ブランド名にフォールバックする）。

```bash
npm run dev
```

## microCMS側の前提（APIスキーマ: `articles`）

管理画面のGUIで以下のフィールドを持つ `articles` エンドポイント（リスト形式）を作成しておくこと。

| フィールドID | 表示名 | 型 | 必須 |
|---|---|---|---|
| title | タイトル | テキスト | ○ |
| body | 本文 | リッチエディタ | ○ |
| stockName | 銘柄名 | テキスト | ○ |
| stockCode | 証券コード | テキスト | ○ |
| dealType | 投資家分類 | セレクトフィールド（個人／創業家の資産管理会社／公益/一般財団法人／プライムブローカー／アクティビスト／VC／PE・メザニンファンド／独立系ブティックAM／国内アセットマネジメント／外資系伝統運用会社／日系証券銀行／事業会社／その他） | ○ |
| dealDate | 取引日 | 日付 | ○ |
| dealAmount | 金額規模（億円） | 数値 | ○ |
| sourceUrl | 出典URL | テキスト | △ |
| tags | タグ | テキスト（カンマ区切り。売り方向の記事には`"売り"`を含める） | △ |
| eyecatch | アイキャッチ画像 | 画像 | △ |
| filerName | 取引企業（提出者名） | テキスト | △（2026-08-15にスキーマ作成済み。下記の注意書きを参照） |
| ratioChangePct | 保有比率の変化幅（ポイント、売りは負値） | 数値 | △（2026-08-15追加。記事詳細のファクトボックス「前回比」に使用） |
| attentionScore | クジラ注目度（0-100） | 数値 | △（買い記事のみ。`lib/attention_score.py`が算出） |
| attentionReasons | 注目度の理由 | テキスト（カンマ区切り、`tags`と同じ運用） | △ |

> **注意（2026-08-15更新）**: `filerName` は長らくmicroCMSスキーマに存在せず（microCMSは
> スキーマに無いフィールドを黙って捨てるため、`web/publish_blog_articles.py`が送っても
> 全記事で値が空だった）、フロント側は提出者名をEDINET開示（Supabase `edinet_large_holdings`）と
> 「銘柄コード×取引日」で突き合わせて補ってきた（`src/lib/investors.ts` の
> `getFilerNamesByStockAndDate()`。同じ銘柄・同じ日に複数の提出者がいる開示＝2026年8月実測で
> 全体の約7%は、誤った帰属を避けるため除外する）。2026-08-15に`filerName`・`ratioChangePct`の
> 両フィールドが管理画面で作成されたため、**以降の新規記事はCMSの値がそのまま入る**
> （フロント側はCMSの値を優先し、値が空の既存記事のみ突合フォールバックを使う実装）。

### クジラ注目度（attentionScore）

「保有比率20%超」「急増」「アクティビスト」等の見た目のインパクトが強い要素は、実際の
株価リターンとは無関係か弱い負の相関しかない（`lib/attention_score.py`のdocstring参照。
tools/filer_win_rate.pyと同じ手法で買い開示4,232件を検証、2026-08-15）ことが判明したため、
直感ではなく実績データで較正した「スコアカード」方式を採用している。保有比率・保有比率の
変化幅・推定取引金額・投資家分類（13分類）をそれぞれ実績の分位ビン平均リターン（または
縮小推定した平均リターン）に変換し、Ridge回帰の重みで線形結合、学習時の予測値分布の
パーセンタイルへ変換して0〜100点にスケーリングする。「過去の取得回数」は検証したが有意な
関係が見られなかったためスコアに含めない。売り方向の記事（スコアカードが買い方向の実績のみで
較正済みのため対象外）・生成タイミングが古い記事は`attentionScore`が未設定になる。

- 生成: `web/publish_blog_articles.py`の`build_and_publish()`が新規記事の投稿時に算出。
- 遡及付与: `tools/backfill_attention_score.py`が既存記事をEDINET開示（Supabase
  `edinet_large_holdings`）と突き合わせて一括計算・PATCH更新する（`--dry-run`で確認可）。
- 表示: `src/components/AttentionScoreBadge.tsx`（一覧・カード用の星付きコンパクトバッジ、
  `FeaturedArticleCard`/`ArticleCard`で使用）、`src/components/AttentionScorePanel.tsx`
  （記事詳細ページ用の大きいスコア＋星＋理由リストのパネル。理由は日本語のみ生成するため
  EN版ではスコア・星のみ表示）。
- 「注目①②③」（`getFeaturedArticles()`、TOP/週次/月別で使用）は、従来の
  「日付優先→同日内は金額降順」の2軸ソートに代えて、直近プール内でこの`attentionScore`が
  高い順に選ぶ（未算出はプール内で最下位扱い、同点は`dealAmount`で比較）。
  `web/publish_blog_articles.py`の`get_featured_article_ids()`（X投稿の対象記事選定）も
  同じロジックで同期させている。

## ページ構成

| パス | 内容 |
|---|---|
| `/` | 記事一覧。見出し「今日の注目取引」の直下に「今日のクジラ」サマリー（`src/components/TodayWhaleSummary.tsx`。最新の取引日・その日の開示件数・推定金額・買い/売り件数を大きく出し、`/date/[date]`へリンクする。毎日更新されていることが一目で分かるようにするための鮮度表示で、件数・金額は初回取得30件で切れないよう`getArticlesByDealDate()`で取り直す）を置き、続けてカテゴリ絞り込み（`src/components/CategoryFilterDetails.tsx`、MUI Accordionで開閉、`/category/[category]`へのボタン）を配置し、その下に金額規模上位の記事をヒーロー枠でピックアップ表示（新着順）。初回30件をサーバー側でレンダリングし、下端までスクロールすると自動で次の10件を読み込むオートスクロール方式 |
| `/weekly` | 大口投資家の動きまとめ（直近7日間の横断要約。「大口投資家の動きを教えて」等の包括的な検索・LLMクエリに直答するための集約ページ。「今週のポイント」で買い/売りの件数・金額、投資家分類別・銘柄別の上位内訳を集計表示し（`lib/weeklyStats.ts`の`buildWeeklySummary()`）、金額規模が大きい上位3件を「注目の取引」としてヒーロー枠で見せ、残りは取引日ごとに件数・金額つきで`/date/[date]`へのリンクに集約する。全記事をカード表示していた旧構成は縦に長すぎるため廃止。ヘッダーから常時リンク） |
| `/articles/[id]` | 記事詳細。銘柄｜取引日｜金額規模｜保有比率（提出者が特定できた記事のみ`getHoldingSnapshot()`でEDINET開示から取得、直前の保有割合があれば「（前回 X%）」を併記）｜前回比（CMSの`ratioChangePct`を優先、無ければEDINET開示の直前保有割合との差。±pt表示で買いは緑・売りは赤）｜取引企業（`/investors/[filer]`への内部リンク。CMSの`filerName`が空の旧記事はEDINET開示との突合で解決する。上記スキーマの注意書きを参照。突合できない場合は列自体を出さない）のdl（ファクトボックス）を表示。本文の下には出典ブロック（情報源=EDINET・提出日・記事公開日・元の開示リンク・免責の定型文。E-E-A-T/AIO対策）と共有ボタン（`src/components/ShareButtons.tsx`、X/LINE/はてブのWeb Intentへのリンク。SDKは読み込まず`ActionButton`の外部リンクとして描画）と、回遊導線として「同じ銘柄の他の記事」（`getArticlesByStockCode()`、最大3件の一行リンク＋銘柄ページへの導線）「この取引をした投資家」（投資家ページへの導線）「関連記事（同じ分類）」（カード最大4件）を重複排除して並べる |
| `/category/[category]` | カテゴリ別一覧（同じく初回30件サーバーレンダリング＋オートスクロール） |
| `/stocks` | 銘柄一覧。見出し直下に業種（セクター）別の絞り込み（Supabase `jpx_stock_list.sector`から集計、`?sector=`クエリでSSRフィルタ、各業種の件数を`(N件)`で表示）を配置し、記事のある銘柄を証券コード順（辞書的に引ける順番）に列挙（`lib/microcms.ts`の`getAllStocksForIndex()`＋`lib/companyInfo.ts`の`getSectorsByCode()`） |
| `/stocks/[code]` | 銘柄ページ。見出し（企業名・証券コード）の直下に事業内容（1文、あれば）を地の文で表示し、続けて直近90営業日の株価推移グラフ・業種・終値・PER/PBR・52週レンジ位置・株主優待有無の会社情報カードを置く。その下にこの銘柄へ大量保有報告書を提出したことがある投資家を1件ずつ改行して一覧表示（`/investors/[filer]`への内部リンク）、続いて「大量保有・自社株買い履歴」の見出しを置き、同一`stockCode`の記事を`-dealDate`順に一覧表示する。記事詳細の「銘柄」欄から内部リンクあり |
| `/investors` | 投資家一覧。見出し直下にカテゴリ別の絞り込み（`?category=`クエリでSSRフィルタ、各カテゴリの件数を`(N件)`で表示）を配置し、EDINET大量保有報告書を提出したことがある投資家を最終開示日が新しい順に列挙（`edinet_filer_summary` Supabaseビュー経由）。投資家は約2,900件あるため`?page=`で200件ずつのページ送り（クロール可能な素の前後リンク。各ページは自分自身をcanonicalにする）。行はMUIコンポーネントではなく素の`ul`/`li`＋`DealTypeLabel`（サーバーコンポーネントの軽量ラベル）で描画する |
| `/investors/[filer]` | 投資家別ページ。プロフィールの下に「主な保有銘柄」（`holdings`を`issuerCode`で重複排除し保有比率つきで列挙）、続けて「最近の取引」の見出しでその投資家が開示した保有銘柄・保有比率の推移を一覧表示（`edinet_large_holdings`/`edinet_filer_classification`）。競合の大量保有報告書データベースには無い「投資家を軸にした横断トラッキング」がこのサイトの差別化ポイント。`edinet_filer_classification.profile`（`web/publish_blog_articles.get_filer_profile()`がClaudeの一般知識から800〜1000字程度で生成・キャッシュ）があれば「{投資家名}について」の解説文として一覧の上に表示する |
| `/ranking` | 投資家別 過去勝率ランキング。`tools/filer_win_rate.py`が週次（GitHub Actions `filer_win_rate.yml`）で再計算するSupabase `filer_win_rate`テーブルを`lib/investors.ts`の`getFilerWinRates()`が収縮後勝率(shrunk_win_rate)降順で取得。買い開示件数(n)が5未満の投資家は表示しない（サンプル不足で勝率のブレが大きいため）。`/investors`と同じ`?category=`クエリでのカテゴリ絞り込みに対応。表示はテーブルではなく「順位＋投資家名／分類バッジ・件数／リターン」の1件1行リスト（全画面幅共通）。5列テーブルだと狭い画面で投資家名が1文字ずつ折り返されて読めなくなるため |
| `/trending` | クジラが急増した銘柄・投資家。直近30日間にEDINETへ提出された大量保有・変更報告書の件数を、その前の30日間と比べて増加件数の多い順にランキング（銘柄別・投資家別の2表、各上位10件。`src/lib/trendingStats.ts`）。集計元は記事(microCMS)ではなくSupabase `edinet_large_holdings`（`getHoldingsInRange()`）。記事の蓄積は2026年7月に始まったばかりで「前30日間」がほぼ空になり比較が成立しないのに対し、開示データは1年分あるため今日時点でも意味のある前期間比が出せる。開示データに推定取引金額は無いため、比較軸は金額ではなく開示件数（本文にもその旨を明記）。銘柄ページ(`/stocks/[code]`)は記事のある銘柄にしか存在しないため、記事の無い銘柄はリンクにせずテキストのまま出す（404へのリンクを作らない）。ヘッダーから常時リンク |
| `/monthly` | 月別アーカイブの入口。記事がある月を新しい順に、開示件数・推定取引金額つきで一覧（`lib/microcms.ts`の`getAllMonthsForIndex()`）。ヘッダーから常時リンク |
| `/monthly/[month]`（`YYYY-MM`） | 月別まとめ。その月の開示件数・推定金額・買い/売りの要約に続けて、「この月に動いた投資家」（提出者別ランキング上位10件、`lib/weeklyStats.ts`の`buildFilerRanking()`）「この月に狙われた銘柄」（銘柄別ランキング上位10件、`buildStockRanking()`）「注目取引」（金額上位3件の`FeaturedArticleCard`）「日別の記事一覧」（`/date/[date]`へのリンク）と前後月ナビを表示。取引日別ページは日数分だけ増える一方で`/weekly`から張られるのは直近7日分だけで、それより古い日付はサイトマップにしか載らない孤立ページになっていた。この月ハブを親に置くことで、全ての取引日別ページが「ヘッダー→月別アーカイブ→各月→各日」でクロールできるようにしている |
| `/date/[date]`（`YYYY-MM-DD`） | 取引日別の大口投資家の動きまとめ（同一`dealDate`の記事を`-dealAmount`順に一覧表示）。先頭（同日内で金額最大の1件）はTOP/`/weekly`と同じ`FeaturedArticleCard`でページ冒頭にハイライト表示し、残りを一覧グリッドに表示する（2026-08-15、「今日の注目」の編集的ハイライトとして追加。英語版`/en`には対応ページなし）。記事詳細のパンくず（トップ＞日付＞記事）から内部リンクあり。このページ自身のパンくずは「トップ＞{月}＞{日}」で、月は`/monthly/[month]`へリンクする |
| `/about` | 運営者情報・データソース・免責事項（E-E-A-T対策）。投資家分類の用語集（`#dealtype-glossary`）と公式X（@kujira_watch）への導線も含む |
| `/privacy`（`/en/privacy`） | プライバシーポリシー（AdSense審査の必須要件）。Google AdSenseによる第三者配信広告・Cookie利用とそのオプトアウト方法、アクセス解析（GA4/Vercel Analytics/独自アクセスログの`kw_vid` cookie）について記載。ヘッダーのハンバーガーメニューから常時リンク |
| `/faq` | よくある質問（FAQPage構造化データ付き、500問。各質問に色分けされたカテゴリバッジを表示し、9カテゴリのタブでも絞り込み可能）。大量保有報告書のしくみ・制度の深掘り・用語解説・投資家分類/業界プレイヤー・本サイトの使い方・初心者/中級者向けの読み方・投資基礎の周辺知識 |
| `/sitemap.xml` | 動的サイトマップ（`src/app/sitemap.ts`、全記事・カテゴリ・銘柄別・取引日別ページを含む） |
| `/robots.txt` | `src/app/robots.ts` |
| `/ads.txt` | AdSenseの販売者情報（`src/app/ads.txt/route.ts`）。`NEXT_PUBLIC_ADSENSE_CLIENT`未設定時は404を返す |
| `/feed.xml` | RSSフィード（新着記事20件、`src/app/feed.xml/route.ts`）。ヘッダーのハンバーガーメニュー・`<head>`の`alternate`リンク・`llms.txt`から参照 |
| `/api/counter` | ヘッダー上部の累計訪問者数カウンター用（POST、`increment_blog_visit_counter` RPCを呼ぶ） |
| `/api/articles` | 記事一覧のオートスクロール用（GET、`offset`/`dealType`クエリでmicroCMSの次のページを返す） |
| `/api/stocks/search` | ヘッダーの企業名・証券コード検索用（GET、`q`クエリで`stockCode`/`stockName`の部分一致を返す） |

## 計測・ログ

- **累計訪問者数カウンター**: ヘッダー上部（サイト名の右側）に表示（`src/components/VisitCounter.tsx`）。ページ読み込み時に `/api/counter` を叩き、Supabaseの `blog_visit_counter`（単一行）をアトミックにインクリメントして返す。
- **アクセスログ**: `src/proxy.ts`（Next.js 16で`middleware`から改称された`proxy`規約）が全リクエストのUser-Agentを見て、Googlebot/Bingbot/GPTBot/ClaudeBot/GoogleOther等の既知クローラーは`bot_name`にその名前、主要ブラウザ（Chrome/Safari/Firefox/Edge/Opera）は`bot_name="Browser"`としてSupabaseの `blog_crawler_log` に記録する（`src/lib/crawlers.ts` の `classifyVisitor()`）。curl等のスクリプト・UA不明のノイズはどちらにも一致しないため記録しない。`bot_name`で絞り込めば「本当のクローラー」と「ブラウザからの実アクセス」を区別できる。`bot_name="Browser"`の行には、`kw_vid`という匿名cookie（初回アクセス時にランダムUUIDを発行、個人情報なし）由来の`visitor_id`も記録するため、`count(DISTINCT visitor_id)`でユニーク訪問者数を集計できる。ログはSupabaseダッシュボードのTable Editorから直接閲覧・CSVエクスポートできる。
- どちらも `SUPABASE_URL`/`SUPABASE_SERVICE_KEY`（トレーディングシステム側と同じSupabaseプロジェクト）が必要。未設定でもビルド・記事表示自体には影響しない（カウンターAPI呼び出し時にのみエラーになるが、フロント側は握りつぶして非表示にする）。

- **表示速度の週次チェック**: GitHub Actions `perf_check.yml`（毎週月曜 09:00 JST / 手動実行も可）が `tools/perf_check.py`（リポジトリルート）で本番の代表9ページを計測する。見るのは「初期表示までに待たされる量」に直結する3つ ―― **TTFB**（キャッシュが効いていないと伸びる。実例: `/investors` が`searchParams`でdynamic renderingになりキャッシュ無しで1.6〜1.9秒）、**HTML転送量**（一覧の全件描画で膨らむ。実例: `/investors` が1.5MB）、**`<head>`内のレンダリングブロッキングCSS**（ウェブフォントの`@font-face`で膨らむ。実例: Noto Sans JPで496個・gzip 130KB）。いずれもgzip後の実転送量で、TTFBは3回計測の最小値。JSは初期表示を直接ブロックしないため参考値のみ（外部ドメインの広告・計測タグは自分たちで削れないので集計対象外）。閾値（TTFB 0.8秒 / HTML 100KB / CSS 30KB）を超えたページがあれば`perf`ラベルのIssueを立てて追記し、全ページが閾値内に戻ったら自動クローズする。**閾値は「調べる価値がある」ラインであって目標値ではない**。計測ロジックは`tests/test_perf_check.py`で保護している（HTMLのパースを間違えると「CSSが軽い」と誤判定して肥大化を見逃すため）。

## 広告（Google AdSense）

- 設定は `src/lib/adsense.ts` の環境変数に集約している。**`NEXT_PUBLIC_ADSENSE_CLIENT` が未設定の間は、広告スクリプト・広告枠・`/ads.txt` のすべてが何も出力しない**（ローカル・プレビュー環境では完全に無効）。スロットIDが未設定の掲載位置にも広告は出ないので、片方だけ先に有効化することもできる。
  - `NEXT_PUBLIC_ADSENSE_CLIENT`: サイト運営者ID（`ca-pub-...`）。ローダースクリプト（`src/components/AdSenseScript.tsx`、両ロケールのlayoutに設置）と`<ins>`の`data-ad-client`に使う。`ca-`を除いた値が`/ads.txt`の販売者IDになる。
  - `NEXT_PUBLIC_ADSENSE_INFEED_SLOT` / `NEXT_PUBLIC_ADSENSE_BOTTOM_SLOT`: 掲載位置ごとの広告ユニットのスロットID（`ADSENSE_SLOTS`の`infeed`/`bottom`）。位置ごとに別ユニットにするのはAdSense管理画面で収益を分けて見るため。どちらもAdSense管理画面で「ディスプレイ広告（レスポンシブ）」として作成する（`data-ad-format="auto"` + `data-full-width-responsive="true"` で描画するため、in-feed広告ユニット固有の`data-ad-layout-key`は不要）。
- 広告枠は `src/components/AdUnit.tsx`（`placement`で位置を指定）の1コンポーネントに統一。記事と広告が地続きに見えると誤クリックを誘発しAdSenseのポリシー違反になるため、必ず「広告」ラベル（英語版は"Sponsored"）を上に添える。
  - **infeed**: 記事一覧（TOP・カテゴリ別・`/en`）のオートスクロールの途中。記事カードの間ではなく**取引日グループの区切り**に、`ARTICLES_PER_AD`（=8）件読み進めるごとに1枠挿入する（`src/components/InfiniteArticleList.tsx`）。最後のグループの後ろは読み込み中のsentinelが続くので置かない。
  - **bottom**: コンテンツ末尾に1枠。記事詳細・銘柄別・投資家別・取引日別・月別・週次・ランキング・銘柄一覧・投資家一覧・トレンド・FAQ・`/about`（日英）に設置。一覧が無限に伸びるTOP・カテゴリ別には置かない（末尾に到達しないため）。**`/privacy`にだけは置かない**（ポリシーページは広告なしで保つ）。
- ローダーは React 19 が `async`+`src` の `<script>` を `<head>` に巻き上げるため、layoutのbody内に置いても審査コードの検出は通る。広告ブロッカー等で `adsbygoogle.push()` が失敗してもページの描画は止めない（例外は握りつぶす）。
- プライバシーポリシー（`/privacy`・`/en/privacy`）はAdSense審査の必須要件。広告ツール・解析ツールを増減させたらこのページの記載も合わせて更新すること。

## SEO/AIO対策

- **metadata**: `src/lib/site.ts` の `SITE_URL`/`SITE_NAME` を起点に、ルートレイアウトで `metadataBase`・タイトルテンプレート（`${SITE_NAME}｜%s` の順。記事タイトルが長いとブラウザタブで末尾が切れるため、サイト名を先頭に置いている）・OGP・Twitter Card・`robots` を設定。記事詳細・カテゴリ別一覧は `generateMetadata` で動的に title/description/canonical/OGPを生成する。
- **アイコン/OGP画像/ロゴ**: `src/app/icon.tsx`（ファビコン、HTMLページの`<head>`にのみ`<link rel="icon">`として注入される）・`src/app/opengraph-image.tsx`（SNSシェア用1200x630）・`src/app/logo/route.ts`（構造化データ用の正方形512x512ロゴ、`/logo`）は `next/og` の `ImageResponse` でクジラ絵文字🐋をブランドネイビー背景に合成して動的生成（画像アセット不要）。`logo`はOGP画像と違い横長ではなく正方形にしてある（構造化データの`logo`にはOGP用の横長比率ではなく正方形〜近い比率の画像を指定するのがGoogleの推奨のため）。加えて`src/app/favicon.ico`（同デザインの静的PNG内蔵ICO、16/32/48/64px）を配置している。Next.jsは`favicon`をコードから生成できず画像ファイルが必須なため、`icon.tsx`だけでは`/sitemap.xml`・`/robots.txt`・`/feed.xml`のような`<head>`を持たないルートを直接開いたときにブラウザが`/favicon.ico`にフォールバックし、ファイルが無いとVercelの既定favicon（三角ロゴ）が表示されてしまう。静的ファイルを置くことでサイト全体のフォールバック先を統一している。
- **構造化データ (JSON-LD)**: ルートレイアウトに `WebSite`/`Organization`（`logo`に`/logo`を指定）。記事詳細には `Article`（`headline`/`url`/`author`＝サイト運営組織/`publisher`＋`publisher.logo`/`image`＝アイキャッチ/`about`に銘柄名・証券コード/`citation`に出典URL）と `BreadcrumbList`（トップ＞取引日＞記事タイトル。取引日は`/date/[date]`へリンク）。トップ・銘柄別・カテゴリ別・取引日別・週次まとめの各一覧ページには `ItemList`（各`itemListElement`に`name`＝記事タイトルを含める）と `BreadcrumbList`（トップ以外）、FAQページには `FAQPage` を埋め込み。Google/AI Overview双方の情報抽出を想定。
- **サイトマップ**: 全記事・カテゴリ・銘柄別・月別（`/monthly/[month]`、日別より高い優先度）・取引日別・投資家別ページと、集約ページ（`/weekly`・`/trending`・`/monthly`）を含む。`src/app/sitemap.ts` はビルド時ではなくリクエスト時に生成（`dynamic = "force-dynamic"`）。microCMSの一時的な障害でVercelのビルド自体が失敗しないようにするため。
- **AIO向け**: `public/llms.txt` にサイトの目的・データソース・主要パスをLLMクローラ向けに明記。
- **公式Xへの導線**: `/about` 冒頭に公式X（@kujira_watch）へのテキストリンクを置く。X公式の埋め込みタイムライン（`platform.twitter.com/widgets.js` + `twitter-timeline`）も一度試したが、X側のsyndication制限でツイートが描画されずフォールバック文言だけが残るうえ、`widgets.js`と2つのiframeの読み込みでページが体感で遅くなったため2026-08-15に撤去した（再挑戦する場合はこの2点を先に検証すること）。
- **E-E-A-T**: `/about` にデータソース・算出方法・免責事項を明記し、ヘッダーのハンバーガーメニューから常時リンク。「情報源について（EDINET）」節でEDINET書類検索・EDINET API仕様書への外部リンクを掲載し、一次情報を読者が自分で検証できるようにしている（日英とも実装、EN側の文言は`src/lib/i18n.ts`の`aboutSource*`）。
- **週次まとめページ**: `/weekly`（`src/app/weekly/page.tsx`、`lib/microcms.ts`の`getRecentArticles()`）が直近7日間の開示を横断要約。「大口投資家の動きを教えて」等の包括的なクエリに個別記事より直接答えられるページとして新設し、ヘッダーから常時リンク・サイトマップに高優先度で登録。「今週のポイント」セクション（`src/lib/weeklyStats.ts`の`buildWeeklySummary()`）は`tags`の売り判定（`isSellArticle()`）・`dealType`・`stockCode`を軸に、買い/売りの件数と金額、投資家分類別・銘柄別の上位3件（各`dealAmount`合計降順）を機械的に集計するのみで、解釈や予測は行わない。全記事をカード表示すると縦に長すぎるため、金額規模上位3件のみ`FeaturedArticleCard`で「注目の取引」として見せ、残りは取引日ごとに件数・金額を添えて`/date/[date]`アーカイブへのリンクに集約する（`ItemList`構造化データもこの可視構成＝注目記事＋日付アーカイブのリンクに合わせている）。
- **FAQPage構造化データ**: `/faq`（独立ページ、計500問）にFAQPage JSON-LDを付与。「大量保有報告書のきほん」（5%ルール・提出期限・保有目的・共同保有者・特例報告など制度の基礎、22問）「用語・投資家分類」（クジラ・13分類それぞれの解説・議決権・EDINET等の用語、20問）「投資用語の補足」（時価総額・ROE/ROA・信用取引・IPO・TOB用語・運用スタイル等、基礎カテゴリでは扱っていない一般的な株式・金融用語のグロッサリー、110問）「制度・法律の深掘り」（5%ルールの細則・共同保有者/特別関係者の認定・課徴金制度・TOB規制の細目・インサイダー規制・コーポレートガバナンス関連法制など、基礎カテゴリより踏み込んだ制度解説、106問）「投資家分類・業界プレイヤー解説」（年金基金・ヘッジファンド類型・PE/VC・ファミリーオフィス・事業会社など、13分類バッジより粒度の細かい投資主体の解説、105問）「投資基礎の周辺知識」（証券口座・NISA・注文方法・分散投資・税金・チャート/ファンダメンタルズの基本など、初心者向けの実践的な周辺知識、105問）「サイトの使い方」（金額規模の算出方法・検索・会社情報カード・銘柄別ページ・週次まとめ等、10問）「読み方・活用法」（大量保有報告書は株価にどう影響するか等、投資初心者・中級者向けの実践的な疑問、20問）「運営・データについて」（海外投資家の扱い・同一銘柄で記事が複数ある理由、2問）の9カテゴリに分類。投資家分類の判定方法・記事の作成方法・更新頻度・タグの意味・カテゴリ別/取引日別ページの説明・投資助言でない旨・免責事項・記事の引用可否・広告掲載の有無など、サイト運営に関する説明的・定型的な質問は`/about`の免責事項セクションや各機能ページ自体の分かりやすさで代替できると判断し`/faq`からは割愛しているほか、使用技術スタック・アクセス解析ツール・利用AIベンダー・トレーディングシステムとの内部関係、レスポンシブ対応やページネーション廃止といった自明・些末なUI仕様、既存の質問と内容が重複するものなど、読者にとって価値が低い質問は含めていない。`src/components/FaqList.tsx`（クライアントコンポーネント）で、各質問に色分けされたカテゴリバッジ（ドット+ラベル、カテゴリごとに固定色）を表示するほか、タブUIでも絞り込み表示できる。タブの初期状態は必ず「すべて」（全件表示）にしているため、SSR済みHTMLには常に全問が含まれ、クローラー・AIOが辿れる内容はJSの実行有無によらず変わらない。回答文はAI Overview等でそのまま引用されても意味が通じるよう、質問文を読まなくても完結する1〜3文の自己完結型の文章にしている。可視コンテンツと構造化データの回答文は一言一句一致させている。`/about`からはリンクのみで誘導。

## 実装メモ

- **Supabaseの1000行上限**: PostgRESTは1リクエスト既定1000行で打ち切り、返る行の順序も保証しない。`src/lib/investors.ts`の`PAGE_SIZE`/`MAX_PAGES`で並び順を固定したページングを行い、全件必要なクエリ（`getAllFilers()`＝投資家2,938件、`getHoldingsInRange()`＝開示60日で2,906行）を取り切る。2026-08-15にこの上限で2件の実害を確認して修正した: ①`/investors`一覧とサイトマップに投資家が1,000件しか載らず、約1,900件の投資家ページがサイトマップから漏れていた ②`/trending`の投資家集計が（打ち切られた1000行に対象期間の行が入らず）まるごと空になっていた。
- **一覧ページの重さ**: `/investors`は全件（約2,900件）を1ページに並べていたためHTMLが1.5MBに達し、本番のTTFBが1.6〜1.9秒だった。1ページ200件のページ送り＋行の軽量化（MUIの`ListItem`/`Chip`/`Tooltip`をやめて素の`ul`/`li`＋`DealTypeLabel`）でHTML約390KB・TTFB 0.03〜0.06秒（`unstable_cache`ヒット時）に改善。分類の色定義は`src/lib/dealTypeInfo.ts`の`DEAL_TYPE_COLORS`に置き、`DealTypeBadge`（Chip+Tooltip、記事カード用）と`DealTypeLabel`（一覧用）で共有する。
- **microCMSの全件取得**: 公式SDKの`getAllContents`はページ間に1秒の固定スリープが入るため使わず、1ページ目の`totalCount`から残りのoffsetを割り出して並列取得する（`fetchAllPagesParallel()`＝銘柄一覧・月別インデックス、`fetchAllArticlesByFilter()`＝`/weekly`・`/monthly/[month]`）。直列だと1往復200〜300msがページ数だけ積み上がる。

- 一覧・カテゴリ別一覧は初回表示分（`INITIAL_ARTICLES_COUNT`＝30件）のみサーバー側で取得し、以降は`src/components/InfiniteArticleList.tsx`（クライアントコンポーネント）が画面下端の要素を`IntersectionObserver`で検知して`/api/articles`から次の10件を都度取得・追記するオートスクロール方式。ページネーションのUIやURLの`?page=`は廃止した。初回件数を10→30に引き上げているのは、オートスクロール分（JS実行後にのみ取得される）はクローラーが辿れない実リンクになるため、クロール可能な記事数の下限を底上げする狙い（クロールログで新着記事の巡回が10件相当に留まっていたための対策）。
- `/api/articles`が返す一覧・初回表示分ともmicroCMSへの `fetch` は `next: { revalidate: 60 }` を指定しており、Next.jsのData Cacheが60秒間キャッシュ・再検証を行う（App RouterにおけるISRの実体）。
- 記事詳細（`/articles/[id]`）は動的APIを使わないため `export const revalidate = 60` をルートセグメントに設定し、オンデマンドISR（初回アクセス時に生成し60秒キャッシュ）として動作する。
- 本文（リッチエディタのHTML）は `dangerouslySetInnerHTML` + Tailwind Typography(`prose`)で描画。ja記事詳細ページは描画前に`linkifyFilerNames()`（`src/lib/format.ts`）で本文中の投資家名（初出のみ）を`/investors/[filer]`へのリンクに変換する。投資家名はCMS側に構造化フィールドが無く自由記述本文の一部でしかないため、`getFilersByStockCode()`でこの銘柄の開示実績がある投資家名一覧を取得し文字列突合する。EDINETのXBRLは提出者名を全角（`Ｏａｓｉｓ　Ｍａｎａｇｅｍｅｎｔ…`）で保持する一方、AI生成本文は半角で書くため、NFKC正規化した文字列上で位置を探し、本文側の表記（半角）はそのまま残しつつリンク先だけDB上の正式表記（全角）でエンコードする。英語版記事（`/en/articles/[id]`）には未適用（`/en/investors/...`相当のページが存在しないため）。
- `eyecatch`（アイキャッチ画像）はカード一覧・ヒーロー枠・記事詳細で表示する（未設定の記事はテキスト中心のレイアウトにフォールバック）。記事詳細では`generateMetadata`のOGP画像としても使う。
- デザインはエディトリアル（雑誌）系。欧文は`next/font/google`のGeist、**和文は端末内蔵フォント**（iOS: Hiragino Sans / Android: Noto Sans CJK JP）を使う。以前はNoto Sans JPをウェブフォントで読んでいたが、`next/font/google`の`subsets`はプリロード範囲の指定でしかなくCJKの`@font-face`は生成CSSから落ちず、4ウェイト分でunicode-range分割の`@font-face`が496個・378KB（gzip 130KB）のレンダリングブロッキングCSSになっていた。加えて本文の漢字に応じて70〜90KBのwoff2スライスをウェイトごとに追加ダウンロードしていた（スマホで表示が重い直接の原因）。フォントスタックは`src/app/globals.css`の`--font-sans`で定義する。配色はクリーム地の紙面(`--background`/`--paper`)＋インクネイビー＋くすみゴールドのアクセント（`src/app/globals.css` のCSS変数で調整可）。バッジ・カテゴリ表示はピル型からドット＋スモールキャップス文字（`.kicker`）のキッカー表記に変更し、カードは影で持ち上げる代わりに罫線区切り＋タイトル下線ホバーのシンプルな見せ方にした。記事詳細の本文冒頭にはドロップキャップ（先頭一文字の大型表示）を適用。ヒーロー枠（注目記事カード）はアイキャッチ画像がある記事のみ大きな高さを取り、無い記事では余白を残さないコンパクトな表示にフォールバックする。
- 記事一覧（TOP・カテゴリ別一覧・銘柄別履歴）は取引日(`dealDate`)の新しい順、同日内は金額規模(`dealAmount`)の大きい順にソートし（`src/lib/microcms.ts` の `orders: "-dealDate,-dealAmount"`）、`src/lib/groupByDealDate.ts` で取引日ごとに見出しを付けて表示する（見出しは`src/components/DealDateHeading.tsx`で3ページ共通）。「いつの話か」が一覧性で分かるようにするため。各見出しの下には、その取引日の全記事を一覧できる`/date/[date]`アーカイブページへのボタン（`src/components/DealDateSeeMoreLink.tsx`）を表示する。
- ヘッダーのロゴ（🐋アイコン）・カテゴリ別一覧のパンくずリストから常にTOPへ戻れる（記事詳細・銘柄別履歴には既存のパンくずリストあり）。
- オートスクロールの導入で記事一覧が際限なく伸び、ページ最下部までスクロールするのが実質困難になったため、独立した`<Footer>`は廃止。運営者情報・免責事項・RSS・累計訪問者カウンターは`src/components/HeaderMenu.tsx`（ヘッダー右上のハンバーガーメニュー）に集約し、スクロール位置によらず常にアクセスできるようにしている。
- ヘッダー上部のハブナビ（今週の動き／大口投資家一覧／株式銘柄一覧）はスマホ幅では折り返さず横スクロール1行にし（`.no-scrollbar`、`src/app/globals.css`）ている。カテゴリ絞り込み（13カテゴリ）は以前ヘッダーに常設していたが、全ページ共通で常時表示すると場所を取り本文の文脈からも離れて見えるため`src/components/CategoryFilterDetails.tsx`に切り出し、TOPページの見出し「今日の注目取引」の直下にのみ表示する構成に変更した。
- カテゴリフィルター（`/category/[category]`、`CategoryFilterDetails.tsx`）はmicroCMS側に別フィールドを持たず、`dealType`の値をそのままカテゴリ名として使う（`src/types/article.ts` の `categoryLabel`/`DEAL_TYPE_BY_CATEGORY`、値はidentity）。CMS側の選択肢リストをdealTypeの分類と別途同期させる必要が無く、選択肢の同期漏れによる不具合が起きない構成にしている。
- `/investors`・`/stocks`・`/ranking`のカテゴリ／業種フィルターはクライアントJSを使わず、`searchParams`（`?category=`/`?sector=`）を読んでサーバー側で絞り込んだ結果を返すシンプルな構成（フィルターの実体は`<Link>`をボタン化した`src/components/FilterButtonNav.tsx`で、選択中のみ塗りつぶし表示）。`/investors`はカテゴリ別、`/stocks`は業種別、`/ranking`は投資家分類別で、切り口をあえて分けている。
- ボタンUI（押す導線）は全ページでMUI Buttonに統一している。共通ラッパは`src/components/ActionButton.tsx`（内部リンクは`component={Link}`、外部リンクは`component="a"`＋`target="_blank"`。Server Componentから関数を渡せないため`"use client"`をこのファイルに置いている）で、配色・角丸・フォントは`src/theme.ts`の`MuiButton`（デフォルト`variant="outlined"` / `size="small"`、色は`globals.css`のCSS変数を参照）で一元管理する。適用箇所は取引日見出し下の「この日の記事を見る」（`DealDateSeeMoreLink.tsx`）・記事詳細の共有ボタンと銘柄履歴への導線・月別アーカイブの前後の月ナビ・TOPのカテゴリ絞り込み・一覧ページの絞り込みナビ。本文中の文脈依存のリンク（`/about`・`/faq`の説明文中のリンクなど）はテキストリンクのまま残す。
- `/stocks/[code]`の会社情報（`src/lib/companyInfo.ts`/`src/components/CompanyInfoCard.tsx`）はトレーディングシステム側が日次で更新するSupabaseの`jpx_stock_list`（業種・事業内容の1文説明・株主優待）と`gen_rankings`（直近90営業日分の終値・PER・PBR・52週レンジ位置）を参照する。事業内容はカード内の付加情報ではなく見出し直下に地の文（クローラーが読む本文）として表示し、`generateMetadata`のdescription（`companyInfo.description`＋`formatStockDealSummary()`の取引サマリー）にも組み込むことで、検索結果スニペット・構造化データの両方に銘柄固有の内容が反映されるようにしている（カード内表示のみではSEO評価にほぼ寄与しないため2026-08-15に変更）。株価推移グラフは外部チャートライブラリを使わず、直近90営業日分の終値をインラインSVGの折れ線（`polyline`）で自前描画している。`gen_rankings`の`drop_prob`（下落確率）・`recommend`（売買シグナル）はstock-alert本体の提供価値そのものなので、ブログ側では意図的に表示しない。取得失敗時（未設定・障害）は記事一覧の表示を止めないよう`null`を返しカードごと非表示にする。ページ全体に`export const revalidate = 300`を設定し、Supabase側のfetchも含めて5分周期で再検証する。
- ヘッダー右上の🔍アイコン（`src/components/StockSearch.tsx`）から企業名・証券コードで検索できる。入力停止から300ms後に`/api/stocks/search`（`lib/microcms.ts`の`searchStocks()`、`stockCode`/`stockName`の`[contains]`部分一致、銘柄単位で重複排除・最大20件）を叩き、結果をクリック（またはEnterで先頭候補）すると`/stocks/[code]`（銘柄別履歴）に遷移する。

## コンテンツの自動生成（任意）

リポジトリルートの `web/publish_blog_articles.py` が、EDINET大量保有報告書（保有比率が増加する取得＝買い方向、減少する譲渡・売却＝売り方向の双方）を基にClaudeで解説記事を生成し、このAPIへ即時投稿する（GitHub Actions `daily_alert.yml` Step 5c、日次）。取得・売却金額(億円)はyfinanceの発行済株式数×株価×保有比率変化からの推定値であることを本文に明記させている。売り方向の記事は`tags`に`"売り"`を追加して買いと区別する（microCMSのセレクトフィールド追加を避け、既存の自由記述`tags`フィールドを流用）。フロント側は`src/lib/format.ts`の`isSellArticle()`が`tags`から判定し、`src/components/DealDirectionBadge.tsx`が`DealTypeBadge`の隣に「売り」バッジを表示する（買い方向の記事にはバッジを出さない）。

`dealType`（提出者の投資家分類）は、Supabaseの`edinet_filer_classification`マスター（Web検索で確認済みの投資家分類テーブル、バックテスト分析とも共用）をまず参照し、未登録の提出者のみClaudeの一般知識で判定して結果をマスターへ保存する（`web/publish_blog_articles.py`の`classify_filer()`）。キーワード一致だけでは日系/外資の区別やスペース無し個人名を正しく判定できないため。判定不能な場合は「その他」に丸める。

`filerName`（記事詳細の「取引企業」欄）には、EDINET開示から取得した提出者名（`filer_name`）をそのまま送信する。フロント側の`/investors/[filer]`は投資家名をパスパラメータにしているため、両者は常に一致する。ただしmicroCMS側にこのフィールドが未作成のため送信値は現状すべて破棄されている（上記スキーマ表の注意書きを参照）。

投稿後の内容確認・修正はmicroCMS管理画面で人間が行う想定。詳細はスクリプト冒頭のdocstringを参照。

## スコープ外

認証・会員機能、コメント機能。
