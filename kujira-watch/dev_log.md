## 2026-08-18 kujira-watch: 文字サイズをTOPに統一し、一覧7ページをカードUI＋ページ送りに

オーナー指摘「なんかトップページだけ文字が大きい？」→「やや小さいぐらいで良い」
→「他のページもTOPにサイズ感を合わせたい」→「リスト形式の情報は全部カードUIにできる？」
→「多い場合はページネーション」への対応。

### 何が起きていたか（実測・幅1280px）
TOPだけが他ページに無い大きさを2つ持っていた。
- `TodayWhaleSummary`の件数・金額が **48px**（MUI `variant="h3"`の既定）。
  サイトの他の最大値はh1の30pxで、48pxはこの1箇所だけだった。
- `FeaturedArticleCard`のタイトルが **24/30px** で、ページのh1と同寸。
  見出し階層が潰れたカードが3枚縦に並ぶため「上半分ぜんぶ大きい」印象になっていた。

一方で通常の記事カード（見出し20px／本文14px／メタ12px）は他ページと完全に一致しており、
スクロールすると急に普通のサイズに戻る＝上だけ浮いた状態だった。

### 実装
1. **文字サイズ**
   - サマリー数値 48px → `{ xs: "1.5rem", sm: "1.75rem" }`(24/28px)、`lineHeight`も1→1.1。
   - 注目カード見出し 24/30px → `{ xs: "1.25rem", sm: "1.5rem" }`(20/24px)。
   - h2セクション見出しを ja/en 18ファイル・61箇所で `text-lg`(18px) → `text-xl`(20px)。
     TOPだけ`text-xl`、他ページは`text-lg`という2系統になっていたためTOP側に寄せた。
     `text-lg`の使用箇所は全て`text-lg font-bold text-brand-navy`のh2で、他用途への巻き込みは無い。
2. **カードUI**（`/disclosures` `/trending` `/ranking` `/activists` `/investors` `/stocks` `/monthly`）
   - `src/app/globals.css`の`@layer components`に`.card`/`.card-grid`/`.card-grid-wide`を新設。
     **MUI CardやTailwindユーティリティの羅列は使わない**: `/faq`では可視HTMLの75%が
     コンテンツではなくマークアップ（`Mui〜`クラスの出現14,160回）だった前例がある。
     `@layer components`に入れるのは、レイヤー無しだとTailwindのユーティリティより優先されて
     `flex`等での上書きが効かなくなるため。
   - グリッドは`repeat(auto-fill, minmax(min(100%, 14rem), 1fr))`。`min()`により、
     カード最小幅より狭い端末では自動的に1列になるのでメディアクエリを持たない。
   - `TrendingTable`はMUI Tableをやめてカードに。`minWidth: 420px`で375px幅では
     横スクロールが必要だったため、数値にラベルを添えてカード内で折り返す形にした。
   - `MonthList`はMUI List＋`"use client"`をやめ、素のカードのサーバーコンポーネントに。
   - `/ranking`は「1行目=順位＋投資家名（全幅）／2行目=分類・件数・リターン（右端）」の2段。
     リターンを名前と同じ行に置くと、長い投資家名が細く4行に折り返されていた。
     表の見出し行（順位・投資家／トータルの推計リターン）はカード2列と対応しないため凡例1行に変更。
3. **ページ送り**
   - `/stocks`（596件を1ページに全件描画していた）に`?page=`を新設。100件/ページ。
   - `/ranking`（131件）に`?page=`を新設。順位はページをまたいで通し番号（`rankOffset`）にし、
     ItemList構造化データの`position`も合わせる。
   - `/investors`は既存のページ送りを200件→100件に。カード化で1件あたりの縦幅が増えたため。
   - 3ページとも`/investors`と同じ規律: 各ページが自分自身をcanonical、絞り込みリンクは
     ページ番号を持ち越さない、サイトマップに載せるのは1ページ目のみ。

### 確認
本番ビルド(`next start`)のHTMLをgzipして本番サイト(変更前)と比較。閾値は
`perf_check.yml`のHTML 100KB。

| ページ | 変更前 | 変更後 |
|---|---|---|
| /disclosures | 31KB | 31KB |
| /trending | 14KB | 13KB |
| /ranking | 47KB | 42KB |
| /activists | 36KB | 36KB |
| /investors | 36KB | 24KB |
| /stocks | 57KB | **21KB** |
| /monthly | 10KB | 9KB |

カード化してもHTMLは増えていない（MUIを使わなかったため）。`/stocks`はページ送りで大幅減。
`/investors`は200件→100件のページ送りで36KB→24KB。`/stocks`のページ高さは 22,110px → 4,894px。
375px/1280pxの両方で7ページを目視確認、横スクロール無しを`scrollWidth`で実測。
`tsc --noEmit`・eslintクリーン、`npm run build`成功。


## 2026-08-18 kujira-watch: 記事の無い銘柄が検索に出ない不具合を修正（6929 日本セラミック）

オーナー報告「6929の銘柄が検索で出てこない。日セラミックのはず」への対応。

### 原因
検索(`/api/stocks/search`)がmicroCMSの記事だけを引いていたため、解説記事が1本も
無い銘柄は検索から完全に消えていた。6929はSupabaseに会社情報(`jpx_stock_list`)も
EDINET大量保有4件(`edinet_large_holdings`)もあるのに記事が0件で、
`/stocks/6929`も記事0件を理由に404にしていた。
記事が無い理由は、日次投稿(`web/publish_blog_articles.py`)が `disc_date >= 今日-3日`
しか見ないのに対し、6929の開示4件は2026-08-10の1年分バックフィルで後から入った行
（`fetched_date`=2026-08-10、`disc_date`は2025-06〜2026-05）だったため。
過去分を埋める`tools/backfill_monthly_articles.py`は月あたり保有比率上位50件のみで、
6929の5.04%は入らなかった。
規模: EDINET開示のある2,982銘柄に対し、記事があるのは599銘柄だけ。
約2,380銘柄が検索・銘柄ページの両方から見えていなかった。

### 対応（記事ベース → 開示・上場銘柄ベースへ）
- `lib/companyInfo.ts`: `searchStockMaster()`（`jpx_stock_list`のコード前方一致・
  社名部分一致）と`getAllListedCodes()`（全上場コードのSet、1時間キャッシュ）を追加。
  `CompanyInfo`に`name`を追加（記事が無い銘柄の表示名の取得元）
- `/api/stocks/search`: 記事検索とマスター検索の両方を引いてコードで重複排除
  （記事のある銘柄を先頭に最大20件）
- `/stocks/[code]`: 記事0件でも会社情報＋開示履歴＋FAQでページを成立させ、
  マスターにも記事にも無いコードだけ404。記事0件のページは薄いので
  `robots: noindex, follow` を付与（sitemapは従来どおり記事ベースなので変更なし）
- `/disclosures`・`/trending`・`/activists`: 銘柄リンクの可否判定を
  「記事がある」から「上場銘柄マスターにある」へ変更（404リンクを作らない規律は維持）

### 確認
- `/api/stocks/search?q=6929` → `日本セラミック`、`?q=日本セラミック` → 6929 が返る
- `/stocks/6929` 200。会社情報（電気機器）・提出投資家3名・保有比率テーブル4行・
  FAQ・`noindex, follow` を実測
- 記事のある銘柄（9235）は `index, follow`・ItemList構造化データ・記事一覧とも従来どおり
- `/disclosures` `/trending` `/activists` `/` 200、存在しないコード（0000/abc）は404
- `tsc --noEmit`・eslintクリーン


## 2026-08-18 kujira-watch: ハンバーガーメニューを上部タブと同順＋見出し付きグループに再編

オーナー指示「ハンバーガーメニューの並びを上タブと同じにして。上タブにないものを
その下に入れて、見出しもつけて」への対応。

### 実装
- `src/lib/nav.ts` 新設: 上部タブとメニューが共用する主要ナビゲーションを一元管理。
  2箇所で別々に定義していたため、ページ改名時にメニューだけ取り残されていた
  （実例: /weekly改名後もメニューは「今週のまとめ」のまま）。今後は構造的に起きない。
- `HeaderMenu.tsx`: フラットな`siteLinks`を廃止し、見出し付き`MenuGroup[]`に。
  ja=「主要ページ（タブと同順9件）／サイト情報（about・FAQ・プライバシー・利用規約）／
  フォロー（X・YouTube・RSS）」、en=「Main pages（Top＋全分類）／Site info／Follow」。
  利用規約はどこからもリンクが無かったため追加（Footerのサイト情報と同構成）。
  enメニューにYouTubeを追加（jaと同じ導線に）。
- `HeaderMenuDrawer.tsx`: `menuGroups`を受け取り、グループごとに
  overline見出し＋Listで描画（言語セクションと同じ見た目）。

### 確認
- 375pxでja/en両ドロワーを開き、並び・見出し・全リンクを実測確認。
  `tsc --noEmit`・eslintクリーン。
- 検証は別セッションのdevサーバー(3002)を利用（Next 16は同一プロジェクトの
  多重起動不可のため）。

## 2026-08-17 kujira-watch: XカードのOGP画像が出ない不具合を修正（サムネなし）

公式X（@kujira_watch）の投稿でリンクカードのサムネイルが表示されない
（グレーのプレースホルダになる）との報告。調査の結果、全ページの`<head>`に
`og:image`/`twitter:image`メタタグ自体が出力されていなかった。

### 原因
`opengraph-image.tsx`を`src/app/`直下に置いていたが、このプロジェクトは
ページがすべて`(ja)`/`(en)`のルートグループ（それぞれルートレイアウトを持つ）
配下にあるため、appルート直下のOGP画像がどのページにも紐付かなかった
（Next.js 16.2.11で確認。`/opengraph-image`のルート自体は200で画像を返すが、
メタタグが注入されない）。`icon.tsx`（ファビコン）はルート直下でも全ページに
効くため気付きにくい。ローカルのdevサーバで`/privacy`等の`<head>`を確認して再現。

### 修正
- `src/app/opengraph-image.tsx` → `src/app/(ja)/opengraph-image.tsx` に移動
  （内容は従来どおり：🐋＋日本語サイト名＋説明文、1200x630）。
- `src/app/(en)/en/opengraph-image.tsx` を新規追加（英語版。`SITE_NAME_EN`/
  `SITE_DESCRIPTION_EN`を使用。従来は英語ページにOGP画像の仕組み自体がなかった）。

### 確認
- devサーバ実測: `/`・`/privacy`に`og:image`/`twitter:image`（ja画像）、
  `/en/about`にen画像のメタタグが出力され、両画像URLとも200/image/pngを返す。
- 記事ページの`generateMetadata`が設定するアイキャッチ（`openGraph.images`）は
  ルートグループのファイルベース画像より優先されることをテストページで実測確認
  （アイキャッチ付き記事のカードは従来どおりアイキャッチが出る）。
- `tsc --noEmit`・eslint（変更ファイル）クリーン。`npm run build`はコンパイル・
  型チェックまで通過（その先のページデータ収集はmicroCMS実キーが無い環境のため
  403で失敗。本修正とは無関係）。
- デプロイ後はX Card Validator等でカード再取得（Xはカードをキャッシュするため
  反映に時間がかかる場合がある）。

## 2026-08-17 kujira-watch: デザインレビューP1修正（重なり・コントラスト・縦割れ）

サイト全体のデザインレビュー（375px/1280px × ja/en を実機確認）で見つけた
実害ありの3件を修正。

### 修正内容
1. **ヘッダーの訪問者数がハンバーガーメニューと重なる**（sm+幅で実測30px重複）
   - `HeaderMenu.tsx`: md+の`position: absolute`配置を撤去しフロー配置に。
     「画面右端に固定」が本来の意図だったが、MUI化でAppBarに付いた
     `backdrop-filter`が包含ブロックを作るため実際はカラム右端に落ちており、
     訪問者数の上に重なるだけで意図は機能していなかった。
2. **クジラ注目度バッジのコントラスト不足**
   - `AttentionScoreBadge.tsx`: gold文字×金12%ティントは明色背景で実効2.7:1
     （WCAG AAの4.5:1未達）。明色地は文字を紺（13:1）・★のみgoldに変更。
     `onDark` propを追加（DealTypeBadgeと同じ方針）し、ダーク地はgoldBright
     （6.1:1）に。`FeaturedArticleCard`から`onDark`を渡す。
3. **記事ページのファクト欄で銘柄名が縦割れ**（375pxで1行3〜4文字×4行）
   - ja/en両方の`articles/[id]/page.tsx`: 2カラムグリッドのうち銘柄・取引企業
     （jaのみ）の項目だけ`gridColumn: 1 / -1`でxs時に全幅化。sm+は4カラム維持。

### 確認
- 375px/1280px × ja/en をブラウザで再確認（重なり解消を座標実測、色は
  computed styleで確認）。`tsc --noEmit`・eslintクリーン。
- 注: `npm run build`の型チェックは別セッションのFAQ機能WIP
  （investorFaq/stockFaq、未コミット）が型エラーで失敗する。本修正とは無関係
  （WIPを退避した状態でビルドが通ることを確認してからpush）。

## 2026-08-15 kujira-watch: 表示速度の週次チェックを追加

「定期的に遅いページを見つけて改善したい」への対応。GitHub Actions `perf_check.yml`
（毎週月曜 09:00 JST / 手動実行も可）で本番の代表9ページを計測する。

### 計測項目（すべてgzip後 = 実際に回線を流れる量）
1. **TTFB**（3回計測の最小値）: キャッシュが効いていないページはここが伸びる。
   実例: `/investors` が searchParams で dynamic rendering になりキャッシュ無しで1.6〜1.9秒。
2. **HTML転送量**: 一覧を全件描画すると膨らむ。実例: `/investors` が1.5MB。
3. **レンダリングブロッキングCSS**（`<head>`内の`<link rel=stylesheet>`のみ）:
   実例: Noto Sans JPで`@font-face`が496個・gzip 130KB。

JSは初期表示を直接ブロックしないので参考値として出すだけで閾値は設けない。
外部ドメインのJS（広告・計測タグ）は自分たちで削れないため集計対象外。

### 閾値と通知
TTFB 0.8秒 / HTML 100KB / CSS 30KB。超過したページがあれば `perf` ラベルの
Issueを立て、既にオープンなら追記する（毎週Issueが増えないように）。
全ページが閾値内に戻ったら自動クローズ。閾値は「調べる価値がある」ラインで目標値ではない。

### 検証
`tests/test_perf_check.py`（8件）。実際にHTTPサーバーを立ててend-to-endで確認する。
開発中に実際に2件バグを見つけた:
- `rel=stylesheet`/`href=`/`src=` のクォート無し属性を取りこぼしていた
  （取りこぼすと「CSSが軽い」と誤判定して肥大化を見逃す）
- `<body>`内のstylesheetを除外できているかの検証で、テストのフィクスチャが
  圧縮で潰れてしまい比較にならなかった（gzipで潰れないCSSを生成するよう修正）

## 2026-08-15 kujira-watch: /faq を分割（1.57MB → 各22KB以下）

週次チェック(perf_check)が本番で唯一フラグを立てた `/faq` を調査・修正。
本番241KB(gzip)をローカル本番ビルドで239KBまで再現できたので、内訳を確定させた。

### 何が起きていたか（実測）
HTML全体 1,570 KB (raw) / 235 KB (gzip)
- 可視HTML          873 KB
- RSCペイロード      471 KB  ← 同じ本文の2回目
- FAQPage構造化データ 225 KB  ← 同じ本文の3回目

**FAQ本文が1つの文書に3回入っていた。**
- 構造化データ: `faqJsonLd.mainEntity` が全502件のquestion/answerを持つ
- RSCペイロード: `FaqList` が `"use client"` で `faqs={FAQS}` を受け取るため、
  ハイドレーション用に全件がシリアライズされる
- 可視HTML: SSR出力

さらに可視HTMLの内訳は、タグを除いた実テキストが220KBに対し
マークアップのオーバーヘッドが653KB（`MuiAccordion-root` 502個、
`Mui〜`クラスの出現14,160回、クラス名の文字列だけで166KB）。
**可視HTMLの75%がコンテンツではなくマークアップだった。**

### 対応
502件を1ページに置いているのが根本原因なので、カテゴリ別ページに分割した。
- `src/lib/faqData.tsx`: データ本体を切り出し（`/faq`と`/faq/[category]`が共用）
- `/faq`: ハブ化。9カテゴリの件数・質問サンプル5件・カテゴリページへのボタン。
  回答本文は置かない（表示していないQ&Aを構造化データに載せるのはGoogleの
  ガイドライン違反になるため、FAQPage構造化データもカテゴリページ側へ移した）
- `/faq/[category]`: カテゴリ別Q&A + そのカテゴリ分だけのFAQPage構造化データ。
  9カテゴリをgenerateStaticParamsで事前生成。サイトマップにも追加
- 分割で「タブによる絞り込み」がURLの役割になったため `FaqList` は削除し、
  タブ無しの `FaqAccordionList` に置き換え（MUI Accordionは維持）

### 結果（gzip後・ローカル本番ビルドで実測）
| ページ | 前 | 後 |
|---|---|---|
| `/faq` | 241 KB | 18 KB (-93%) |
| `/faq/basics` | - | 22 KB |
| `/faq/terms` | - | 22 KB |

TTFBも 0.35s → 0.04s。

## 2026-08-17 kujira-watch: CDNキャッシュの明示活用（API・RSS・画像）

CDNの勉強を兼ねて、Vercel Edge Networkのキャッシュを「自動で効いている部分」だけでなく
明示的に設計した。座学メモは `docs/cdn_study.md`（一般論→自サイトの実例の対応表つき）。

### 現状整理
- ページはISR（`export const revalidate`）で既にCDNキャッシュ済み
  （ISRはCDN的には `s-maxage={revalidate} + stale-while-revalidate` として配信される）
- 一方Route HandlerはNext.js 15以降デフォルト非キャッシュで、
  `/api/articles`（無限スクロール）・`/api/stocks/search`（ヘッダー検索）・
  `/api/watchlist-latest`・`/feed.xml` は毎リクエストがオリジン
  （Vercel関数→microCMS/Supabase）まで到達していた

### 対応
1. APIルート3本に `Cache-Control: public, s-maxage=N, stale-while-revalidate=M` を付与
   - `/api/articles`: 60/300（記事ページのISR 60sと同鮮度）
   - `/api/stocks/search`: 300/3600（検索対象の増減は記事投稿時のみ＝最短毎時）
   - `/api/watchlist-latest`: 300/600（開示スキャンが毎時なので5分で十分）
   - `/api/counter` はPOST＋副作用ありのため対象外（キャッシュしない設計判断も記録）
2. `/feed.xml` を `export const revalidate = 300` でISR化（RSSリーダーの定期巡回対策）
3. `next.config.ts` の `images.minimumCacheTTL` を既定4時間→31日
   （microCMSは画像差し替えでURLが変わる＝実質immutableなので安全）

### 検証
- `tsc --noEmit`・`eslint` パス。`next build` はコンパイル・型検査を通過
  （ページデータ収集はmicroCMSキー未設定の環境のため実行不可）
- デプロイ後は `x-vercel-cache` ヘッダー（MISS→HIT→STALE）で実測する
  （手順は `docs/cdn_study.md` §5）
