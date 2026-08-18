## 2026-08-18 kujira-watch: PCのヘッダー検索を常時表示の検索窓に

オーナー指示「PCの検索ボタンは検索窓にしてほしい。スマホはそのままで良い」への対応。

- md以上では虫眼鏡アイコンをやめ、幅220pxの検索窓をヘッダーに常時表示する。
  モバイル（md未満）は横幅をロゴに使いたいので従来どおり🔍タップで開く。
- 平常時は素の`<input>`のまま。フォーカスした時点で`StockSearchPanel`
  （MUI Autocomplete）を`next/dynamic`で読み込んで差し替えるので、
  「重いコンポーネントは開くまで読み込まない」既存の方針を崩さずに常時表示にできる。
- 差し替え中のキー入力が落ちないよう、パネル側の`onReady`（マウント通知）が来るまで
  素のinputを残す。入力値は`StockSearch`のstateに集約し、Autocompleteには`inputValue`で
  渡す。マウント直後にMUIが`reason="reset"`で空文字を流してくるので、`onInputChange`は
  `input`/`clear`だけ拾う（この2点を入れる前は、クリック直後に打った先頭の数文字が
  消えてPlaywrightで「トヨタ」→「タ」になっていた）。

### 検証
- Playwright（1280px / 390px）でヘッダーを実描画して確認。PCは検索窓が常時表示され、
  クリック直後に「トヨタ」と打っても全文字がAutocompleteに引き継がれる（`/api/stocks/search`も発火）。
  モバイルは従来どおり🔍→ドロップダウン。
- `tsc --noEmit`・`eslint src` パス

## 2026-08-18 kujira-watch: 見出しを「月間ランキング」に改名し、報告書件数ランキングを廃止

オーナー指示2件（「タブの投資家ランキングは月間ランキングにして」「報告書の件数ランキングの
使い道がわかってない。あまり有益じゃないなら消してもいいかも」）への対応。

### 改名
`/ranking`・`/ranking/[slug]`・`/ranking/trending` 共通のh1・パンくず・ヘッダー/フッターの
ナビ表記を「投資家ランキング」→「月間ランキング」に統一。個別ランキング名は従来どおり
タブ直下のh2に置く。タブ配下は直近30日の集計が主なので「月間」を冠しているが、既定タブの
3ヶ月リターンだけは63営業日の評価である点はREADMEに注記した。

### 報告書件数ランキング(/ranking/filings)の廃止
残す価値が無いと判断して削除した。

- 中身が `/ranking/trending`（開示急増投資家）とほぼ重複する。どちらも直近30日の開示件数を
  投資家別に数えるもので、trendingはさらに前30日との差分まで出す上、集計元が記事(2026年7月〜)
  ではなくEDINET開示(1年分)で精度も高い
- 件数の絶対値は毎月ほぼ同じ常連の提出者（大量の変更報告書を出す運用会社）が並ぶだけで、
  月ごとの変化が出ず読み手の判断材料にならない
- タブ・sitemap・記事詳細の「関連ランキング」navからも導線を削除
- 2026-08-15公開でインデックス済みのURLなので404にはせず、`next.config.ts`の`redirects()`で
  `/ranking/trending`へ301。`buildFilerRows()`のslugは`buys`/`sells`のみになり、
  件数優先のタイブレーク分岐も畳んだ

### 検証
- 集計関数を素のnodeで再確認（buys/sellsの投資家別合計・代表開示・最終開示日・limit・
  activistが銘柄別のままであること）。全PASS
- `tsc --noEmit`・`eslint src` パス

## 2026-08-18 kujira-watch: 投資家ランキングのタブ3種が銘柄ランキングになっていた不具合を修正

オーナー指摘「投資家ランキングが、銘柄ランキングになってる。買い増しと売りましと報告件数」への対応。

### 症状
`/ranking/buys`・`/ranking/sells`・`/ranking/filings` は h1「投資家ランキング」＋
`RankingTabNav` のタブ配下にあるのに、中身は開示1件＝1行（buys/sells）・銘柄別集計（filings）で
並んでおり、順位が付いていたのは投資家ではなく銘柄だった。
同じタブ内で `/ranking`（3ヶ月リターン）・`/ranking/trending`（開示急増）は投資家別なので、
タブを切り替えるとランキングの軸が投資家↔銘柄で入れ替わる状態だった。

### 対応
集計を `src/lib/rankingStats.ts` に切り出し、3ランキングを投資家別に積み直した
（`/trending` の `trendingStats.ts` と同じ置き方）。

- `buys`: 買い開示を投資家ごとに合計し推定取得金額の降順（同額は件数降順）
- `sells`: 売り開示を投資家ごとに合計し推定売却金額の降順（同額は件数降順）
- `filings`: 投資家ごとの開示件数の降順（同数は合計金額降順）
- 提出者名(`filerName`)が無い過去記事は投資家別に積めないため集計対象外にした
- 各行のメタは「分類ラベル・N件の開示・代表銘柄（金額が最大の開示）・解説記事・最終開示日・
  合計金額」。投資家／銘柄／記事の3リンクは維持している（内部リンクを減らさない）
- ItemList構造化データも投資家ページを指すよう軸を合わせた
- `activist` はタブに含まれない「アクティビストが動いた銘柄」なので銘柄別のまま
  （`/activists` と対の関係。ページ内で軸が2種類あるためレンダリングを分岐させている）
- `filings`（直近30日の件数そのもの）と `/ranking/trending`（前30日比の増加件数）は
  指標が別なので併存させた

### 検証
- 集計関数を素のnodeに落として8ケースを確認（投資家別の合計・件数、同数時のタイブレーク、
  代表開示＝金額最大の開示、最終開示日、`filerName`なしの除外、`limit`打ち切り、
  activistが銘柄別のままであること）。いずれもPASS
- `tsc --noEmit`・`eslint` パス。`next build` はコンパイル・型検査を通過
  （ページデータ収集はmicroCMSキー未設定の環境のため実行不可）

## 2026-08-18 kujira-watch: /trendingをオートページャー化（HTML 1.08MB → 225KB）

オーナー指示「/trendingはオートページャーにして」への対応。
別セッションが急増銘柄の件数制限を外して全件表示にした結果、480件が一度に描画され
HTMLが1.08MB(gzip 106KB)まで膨らみ、perf_check.ymlの閾値100KBを超えていた。

### 実装
`TrendingTable`を`"use client"`にし、初回30件だけ描画して下端のsentinelが見えたら
30件ずつ増やす。TOPの`InfiniteArticleList`と同じIntersectionObserver方式（rootMargin 600px）。

`/api/articles`のような追加取得は作っていない。集計結果（480件）はサーバー側で
すでに全件手元にあり、重いのはデータではなくカードのマークアップだったため
（480件で1.08MB＝1件あたり約2.3KB）、描画する件数だけを絞れば足りる。

- `hrefOf`/`noteOf`の関数propsは廃止し、`items: TrendingItem[]`（href・noteを解決済み）
  で受け取る。クライアントコンポーネントの境界を関数は越えられないため。
- ItemList構造化データは初回SSR分の30件のみに揃える（追加分はクライアント描画で
  クロール時点のHTMLには無いため。TOPのItemListと同じ規律）。
- sentinelは同時に「もっと見る（残りN件）」ボタンにしてある。IntersectionObserverが
  働かない環境（バックグラウンドタブ等）でも先に進めるようにするため。

### 確認
| ページ | 変更前 | 変更後 |
|---|---|---|
| /trending | 1,079KB raw / 106KB gzip / カード480枚 | **225KB raw / 46KB gzip / SSRカード30枚** |
| /ranking/trending | 125KB raw / 16KB gzip | 106KB raw / 16KB gzip |

ページ送りの状態遷移は実ブラウザで確認（30件→60件、ラベル「480件中60件」・
残り件数の更新まで）。IntersectionObserverの発火自体は、検証に使うプレビューペインが
`document.visibilityState = "hidden"`のため確認できていない（この環境ではObserverの
コールバックが配送されない）。実装はTOPで本番稼働中の`InfiniteArticleList`と同じ。
`tsc --noEmit`・eslintクリーン、`npm run build`成功。

## 2026-08-18 kujira-watch: デザインレビューP2（パンくず・チャート・分類バッジ）

デザインレビューの優先度2（一貫性・洗練）の3件を実装。P1は2026-08-17に対応済み。

### 修正内容
1. **記事パンくずが3行を占有**（375pxでフルタイトルが折り返し、本文到達前のノイズ）
   - ja/en の `articles/[id]/page.tsx`: flex＋`truncate`で1行ellipsisに。
     「トップ / 日付」側は`flex-none`で固定し、タイトルだけが縮む。
     SEO用のBreadcrumbList（JSON-LD）はフルタイトルのまま変更なし。
2. **株価チャートが線だけで水準感が読めない**
   - `CompanyInfoCard.tsx`: 期間高値・安値の点線ガイド（`--rule`）と終値のドット
     （`--color-brand-blue`）、右上に「期間高値◯円・安値◯円」のキャプションを追加。
     文字はSVG内に置くと`preserveAspectRatio="none"`の横伸縮で歪むためHTML側に出す。
     新規ライブラリ・アセットは追加していない（既存のインラインSVGのまま）。
3. **同じ「分類」が一覧と記事詳細で別デザイン**（一覧=色ドット、記事詳細=枠付きChip）
   - `CategoryBadge.tsx`: `DEAL_TYPE_COLORS[].dot` の分類色ドットを追加し
     `DealTypeBadge`と視覚的に揃えた。枠とリンクはカテゴリ一覧への導線として維持。

### 見送り
- ランキング冒頭の説明文の畳み込み: 別作業で既に2文＋FAQリンクに短縮済み。
- デスクトップの2カラム化: 効果は見込めるがRSC境界・ja/en両対応で大規模になるため、
  計測（回遊率）を整えてから判断する。

### 確認
- 375px/1280px × ja/en で目視確認。パンくずは41px(1行)、横スクロール無しを
  `scrollWidth`で実測。`tsc --noEmit`・eslintクリーン。

## 2026-08-18 kujira-watch: カード一覧の高さのばらつきを解消

オーナー指摘「カードUIの大きさのずれが気になる。統一して」への対応。
`/investors`で、同じ行の左右のカードの下端が揃っていなかった（実例: 「三光起業株式会社」
87px と「寺井 秀藏」67px が隣り合う）。

### 原因
2つ重なっていた。
1. **カードが行の高さいっぱいに伸びていなかった**。`<li><a class="card">`という構造で、
   グリッドアイテムである`li`は引き伸ばされるが、中の`a`は内容分の高さしか持たない。
2. **カード内の行数がカードごとに変わっていた**。分類ラベルと開示メタを同じ行に
   flex-wrapで流していたため、「創業家の資産管理会社」のような長い分類のときだけ
   2行に折り返り、カードが1行ぶん高くなっていた。

### 実装
- `globals.css`: `.card-grid > li:not(.card)`を`display: flex`、その直下の`.card`を
  `flex-grow: 1`に。`li`そのものが`.card`の一覧（/disclosures・/activists・/ranking/[slug]）は
  グリッドアイテムとして自動で揃うので`:not(.card)`で除外する。
- `/investors`: 「1行目=名前／2行目=分類／3行目=開示メタ」の3行固定に変更。

### 確認
1280pxで実測。`/investors`は行内の高さ不一致0件、`/monthly`は全15枚が95pxで完全に一致、
`/stocks`も行内不一致0件。行をまたいだ高さの差（`/stocks` 82px/106px、`/investors` 90px/114px）は
銘柄名・投資家名が2行に折り返すかどうかによるもので、名前を省略しない限り解消できない。
`tsc --noEmit`・eslintクリーン、`npm run build`成功。

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
2. **カードUI**（`/disclosures` `/trending` `/ranking` `/ranking/[slug]` `/activists` `/investors` `/stocks` `/monthly`）
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
   - `/ranking/[slug]`（buys/sells/filings/activist）も同じ2段カードに。上位30件固定でページ送りは無い。
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
