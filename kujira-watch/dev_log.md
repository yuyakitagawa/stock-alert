
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
