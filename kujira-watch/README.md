# クジラウォッチ

EDINET大量保有報告書などの公開情報をもとに、機関投資家・インサイダー・自社株買いなど
「クジラ」（相場を動かすほどの資金力を持つ大口投資家の俗称）の動きを監視・解説するブログ。
SEO/AIO（AI Overview・LLM引用）対策済み。

デプロイ先: https://kujira-watch.com/ （旧URL: https://stock-alert-lyart.vercel.app/ 。
進捗はリポジトリルートの `docs/progress_blog_seo_aio.md` を参照）

## スタック

- Next.js 16 (App Router) + TypeScript
- Tailwind CSS v4（`@tailwindcss/typography` でリッチテキスト本文を装飾）
- microCMS（`microcms-js-sdk`）
- Supabase（`@supabase/supabase-js`。フッターの累計訪問者カウンター用。トレーディングシステム側と同じプロジェクトの`blog_visit_counter`テーブル+`increment_blog_visit_counter` RPC）
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
NEXT_PUBLIC_SITE_NAME=クジラウォッチ
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
| tags | タグ | テキスト（カンマ区切り） | △ |
| eyecatch | アイキャッチ画像 | 画像 | △ |

## ページ構成

| パス | 内容 |
|---|---|
| `/` | 記事一覧（先頭記事はヒーロー枠でピックアップ表示、新着順10件ページネーション、`?page=`） |
| `/articles/[id]` | 記事詳細 |
| `/category/[category]` | カテゴリ別一覧（`?page=` 対応） |
| `/stocks/[code]` | 銘柄別の大量保有・自社株買い履歴まとめ（同一`stockCode`の記事を`-dealDate`順に一覧表示）。記事詳細の「銘柄」欄から内部リンクあり |
| `/about` | 運営者情報・データソース・免責事項（E-E-A-T対策） |
| `/sitemap.xml` | 動的サイトマップ（`src/app/sitemap.ts`、全記事・カテゴリ・銘柄別ページを含む） |
| `/robots.txt` | `src/app/robots.ts` |
| `/feed.xml` | RSSフィード（新着記事20件、`src/app/feed.xml/route.ts`）。フッター・`<head>`の`alternate`リンク・`llms.txt`から参照 |
| `/api/counter` | フッターの累計訪問者カウンター用（POST、`increment_blog_visit_counter` RPCを呼ぶ） |

## 計測・ログ

- **累計訪問者カウンター**: フッターに表示（`src/components/VisitCounter.tsx`）。ページ読み込み時に `/api/counter` を叩き、Supabaseの `blog_visit_counter`（単一行）をアトミックにインクリメントして返す。
- **クローラーアクセスログ**: `src/proxy.ts`（Next.js 16で`middleware`から改称された`proxy`規約）が全リクエストのUser-Agentを見て、Googlebot/Bingbot/GPTBot/ClaudeBot等の既知クローラーだけをSupabaseの `blog_crawler_log` に記録する（一般訪問者は記録しない）。パターン一覧は `src/lib/crawlers.ts`。ログはSupabaseダッシュボードのTable Editorから直接閲覧・CSVエクスポートできる。
- どちらも `SUPABASE_URL`/`SUPABASE_SERVICE_KEY`（トレーディングシステム側と同じSupabaseプロジェクト）が必要。未設定でもビルド・記事表示自体には影響しない（カウンターAPI呼び出し時にのみエラーになるが、フロント側は握りつぶして非表示にする）。

## SEO/AIO対策

- **metadata**: `src/lib/site.ts` の `SITE_URL`/`SITE_NAME` を起点に、ルートレイアウトで `metadataBase`・タイトルテンプレート（`${SITE_NAME}｜%s` の順。記事タイトルが長いとブラウザタブで末尾が切れるため、サイト名を先頭に置いている）・OGP・Twitter Card・`robots` を設定。記事詳細・カテゴリ別一覧は `generateMetadata` で動的に title/description/canonical/OGPを生成する。
- **アイコン/OGP画像**: `src/app/icon.tsx`（ファビコン）・`src/app/opengraph-image.tsx`（SNSシェア用1200x630）は `next/og` の `ImageResponse` でクジラ絵文字🐋をブランドブルー背景に合成して動的生成（Next.jsのファイルベースmetadata規約、画像アセット不要）。
- **構造化データ (JSON-LD)**: ルートレイアウトに `WebSite`/`Organization`、記事詳細に `Article`（`about`に銘柄名・証券コード、`citation`に出典URL）と `BreadcrumbList` を埋め込み。Google/AI Overview双方の情報抽出を想定。
- **サイトマップ**: `src/app/sitemap.ts` はビルド時ではなくリクエスト時に生成（`dynamic = "force-dynamic"`）。microCMSの一時的な障害でVercelのビルド自体が失敗しないようにするため。
- **AIO向け**: `public/llms.txt` にサイトの目的・データソース・主要パスをLLMクローラ向けに明記。
- **E-E-A-T**: `/about` にデータソース・算出方法・免責事項を明記し、フッター/ヘッダーから常時リンク。

## 実装メモ

- 一覧・カテゴリ別一覧はページネーションに `searchParams`（`?page=`）を使うため、その2ルートは実行時に動的レンダリングされる。ただしmicroCMSへの `fetch` 自体は `next: { revalidate: 60 }` を指定しており、Next.jsのData Cacheが60秒間キャッシュ・再検証を行う（App RouterにおけるISRの実体）。
- 記事詳細（`/articles/[id]`）は動的APIを使わないため `export const revalidate = 60` をルートセグメントに設定し、オンデマンドISR（初回アクセス時に生成し60秒キャッシュ）として動作する。
- 本文（リッチエディタのHTML）は `dangerouslySetInnerHTML` + Tailwind Typography(`prose`)で描画。
- `eyecatch`（アイキャッチ画像）はカード一覧・ヒーロー枠・記事詳細で表示する（未設定の記事はテキスト中心のレイアウトにフォールバック）。記事詳細では`generateMetadata`のOGP画像としても使う。
- デザインは東京ガス公式サイトを参考に、ブランドブルー(`#0068b7`)＋ネイビー＋ゴールドのアクセント、アウトライン型ピルバッジを採用（`src/app/globals.css` のCSS変数で調整可）。カードはホバーで浮き上がるトランジション付き、本文フォントは`next/font/google`のNoto Sans JPを使用（日本語の表示品質向上のため、旧来のArial/Helveticaフォールバックから変更）。
- 記事一覧（TOP・カテゴリ別一覧・銘柄別履歴）は取引日(`dealDate`)の新しい順、同日内は金額規模(`dealAmount`)の大きい順にソートし（`src/lib/microcms.ts` の `orders: "-dealDate,-dealAmount"`）、`src/lib/groupByDealDate.ts` で取引日ごとに見出しを付けて表示する（見出しは`src/components/DealDateHeading.tsx`で3ページ共通）。「いつの話か」が一覧性で分かるようにするため。
- ヘッダーのロゴ（🐋アイコン）・カテゴリ別一覧のパンくずリストから常にTOPへ戻れる（記事詳細・銘柄別履歴には既存のパンくずリストあり）。運営者情報リンクはフッターのみに配置（ヘッダーはロゴとカテゴリのみでシンプルに保つ）。
- ヘッダーのカテゴリフィルターはスマホ幅では折り返さず横スクロール1行にし（`.no-scrollbar`、`src/app/globals.css`）、13カテゴリぶんが縦に何行も積み重なって本文を押し下げないようにしている。sm以上（タブレット・PC幅）では通常の折り返し表示に戻る。
- カテゴリフィルター（`/category/[category]`）はmicroCMS側に別フィールドを持たず、`dealType`の値をそのままカテゴリ名として使う（`src/types/article.ts` の `categoryLabel`/`DEAL_TYPE_BY_CATEGORY`、値はidentity）。CMS側の選択肢リストをdealTypeの分類と別途同期させる必要が無く、選択肢の同期漏れによる不具合が起きない構成にしている。

## コンテンツの自動生成（任意）

リポジトリルートの `web/publish_blog_articles.py` が、EDINET大量保有報告書（買い方向のみ）を基にClaudeで解説記事を生成し、このAPIへ即時投稿する（GitHub Actions `daily_alert.yml` Step 5c、日次）。取得金額(億円)はyfinanceの発行済株式数×株価×保有比率変化からの推定値であることを本文に明記させている。

`dealType`（提出者の投資家分類）は、Supabaseの`edinet_filer_classification`マスター（Web検索で確認済みの投資家分類テーブル、バックテスト分析とも共用）をまず参照し、未登録の提出者のみClaudeの一般知識で判定して結果をマスターへ保存する（`web/publish_blog_articles.py`の`classify_filer()`）。キーワード一致だけでは日系/外資の区別やスペース無し個人名を正しく判定できないため。判定不能な場合は「その他」に丸める。

投稿後の内容確認・修正はmicroCMS管理画面で人間が行う想定。詳細はスクリプト冒頭のdocstringを参照。

## スコープ外

認証・会員機能、検索機能、コメント機能。
