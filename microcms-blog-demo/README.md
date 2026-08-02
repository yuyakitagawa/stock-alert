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
| dealType | 大口取引種別 | セレクトフィールド（機関投資家買い[レガシー]／インサイダー買い／日系ファンド買い／外資系ファンド買い／ベンチャーキャピタル買い／財団買い／日系企業買い／外資系企業買い／自社株買い／ETFフロー／その他） | ○ |
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
| `/about` | 運営者情報・データソース・免責事項（E-E-A-T対策） |
| `/sitemap.xml` | 動的サイトマップ（`src/app/sitemap.ts`、全記事・カテゴリを含む） |
| `/robots.txt` | `src/app/robots.ts` |

## SEO/AIO対策

- **metadata**: `src/lib/site.ts` の `SITE_URL`/`SITE_NAME` を起点に、ルートレイアウトで `metadataBase`・タイトルテンプレート・OGP・Twitter Card・`robots` を設定。記事詳細・カテゴリ別一覧は `generateMetadata` で動的に title/description/canonical/OGPを生成する。
- **構造化データ (JSON-LD)**: ルートレイアウトに `WebSite`/`Organization`、記事詳細に `Article`（`about`に銘柄名・証券コード、`citation`に出典URL）と `BreadcrumbList` を埋め込み。Google/AI Overview双方の情報抽出を想定。
- **サイトマップ**: `src/app/sitemap.ts` はビルド時ではなくリクエスト時に生成（`dynamic = "force-dynamic"`）。microCMSの一時的な障害でVercelのビルド自体が失敗しないようにするため。
- **AIO向け**: `public/llms.txt` にサイトの目的・データソース・主要パスをLLMクローラ向けに明記。
- **E-E-A-T**: `/about` にデータソース・算出方法・免責事項を明記し、フッター/ヘッダーから常時リンク。

## 実装メモ

- 一覧・カテゴリ別一覧はページネーションに `searchParams`（`?page=`）を使うため、その2ルートは実行時に動的レンダリングされる。ただしmicroCMSへの `fetch` 自体は `next: { revalidate: 60 }` を指定しており、Next.jsのData Cacheが60秒間キャッシュ・再検証を行う（App RouterにおけるISRの実体）。
- 記事詳細（`/articles/[id]`）は動的APIを使わないため `export const revalidate = 60` をルートセグメントに設定し、オンデマンドISR（初回アクセス時に生成し60秒キャッシュ）として動作する。
- 本文（リッチエディタのHTML）は `dangerouslySetInnerHTML` + Tailwind Typography(`prose`)で描画。
- `eyecatch`（アイキャッチ画像）はカード一覧・ヒーロー枠・記事詳細で表示する（未設定の記事はテキスト中心のレイアウトにフォールバック）。記事詳細では`generateMetadata`のOGP画像としても使う。
- デザインは東京ガス公式サイトを参考に、ブランドブルー(`#0068b7`)＋ネイビー＋ゴールドのアクセント、アウトライン型ピルバッジを採用（`src/app/globals.css` のCSS変数で調整可）。
- カテゴリフィルター（`/category/[category]`）はmicroCMS側に別フィールドを持たず、`dealType`から「買い」を除いた値をフロントエンドでその場で導出する（`src/types/article.ts` の `categoryLabel`/`DEAL_TYPE_BY_CATEGORY`）。CMS側の選択肢リストをdealTypeの分類と別途同期させる必要が無く、選択肢の同期漏れによる不具合が起きない構成にしている。

## コンテンツの自動生成（任意）

リポジトリルートの `web/publish_blog_articles.py` が、EDINET大量保有報告書（買い方向のみ）を基にClaudeで解説記事を生成し、このAPIへ即時投稿する（GitHub Actions `daily_alert.yml` Step 5c、日次）。取得金額(億円)はyfinanceの発行済株式数×株価×保有比率変化からの推定値であることを本文に明記させている。

`dealType`（提出者が個人/日系ファンド/外資系ファンド/VC/財団/日系企業/外資系企業のどれか）はキーワード一致ではなく、記事本文生成と同じClaude呼び出しで提出者名から一般知識で判定させている（キーワード一致だけでは日系/外資の区別や、スペース無し個人名を正しく判定できないため）。判定不能な場合は「その他」に丸める。

投稿後の内容確認・修正はmicroCMS管理画面で人間が行う想定。詳細はスクリプト冒頭のdocstringを参照。

## スコープ外

認証・会員機能、検索機能、コメント機能。
