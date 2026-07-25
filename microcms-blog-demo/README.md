# 大口取引解説ブログ（microCMS検証用ダミーサイト）

microCMSの操作感・API設計を検証するためのダミーサイト。本番運用は想定しない。

## スタック

- Next.js 16 (App Router) + TypeScript
- Tailwind CSS v4（`@tailwindcss/typography` でリッチテキスト本文を装飾）
- microCMS（`microcms-js-sdk`）
- Vercel想定（ISR: `revalidate = 60`）

## セットアップ

```bash
npm install
cp .env.local.example .env.local
```

`.env.local` に microCMS サービスの値を設定する。

```
MICROCMS_SERVICE_DOMAIN=xxxx
MICROCMS_API_KEY=xxxx
```

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
| dealType | 大口取引種別 | セレクトフィールド（機関投資家買い／インサイダー買い／自社株買い／ETFフロー／その他） | ○ |
| dealDate | 取引日 | 日付 | ○ |
| dealAmount | 金額規模（億円） | 数値 | ○ |
| sourceUrl | 出典URL | テキスト | △ |
| category | カテゴリ | セレクトフィールド（決算前動向／インサイダー／ETFフロー／その他） | ○ |
| tags | タグ | テキスト（カンマ区切り） | △ |
| eyecatch | アイキャッチ画像 | 画像 | △ |

## ページ構成

| パス | 内容 |
|---|---|
| `/` | 記事一覧（新着順、10件ページネーション、`?page=`） |
| `/articles/[id]` | 記事詳細 |
| `/category/[category]` | カテゴリ別一覧（`?page=` 対応） |

## 実装メモ

- 一覧・カテゴリ別一覧はページネーションに `searchParams`（`?page=`）を使うため、その2ルートは実行時に動的レンダリングされる。ただしmicroCMSへの `fetch` 自体は `next: { revalidate: 60 }` を指定しており、Next.jsのData Cacheが60秒間キャッシュ・再検証を行う（App RouterにおけるISRの実体）。
- 記事詳細（`/articles/[id]`）は動的APIを使わないため `export const revalidate = 60` をルートセグメントに設定し、オンデマンドISR（初回アクセス時に生成し60秒キャッシュ）として動作する。
- 本文（リッチエディタのHTML）は `dangerouslySetInnerHTML` + Tailwind Typography(`prose`)で描画。
- 画像は `next/image` を使用し、`next.config.ts` の `images.remotePatterns` で `images.microcms-assets.io` を許可。

## 検証観点

- スキーマ変更時のフロントエンド追従のしやすさ → `src/types/article.ts` の `Article` 型と `src/lib/microcms.ts` を変更すれば追従できる構成にしている。
- リッチエディタ出力HTMLの扱いやすさ → `@tailwindcss/typography` の `prose` クラスで装飾。
- REST APIレスポンス速度・データ構造の使い勝手 → `microcms-js-sdk` の `getList` / `getListDetail` をそのまま利用。
- 管理画面での記事投稿のしやすさ（非エンジニア視点） → 人間側で確認。

## スコープ外

認証・会員機能、検索機能、コメント機能、本番デプロイ後の運用・監視。
