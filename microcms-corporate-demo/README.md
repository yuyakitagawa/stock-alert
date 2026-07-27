# コーポレートサイト リプレイスDEMO（microCMS + Vercel）

「よくあるコーポレートサイト（東京ガス公式サイトのような構成）を、microCMS + Vercel で作り替えたらどうなるか」を
社内・クライアント向けに見せるためのデモサイト。架空の企業「みらいエネルギー株式会社」のサイトとして構築しており、
実在の企業・団体とは関係ない。本番運用は想定しない。

## 見せたいポイント

- **トップページの構成**: ヒーロービジュアル／お知らせバー／サービス一覧／IR・サステナビリティ導線／ニュースリリースといった、
  よくあるコーポレートサイトの型をNext.js + Tailwind CSSで再現。
- **ニュースリリースのCMS化**: 「お知らせ」「プレスリリース」「IR情報」「サステナビリティ」のカテゴリを持つニュースを
  microCMSで管理し、ノーコードで更新できる導線（一覧・詳細・ページネーション）を再現。
- **Vercelへのデプロイ前提**: ISR（`revalidate = 60`）でmicroCMSの更新を数十秒〜1分程度で反映。

## スタック

- Next.js 16 (App Router) + TypeScript
- Tailwind CSS v4（`@tailwindcss/typography` でリッチテキスト本文を装飾）
- microCMS（`microcms-js-sdk`）
- Vercel想定（ISR: `revalidate = 60`）

## セットアップ

```bash
npm install
npm run dev
```

**`MICROCMS_SERVICE_DOMAIN` / `MICROCMS_API_KEY` を設定しなくても、モックデータで即座に動作する。**
デモをその場で見せる用途ではこのままでよい。

実際のmicroCMSサービスに接続する場合は以下を設定する。

```bash
cp .env.local.example .env.local
```

```
MICROCMS_SERVICE_DOMAIN=xxxx
MICROCMS_API_KEY=xxxx
```

## microCMS側の前提（APIスキーマ: `news`）

管理画面のGUIで以下のフィールドを持つ `news` エンドポイント（リスト形式）を作成する。

| フィールドID | 表示名 | 型 | 必須 |
|---|---|---|---|
| title | タイトル | テキスト | ○ |
| summary | 概要 | テキスト | ○ |
| body | 本文 | リッチエディタ | ○ |
| category | カテゴリ | セレクトフィールド（お知らせ／プレスリリース／IR情報／サステナビリティ） | ○ |

環境変数が未設定の間は `src/lib/mock-news.ts` の12件のモックデータがそのまま表示される。

## ページ構成

| パス | 内容 |
|---|---|
| `/` | トップページ（ヒーロー・お知らせ・サービス一覧・IR/サステナビリティ導線・最新ニュース5件） |
| `/news` | ニュースリリース一覧（新着順、6件ページネーション、`?page=`） |
| `/news/[id]` | ニュースリリース詳細 |

## 実装メモ

- `src/lib/microcms.ts` は環境変数の有無で実クライアント／モックデータを自動切り替えする。microCMSアカウントを
  用意しなくても `npm run dev` だけでフルのデモ画面を確認できる。
- 一覧はページネーションに `searchParams`（`?page=`）を使うため実行時に動的レンダリングされるが、microCMSへの
  `fetch` 自体は `next: { revalidate: 60 }` を指定しており、Next.jsのData Cacheが60秒間キャッシュ・再検証を行う
  （App RouterにおけるISRの実体）。
- ニュース詳細（`/news/[id]`）は動的APIを使わないため `export const revalidate = 60` をルートセグメントに設定し、
  オンデマンドISR（初回アクセス時に生成し60秒キャッシュ）として動作する。
- 本文（リッチエディタのHTML）は `dangerouslySetInnerHTML` + Tailwind Typography(`prose`)で描画。
- デザインは東京ガス公式サイトの構成を参考に、ブランドブルー(`#0068b7`)＋ネイビーを基調に、IR/サステナビリティで
  ゴールド・グリーンのアクセントを添えている（`src/app/globals.css` のCSS変数で調整可）。

## スコープ外

認証・会員機能、検索機能、コメント機能、実在企業のロゴ・写真素材の再現、本番デプロイ後の運用・監視。
