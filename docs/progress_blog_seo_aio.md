# ブログ SEO/AIO最適化 + トップページ改善 + ドメイン変更

対象: `microcms-blog-demo`（Vercelデプロイ: https://stock-alert-lyart.vercel.app/）

## 背景・ゴール
現状は「microCMS検証用ダミーサイト」という位置付けだが、実際に読まれるブログへ格上げする。
- Google/AIO（AI Overview・LLM引用）両方で見つかりやすくする
- トップページを人気ブログのような見た目・構成にする
- 独自ドメインへ切替える

## 未確定事項（ユーザー判断待ち・作業はブロックしない設計にする）
- [ ] 独自ドメイン名: 未取得。取得・Vercelへの追加・DNS設定はダッシュボード作業のためユーザー側で実施が必要。
- [ ] ブランド名: 現状「大口取引解説ブログ」を維持するか改名するか未定。
- 対応方針: `NEXT_PUBLIC_SITE_URL` / `NEXT_PUBLIC_SITE_NAME` の環境変数化で、後から値を変えるだけで全ページに反映される設計にする（コード変更なしでドメイン・ブランド名切替可能）。

## ステージ一覧（トークン分割実行・各ステージ単位でコミット）
- [x] Stage 0: 計画・進捗ファイル作成
- [x] Stage 1: 技術的SEO基盤（metadata/canonical/OGP/sitemap/robots/構造化データ）
- [x] Stage 2: AIO対策（llms.txt、E-E-A-T用の運営者情報・免責事項ページ、パンくず）
- [x] Stage 3: トップページの「人気ブログ」化（ヒーロー/注目記事、カード強化、フッター導線）— 一次対応。タグクラウド・検索・アイキャッチ画像表示は未着手。
- [ ] Stage 4: 計測（Vercel Analytics導入）
- [ ] Stage 5: 独自ドメイン切替（ユーザー作業 + 切替後のコード側対応）

## Stage詳細

### Stage 1: 技術的SEO基盤
- `src/lib/site.ts`: SITE_URL / SITE_NAME / SITE_DESCRIPTION を環境変数優先で定義
- `layout.tsx`: metadataBase, title template, OGP, Twitter Card, canonical, robots, JSON-LD (Organization/WebSite)
- `articles/[id]/page.tsx`: generateMetadata（動的title/description/OGP article/canonical）+ JSON-LD (Article) + BreadcrumbList
- `category/[category]/page.tsx`: generateMetadata + canonical
- `app/sitemap.ts`: 全記事・カテゴリ・トップを含む動的サイトマップ
- `app/robots.ts`
- README更新（ダミーサイト表記の見直し、SEO対応の追記）

### Stage 2: AIO対策
- `public/llms.txt`: サイトの目的・データソース(EDINET)・主要パスをLLMクローラ向けに明記
- `/about`（運営者情報）: データソース・更新頻度・免責事項（投資助言ではない旨）→ YMYL領域のE-E-A-T対策
- パンくずリスト表示（UI）

### Stage 3: トップページ「人気ブログ」化
- 最新記事の中からピックアップ（ヒーロー枠）
- ArticleCardのビジュアル強化（NEWバッジ等）
- フッターに About/免責事項/カテゴリ導線を追加

### Stage 4: 計測
- `@vercel/analytics` 導入（ドメイン非依存で先行導入可能）

### Stage 5: 独自ドメイン切替（要ユーザー作業）
1. （ユーザー）ドメイン取得
2. （ユーザー）Vercelプロジェクト設定 → Domains に追加、指示されたDNSレコードをレジストラ側に設定
3. （私）Vercelの環境変数 `NEXT_PUBLIC_SITE_URL` を新ドメインに更新して再デプロイ
4. （私）canonical/sitemap/OGP/構造化データが新ドメインを指すことを確認
5. （ユーザー）Google Search Consoleで新ドメインを検証・サイトマップ再送信
