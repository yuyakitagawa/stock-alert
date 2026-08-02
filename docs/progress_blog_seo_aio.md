# ブログ SEO/AIO最適化 + トップページ改善 + ドメイン変更

対象: `kujira-watch/`（旧ディレクトリ名: `microcms-blog-demo`。デプロイ先: https://kujira-watch.com/ 、旧URL: https://stock-alert-lyart.vercel.app/）

## 追加タスク: ディレクトリリネーム
- [x] `microcms-blog-demo/` → `kujira-watch/` にリネーム（`git mv`でhistory維持）、`package.json`/`package-lock.json`の`name`も追従
- [ ]（ユーザー）Vercelプロジェクト設定 → General → Root Directory を `microcms-blog-demo` から `kujira-watch` に変更（**これをしないと次回デプロイが失敗する**）

## 背景・ゴール
現状は「microCMS検証用ダミーサイト」という位置付けだが、実際に読まれるブログへ格上げする。
- Google/AIO（AI Overview・LLM引用）両方で見つかりやすくする
- トップページを人気ブログのような見た目・構成にする
- 独自ドメインへ切替える

## 未確定事項（ユーザー判断待ち・作業はブロックしない設計にする）
- [x] ブランド名: 「クジラウォッチ」に決定・反映済み（`NEXT_PUBLIC_SITE_NAME`のデフォルト値、「クジラ」=大口投資家の俗称という説明を`/about`に追記）。
- [x] 独自ドメイン名: `kujira-watch.com` に決定。コード側（`.env.local.example`/README記載）は反映済み。
- [x] Vercel側のドメイン接続・DNS設定: 完了（Vercel Domainsで`kujira-watch.com`がValid）。`src/lib/site.ts`のフォールバックデフォルトも`kujira-watch.com`に切り替え済み。
- [ ] ブラウザで「保護されていない通信」の警告が出る問題を調査中（Vercel側はValid表示。ブラウザの古いセキュリティ例外の可能性が高いが未解決）。
- [ ] Vercelの`NEXT_PUBLIC_SITE_URL`環境変数への明示的な設定（未設定でもコード側のデフォルトが`kujira-watch.com`になったため必須ではないが、明示設定を推奨）。
- 対応方針: `NEXT_PUBLIC_SITE_URL` / `NEXT_PUBLIC_SITE_NAME` の環境変数化で、後から値を変えるだけで全ページに反映される設計にする（コード変更なしでドメイン・ブランド名切替可能）。

## ステージ一覧（トークン分割実行・各ステージ単位でコミット）
- [x] Stage 0: 計画・進捗ファイル作成
- [x] Stage 1: 技術的SEO基盤（metadata/canonical/OGP/sitemap/robots/構造化データ）
- [x] Stage 2: AIO対策（llms.txt、E-E-A-T用の運営者情報・免責事項ページ、パンくず）
- [x] Stage 3: トップページの「人気ブログ」化（ヒーロー/注目記事、カード強化、フッター導線）— 一次対応。タグクラウド・検索・アイキャッチ画像表示は未着手。
- [x] Stage 4: 計測（`@vercel/analytics` + `@vercel/speed-insights`導入）+ アイキャッチ画像表示（カード一覧/ヒーロー/記事詳細/OGP画像）
- [x] Stage 5: 独自ドメイン切替（`kujira-watch.com`。DNS/Vercel接続はユーザー側で完了、コード側のデフォルト値切替も完了。SSL警告の原因調査とGSC登録が残タスク）

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
- `@vercel/analytics` 導入（ページビュー等のアクセス解析。Vercelダッシュボード側で「Web Analytics」を有効化する必要あり）
- `@vercel/speed-insights` 導入（Core Web Vitals計測。SEOのランキング要因にも直結）

### Stage 5: 独自ドメイン切替（`kujira-watch.com`）
1. [x]（ユーザー）ドメイン`kujira-watch.com`をお名前.comで取得
2. [x]（ユーザー）Vercelプロジェクト設定 → Domains に`kujira-watch.com`を追加、DNSレコード(A/CNAME)をお名前.com側に設定 → Valid確認済み
3. [ ]（ユーザー）VercelのEnvironment Variablesに `NEXT_PUBLIC_SITE_URL=https://kujira-watch.com` を明示設定（任意。コード側のデフォルト値は既に切替済み）
4. [x]（私）`src/lib/site.ts`のフォールバックデフォルトを`kujira-watch.com`に切り替え
5. [ ]（ユーザー）Google Search Consoleで新ドメインを検証・サイトマップ再送信
6. [ ]（調査中）ブラウザで「保護されていない通信」警告が出る件の原因切り分け（Vercel側はValidのため、ブラウザの古いセキュリティ例外の可能性が高い）
