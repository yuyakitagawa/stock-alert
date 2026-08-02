# ブログ SEO/AIO最適化 + トップページ改善 + ドメイン変更

対象: `kujira-watch/`（旧ディレクトリ名: `microcms-blog-demo`。デプロイ先: https://kujira-watch.com/ 、旧URL: https://stock-alert-lyart.vercel.app/）

## 追加タスク: ディレクトリリネーム
- [x] `microcms-blog-demo/` → `kujira-watch/` にリネーム（`git mv`でhistory維持）、`package.json`/`package-lock.json`の`name`も追従
- [x]（ユーザー）Vercelプロジェクト設定 → General → Root Directory を `microcms-blog-demo` から `kujira-watch` に変更済み、デプロイ成功確認済み

## 背景・ゴール
現状は「microCMS検証用ダミーサイト」という位置付けだが、実際に読まれるブログへ格上げする。
- Google/AIO（AI Overview・LLM引用）両方で見つかりやすくする
- トップページを人気ブログのような見た目・構成にする
- 独自ドメインへ切替える

## 未確定事項（ユーザー判断待ち・作業はブロックしない設計にする）
- [x] ブランド名（日本語表示名）: 「大口投資家の監視ブログ」に最終決定（一時「クジラウォッチ」にしていたが差し戻し）。`NEXT_PUBLIC_SITE_NAME`のデフォルト値、「クジラ」=大口投資家の俗称という説明は`/about`・ヘッダーサブ見出しに残す。
- [x] 独自ドメイン名: `kujira-watch.com` に決定（ブランド名とは独立して維持）。コード側（`.env.local.example`/README記載）は反映済み。
- [x] Vercel側のドメイン接続・DNS設定: 完了（Vercel Domainsで`kujira-watch.com`がValid）。`src/lib/site.ts`のフォールバックデフォルトも`kujira-watch.com`に切り替え済み。
- [ ] ブラウザで「保護されていない通信」の警告が出る問題を調査中（Vercel側はValid表示。ブラウザの古いセキュリティ例外の可能性が高いが未解決）。
- [ ] Vercelの`NEXT_PUBLIC_SITE_URL`環境変数への明示的な設定(未設定でもコード側のデフォルトが`kujira-watch.com`になったため必須ではないが、明示設定を推奨)。
- [x] Google Search Console ドメイン検証（DNS TXTレコード）・サイトマップ送信完了。
- [x] SSL「保護されていない通信」警告の原因判明: サイト自体は正常(スマホ・別端末では警告なし)。PCブラウザ(Chrome)に古いセキュリティ例外がキャッシュされていたのが原因。ユーザー側でリセット案内済み。
- 対応方針: `NEXT_PUBLIC_SITE_URL` / `NEXT_PUBLIC_SITE_NAME` の環境変数化で、後から値を変えるだけで全ページに反映される設計にする（コード変更なしでドメイン・ブランド名切替可能）。

## ステージ一覧（トークン分割実行・各ステージ単位でコミット）
- [x] Stage 0: 計画・進捗ファイル作成
- [x] Stage 1: 技術的SEO基盤（metadata/canonical/OGP/sitemap/robots/構造化データ）
- [x] Stage 2: AIO対策（llms.txt、E-E-A-T用の運営者情報・免責事項ページ、パンくず）
- [x] Stage 3: トップページの「人気ブログ」化（ヒーロー/注目記事、カード強化、フッター導線）— 一次対応。タグクラウド・検索・アイキャッチ画像表示は未着手。
- [x] Stage 4: 計測（`@vercel/analytics` + `@vercel/speed-insights`導入）+ アイキャッチ画像表示（カード一覧/ヒーロー/記事詳細/OGP画像）
- [x] Stage 5: 独自ドメイン切替（`kujira-watch.com`。DNS/Vercel接続・GSC検証・サイトマップ送信・SSL警告の原因判明まで完了）
- [x] Stage 6: 訪問者カウンター＋クローラーログ（Supabase連携、下記詳細）
- [ ] Stage 7: Google Analytics(GA4)導入（要ユーザーの測定ID）
- [x] Stage 8: 閲覧者数向上 Cycle 1 — 銘柄別まとめページ＋記事本数上限引き上げ（下記詳細）

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
5. [x]（ユーザー）Google Search ConsoleでDNS TXTレコード検証・サイトマップ送信
6. [x] SSL警告の原因判明: サイトは正常、PCブラウザのキャッシュが原因

### Stage 6: 訪問者カウンター＋クローラーログ（Supabase連携）
- Supabaseプロジェクト`stock-alert`（`kxrgyguowxtjqexvmlgx`、トレーディングシステムと共用）に2テーブル追加
  - `blog_visit_counter`: 単一行のカウンター。`increment_blog_visit_counter()` RPCでアトミックにインクリメント
  - `blog_crawler_log`: Googlebot/GPTBot/ClaudeBot等の既知クローラーのアクセスログ（一般訪問者は記録しない）
- `src/lib/supabase.ts`: サーバー専用クライアント（`SUPABASE_URL`/`SUPABASE_SERVICE_KEY`）
- `src/app/api/counter/route.ts`: カウンター増分API、`src/components/VisitCounter.tsx`をフッターに表示
- `src/proxy.ts`（Next.js 16で`middleware`から改称）+ `src/lib/crawlers.ts`: User-Agentでクローラー判定しログ記録
- ログはSupabaseダッシュボードのTable Editorから閲覧・CSVエクスポート可能
- Vercelの環境変数に`SUPABASE_URL`/`SUPABASE_SERVICE_KEY`の設定が必要（ユーザー側の作業、値はリポジトリルートの`.env`と同じ）

### Stage 7: Google Analytics(GA4)導入（未着手）
- ユーザーがGA4プロパティを作成し測定ID(`G-XXXXXXXXXX`)を発行後、`@next/third-parties`の`GoogleAnalytics`コンポーネントで組み込み予定

### Stage 8: 閲覧者数向上 PDCAサイクル（進行中）

**運用方針**: ユーザーからの明示的な開始指示で1サイクル進める。実際のユーザーには接触できないため
「AIペルソナによるコードベースレビュー」を代替手段とし、Google Search Console等の実データが
溜まり次第そちらを優先する（本物のユーザーリサーチの代替にはならない点は都度明記する）。

**Cycle 1（完了）**
- Plan: コードレビューによるペルソナ所見 — (1)専門用語（dealType）の説明導線が記事から辿れない、
  (2)検索流入者向けの内部リンク（銘柄の他の記事）が無い、(3)銘柄を継続的に追う手段が無い。
  直近14日のEDINET候補が1日25〜164件に対し投稿上限が3件と大きく余裕があることも確認。
- Do:
  - `/stocks/[code]`（銘柄別の大量保有・自社株買い履歴まとめページ）を新設。`getArticlesByStockCode()`
    追加、`ItemList`+`BreadcrumbList`のJSON-LD、sitemap.tsに追加
  - 記事詳細ページの「銘柄」欄から `/stocks/[code]` への内部リンクを追加
  - `web/publish_blog_articles.py`の`MAX_ARTICLES_PER_RUN`を3→10に引き上げ
- Check: `npx tsc --noEmit`/`npm run lint`/`npm run build`成功。Pythonの既存テスト全件(tests/test_*.py、
  100件超)成功、今回の変更による新規失敗なし。
- Act: PR #192としてマージ済み。

**Cycle 2（完了）**
- Plan: 投資家分類（14分類）がバッジで表示されるが、初心者には意味が分からず記事から説明に
  辿る導線もない（Cycle 1の所見(1)）。他セッションの分類14分類化(`classify_filer()`)と
  タイミングが合ったため定義文言をフロントエンドに移植する形で対応。
- Do:
  - `src/lib/dealTypeInfo.ts`: `classify_filer()`の判断基準に準拠した14分類の説明文を追加
  - `DealTypeBadge`にHTML `title`属性でツールチップ表示
  - 記事詳細ページのバッジ直下に該当分類の説明1文＋`/about#dealtype-glossary`へのリンクを追加
  - `/about`に用語集セクション（`#dealtype-glossary`）を新設し全14分類を一覧表示
- Check: `npx tsc --noEmit`/`npm run lint`/`npm run build`成功。
- Act: PR #193としてマージ済み。

**Cycle 3（Do完了・Act未実施）**
- Plan: ユーザーが`.claude/skills/note-cover/`（note記事用カバー画像生成スキル、Pillow合成）を
  参照しつつ「アイキャッチ画像をやりたい」と要望。Pexels写真背景版を選択。
- Do:
  - `web/publish_blog_articles.py`: `search_pexels_photo()`（Pexels検索API）、
    `generate_eyecatch_image()`（写真+黒帯+Noto Sans CJK Bold太字白文字を合成、1200x630）、
    `upload_eyecatch()`（microCMSメディアアップロードAPI）、`build_eyecatch_for_article()`
    （上記をまとめてimage型フィールド用の`{"url":...}`を返す）を追加し`build_and_publish()`に統合
  - 投資家分類ごとのPexels検索クエリ`EYECATCH_QUERY_BY_CATEGORY`を定義（銘柄固有の写真は
    現実的でない＋商標リスク回避のため、分類のイメージに合う汎用写真を使用。個別銘柄名は
    画像内テキストの方で表現）
  - `.github/workflows/daily_alert.yml`: `Pillow`追加、`fonts-noto-cjk`インストールステップ追加、
    `PEXELS_API_KEY`シークレットを`.env`に追加
  - `.github/workflows/ci.yml`: `Pillow`追加（テストの折り返しロジック等で使用）
  - `tests/test_publish_blog_articles.py`: 上記関数のユニットテスト12件を追加（22→34件）
- Check: `python3 tests/test_publish_blog_articles.py`含む全Pythonテスト成功。実際に
  `generate_eyecatch_image()`をダミー写真で呼び出し、日本語タイトルの折り返し・合成を目視確認済み
  （このサンドボックスはPexels自体には接続不可のため、実写真での確認はGitHub Actions実行時が初回）
- Act: PR #196としてマージ済み。PEXELS_API_KEY発行・microCMSメディアアップロード権限設定は
  ユーザー側で完了。撮影者クレジット（Pexels API利用ガイドライン推奨）を画像右下に追加で焼き込み。

**Cycle 4（完了）**
- Plan: RSSフィードが無く、フィードリーダー経由の継続読者・外部サイトへのシンジケーション導線が
  存在しなかった。実装コストが低く「ちゃんとしたブログ」の標準機能でもあるため対応。
- Do:
  - `src/app/feed.xml/route.ts`: 新着記事20件のRSS 2.0フィードを生成するRoute Handler
  - `layout.tsx`の`alternates.types`に`application/rss+xml`を追加（`<head>`に自動でlink要素が出る）
  - フッターに`/feed.xml`へのリンクを追加
  - `llms.txt`にRSSフィードのパスを追記
- Check: `npx tsc --noEmit`/`npm run lint`/`npm run build`成功（`/feed.xml`ルートが生成されることを確認）。
- Act: PR #198としてマージ済み。

**Cycle 5（完了）**
- Plan: (1)記事本文が250〜400字と薄くSEO的に弱い、(2)記事詳細から他記事への回遊導線が
  「銘柄別」しか無く「同じ投資家分類の他の記事」が無い、の2点に対応。
- Do:
  - `web/publish_blog_articles.py`: `generate_article_body()`の目標文字数を250〜400字→
    500〜700字（3〜4段落）に緩和し`max_tokens`を800→1400に増加。`classify_filer()`が
    既に返している`description`（提出者の一言説明、これまで未使用だった）を`fact_sheet
    ['filer_description']`としてプロンプトに渡し、事実の範囲内で投資家の種類を1文補足させる
    （新しい事実の創作はさせない）
  - 記事詳細ページ（`articles/[id]/page.tsx`）に「関連記事（同じ投資家分類）」セクションを追加。
    既存の`getArticleList({dealType, limit})`を再利用し自分自身を除いて最大4件表示
- Check: Pythonテスト全件成功。`npx tsc --noEmit`/`npm run lint`/`npm run build`成功。
- Act: PR #200としてマージ済み。

**Cycle 6（完了）**
- Plan: ユーザーからの直接要望「ページ一番上のタイトルにサブタイトルをつけたい」。ヘッダーの
  ロゴ＋サイト名にタグラインが無かったため追加。あわせてホームページ先頭の「ようこそ」説明ブロック
  （やや長文）をユーザー指摘により記事一覧の下に移動し、SEO上必須なH1は`sr-only`で維持。
- Do:
  - `Header.tsx`: ロゴ＋サイト名の下にサブ見出しを追加（後にブランド名を「大口投資家の監視
    ブログ」へ差し戻したため「EDINET大量保有報告書から読む「クジラ」の動き」に変更、名前との
    重複を避けた）
  - `page.tsx`（ホームページ）: 先頭のH1「ようこそ」ブロックを撤去し、代わりに`sr-only`の
    H1（`{SITE_NAME}｜新着記事`）を設置。旧ブロックはh2に格下げして記事一覧・ページネーションの
    下に移動（新規訪問者がまず記事一覧を見られるようにするため）
- Check: `npx tsc --noEmit`/`npm run lint`/`npm run build`成功、Pythonテストへの影響なし（Next.js側のみの変更）。
- Act: PR #204としてマージ済み。

**ブランド名の差し戻し（Cycle 6の後）**
「クジラウォッチ」はダサいので却下→再考の末に一度採用したが、最終的にユーザーが
「大口投資家の監視ブログ」＋ドメイン`kujira-watch.com`という組み合わせを希望。
`SITE_NAME`のデフォルト値と、ハードコードされていた「クジラウォッチ」表記（`llms.txt`・
各READMEの説明文・`publish_blog_articles.py`等のdocstring・`daily_alert.yml`のステップ名）を
「大口投資家の監視ブログ」に統一。ドメイン(`SITE_URL`)は変更なし。

**未着手・保留**
- X(Twitter)自動投稿（サイトの基盤が整ったので着手を検討してよい段階）
