# 自動動画投稿パイプライン（YouTube Shorts / TikTok）

## 決定事項（2026-08-15 オーナー指示）
- 投稿先: **YouTube Shorts** ＋ **TikTok**
- 中身: **公開済みブログ記事（microCMS `articles`）の要約**
- 生成: **Remotion**（React で 1080x1920 縦動画）→ YouTube Data API v3 / TikTok Content Posting API

## 設計方針
- 既存の `web/publish_blog_articles.py`（毎時実行）とは**別ワークフロー**にする。
  Remotion のレンダリングは Chrome Headless が必要で毎時回すのは重いため、1日1回のバッチにする。
- 投稿対象の選び方は X 投稿（`web/x_client.py`）と同じ思想:
  「その日 microCMS に公開された記事」×「サイトの注目枠に入っている記事」の積集合の先頭1件。
  該当が無い地味な日は投稿0件（無理に投稿しない）。
- 動画の台本は Claude が記事本文から生成する（新しい事実は足さない。記事にある事実だけを縦動画用に圧縮）。
- 認証情報が未設定のプラットフォームは**スキップ**して他を止めない（x_client.py と同じ挙動）。

## ステップ
- [x] 1. 設計確定・進捗ファイル作成
- [x] 2. Remotion プロジェクト雛形（`video/remotion/`、1080x1920 / 30fps）
- [x] 3. 動画コンポジション実装（タイトル→数字→要約→CTA の4シーン）
- [x] 4. 記事取得＋台本生成（`video/build_script.py`: microCMS取得 → Claude で縦動画台本 → props JSON）
- [x] 5. レンダリング実行ラッパ（`video/render.py`: npx remotion render → mp4）
- [x] 6. YouTube アップロード（`video/youtube_client.py`, OAuth2 refresh token）
- [x] 7. TikTok アップロード（`video/tiktok_client.py`, Content Posting API）
- [x] 8. オーケストレーター（`video/publish_video.py`, `--dry-run` 対応）
- [x] 9. ローカル検証（台本生成 → レンダリング → mp4 確認）
- [x] 10. GitHub Actions ワークフロー（`.github/workflows/video_post.yml`）
- [ ] 11. Secrets 登録（オーナー作業）→ 初回ライブ投稿の確認
- [x] 12. README / dev_log 更新・コミット

## 必要な Secrets（オーナー作業）
| Secret | 用途 | 取得元 |
| --- | --- | --- |
| `YOUTUBE_CLIENT_ID` | YouTube OAuth2 | Google Cloud Console → OAuth クライアント（デスクトップアプリ） |
| `YOUTUBE_CLIENT_SECRET` | 同上 | 同上 |
| `YOUTUBE_REFRESH_TOKEN` | 無人アップロード | `python video/youtube_auth.py` をローカルで1回実行して取得 |
| `TIKTOK_CLIENT_KEY` | TikTok OAuth | TikTok for Developers → アプリ作成 |
| `TIKTOK_CLIENT_SECRET` | 同上 | 同上 |
| `TIKTOK_REFRESH_TOKEN` | 無人アップロード | `python video/tiktok_auth.py` をローカルで1回実行して取得 |

### TikTok の制約（重要）
TikTok Content Posting API は**アプリ審査（audit）を通るまで直接公開ができない**。
未審査アプリは `SELF_ONLY`（自分だけ閲覧可）でしか投稿できないため、本パイプラインは
既定で **inbox（下書き）へアップロード**する。TikTok アプリ側に通知が届き、
オーナーが手動で公開する運用になる。審査通過後に `TIKTOK_DIRECT_POST=1` を設定すると
直接公開に切り替わる。

## 状態
- 2026-08-15: パイプライン一式を実装し main へマージ済み（コミット `102b7868`。
  同じ作業ツリーで別セッションが AdSense 対応を進めており、そちらのコミットに
  動画パイプラインのファイルも同梱される形で push された）。
  ローカルで実データ（東陽テクニカ 8151 の売却記事）の台本生成〜mp4書き出しまで確認済み。
  `tests/test_video_pipeline.py` 22件、リポジトリ全体 209件 pass。
- Secrets 未登録のためアップロードは未実施。この状態でワークフローが走った場合は
  動画を生成してアーティファクトに残すだけで、投稿はせず正常終了する。

## v2改修（2026-08-15 オーナー指示「動画が面白くない」→ TikTok運用視点で全面改修）
- [x] 台本を「記事をほぼ読む」ナレーション構成に変更（7シーン: hook→company→deal→filer→change→outlook→cta。
      各シーン narration+caption の対。銘柄の事業内容・投資家プロフィールはSupabaseキャッシュを再利用）
- [x] TTS導入。Google Cloud TTSで実装→**GCPに請求先が無くAPI有効化不可**→VOICEVOX（ずんだもん）へ切替。
      CIは公式Dockerイメージでエンジン起動。クレジット表記はCTAシーン・説明文・キャプションに自動挿入
- [x] シーンの尺をナレーション音声の長さに連動（固定20秒を廃止、calculateMetadata）
- [x] TikTok運用の定石を反映: 冒頭は金額の一撃から／safeArea（下部470px・右190px）遵守／
      無音視聴者向け大型字幕常時表示／進行バー／背景の常時ドリフト／ループに繋がる締め
- [x] ナレーションの切り詰めを句点境界に（文の途中切りが読み上げられる不具合の防止）
- [x] テスト29件に更新・全209+件pass
- [x] ローカルでVOICEVOXエンジンを立てて音声付きの通し確認（2026-08-15、エンジン0.25.2を
      ~/voicevox/macos-arm64/ に導入。実データで79秒・AAC音声入りmp4を確認）
- [x] 冒頭の「クジラがいる」を削除し動詞の言い切りに（オーナー指摘）

## v3改修（2026-08-15 オーナー指示: 背景を自然映像に・最後に株価推移・望遠鏡シーン削除）
- [x] 背景をPexels Videos（海系・縦向き・商用可）のループ再生＋暗幕オーバーレイに。
      `video/background.py` が毎回ランダムに1本取得、PEXELS_API_KEY未設定/失敗時は
      従来のグラデーション背景へフォールバック（ローカル.envにはキー未設定のため、
      ローカルでの実映像確認にはオーナーがキーを.envへ追加する必要あり）
- [x] Pexels APIキーをオーナーが新規取得し .env / GitHub Secrets へ登録（2026-08-16。
      それまでPEXELS_API_KEYはどこにも存在せず、アイキャッチ画像も実は無効だった）
- [x] 実映像（海面11秒素材）で通しレンダリング確認。CRF 18→23（実写背景で190MB→105MB）

## v4改修（2026-08-16 オーナー指示: 人物素材の追加・シーンごとに背景を切り替え）
- [x] 背景プールを「自然系9クエリ＋人物系3クエリ（女性）」に拡張、人物枠1本を確保して最大4本取得
- [x] シーン（セリフの区切り）ごとに背景をランダム割当。隣接シーンで同じ映像は使わない
- [x] 7秒未満の素材を除外（短尺ループの継ぎ目が目立つため）
- [x] テスト38件・実データで森→海→人物と切り替わる55MBのmp4を確認

## v5改修（2026-08-16 オーナー指示: 人物比率を上げる・冒頭は人物）
- [x] 人物枠を1本→2本（プールの半分）に拡大、人物クエリも3→5種に増強
- [x] 先頭シーン（hook）はプール内の人物素材を優先割当
- [x] 明るい実写素材で金色の文字が沈まないよう暗幕オーバーレイを強化
- [x] テスト40件pass
- [x] 株価推移シーン（kind="chart"）をctaの直前に追加。yahoo_price_cacheの直近3ヶ月終値を
      線が伸びるアニメーションで描画、終値と騰落率を表示。ナレーションは数値創作を避けるため
      Claudeではなくテンプレート生成
- [x] outlookシーンの中央ビジュアル（🔭「この買いが意味するもの」）を削除（オーナー指摘）
- [x] テスト35件・全230件pass。実データ（アインHD）で音声・チャート込み89秒のmp4を確認

## 投稿フロー構築（2026-08-16）
- [x] YouTube Data API v3 をサービスアカウント経由で有効化（課金不要だった）
- [x] オーナーがOAuthクライアント（デスクトップ）作成・テストユーザー登録 → youtube_auth.py で
      リフレッシュトークン取得 → .env と GitHub Secrets（gh secret set）に登録・動作確認済み
- [x] チャンネル開設・体裁（バナー/アイコン/透かし/説明文。バナーと透かしはPILで生成して提供）
- [x] **初投稿完了**: https://youtube.com/shorts/MqCMZqa91t4 （アインHD/Oasis記事。
      日曜で36h窓に対象が無かったため72hに広げた1回きりのローカル実行。タイトルの
      filerName欠落によるねじれはオーナーがStudioで手動修正、コードにはフォールバック追加）
- [x] TikTok: アプリ作成・Sandbox認可完了（2026-08-16）。URLプロパティ検証用に
      /terms ページと検証ファイル配信ルートを新設。Sandbox の Client Key/Secret +
      リフレッシュトークン（365日有効）を .env / GitHub Secrets に登録、トークン更新フロー確認済み。
      現在は未審査のため inbox（下書き）投稿のみ。**審査提出時に必要なもの**: 説明文（作成済み・
      フォームに入力済み）と Sandbox 投稿フローのデモ動画（未作成。仮動画でSaveを通した状態なので
      Submit for review は押さないこと）。審査通過後に TIKTOK_DIRECT_POST=1 で直接公開へ
- [x] ドメイン障害対応（2026-08-16）: kujira-watch.com が clientHold（ICANN メール認証未完了）で
      全世界NXDOMAINになっていたのを発見、オーナーが認証して復旧

## 記事ID指定の手動実行（2026-08-18 オーナー指示「気に入った記事も自動で動画にしたい」）
- [x] 銘柄コード指定モード（PR #269、stockCode＋直近14日で金額最大1件）を**記事ID指定に置き換え**。
      銘柄指定だと同一銘柄に複数記事がある場合に狙った記事とは限らない問題があったため
      （実例: 2026-08-18に `stock_code=2413` で実行したところ、意図とは別のエムスリー記事が選ばれた）。
      `publish_video.py --article-id` / workflow_dispatch の `article_id` 入力から使う。
      記事URL（https://kujira-watch.com/articles/xxxx）を丸ごと貼っても ID を抜き出す。
      公開時刻・注目枠は問わないので、いつの記事でも動画にできる。

## 状態（2026-08-18）
- Anthropic APIの使用量上限は解消済み。8/17の定時便でYouTube投稿成功
  （電通総研×Oasis: https://youtube.com/shorts/-_-Z4m_6HFU ）。
- 8/18 19:30の定時便は上限エラーで空振り（対象はエムスリー×Oasis）。その後オーナーが上限を
  上げ、手動実行で投稿成功（ https://youtube.com/shorts/CYl84alBoYo ）。
- TikTokは引き続き未審査のため inbox（下書き）投稿。アプリの通知から手動公開が必要。
