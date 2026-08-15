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

## 次にやること（オーナー作業）
1. 上表の Secrets を GitHub に登録する（`video/youtube_auth.py` / `video/tiktok_auth.py` を
   ローカルで1回ずつ実行してリフレッシュトークンを取得）
2. Actions から `Short Video Post` を「render_only=true」で手動実行し、
   アーティファクトの mp4 を目視確認する
3. 問題なければ render_only なしで手動実行し、初回の実投稿を確認する
