# TikTok アプリ審査の提出パッケージ

未審査の間は inbox（下書き）投稿のみ・SELF_ONLY 制限のため、審査を通して
`TIKTOK_DIRECT_POST=1`（キャプション込みの全自動公開）に切り替えるのがゴール。

## 提出フォームに入れる説明文（英語・確定版）

```
Kujira Watch (https://kujira-watch.com) is a Japanese stock-market information website. It tracks large shareholding disclosures filed with EDINET (Japan's official corporate disclosure system) and publishes summary articles. Our server automatically generates a short vertical video (1080x1920 mp4, about 60-90 seconds) that summarizes one published article per day, with narration and captions.

How each product/scope is used:

- Login Kit: Used once by the site owner to authorize our own TikTok account via OAuth (authorization code flow, redirect to https://kujira-watch.com/tiktok-callback). No third-party users log in; the integration posts only to our own account.

- Content Posting API (video.upload): Our automated pipeline uploads the generated daily summary video to the authorized account's inbox as a draft. The owner then reviews it in the TikTok app and decides whether to publish. Videos contain only factual summaries of public regulatory filings.
```

※直接公開（video.publish スコープ）まで審査で取りたい場合は、スコープに `video.publish` を
追加した上で、上の2段落目を「uploads and publishes the video directly to the authorized
account」に変えて提出する。まずは video.upload のみで通し、後から video.publish を
追加審査する二段構えでもよい。

## デモ動画の構成（ショットリスト）

要件: mp4/mov、50MB以下、Sandbox 環境での end-to-end フロー、実際の UI と操作が見えること。
以下を1本に繋いで2〜3分にする。

| # | 撮影者 | 内容 | 撮り方 |
|---|---|---|---|
| 1 | Mac | kujira-watch.com を開き、記事ページをスクロール（動画の元になるサイトを見せる。ドメインが URL バーに写ること） | 画面録画 |
| 2 | Mac | Developer Portal の Sandbox 設定画面（アプリ名・Sandbox である事がわかる部分） | 画面録画 |
| 3 | スマホ or Mac | OAuth 認可: 認可URLを開く → TikTok の許可画面 → 「許可」を押す → kujira-watch.com/tiktok-callback にリダイレクトされる様子 | 画面録画 |
| 4 | Mac | ターミナルで `python video/publish_video.py` 相当を実行し、「TikTok下書き送信: publish_id=...」のログが出るところ | 画面録画 |
| 5 | スマホ | TikTok アプリ: 受信トレイ → システム通知「動画が準備できました」→ タップ → 動画が開く → 編集画面 → キャプション入力 → 下書き保存（または公開） | iPhoneの画面収録 |

- 結合・圧縮は ffmpeg でこちらで実施（オーナーは素材を撮って渡すだけ）
- 各ショットの間に1秒程度の説明テロップ（英語）をこちらで挿入する

## 提出手順

1. デモ動画をフォームの Upload に差し替え（いまは仮の動画が入っている）
2. 説明文を上記に置き換え（入力済みなら確認のみ）
3. **Submit for review** を押す
4. 結果は History / Review comments タブに届く（通常数日〜2週間）
5. 承認後: GitHub Secrets に `TIKTOK_DIRECT_POST=1` を追加（`gh secret set TIKTOK_DIRECT_POST --body 1`）。
   video.publish スコープを追加した場合は再認可（tiktok_auth.py の scope に video.publish を足して
   認可URL再発行→トークン差し替え）も必要

## 状態

- [x] 説明文の確定
- [x] LINE キャプション通知の実装（審査待ちの間の運用改善。video/line_notify.py）
- [x] デモ動画素材の録画（2026-08-16。Mac側4カットはscreencapture -v、スマホ1カットはオーナー撮影）
- [x] 結合・テロップ・圧縮（video/out/tiktok_review_demo.mp4、2分41秒・4.1MB）
- [x] 本体アプリ（Production）側にも Products（Login Kit + Content Posting API）と
      Scopes（video.upload）を追加（審査は本体構成に対して行われるため。Sandboxとは別設定）
- [x] **審査提出完了（2026-08-16）** — ステータス「in review」。結果待ち（通常数日〜2週間、
      混雑時はさらに遅延の告知あり）。結果は Developer Portal の History / Review comments に届く
- [ ] 承認後: `gh secret set TIKTOK_DIRECT_POST --body 1` で直接公開へ切替（本体側の
      Client Key/Secret への差し替えと再認可も必要になる場合がある — 承認時に確認）
