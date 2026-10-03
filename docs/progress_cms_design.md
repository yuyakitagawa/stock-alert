# 進捗: Storyblokでデザイン編集（TOP・サイト設定）

ブランチ: `feat/storyblok-cms`（worktree: `../stock-alert-storyblok`）

- [x] Storyblokスペース作成（オーナー、Growth Plus 45日トライアル→終了後は無料Starterへ）
- [x] CDNトークン（Preview/Public）を `.env.local` に設定、疎通確認
- [x] コード: `lib/storyblok.ts`・`lib/siteTheme.ts`・TOPのセクション化・layoutで配色適用・preview/exit/revalidate ルート・Bridge
- [x] `next build` で全ページの静的/ISR判定が変わらないことを確認
- [x] ローカルで既定表示（Storyblok空）と preview ルート（401/307・Draft Mode）を確認
- [ ] `STORYBLOK_MANAGEMENT_TOKEN` を `.env.local` に設定（オーナー）
- [ ] `scripts/storyblok-setup.mjs` 実行（ブロック定義・初期ストーリー・プレビューURL・Webhook）
- [ ] ローカルで配色5種・明朝・セクション並べ替えを確認
- [ ] Vercelに環境変数4つ（PUBLIC/PREVIEW_TOKEN・PREVIEW/REVALIDATE_SECRET）を登録
- [ ] PR作成 → マージ（オーナー確認）→ 本番でビジュアルエディタ動作確認
- [ ] 第2段: about・運営者ページ
