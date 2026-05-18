# Next.js フロントエンド構築進捗

最終更新: 2026-05-19

## ゴール
`stocksignal.jp` で動くNext.jsアプリを作り、Vercelにデプロイする。

---

## フェーズ 1: ローカル実装 ✅

- [x] `frontend/` ディレクトリ作成
- [x] Next.js 15 + Tailwind CSS + TypeScript セットアップ
- [x] ランキングテーブルUI（フィルター付き）
- [x] サマリーカード（S買い/A買い/保有継続/売り検討）
- [x] Web Pushボタン（PushButton.tsx）
- [x] `/api/push/subscribe` エンドポイント
- [x] `/api/push/send` エンドポイント（Python側から呼ばれる）
- [x] Service Worker (`public/sw.js`)
- [x] PWA manifest
- [x] VAPIDキー生成済み
- [x] `npm run build` 成功確認済み

---

## フェーズ 2: Vercelデプロイ

- [ ] Vercel アカウント作成（vercel.com）
- [ ] GitHub連携（Import Project → yuyakitagawa/stock-alert）
- [ ] Root Directory を `frontend` に設定
- [ ] 環境変数を Vercel に登録（下記参照）
- [ ] `stocksignal.jp` ドメインを Vercel に追加

### Vercel 環境変数（Project Settings → Environment Variables）

| Name | Value |
|------|-------|
| `NEXT_PUBLIC_SUPABASE_URL` | `https://kxrgyguowxtjqexvmlgx.supabase.co` |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | `sb_publishable_A7eK3pY7Yi12bxjQk-QzGA_sQJjY8Xn` |
| `SUPABASE_SERVICE_KEY` | `.env.local` 参照 |
| `NEXT_PUBLIC_VAPID_PUBLIC_KEY` | `BJDJPN6v7XsiFi_oA48ysk7asQT0hNcgHeRmcGu2Qk68c4c4knn0HOlufsZssw0Mztwr35eQYGup20siia-ZGM0` |
| `VAPID_PRIVATE_KEY` | `.env.local` 参照 |
| `NEXT_PUBLIC_SITE_URL` | `https://stocksignal.jp` |
| `INTERNAL_SEND_SECRET` | `.env.local` 参照 |

---

## フェーズ 3: ドメイン設定

- [ ] Vercel の Project Settings → Domains → `stocksignal.jp` 追加
- [ ] DNS を Vercel のネームサーバーに変更（またはCNAMEレコード追加）
- [ ] HTTPS 自動発行確認

---

## フェーズ 4: GitHub Secretsの `INTERNAL_SEND_SECRET` 更新

`.env.local` の `INTERNAL_SEND_SECRET` をGitHub Secretsに登録（Step 6 が正常に呼べるようになる）

- [x] GitHub Secrets に `INTERNAL_SEND_SECRET` 登録済み（6bcdb2c...）

---

## フェーズ 5: アイコン画像の追加

- [ ] `frontend/public/icon-192.png` を追加（192×192px）
- [ ] `frontend/public/icon-512.png` を追加（512×512px）

---

## 現在のステータス

**2026-05-19**: フロントエンド実装完了・ビルド確認済み。
次は Vercel デプロイ（フェーズ2）から再開する。
