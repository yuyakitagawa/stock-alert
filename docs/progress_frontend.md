# Next.js フロントエンド構築進捗

最終更新: 2026-05-19

## ゴール
`stocksignal.jp` で動くNext.jsアプリを作り、Vercelにデプロイする。

---

## フェーズ 1: ローカル実装 ✅
## フェーズ 2: Vercelデプロイ ✅
## フェーズ 3: デザイン全面刷新 ✅

**2026-05-19 本番稼働中**: https://stock-alert-web.vercel.app

---

## 完了済み項目

- [x] Next.js 15 + Tailwind CSS + TypeScript
- [x] fetch()ベースのSupabase REST API（JSクライアント不使用）
- [x] ホームページ（サマリーカード・注目銘柄グリッド・売り検討一覧・凡例）
- [x] ランキングページ（タブフィルター・検索・ソート・モバイルカードビュー）
- [x] 銘柄詳細ページ（AIスコア・AI解析・決算日・バリュエーション）
- [x] ダークUI（Bloomberg/Yahoo Finance風、シグナル別7色）
- [x] Navbar・Footer・StockCard・Skeleton・RecommendBadge
- [x] ローディングスケルトン / エラーページ / 404ページ
- [x] SEO metadata（全ページ）
- [x] Web Push通知（/api/push/send・/api/push/subscribe）
- [x] Service Worker（/public/sw.js）
- [x] Supabase recommend値から絵文字除去（ランキング0銘柄バグ修正）
- [x] Vercel rootDirectory=frontend をAPIで設定
- [x] GitHub連携によるCI/CDデプロイ確立

---

## 残タスク

- [ ] `stocksignal.jp` カスタムドメインの接続
- [ ] `INTERNAL_SEND_SECRET` を GitHub Secrets に設定済み → Vercelにも登録
- [ ] アイコン画像（icon-192.png / icon-512.png）追加
- [ ] AI解析データが蓄積されたら銘柄詳細ページで確認

---

## 環境変数（Vercel登録済み）
| 変数名 | 用途 |
|--------|------|
| NEXT_PUBLIC_SUPABASE_URL | Supabase接続URL |
| NEXT_PUBLIC_SUPABASE_ANON_KEY | 読み取り用キー |
| SUPABASE_SERVICE_KEY | 書き込み用キー |
| NEXT_PUBLIC_VAPID_PUBLIC_KEY | Web Push公開鍵 |
| VAPID_PRIVATE_KEY | Web Push秘密鍵 |
| VAPID_CONTACT_EMAIL | Web Push連絡先 |
| NEXT_PUBLIC_SITE_URL | サイトURL |
| INTERNAL_SEND_SECRET | Push送信認証 |
