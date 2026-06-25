# Next.js フロントエンド構築進捗

最終更新: 2026-06-25

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
- [x] EN/JP切替機能の完全削除（i18n基盤・LanguageContext・切替ボタン全撤去、全コンポーネント日本語ハードコード化）
- [x] 💎買い銘柄なし時の表示改善（注目銘柄セクション非表示、該当なしメッセージ、日経225上昇基調時はETF推奨）
- [x] HomeContent.tsxをクライアントコンポーネントとして分離（page.tsxはサーバーコンポーネント専任）
- [x] CI smoke testの/review→/model修正（PR #44）
- [x] CIにVercel本番デプロイ確認ステップ追加（VERCEL_TOKEN使用）（PR #44）
- [x] CI障害通知メール停止（ユーザー依頼 2026-06-18）

---

## 残タスク

- [ ] `stocksignal.jp` カスタムドメインの接続
- [ ] `INTERNAL_SEND_SECRET` を GitHub Secrets に設定済み → Vercelにも登録
- [ ] アイコン画像（icon-192.png / icon-512.png）追加
- [ ] AI解析データが蓄積されたら銘柄詳細ページで確認

---

## 変更履歴（2026-06）

### 2026-06-24: EN/JP切替削除 & 💎なし表示改善 & CI修正
- **削除**: `frontend/contexts/LanguageContext.tsx`, `frontend/lib/i18n.ts`
- **変更**: Navbar, StockCard, SimulationPanel, SectorPerformancePanel, RiskRegimeBanner, Footer, StockChart, RankingsTable, layout.tsx — i18n依存除去、日本語ハードコード化
- **変更**: `HomeContent.tsx` — 💎買い銘柄0件時は注目銘柄セクション非表示、「該当銘柄なし」メッセージ表示。日経225上昇基調時はETF推奨
- **変更**: `page.tsx` — サーバーコンポーネント専任化
- **修正**: `.github/workflows/frontend_build.yml` — smoke test `/review`→`/model`、Vercelデプロイ確認ステップ追加、障害通知メール停止
- **GitHub Secrets追加**: `VERCEL_TOKEN`（Vercel API経由のデプロイ状態確認用）
- **PR #44**: マージ済み（2026-06-25）

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
