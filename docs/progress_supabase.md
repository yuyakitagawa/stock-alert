# Supabase セットアップ進捗

最終更新: 2026-05-19

## ゴール
`web/export_to_web.py`（Step 5）と `web/send_user_alerts.py`（Step 6）が
Supabase に接続できる状態にする。

---

## フェーズ 1: Supabase プロジェクト作成 ✅

- [x] supabase.com でプロジェクト作成（リージョン: Northeast Asia (Tokyo)）
- [x] `SUPABASE_URL` → `https://kxrgyguowxtjqexvmlgx.supabase.co`
- [x] `SUPABASE_SERVICE_KEY`（service_role）取得済み

---

## フェーズ 2: テーブル作成 ✅

- [x] `web_rankings` 作成
- [x] `web_stock_meta` 作成
- [x] `web_earnings` 作成
- [x] `ai_analyses` 作成
- [x] `push_subscriptions` 作成

---

## フェーズ 3: RLS（Row Level Security）設定 ✅

- [x] RLS 設定完了

---

## フェーズ 4: 環境変数の設定 ✅

- [x] ローカル `.env` に追記（SUPABASE_URL / SUPABASE_SERVICE_KEY / SITE_URL）
- [x] GitHub Secrets 登録（SUPABASE_URL / SUPABASE_SERVICE_KEY / ANTHROPIC_API_KEY / SITE_URL）
- [ ] `INTERNAL_SEND_SECRET` → Next.js 側実装後に設定

---

## フェーズ 5: 動作確認 ✅

- [x] `export_to_web.py` → 接続成功（本日データなしで正常終了）
- [x] `send_user_alerts.py --dry-run` → 有効サブスクリプション 0 件（正常）

---

## 現在のステータス

**2026-05-19**: バックエンド側の Supabase セットアップ完了。
次のステップは Next.js フロントエンドの実装（別リポジトリ）。
`INTERNAL_SEND_SECRET` は Next.js の `/api/push/send` 実装後に GitHub Secrets へ追加する。
