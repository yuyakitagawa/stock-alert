# 進捗: ブックマークの Supabase 永続化

オーナー方針: 端末ローカル(localStorage)のみだったブックマークを Supabase に永続化する。
認証は導入しない（匿名 client_id 方式）。

## 設計
- テーブル `web_bookmarks(client_id text, code text, created_at timestamptz default now(), PK(client_id, code))`。
- 識別: ブラウザごとに UUID の `client_id` を localStorage に発行（`stocksignal:client-id`）。
- 同期: localStorage を即時反映（オフラインファースト）→ `/api/bookmarks` 経由で Supabase に非同期反映。
- 書き込みは service key を使う API ルートのみ（フロントは anon key で read のみのため）。
- **制約**: 認証が無いため端末間同期は不可（client_id はブラウザ単位）。

## オーナー保有株(43銘柄)の移行
- 43コードを `frontend/lib/owner-holdings.ts` に定数化。
- ウォッチリストページに「保有株を取り込む」ボタンを設置し、オーナーが自分のブラウザで一度押して取り込む。

## ステップ
- [x] Supabase `web_bookmarks` テーブル作成
- [x] 進捗ファイル作成
- [x] `/api/bookmarks` ルート（GET/POST/DELETE）
- [x] `lib/bookmarks.ts` を client_id + Supabase 同期に改修
- [x] `lib/owner-holdings.ts`（43コード）＋ 取り込みボタン
- [x] README 更新
- [x] tsc / ビルド確認（tsc clean・next build OK・/api/bookmarks 登録）
- [x] commit & push
- [ ] 本番(Vercel)反映確認（/watchlist HTTP200・取り込み動作）

## held_scores 削除（完了）
- held_scores テーブルは削除。保有日数計算は Google Sheets の購入日列のみを使用するよう変更。
- 関連コード（save_held_scores / get_holding_days / migrate のマッピング）も除去。
- ブックマークは手動で取り込み直す運用。
