# 進捗: 銘柄の会社説明（スプシ管理 → Web表示）

オーナー方針: 銘柄詳細ページ（/stocks/[code]）に「この会社は何の会社か」の説明を載せる。
スプレッドシートで手動管理し、Webアプリにも反映する。

## 設計（最小変更）
- 既存の説明表示: `StockLivePanel`「この会社について」→ `/api/stock/[code]/description`
  → Supabase `ai_analyses`(model_version=`company-desc-v1`, date=`1970-01-01`) をキャッシュ参照。
- スプシ専用シート **「📝 会社説明」**（コード/銘柄名/説明）を新設し手動管理（自動上書きされない）。
- `web/sync_descriptions.py`: シートを読み、説明が入っている行を `ai_analyses` の company-desc-v1 へ upsert。
  → 既存の description API がそれをキャッシュとして返す＝手動説明が最優先・AIは未記入銘柄のフォールバック。
- フロント・API・DBマイグレーションの変更は不要。

## ステップ
- [x] `web/sync_descriptions.py` 作成（シート作成/seed + Supabase upsert）
- [x] スプシに「📝 会社説明」シート作成（28行seed、ウォッチ10件は説明記入済み）
- [x] 同期実行 → Supabase 10件 反映確認
- [x] Web（/stocks/2811等）で「この会社について」表示を確認
- [x] README 更新 + 日次パイプライン Step 5b で自動同期
- [x] commit
