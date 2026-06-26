# 全データ取得をSupabase DB経由に統一

最終更新: 2026-06-25

## ゴール
Yahoo Finance API直接取得・_local_prices.pklを全廃し、Supabase(yahoo_price_cache / yahoo_market_index)を唯一のデータソースにする。

---

## 対象ファイル・関数

### Phase 1: 価格キャッシュ（_local_prices.pkl廃止）
- [ ] `tools/optimize_net_weights.py` — _local_prices.pkl → yahoo_price_cache
- [ ] `_train_local.py` — _local_prices.pkl → yahoo_price_cache

### Phase 2: 市場指数（Yahoo API → yahoo_market_index）
- [x] `tools/optimize_net_weights.py` — N225 yfinance → load_market_index_data (済)
- [x] `tools/sell_signal_backtest.py` — N225 yfinance → load_market_index_data (済)
- [x] `tools/export_report_to_sheets.py` — N225 yfinance → load_market_index_data (済)
- [ ] `core/rank_stocks.py:239-249` — N225 inline fetch → load_market_index_data
- [ ] `core/rf_train_v3.py:59-71` — _fetch_index_df() → load_market_index_data
- [ ] `tools/backfill_history.py:75-96` — fetch_nikkei_history() → load_market_index_data

### Phase 3: 個別株価格（Yahoo API → yahoo_price_cache）
- [ ] `lib/utils.py:24-52` — get_prices() → DB読み
- [ ] `lib/utils.py:55-82` — get_price_at_date() → DB読み
- [ ] `core/rf_train_v3.py:101-120` — get_prices() → DB読み
- [ ] `tools/backtest.py:101-130` — _fetch_yahoo() → DB読み

### Phase 4: US ETF・セクター（新テーブル or yahoo_market_indexに追加）
- [ ] `core/rank_stocks.py:84-99` — fetch_us_sector_etf_returns() → DB
- [ ] `tools/backfill_history.py:99-115` — fetch_etf_history() → DB

### Phase 5: データ投入パイプライン
- [ ] `tools/fetch_history.py` — Yahoo→DB投入は残す（唯一のデータフィーダー）
- [ ] `lib/utils.py:get_market_index_df_cached()` — 既にDB-first。差分取得→DB保存のフィーダーとして残す

### 対象外（フロントエンド）
- frontend/lib/yahoo.ts, frontend/app/api/chart/ — ブラウザ用チャートAPI、DB経由不要

---

## 前提条件
- yahoo_price_cache に十分な過去データ（2022年〜）が必要
- yahoo_market_index に N225/VIX/SP500/USDJPY の2022年〜データが必要
- 現状DBは2025-01-06〜しかない → ローカルでバックフィル要
