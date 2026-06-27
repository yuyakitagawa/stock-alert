# DB Cleanup 進捗

## 完了済み
- [x] gen_rankings NULLバックフィル（PER/PBR/piotroski/bps_growth/eps_surprise/actual_return_63d）
- [x] gen_risk_regime 11→117行バックフィル
- [x] gen_ai_analyses 日付修正（1970-01-01→生成日、3579行）
- [x] gen_dividend_strategy / gen_qv_sim テーブル削除 + シミュレーション機能コード撤去
- [x] jpx_margin_balance の margin_buy_chg/margin_sell_chg カラム削除（100% NULL）
- [x] jpx_stock_list の name=NULL 870行削除
- [x] 全テーブルNULL率調査完了
- [x] gen_rankings.pos52 バックフィル（91.3%→5.8% NULL、yahoo_price_cacheから52週高値安値位置を計算、348,246行更新）

## 対応不要と判断
- [x] ext_tdnet_disclosures.xbrl_url — カラム削除済み（前セッション）
- [x] ext_tdnet_disclosures.category — 73.6% NULL。TDnetの取得仕様上、過去データの遡及困難。放置。
- [x] jquants_fin_summary の bps 39.4%、予想系54%、div_ann 51.8% — J-Quantsデータソース由来で正常（中間決算BPS未記載、予想非開示企業等）。対応不要。
