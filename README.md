# stock-alert

東証上場株式の機械学習スクリーニング・アラートシステム。毎日自動でランキングを生成し、Gmailでメール通知する。

## システム概要

平日にGitHub Actionsが自動実行される。

```
【20:00 JST】アラートパイプライン（daily_alert.yml）
core/screener.py → core/rank_stocks.py
core/rf_train_v3.py は金曜 or モデル未存在時のみ実行
web/export_to_web.py（Supabase同期）→ web/market_timing_alert.py（LINE通知）

その他ワークフロー: ci.yml（テスト）、
keepalive.yml（Supabase keepalive）、watchdog.yml（パイプライン監視）、
data_backfill.yml（JPX/TDnet/EDINET手動遡及）、backfill_rankings.yml（株価キャッシュ更新+ランキング遡及・手動実行）
```

ユーザー向けの通知・操作は LINE Messaging API 経由（Supabase Edge Function `supabase/functions/line-webhook`）で提供する。Web/Vercelアプリは廃止済み。

---

## ファイル構成

| ファイル | 役割 |
|---|---|
| `core/screener.py` | JPX全銘柄から条件通過銘柄を抽出して `data/screeners/` に保存 |
| `tools/fetch_history.py` | Yahoo Finance で全銘柄株価四本値を取得し `yahoo_price_cache` を差分更新（daily_alert.yml Step 0で毎日 `--years 1` 実行。`rank_stocks.py`の「直近株価」の鮮度に直結。既存(code,date)は insert_ignore で保護されるため初回10年分バックフィルにも日次更新にも使える） |
| `tools/backfill_history.py` | 指定期間の過去営業日ぶんランキングを再生成し`gen_rankings`へupsert（アラート送信はしない。`--start`/`--end`指定可。既存日付は既定でスキップするため、価格データ修正後に再生成したい場合は`--force`で上書き。生成後に`check_price_freshness`で複数日にまたがるclose凍結（更新漏れ）を検査）|
| `core/rf_train_v3.py` | XGBoostの下落モデルを東証全銘柄×5年データで学習（金曜のみ。上昇モデルは廃止済み）。`--cutoff YYYY-MM-DD` でウォークフォワード用モデルも生成可能 |
| `core/rank_stocks.py` | スクリーナー通過銘柄に下落確率をつけてランキング生成・DB保存。フェーズ5(優待権利落ち)→フェーズ7(米国ETFリードラグフィルター)→フェーズ8(相場リスク管制官) |
| `web/export_to_web.py` | Supabaseへランキング・日経 vs S&P500判定をエクスポート（Step 4）|
| `web/market_timing_alert.py` | LINE Messaging APIで日次プッシュ通知（Step 5b）。N225シグナル（平均下落確率→投資/キャッシュ）・🌐日経 vs S&P500相対強弱・🏦直近のEDINET大口保有動向（自己申告のみ除外、譲渡/売却も📈買い・📉売りを明示して表示。ウォッチ銘柄→法人/ファンド→保有比率が大きい順に優先し最大5件、個人名の提出者は後回し。残りはLINEで「大量保有」と聞けば`check_catalyst`ツールで個別回答）・ユーザー別ウォッチリストのdp閾値アラートを配信 |
| `config.py` | 戦略パラメータの一元管理（閾値・フィルター値）|
| `lib/utils.py` | 共通関数（get_prices, extract_features, add_cs_rank_features, recommend_from_scores 等）|
| `lib/db.py` | Supabase永続化層（gen_rankings / jpx_stock_list / yahoo_price_cache ほか）。`lib/supabase_client.py` のREST API経由 |
| `lib/sheets_helper.py` | Googleスプレッドシート連携 |
| `lib/data_sanity.py` | **Quality Assurance (QA)** ロール。リリースのたびにデータを検証。`check_ranking`（下落確率レンジ・予測多様性等の行レベル、rank_stocks/export_to_webで使用）＋`check_price_freshness`（複数日にまたがるclose凍結=更新漏れ検知、backfill_historyで使用）（alert-only：違反でも更新は止めずメール通知）|
| `lib/kabutan_earnings.py` | kabutan.jpから決算業績を取得（AI解析プロンプト用）|
| `lib/risk_regime.py` | **相場リスク管制官**。日経20日・VIX・ドル円・S&P500からリスクオン/オフを判定。rank_stocksのフェーズ8でリスクオフ日はS買いを自動見送り、判定を `data/risk_regime.json` に保存しメールに警告表示 |
| `lib/market_compare.py` | **日経 vs S&P500 相対強弱アドバイザー**。日経225とS&P500の20日・60日リターン差から「日本株優位／米国株優位／拮抗」を判定(売買シグナルには影響しない参考情報)。rank_stocksのフェーズ8bで判定し `data/market_compare.json` に保存、`gen_market_compare`経由でLINE(`market_timing_alert.py`)に表示 |
| `tools/backtest.py` | バックテスト（先読みバイアスなし）。下落確率が低い順に選定。結果は `simulations/backtests/` に保存。`--drop-max`で下落確率上限、`--model-cutoff YYYY-MM-DD` でウォークフォワード用モデル指定可能 |
| `tools/multi_backtest.py` | 33期間一括バックテスト＋下落確率閾値比較分析（ウォークフォワード対応） |
| `tools/screen_catalyst_candidates.py` | カタリスト候補スクリーン（GARP補助）。PBR<1.0・ROE<8%・自己資本比率>50%・流動性の「安い箱」抽出は Postgres RPC `screen_catalyst_candidates()` でサーバーサイド集計（J-Quants財務データ使用）。通過候補に **利益の質フィルター(A/B)** で化粧決算（営業赤字・純利益>営業益×1.5）と斜陽事業（本業減益）を除外し、売上CAGR・営業利益率・会社予想方向で加減点。`data/catalyst_candidates.csv`（残）＋ `data/catalyst_excluded.csv`（除外理由付き・レビュー用）。`--no-quality` で品質フィルター無効 |
| `tools/catalyst_backtest.py` | カタリスト候補スクリーンのヒストリカルBT（point-in-time・disc_date≤基準日）。A/Bあり/なしで平均・勝率・大勝率を比較。データは J-Quants財務＋yahoo_price_cache |
| `lib/earnings_quality.py` | カタリスト候補の利益の質・本業方向性を判定（年次の営業益/売上/純益から化粧決算/斜陽を機械判定）。データ源は kabutan 優先、取れない環境（クラウドはkabutanがIPブロック）では J-Quants 実績にフォールバック |
| `lib/edinet.py` + `tools/scan_large_holdings.py` | **EDINET大量保有スキャナー**（イベント駆動）。EDINET APIから大量保有報告書(350)/変更報告書(360)を日次スキャンして `edinet_large_holdings` に蓄積し、カタリスト候補と突合（構造的候補×実際の買い集め＝先回り候補）。突合時に自己申告（提出者≒対象企業）と譲渡/売却の報告を除外し、外部の買い集めだけ残す（`--no-exclude` で無効化可）。`is_sell_disclosure`/`is_individual_filer` は `market_timing_alert.py` のLINE通知セクションでも再利用（売却を除外せず方向性表示、個人名提出者を優先度で後回し）。`EDINET_API_KEY` 必須 |
| `tests/test_earnings_quality.py` | 利益の質フィルター（化粧・赤字・減益・加減点）のユニットテスト（8件）|
| `tests/test_screener.py` | スクリーナー条件のユニットテスト（9件）|
| `tests/test_data_sanity.py` | QA（データ整合性・価格凍結検知）のユニットテスト（14件）|
| `tests/test_market_compare.py` | 日経 vs S&P500 相対強弱アドバイザーのユニットテスト（4件）|
| `tests/test_market_timing_alert.py` | LINE通知の大口保有動向セクション整形のユニットテスト（9件）|
| `tests/test_scan_large_holdings.py` | EDINET大量保有スキャナーの判定ロジック（売却検知・個人名判定・ノイズ除外）のユニットテスト（6件）|

---

## S買い 発令条件（passes_buy_filter + rank_stocks.py フェーズ5・7・8）

下落モデルのみに一本化済み（上昇モデル・netスコアは廃止。詳細は `dev_log.md` 参照）。

品質フィルター（`passes_buy_filter`）:

| 条件 | 値 | 意図 |
|---|---|---|
| 株価 ≥ | 300円 | 低位株除外 |
| 3ヶ月モメンタム ≥ | +8% | 上昇トレンド確認（5%→8%: 10期間BTで勝率+7pp）|
| 2年モメンタム | プラス | 長期下落株を除外（2年<0は勝率25%・avg-3.1%）|
| 2年トレンド R²（504日） ≥ | 0.4 | 長期トレンド一貫性確保（R²<0.4は勝率18%・avg-2.8%）|
| RSI（14日） | < 75 | 過熱除外のみ（下限撤廃: 30〜45帯が有効と判明）|
| 出来高比 vr2060 ≥ | 1.0 | 出来高増加トレンド確認 |
| 直近20日ボラ (vol20) ≤ | 22% | 高ボラ時は見送り（BT: vol>22%は平均▼0.9pp）|
| 連続下落日数 ≤ | 3日 | 急落継続銘柄の除外 |
| 60日ドローダウン ≥ | −15% | 深い下落銘柄の除外 |
| 20日平均売買代金 ≥ | 50百万円 | 流動性確保（板薄銘柄除外）|

モデル予測フィルター（`recommend_from_scores`）:

| 条件 | S買い |
|---|---|
| 下落確率 | < 8% |
| 年率ボラティリティ | ≤ 20% |

フェーズ5・7・8 追加フィルター（`rank_stocks.py`）:
- フェーズ5: 株主優待権利落ち21日前以内の銘柄はS買い→方向感なしに降格
- フェーズ7: 対応する米国セクターETF（XLK/XLF/XLI/XLB/XLV/XLY）の前日リターンがマイナスならS買い→方向感なしに降格。リードラグ効果（US→JP翌日）を活用。21,416サンプル(2023-2026)で全26ペア正相関・avg +0.64pp効果を確認。キャッシュは `data/sector_map.json`。
- フェーズ8: 相場リスク管制官がリスクオフ地合いと判定した日は、S買いを全件見送り（自動防御）

**推奨ラベル**:
- 💎 買い: QV条件+ファンダ品質+モデルスコア全条件クリア
- 🔴 売り検討: drop_prob≥10% / drawdown60<-20% / 連続下落≥5日
- —: それ以外

## スクリーナー条件（screener.py / 前段フィルター）

`screener.py → rank_stocks.py` の前段として動作する追加フィルター:

| 条件 | 値 |
|---|---|
| 3ヶ月相対強度 ≥ | 0%（通常）/ +5%（下落相場：日経20日 < -5%）|
| セクター集中除外 | 同一業種3銘柄以上でセクター全除外（バブル兆候回避）|

> 下落モデル一本化後のバックテスト再検証は未実施（別環境でのbacktest.py実行が必要。詳細は `dev_log.md`）。

---

## モデル詳細（core/rf_train_v3.py）

### 学習データ
- 対象：東証プライム・スタンダード全銘柄（約3,500〜4,000銘柄）
- 期間：過去5年分（約1,800日）
- サンプリング：20営業日ごと（自己相関低減）
- 分割：cutoff日より前→学習 / 以降→テスト（ウォークフォワード）

### 特徴量（61次元 = 54基本 + クロスセクション7次元）

**テクニカル10**: ret5, ret20, ret60, ret90, ma5_25, ma25_75, rsi, vol20, vol60, pos52

**トレンド反転5**: drawdown60, from_hi52, down_streak, momentum_accel, ma_cross_dir

**出来高3**: vr520, vr2060, vsurge

**日経マクロ3**: 日経225の5/20/60日リターン

**60日系列要約7**: autocorr_lag1, skew, max_ret, min_ret, pos_ratio, trend_slope, recent_vs_early

**日経相対アルファ4**: rel5, rel20, rel60, alpha_momentum

**ファンダメンタル11**: per, pbr, roe, days_to_earnings, days_since_div_ex, sin/cos_month, div_yield, eps_growth, dps_growth

**マクロ拡張4**: vix, us5, us20 (SP500), jpy5 (USD/JPY)

**新規IB特徴量8**: amihud_f (非流動性), fx_beta, jpy5, eps_surprise, bps_growth, piotroski, payout, accruals (Sloan正確版)

**EDINET1**: edinet_hold_f（大量保有報告書の保有比率）

**クロスセクショナルランク7**: cs_ret5, cs_ret20, cs_ret60, cs_rsi, cs_vol20, cs_pos52, cs_sector_ret60

### 予測ラベル
- 下落モデル：63日後（約3ヶ月）に **15%以上下落**（絶対リターン、DROP_THRESHOLD=15.0）
- 上昇モデル（rf_model.pkl）は廃止済み。下落モデルのみ学習・保存する

### ウォークフォワードモデル

先読みバイアスなしのバックテストのため、期間開始日に応じて学習済みモデルを切り替える。

| 期間開始日 | 使用モデル（cutoff） | テストAUC（下落）|
|---|---|---|
| ≥ 2025-07-01 | rf_drop_model_2025-07-01.pkl | 0.818 |
| ≥ 2025-05-01 | rf_drop_model_2025-05-01.pkl | 0.803 |
| ≥ 2025-03-01 | rf_drop_model_2025-03-01.pkl | 0.806 |
| < 2025-03-01 | rf_drop_model.pkl（cutoff 2026-01-01）| 0.766 |

キャリブレーション：IsotonicRegression で確率値を実績頻度に補正済み

---

## ランキングロジック（core/rank_stocks.py）

下落確率(%)の昇順（低い順）でランキングし、`drop_prob < 8%` を買い候補の主条件とする（詳細は上の「S買い 発令条件」）。

> **下落確率の表示について**：モデルの確率はIsotonic較正の特性上、数十段の階段値（例: 3,566銘柄が約31個の値に収束）になり、小数第1位まで出すと多数の銘柄が同じ値（例「20.3%」）に見えてしまう。そのためWeb・メール・LINEの画面表示では小数%ではなく **高 / やや高 / 中 / やや低 / 低** の5段階で示す（しきい値: 30/22/14/7%）。並び順・スコア計算は引き続き数値の下落確率を使用。

### ハードフィルター（除外）
- 連続下落日数 > 3日（down_streak > 0.15）
- 直近60日高値から-15%超（drawdown60 < -0.15）

---

## データ永続化（Supabase / Postgres）

全データを **Supabase（Postgres）** に一元管理する（旧 `stock_alert.db` SQLite から全面移行済み）。
`lib/db.py` が Supabase REST API（`lib/supabase_client.py`）経由で読み書きする。GitHub Actions の
DBキャッシュは廃止。

| テーブル | 内容 |
|---|---|
| `gen_rankings` | 毎日のランキングスコア（コード・下落確率・推奨・rank）|
| `jpx_stock_list` | 業種分類・優待月ほかメタ |
| `gen_market_compare` | 日経 vs S&P500 相対強弱判定 |
| `jquants_fin_summary` | 四半期財務サマリ（EDINET決算XBRLから抽出。テーブル名は旧J-Quants由来）|
| `yahoo_price_cache` | 株価履歴キャッシュ（バックテスト高速化用）|
| `yahoo_market_index` | VIX/S&P500/USDJPY 日次 |
| `edinet_large_holdings` | EDINET大量保有/変更報告書の日次蓄積（先回り突合用）|
| `ext_tdnet_disclosures` | TDnet適時開示（やのしん・⚠️個人運営ソースのため `ext_` で隔離）|
| `jpx_short_selling` | JPX空売り残高報告（0.5%以上）|
| `jpx_margin_balance` | JPX個別銘柄信用取引週末残高 |
| `line_chat_history` | LINE Bot会話履歴（直近3往復、文脈保持用） |
| `line_users` | LINE Bot登録ユーザー |
| `dp_watchlist` | ユーザー別ウォッチ銘柄・dp閾値（LINE Bot）|

全銘柄スクリーン（カタリスト候補）は Postgres RPC `screen_catalyst_candidates()` でサーバーサイド集計する
（REST per-code を避け高速化）。

ランキングCSV（`data/rankings/`）と スクリーナーCSV（`data/screeners/`）も日付付きで保存されるがgitignore対象。

---

## 外部API・データソース一覧

### 利用API

| API | 取得データ | 用途 | 利用ファイル |
|---|---|---|---|
| **Yahoo Finance** (非公式REST) | 株価OHLCV（日次）、日経225/VIX/S&P500/USD/JPY | テクニカル特徴量・マクロ特徴量・バックテスト | `lib/utils.py` (`get_prices`, `get_market_index_df`) |
| **kabutan.jp** (スクレイピング) | PER/PBR/ROE、株主優待月、業績テキスト | ファンダ特徴量・NLP感情分析 | `lib/utils.py`, `lib/alt_data.py`, `lib/kabutan_earnings.py` |
| **EDINET API v2** | 大量保有報告書(350)/変更報告書(360)、有報/四半期報の決算XBRL(BS/PL/CF) | 先回りシグナル・財務サマリ（EPS/BPS/ROE/CFO/売上/営業益/予想）本体 | `lib/edinet.py`, `lib/edinet_financials.py`, `tools/scan_large_holdings.py`, `tools/fetch_edinet_financials.py` |
| **TDnet適時開示** (やのしんWEB-API・⚠️個人運営) | 適時開示（業績修正/増配/自社株買い/M&A等のカタリスト） | 企業イベント情報（LINE通知用）。停止リスク隔離のため `ext_` テーブルに保存 | `lib/tdnet.py`, `tools/fetch_tdnet.py` |
| **JPX 空売り残高/信用取引残高** (公式Excel/CSV) | 空売り残高報告(0.5%以上)、個別銘柄信用週末残高 | 需給シグナル（逆張り/買い残） | `lib/jpx_market_data.py`, `tools/fetch_jpx_market.py` |
| **JPX 東証上場銘柄一覧** (Excel) | 銘柄コード・名前・市場区分・33業種分類 | スクリーニング母集団・セクター分類 | `lib/utils.py`, `core/screener.py` |
| **yfinance** | セクターマッピング（米国ETF対応用） | 米国ETFリードラグフィルター（フェーズ7） | `core/rank_stocks.py` |
| **Supabase REST API** | 全テーブルCRUD | データ永続化（DB一元管理） | `lib/supabase_client.py` |
| **Claude API** (Anthropic) | テキスト生成 | 決算テキスト感情分析（Haiku × kabutan） | `lib/nlp_sentiment.py` |

### 特徴量が使うデータと出所

| カテゴリ | 特徴量 | データ出所 |
|---|---|---|
| **テクニカル (10)** | ret5/20/60/90, ma5_25, ma25_75, rsi, vol20/60, pos52 | Yahoo Finance 株価 |
| **トレンド反転 (5)** | drawdown60, from_hi52, down_streak, momentum_accel, ma_cross_dir | Yahoo Finance 株価 |
| **出来高 (3)** | vr520, vr2060, vsurge | Yahoo Finance 出来高 |
| **日経マクロ (3)** | nk5, nk20, nk60 | Yahoo Finance 日経225 |
| **60日系列要約 (7)** | autocorr, skew, max/min_ret, pos_ratio, slope, recent_vs_early | Yahoo Finance 株価 |
| **相対アルファ (4)** | rel5/20/60, alpha_momentum | Yahoo Finance 株価＋日経 |
| **ファンダメンタル (11)** | per, pbr, roe, earn_feat, div_ex_feat, sin/cos_month, div_yield, eps/dps_growth, dividend_relevant | jquants_fin_summary (EPS/BPS/ROE/決算日), kabutan (優待月) |
| **マクロ拡張 (4)** | vix, us5, us20, jpy5 | Yahoo Finance (^VIX, ^GSPC, JPY=X) |
| **IB特徴量 (8)** | amihud, fx_beta, jpy5, eps_surprise, bps_growth, piotroski, payout, accruals | Yahoo Finance (株価/出来高/為替), jquants_fin_summary (CFO/NP/TA/equity) |
| **EDINET (1)** | edinet_hold_f | edinet_large_holdings（大量保有報告書の保有比率） |
| **クロスセクショナル (7)** | cs_ret5/20/60, cs_rsi, cs_vol20, cs_pos52, cs_sector_ret60 | 上記テクニカル特徴量の日次グループ内正規化 |

### フィルターが使うデータ

| フィルター | 条件 | データ出所 |
|---|---|---|
| **品質フィルター** (`passes_buy_filter`) | 株価≥300, drawdown60≥-20%, down_streak≤4日, RSI<80, 売買代金≥50M | Yahoo Finance 株価・出来高 |
| **💎買い条件** (`recommend_from_scores`) | QV条件(Piotroski≥6/9, pos52<45%, EPS surprise>2% or BPS成長+) + 品質(CFOマージン>0, レバレッジ<5x) + drop_prob<8%, vol≤20%, ret90>-25%, 売買代金≥50M, bear時は💎抑制 | モデル予測＋jquants_fin_summary |
| **🔴売り検討** (`recommend_from_scores`) | drop_prob≥10% / drawdown60<-20% / 連続下落≥5日（いずれか該当で警告） | モデル予測＋株価データ |
| **優待フィルター** (フェーズ5) | 権利落ち21日前以内→S買い降格 | kabutan 優待月 |
| **米国ETFフィルター** (フェーズ7) | 対応セクターETF前日リターン<0→S買い降格 | Yahoo Finance (XLK/XLF/XLI等) |
| **レジーム調整** | 日経20日<-5%→下落相場、VIX>30→高恐怖 | Yahoo Finance (日経/VIX) |
| **カタリストスクリーン** (RPC) | PBR<1.0, ROE<8%, 自己資本比率>50%, 売買代金≥指定値 | jquants_fin_summary |
| **利益の質フィルター** (A/B) | 営業赤字/化粧決算/本業減益を除外 | jquants_fin_summary (営業益/売上/純利益) |
| **EDINET突合** | 大量保有報告×カタリスト候補マッチ（自己申告・売り除外） | EDINET API |

---

## セットアップ

### 必要なSecrets（GitHub Settings → Secrets）

| Secret名 | 内容 |
|---|---|
| `GMAIL_ADDRESS` | 送信元Gmailアドレス |
| `GMAIL_APP_PASSWORD` | Gmailアプリパスワード |
| `SUPABASE_URL` | Supabase プロジェクトURL（全データ永続化の宛先）|
| `SUPABASE_SERVICE_KEY` | Supabase service_role キー（バックエンド書込用）|
| `EDINET_API_KEY` | EDINET API v2 サブスクリプションキー（日次の大量保有スキャン用。未登録ならスキャンはスキップ）|

### 依存パッケージ
```
requests pandas numpy scikit-learn joblib xgboost python-dotenv openpyxl yfinance
```

### パス設定（ローカル実行）

各スクリプトはデフォルトで「実行中のプロジェクトディレクトリ」を参照する。  
別ディレクトリに `.env` / モデル / CSV を置く場合は `STOCK_ALERT_HOME` を設定する。

```bash
export STOCK_ALERT_HOME=/path/to/stock-alert
```

### 手動実行コマンド
```bash
python3 screener.py              # スクリーニング（全銘柄、約30分）
python3 screener.py --test       # テストモード（5銘柄のみ）
python3 rf_train_v3.py           # モデル学習（40〜70分）
python3 rf_train_v3.py --cutoff 2025-07-01  # ウォークフォワード用モデル学習
python3 rank_stocks.py           # ランキング生成
python3 backtest.py              # バックテスト（通常期）→ simulations/backtests/ に保存
python3 backtest.py bear         # 下落相場テスト（2024年8月クラッシュ期）
python3 backtest.py --start 2025-01-01 --end 2025-04-01  # 任意期間指定
python3 backtest.py --start 2025-03-01 --end 2025-06-01 --model-cutoff 2025-03-01  # ウォークフォワード指定

python3 multi_backtest.py        # 33期間一括バックテスト＋フィルター比較（ウォークフォワード）
python3 multi_backtest.py --skip-run  # 既存CSVのみ集計（バックテスト実行なし）

python3 tests/test_screener.py     # スクリーナーユニットテスト
```

---

## 設計上の注意点

- **モデルの限界**：AUC 0.766（下落）はランダム（0.50）よりわずかに良い程度。参考指標として使い、最終判断は自分で行う。
- **2段階構造が必須**：モデル単体を全銘柄に適用しても効果なし。スクリーナー→モデル→下落確率フィルターの順で使うことでアルファが出る。
- **下落相場では慎重に**：日経20日 < -5% のとき赤バナー警告。3ヶ月相対強度閾値も自動引き上げ。
- **日経急騰時の限界**：大型株主導の急騰相場（例：2025年7月 日経+21%超）では中小型株主体の選定が相対的に不利。日経60日 ≥ +15% のときオレンジバナーで警告（新規の日経超え率: 7% vs 通常時59%）。
- **季節性**：3〜5月エントリーが最も好成績（avg+8〜10%、勝率75〜82%）。8〜9月は低調（avg−2.6〜+2.7%）。
- **主要特徴量**：下落モデルはcs_vol20（ボラ相対ランク、8%）・sin_month（7%）・div_ex_feat（7%）が上位。
