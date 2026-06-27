# stock-alert

東証上場株式の機械学習スクリーニング・アラートシステム。毎日自動でランキングを生成し、Gmailでメール通知する。

## システム概要

平日にGitHub Actionsが自動実行される。

```
【20:00 JST】アラートパイプライン（daily_alert.yml）
core/screener.py → core/rank_stocks.py
core/rf_train_v3.py は金曜 or モデル未存在時のみ実行
web/export_to_web.py → web/send_user_alerts.py（Webアプリ向け）

その他ワークフロー: ci.yml（テスト）、frontend_build.yml（ビルド検証）、
keepalive.yml（Supabase keepalive）、watchdog.yml（パイプライン監視）
```

**ウォッチリスト**（/watchlist）は、ユーザーが自分で銘柄をブックマークして監視する**マイ・ウォッチリスト**。
ログイン不要で、ブックマークは `localStorage`（即時の正本・オフラインファースト）に保存しつつ、ブラウザ発行の匿名 `client_id`（キー `stocksignal:client-id`）を識別子に
Supabase `app_bookmarks` へ非同期同期する（実装: `frontend/lib/bookmarks.ts` の `useBookmarks` フック、キー `stocksignal:bookmarks`、API ルート `app/api/bookmarks` の GET/POST/DELETE・service key）。
認証が無いため `client_id` はブラウザ単位で、端末間同期は不可（サーバーに記録が残るのみ）。銘柄詳細ページのヘッダーとランキング各行の**しおりアイコン**（`frontend/components/BookmarkButton.tsx`）で追加/削除する。
ウォッチリストには**「保有株を取り込む」ボタン**があり、オーナーの保有43銘柄（`frontend/lib/owner-holdings.ts`）を自分のブラウザで一括ブックマークできる。
ウォッチリスト本体は当日の `gen_rankings`（全銘柄スコア）から該当コードを引き、銘柄名・**直近1ヶ月ミニチャート**（Sparkline・緑=上昇/赤=下落）・
**お得度**（52週高値からの下落率: −30%↓=🔥大お得 / −20%↓=お得 / −10%↓=やや安 / それ以外=高値圏）・**PER/PBR**・**ネットスコア（上昇↑/下落↓確率）**・推奨ラベルを表示する
（`frontend/components/WatchlistClient.tsx`、デスクトップはテーブル/モバイルはカード）。ミニチャート・お得度・PER/PBR の市場データは API ルート `app/api/watch-metrics`（`fetchWatchMetricsMap`・Yahoo・ISR 1h）から
ブックマークしたコードぶんだけ取得する（ブックマークは client 管理のため）。当日スコアが無い銘柄（上場廃止・売買停止・対象外等）は「未取得」と表示。ブックマークが0件のときは空状態を表示する。
初回訪問時は旧キュレーション（toC独占ブランド10銘柄＝`data/pricing_power_watchlist.csv` 由来）を**デフォルトとして一度だけ種まき**する（`DEFAULT_BOOKMARKS`、フラグ `stocksignal:bookmarks-seeded`）。以後ユーザーが削除したものは復活しない。

**会社説明**（銘柄詳細ページ /stocks/[code] の「この会社について」）は**全銘柄が対象**。
- `web/generate_descriptions.py`：**2段階生成**で事実精度を重視。
  Phase1: Yahoo Finance JPの「特色」（会社四季報データ）をスクレイプ → 事実確実なテキストを直接保存。
  Phase2: スクレイプ失敗分のみ Claude Haiku でバッチ生成（20件/リクエスト・429リトライ・フォールバック）。
  → Supabase `gen_ai_analyses`(model_version=`company-desc-v1`) に保存。
  日次パイプライン Step 5a2 で差分補完（`--limit` で1回の上限）。`--all` で全件。`--refresh` で全銘柄を再スクレイプ（事実精度向上）。
- `web/sync_descriptions.py`：スプレッドシートのシート **「📝 会社説明」**（コード/銘柄名/説明）の手動説明を
  同じテーブルへ upsert（手動が AI 生成を上書き＝最優先）。日次パイプライン Step 5b。
- 既存の `/api/stock/[code]/description` がこのキャッシュを返す。QA `description_coverage` が全銘柄のカバレッジ欠損を指摘。

**企業インサイト**（銘柄詳細ページ /stocks/[code] の「企業インサイト」・**AI生成/参考**）は、事業概要(拡張)・主要取引先・カタリスト評価・リスクの定性解説。
- API `/api/stock/[code]/insight`（Claude Haiku、Supabase `gen_ai_analyses` の model_version=`company-insight-v1` にJSONキャッシュ）→ `components/StockInsightPanel.tsx` が「🤖 AI生成・参考」免責付きで表示。取引先は要確認（一次情報＝有価証券報告書）の注記あり。
- 「利益の質（化粧／本業シュリンク／営業益トレンド）」の**実データ表示はフェーズ2**（営業利益等のSupabaseエクスポートが必要・`docs/handoff_web_quality_section.md`）。

`frontend/` を変更して push すると `.github/workflows/frontend_build.yml` が自動でビルド検証し、
失敗時（=Vercelデプロイも失敗する）は Gmail に通知する。フロント変更前のローカル `npm run build` 確認も推奨。

---

## ファイル構成

| ファイル | 役割 |
|---|---|
| `core/screener.py` | JPX全銘柄から条件通過銘柄を抽出して `data/screeners/` に保存 |
| `core/rf_train_v3.py` | XGBoostモデルを東証全銘柄×5年データで学習（金曜のみ）。`--cutoff YYYY-MM-DD` でウォークフォワード用モデルも生成可能 |
| `core/rank_stocks.py` | スクリーナー通過銘柄に上昇/下落確率をつけてランキング生成・DB保存。フェーズ5(決算チェック)→フェーズ6(3件cap)→フェーズ7(米国ETFリードラグフィルター) |
| `web/export_to_web.py` | Supabaseへランキング・AI解析をエクスポート（Step 5）|
| `web/generate_descriptions.py` | 全銘柄の会社説明を2段階で生成（Phase1: Yahoo特色スクレイプ／Phase2: Haiku Haikuフォールバック）→ `gen_ai_analyses`(company-desc-v1)。`--refresh` で全銘柄再スクレイプ。Step 5a2 |
| `web/sync_descriptions.py` | スプシ「📝 会社説明」の手動説明を `gen_ai_analyses`(company-desc-v1) へ同期（AI生成を上書き）。Step 5b |
| `web/send_user_alerts.py` | Webアプリユーザーへのプッシュ通知送信（Step 6）|
| `config.py` | 戦略パラメータの一元管理（閾値・フィルター値）|
| `lib/utils.py` | 共通関数（get_prices, extract_features, add_cs_rank_features, recommend_from_scores 等）|
| `lib/db.py` | Supabase永続化層（gen_rankings / gen_stock_meta / yahoo_price_cache ほか）。`lib/supabase_client.py` のREST API経由 |
| `lib/sheets_helper.py` | Googleスプレッドシート連携 |
| `lib/data_sanity.py` | **Quality Assurance (QA)** ロール。リリースのたびにデータを検証。`check_ranking`（net=rise−drop整合・確率レンジ・予測多様性等の行レベル）＋`check_site`（テーブル横断のカバレッジ・鮮度・欠損＋会社説明カバレッジ `description_coverage`：全銘柄（当日ランキング）に会社説明が無いと指摘）＋`check_pages`（全Webページのスモーク検査：HTTPステータス・エラー画面・空ページ・期待文言の欠落を検知）。全リリース地点（rank_stocks/export_to_web/send_user_alerts）で使用（alert-only：違反でも更新は止めずメール通知）|
| `web/qa_pages.py` | QA: 本番サイトの全ページ（/ /rankings /watchlist ＋サンプル銘柄ページ）を巡回し `check_pages` で検査。日次パイプライン Step 5c で実行 |
| `lib/kabutan_earnings.py` | kabutan.jpから決算業績を取得（AI解析プロンプト用）|
| `lib/risk_regime.py` | **相場リスク管制官**。日経20日・VIX・ドル円・S&P500からリスクオン/オフを判定。rank_stocksのフェーズ8でリスクオフ日はS買いを自動見送り、判定を `data/risk_regime.json` に保存しメールに警告表示 |
| `tools/backtest.py` | バックテスト（先読みバイアスなし）。結果は `simulations/backtests/` に保存。`--model-cutoff YYYY-MM-DD` でウォークフォワード用モデル指定可能 |
| `tools/multi_backtest.py` | 33期間一括バックテスト＋フィルター比較分析（ウォークフォワード対応） |
| `tools/simulate_monthly.py` | 月次シミュレーション（保有シナリオ分析）|
| `tools/screen_catalyst_candidates.py` | カタリスト候補スクリーン（GARP補助）。PBR<1.0・ROE<8%・自己資本比率>50%・流動性の「安い箱」抽出は Postgres RPC `screen_catalyst_candidates()` でサーバーサイド集計（J-Quants財務データ使用）。通過候補に **利益の質フィルター(A/B)** で化粧決算（営業赤字・純利益>営業益×1.5）と斜陽事業（本業減益）を除外し、売上CAGR・営業利益率・会社予想方向で加減点。`data/catalyst_candidates.csv`（残）＋ `data/catalyst_excluded.csv`（除外理由付き・レビュー用）。`--no-quality` で品質フィルター無効 |
| `tools/catalyst_backtest.py` | カタリスト候補スクリーンのヒストリカルBT（point-in-time・disc_date≤基準日）。A/Bあり/なしで平均・勝率・大勝率を比較。データは J-Quants財務＋yahoo_price_cache |
| `lib/earnings_quality.py` | カタリスト候補の利益の質・本業方向性を判定（年次の営業益/売上/純益から化粧決算/斜陽を機械判定）。データ源は kabutan 優先、取れない環境（クラウドはkabutanがIPブロック）では J-Quants 実績にフォールバック |
| `lib/edinet.py` + `tools/scan_large_holdings.py` | **EDINET大量保有スキャナー**（イベント駆動）。EDINET APIから大量保有報告書(350)/変更報告書(360)を日次スキャンして `edinet_large_holdings` に蓄積し、カタリスト候補と突合（構造的候補×実際の買い集め＝先回り候補）。突合時に自己申告（提出者≒対象企業）と譲渡/売却の報告を除外し、外部の買い集めだけ残す（`--no-exclude` で無効化可）。`EDINET_API_KEY` 必須 |
| `tests/test_earnings_quality.py` | 利益の質フィルター（化粧・赤字・減益・加減点）のユニットテスト（8件）|
| `tests/test_screener.py` | スクリーナー条件のユニットテスト（9件）|
| `tests/test_data_sanity.py` | QA（データ整合性・サイト全体・会社説明カバレッジ・全ページスモーク）のユニットテスト（29件）|

---

## S買い 発令条件（passes_buy_filter + rank_stocks.py フェーズ5〜7）

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
| ネットスコア (上昇確率−下落確率) | 17% ≤ net ≤ 24% |
| 下落確率 | < 4% |
| 年率ボラティリティ | ≤ 25% |

フェーズ5〜7 追加フィルター（`rank_stocks.py`）:
- フェーズ5: 株主優待権利落ち21日前以内の銘柄はS買い→方向感なしに降格
- フェーズ5b: 株主優待権利落ち21日前以内の銘柄はS買い→方向感なしに降格（優待クロス売り圧力を回避）
- フェーズ6: S買い1日最大3件のキャップ（net降順）。4件目以降は方向感なしに降格
- フェーズ7: 対応する米国セクターETF（XLK/XLF/XLI/XLB/XLV/XLY）の前日リターンがマイナスならS買い→方向感なしに降格。リードラグ効果（US→JP翌日）を活用。21,416サンプル(2023-2026)で全26ペア正相関・avg +0.64pp効果を確認。キャッシュは `data/sector_map.json`。

**推奨ラベル**:
- 💎 買い: QV条件+ファンダ品質+モデルスコア全条件クリア
- 🔴 売り検討: drop_prob≥10% / net<-5 / drawdown60<-20% / 連続下落≥5日
- —: それ以外

## スクリーナー条件（screener.py / 前段フィルター）

`screener.py → rank_stocks.py` の前段として動作する追加フィルター:

| 条件 | 値 |
|---|---|
| 3ヶ月相対強度 ≥ | 0%（通常）/ +5%（下落相場：日経20日 < -5%）|
| セクター集中除外 | 同一業種3銘柄以上でセクター全除外（バブル兆候回避）|

バックテスト（33期間ウォークフォワード: 2021〜2025年）: 現行フィルター avg+6.7% / 日経アルファ+4.4%（有効16/33期間）
net10〜13%帯（実運用相当）に絞ると avg+7.4% 勝率73%と更に良化。3〜5月エントリーが最も好成績（avg+8〜10%）。

### Top10シミュレーション（Webアプリ・メールに表示）

毎日のネットスコア上位10銘柄（`gen_rankings.rank ≤ 10`）を100株ずつ購入し、
ネットスコアが **5%未満** に下がった日に売却するロジック。手数料・税金は含まない。

- Web版（`frontend/lib/simulation.ts`）: `gen_rankings` の履歴から遡って集計（即時表示）
- メール版（`web/export_to_web.py` の `update_top10_simulation`）: `gen_top10_sim` テーブルで日次トラッキング（フォワード）
- 売却基準は `SELL_NET_THRESH = 5`（信号消滅）。銘柄ごとに最初のtop10入り日を購入日とする。

### シミュレーション実績（2025/05〜11月エントリー組・2026-05-16時点）

37銘柄を候補選出日の終値でエントリーし、現在まで保有した場合の実績。

| 指標 | 値 |
|---|---|
| 平均リターン | **+18.0%** |
| 中央値 | +17.4% |
| 勝率（+0%以上） | **89.2%**（33/37銘柄）|
| 大勝率（+15%以上） | **54.1%**（20/37銘柄）|
| 最高 | ニッポンインシュア +75.3%（337日保有）|
| 最低 | キッセイ薬品 −4.8%（276日保有）|

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
- 上昇モデル：63日後（約3ヶ月）に日経225を **+5%以上上回る**（アルファ ≥ +5%）
- 下落モデル：63日後（約3ヶ月）に日経225を **5%以上下回る**（アルファ ≤ -5%）
- 絶対リターンではなく日経比相対リターン（アルファ）でラベル付けし、相場全体に依存しない銘柄選定力を学習

### ウォークフォワードモデル

先読みバイアスなしのバックテストのため、期間開始日に応じて学習済みモデルを切り替える。

| 期間開始日 | 使用モデル（cutoff） | テストAUC（上昇/下落）|
|---|---|---|
| ≥ 2025-07-01 | rf_model_2025-07-01.pkl | 0.655 / 0.818 |
| ≥ 2025-05-01 | rf_model_2025-05-01.pkl | 0.646 / 0.803 |
| ≥ 2025-03-01 | rf_model_2025-03-01.pkl | 0.645 / 0.806 |
| < 2025-03-01 | rf_model.pkl（cutoff 2026-01-01）| 0.642 / 0.766 |

キャリブレーション：IsotonicRegression で確率値を実績頻度に補正済み

---

## ランキングロジック（core/rank_stocks.py）

**ネットスコア = 上昇確率(%) − 下落確率(%)**

> **上昇確率/下落確率の表示について**：モデルの確率はIsotonic較正の特性上、数十段の階段値（例: 3,566銘柄が約31個の値に収束）になり、小数第1位まで出すと多数の銘柄が同じ値（例「20.3%」）に見えてしまう。そのためWeb・メール・LINEの画面表示では小数%ではなく **高 / やや高 / 中 / やや低 / 低** の5段階で示す（しきい値: 30/22/14/7%）。並び順・スコア計算は引き続き数値のネットスコアを使用。

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
| `gen_rankings` | 毎日のランキングスコア（コード・確率・ネット・推奨・rank）|
| `gen_stock_meta` | 業種分類・優待月ほかメタ |
| `gen_ai_analyses` | AI分析（会社説明・企業インサイト）|
| `gen_top10_sim` | Top10シミュレーション（日次トラッキング）|
| `gen_simulation` | バックテスト結果 |
| `gen_risk_regime` | リスクオン/オフ判定 |
| `gen_dividend_strategy` | 配当戦略 |
| `gen_qv_sim` | QV戦略バックテスト結果 |
| `jquants_fin_summary` | 四半期財務サマリ（J-Quants）|
| `yahoo_price_cache` | 株価履歴キャッシュ（バックテスト高速化用）|
| `yahoo_market_index` | VIX/S&P500/USDJPY 日次 |
| `edinet_large_holdings` | EDINET大量保有/変更報告書の日次蓄積（先回り突合用）|
| `ext_tdnet_disclosures` | TDnet適時開示（やのしん・⚠️個人運営ソースのため `ext_` で隔離）|
| `jpx_short_selling` | JPX空売り残高報告（0.5%以上）|
| `jpx_margin_balance` | JPX個別銘柄信用取引週末残高 |
| `simulation_results` | 月次シミュレーション結果 |
| `app_bookmarks` | ウォッチリスト（ブックマーク）|
| `app_push_subscriptions` | プッシュ通知サブスクリプション |
| `line_chat_history` | LINE Bot会話履歴（直近3往復、文脈保持用） |

全銘柄スクリーン（カタリスト候補）は Postgres RPC `screen_catalyst_candidates()` でサーバーサイド集計する
（REST per-code を避け高速化）。

ランキングCSV（`data/rankings/`）と スクリーナーCSV（`data/screeners/`）も日付付きで保存されるがgitignore対象。

---

## 外部API・データソース一覧

### 利用API

| API | 取得データ | 用途 | 利用ファイル |
|---|---|---|---|
| **Yahoo Finance** (非公式REST) | 株価OHLCV（日次）、日経225/VIX/S&P500/USD/JPY | テクニカル特徴量・マクロ特徴量・バックテスト | `lib/utils.py` (`get_prices`, `get_market_index_df`) |
| **J-Quants API v2** (Freeプラン) | 財務サマリ（EPS/BPS/ROE/CFO/売上/営業益/予想） | ファンダ特徴量・IB特徴量・カタリストスクリーン | `tools/fetch_jquants_fin.py` |
| **kabutan.jp** (スクレイピング) | PER/PBR/ROE、株主優待月、業績テキスト | ファンダ特徴量・NLP感情分析 | `lib/utils.py`, `lib/alt_data.py`, `lib/kabutan_earnings.py` |
| **EDINET API v2** | 大量保有報告書(350)/変更報告書(360)、有報/四半期報の決算XBRL(BS/PL/CF) | 先回りシグナル・J-Quants期限切れ後の財務データ補完 | `lib/edinet.py`, `lib/edinet_financials.py`, `tools/scan_large_holdings.py`, `tools/fetch_edinet_financials.py` |
| **TDnet適時開示** (やのしんWEB-API・⚠️個人運営) | 適時開示（業績修正/増配/自社株買い/M&A等のカタリスト） | 企業イベント情報（LINE通知用）。停止リスク隔離のため `ext_` テーブルに保存 | `lib/tdnet.py`, `tools/fetch_tdnet.py` |
| **JPX 空売り残高/信用取引残高** (公式Excel/CSV) | 空売り残高報告(0.5%以上)、個別銘柄信用週末残高 | 需給シグナル（逆張り/買い残） | `lib/jpx_market_data.py`, `tools/fetch_jpx_market.py` |
| **JPX 東証上場銘柄一覧** (Excel) | 銘柄コード・名前・市場区分・33業種分類 | スクリーニング母集団・セクター分類 | `lib/utils.py`, `core/screener.py` |
| **yfinance** | セクターマッピング（米国ETF対応用） | 米国ETFリードラグフィルター（フェーズ7） | `core/rank_stocks.py` |
| **Supabase REST API** | 全テーブルCRUD | データ永続化（DB一元管理） | `lib/supabase_client.py` |
| **Claude API** (Anthropic) | テキスト生成 | 会社説明(Phase2)・企業インサイト | `web/generate_descriptions.py` |

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
| **💎買い条件** (`recommend_from_scores`) | QV条件(Piotroski≥6/9, pos52<45%, EPS surprise>2% or BPS成長+) + 品質(CFOマージン>0, レバレッジ<5x) + drop_prob<8%, net≥10, vol≤20%, ret90>-25%, 売買代金≥50M, bear時は💎抑制 | モデル予測＋jquants_fin_summary |
| **🔴売り検討** (`recommend_from_scores`) | drop_prob≥10% / net<-5 / drawdown60<-20% / 連続下落≥5日（いずれか該当で警告） | モデル予測＋株価データ |
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

- **モデルの限界**：AUC 0.642（上昇）/ 0.766（下落）はランダム（0.50）よりわずかに良い程度。参考指標として使い、最終判断は自分で行う。
- **2段階構造が必須**：モデル単体を全銘柄に適用しても効果なし。スクリーナー→モデル→ネット範囲フィルターの順で使うことでアルファが出る。
- **下落相場では慎重に**：日経20日 < -5% のとき赤バナー警告。3ヶ月相対強度閾値も自動引き上げ。
- **日経急騰時の限界**：大型株主導の急騰相場（例：2025年7月 日経+21%超）では中小型株主体の選定が相対的に不利。日経60日 ≥ +15% のときオレンジバナーで警告（新規の日経超え率: 7% vs 通常時59%）。
- **季節性**：3〜5月エントリーが最も好成績（avg+8〜10%、勝率75〜82%）。8〜9月は低調（avg−2.6〜+2.7%）。
- **主要特徴量**：上昇モデルはjpy5（USD/JPY 5日変動、寄与15%）・sin_month（季節性、12%）・div_ex_feat（配当権利日、8%）が上位。下落モデルはcs_vol20（ボラ相対ランク、8%）・sin_month（7%）・div_ex_feat（7%）が上位。
