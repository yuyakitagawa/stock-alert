# stock-alert

東証上場株式の機械学習スクリーニング・アラートシステム。毎日自動でランキングを生成し、Gmailでメール通知する。

## システム概要

平日に2本のGitHub Actionsが自動実行される。

```
【17:30 JST】アラートパイプライン
core/screener.py → core/rank_stocks.py → email/alert_email.py
core/rf_train_v3.py は金曜 or モデル未存在時のみ実行
web/export_to_web.py → web/send_user_alerts.py（Webアプリ向け）

【18:00 JST】PDCA自律改善ループ（pdca/orchestrate.py）
オーナー(Human) → Fund Manager → 各メンバーへ命令展開
  Human が feedback.md に方針を記入 → FM が各メンバーへの具体的命令に翻訳
  Quant(改善提案) / 相場リスク管制官(リスクオン/オフ判定・買い見送り) / Engineer(実装・検証) / QA(整合性検査)
  ↳ パラメータを変更 → backtest.py bear で検証
  ↳ 改善したらcommit/push、ダメなら即revert

【月曜 10:00 JST】週次チームレビュー（pdca/weekly_review.py）
5ロールが相互評価 → 来週の改善アクションを feedback.md に反映
```

全アクション（実施中・実施済み）は `pdca/activity.py` 経由で Supabase `activity_log` に記録され、
Webの **活動ログ**（/activity）・**チームレビュー**（/review）で誰でも状況把握できる。

**値上げ力ウォッチリスト**（/watchlist）は、シェアを独占しインフレ下でも値上げを通せる toC ブランド銘柄の
将来買い候補リスト（オーナー選定）。正本は `data/pricing_power_watchlist.csv`、フロントは `frontend/lib/watchlist.ts`
の静的定数で持ち、当日の `web_rankings`（netスコア・シグナル）と銘柄コードで突き合わせて表示する。
長期の押し目買い向けに、**直近1ヶ月のミニチャート**（Sparkline・緑=上昇/赤=下落）、**お得度**（52週高値からの下落率:
−30%↓=🔥大お得 / −20%↓=お得 / −10%↓=やや安 / それ以外=高値圏）、**PER / PBR**、**上昇↑/下落↓確率** を表示する。
株価・チャート・52週高値は Yahoo Finance chart API（認証不要・`fetchWatchMetricsMap` で並列取得、各fetchにタイムアウト、ISR 1h）、
PER は当日 `web_rankings` のファンダ値（Yahoo補完）。PBR は bps データ未整備の銘柄が多く「—」になりやすい。
netスコア・確率は当日ランキング（全銘柄スコアリング）から取得。市場価格データが無い銘柄（上場廃止・売買停止等）は「データ未取得」と表示。シェア率・海外比率は公知ベースの概算。

**会社説明**（銘柄詳細ページ /stocks/[code] の「この会社について」）は、スプレッドシートのシート
**「📝 会社説明」**（コード/銘柄名/説明）で手動管理する。`web/sync_descriptions.py` がシートを読み、
Supabase `ai_analyses`(model_version=`company-desc-v1`) へ upsert → 既存の `/api/stock/[code]/description`
がそれを最優先で返す（未記入銘柄はClaude Haiku自動生成にフォールバック）。日次パイプライン Step 5b で自動同期。

`frontend/` を変更して push すると `.github/workflows/frontend_build.yml` が自動でビルド検証し、
失敗時（=Vercelデプロイも失敗する）は Gmail に通知する。フロント変更前のローカル `npm run build` 確認も推奨。

---

## ファイル構成

| ファイル | 役割 |
|---|---|
| `core/screener.py` | JPX全銘柄から条件通過銘柄を抽出して `data/screeners/` に保存 |
| `core/rf_train_v3.py` | XGBoostモデルを東証全銘柄×5年データで学習（金曜のみ）。`--cutoff YYYY-MM-DD` でウォークフォワード用モデルも生成可能 |
| `core/rank_stocks.py` | スクリーナー通過銘柄に上昇/下落確率をつけてランキング生成・DB保存。フェーズ5(決算チェック)→フェーズ6(3件cap)→フェーズ7(米国ETFリードラグフィルター) |
| `email/alert_email.py` | チェック銘柄の評価＋新規候補をGmailで送信 |
| `email/email_html.py` | メールHTML生成ヘルパー（スパークライン・優先アクション等）|
| `web/export_to_web.py` | Supabaseへランキング・AI解析をエクスポート（Step 5）|
| `web/send_user_alerts.py` | Webアプリユーザーへのプッシュ通知送信（Step 6）|
| `pdca/orchestrate.py` | PDCA自律改善ループ（Human→FM→Quant/Consultant/Engineer→QA の指揮系統）。QAがモデル退化・データ不整合を検出（通知のみ）|
| `pdca/move_history.py` | PDCA棋譜（過去の変更履歴・採否を抽出、振り子＝逆戻し提案を検出）|
| `pdca/weekly_review.py` | 週次チームレビュー（5ロール相互評価＋オーナーへのフィードバック）|
| `pdca/activity.py` | アクティビティlog（各担当者の実施中・実施済みを Supabase に記録）|
| `pdca/feedback.md` | オーナー(Human)からチームへの方針・指示。FMが各メンバーへの命令に翻訳する |
| `pdca/owner_directive.py` | オーナーの自然文の方針を、FMが feedback.md の要点に整理して書き込む（`python3 pdca/owner_directive.py "方針"`）|
| `config.py` | 戦略パラメータの一元管理（閾値・フィルター値）|
| `lib/utils.py` | 共通関数（get_prices, extract_features, add_cs_rank_features, recommend_from_net 等）|
| `lib/db.py` | SQLite操作（daily_ranking / held_scores / earnings_cache / sector_cache / price_cache）|
| `lib/sheets_helper.py` | Googleスプレッドシート連携 |
| `lib/data_sanity.py` | **Quality Assurance (QA)** ロール。リリースのたびにデータを検証。`check_ranking`（net=rise−drop整合・確率レンジ・予測多様性等の行レベル）＋`check_site`（テーブル横断のカバレッジ・鮮度・欠損＋会社説明カバレッジ `description_coverage`：ウォッチリスト＋保有株に説明が無いと指摘）。全リリース地点（rank_stocks/export_to_web/alert_email/send_user_alerts）とPDCAで使用（alert-only：違反でも更新は止めずメール通知）|
| `lib/kabutan_earnings.py` | kabutan.jpから決算業績を取得（AI解析プロンプト用）|
| `lib/risk_regime.py` | **相場リスク管制官**。日経20日・VIX・ドル円・S&P500からリスクオン/オフを判定。rank_stocksのフェーズ8でリスクオフ日はS買いを自動見送り、判定を `data/risk_regime.json` に保存しメールに警告表示 |
| `tools/backtest.py` | バックテスト（先読みバイアスなし）。結果は `simulations/backtests/` に保存。`--model-cutoff YYYY-MM-DD` でウォークフォワード用モデル指定可能 |
| `tools/multi_backtest.py` | 33期間一括バックテスト＋フィルター比較分析（ウォークフォワード対応） |
| `tools/simulate_monthly.py` | 月次シミュレーション（保有シナリオ分析）|
| `tests/test_screener.py` | スクリーナー条件のユニットテスト（9件）|
| `tests/test_alert_email.py` | メール生成ヘルパーのユニットテスト（36件）|
| `tests/test_data_sanity.py` | QA（データ整合性・サイト全体・会社説明カバレッジ）のユニットテスト（22件）|

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
- フェーズ5: 決算22日以内の銘柄はS買い→方向感なしに降格
- フェーズ5b: 株主優待権利落ち21日前以内の銘柄はS買い→方向感なしに降格（優待クロス売り圧力を回避）
- フェーズ6: S買い1日最大3件のキャップ（net降順）。4件目以降は方向感なしに降格
- フェーズ7: 対応する米国セクターETF（XLK/XLF/XLI/XLB/XLV/XLY）の前日リターンがマイナスならS買い→方向感なしに降格。リードラグ効果（US→JP翌日）を活用。21,416サンプル(2023-2026)で全26ペア正相関・avg +0.64pp効果を確認。キャッシュは `data/sector_map.json`。

**推奨ラベル**:
- 🥇 S買い: 全条件クリア（1日上位3件）
- ⏳ 方向感なし: フィルター非通過 or ネット弱 or ETF/決算で降格
- ⚠️ 弱気シグナル: ネット -10〜-5%
- 🔴 下降シグナル: ネット < -10%

## スクリーナー条件（screener.py / 前段フィルター）

`screener.py → rank_stocks.py` の前段として動作する追加フィルター:

| 条件 | 値 |
|---|---|
| 3ヶ月相対強度 ≥ | 0%（通常）/ +5%（下落相場：日経20日 < -5%）|
| セクター集中除外 | 同一業種3銘柄以上でセクター全除外（バブル兆候回避）|

バックテスト（33期間ウォークフォワード: 2021〜2025年）: 現行フィルター avg+6.7% / 日経アルファ+4.4%（有効16/33期間）
net10〜13%帯（実運用相当）に絞ると avg+7.4% 勝率73%と更に良化。3〜5月エントリーが最も好成績（avg+8〜10%）。

### Top10シミュレーション（Webアプリ・メールに表示）

毎日のネットスコア上位10銘柄（`web_rankings.rank ≤ 10`）を100株ずつ購入し、
ネットスコアが **5%未満** に下がった日に売却するロジック。手数料・税金は含まない。

- Web版（`frontend/lib/simulation.ts`）: `web_rankings` の履歴から遡って集計（即時表示）
- メール版（`web/export_to_web.py` の `update_top10_simulation`）: `top10_sim` テーブルで日次トラッキング（フォワード）
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

### 特徴量（60次元 = 53次元 + クロスセクション7次元）

**テクニカル10**: ret5, ret20, ret60, ret90, ma5_25, ma25_75, rsi, vol20, vol60, pos52

**トレンド反転5**: drawdown60, from_hi52, down_streak, momentum_accel, ma_cross_dir

**出来高3**: vr520, vr2060, vsurge

**日経マクロ3**: 日経225の5/20/60日リターン

**60日系列要約7**: autocorr_lag1, skew, max_ret, min_ret, pos_ratio, trend_slope, recent_vs_early

**日経相対アルファ4**: rel5, rel20, rel60, alpha_momentum

**ファンダメンタル11**: per, pbr, roe, days_to_earnings, days_since_div_ex, sin/cos_month, div_yield, eps_growth, dps_growth

**マクロ拡張4**: vix, us5, us20 (SP500), fx_beta (USD/JPY)

**新規IB特徴量8**: amihud_f (非流動性), fx_beta, jpy5, eps_surprise, bps_growth, piotroski, payout, accruals (Sloan正確版)

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
| < 2025-03-01 | rf_model.pkl（cutoff 2025-01-01）| 0.663 / 0.791 |

キャリブレーション：IsotonicRegression で確率値を実績頻度に補正済み

---

## ランキングロジック（core/rank_stocks.py）

**ネットスコア = 上昇確率(%) − 下落確率(%)**

### ハードフィルター（除外）
- 連続下落日数 > 3日（down_streak > 0.15）
- 直近60日高値から-15%超（drawdown60 < -0.15）

### 損切りライン
各銘柄に自動計算：`損切り価格 = 現値 × (1 − 1.5 × 年率ボラ × √(20/252))`

---

## メール機能（alert_email.py）

- **チェック銘柄一覧**：保有銘柄をネット降順で表示（#ランク・銘柄・上昇確率・下落確率・ネット・推奨・日経差(20日)・ボラ・保有日数）
- **売り検討**：段階的閾値で保有期間に応じた売りシグナル（後述）。赤枠で強調
- **新規候補 Top5**：スクリーナー通過 + モデル予測ネットスコア ≥10% かつ下落確率 <8%（コンフリクト除外済み）の銘柄
- **損切り価格**：新規候補に損切りラインを自動表示
- **決算警告**：14日以内に決算がある保有銘柄に「決算X日前」バッジ表示（新規候補は7日以内を自動除外）
- **昨日比差分**：ネットスコアが±3%以上変動した銘柄を表示
- **セクター集中警告**：保有銘柄に同一業種が3銘柄以上集中している場合に警告（新規候補側はスクリーナーで除外済み）
- **下落相場バナー**：日経20日 < -5% のとき赤バナーで「新規買いは見送り推奨」を表示
- **急騰相場バナー**：日経60日 ≥ +15% のときオレンジバナーで「新規エントリーは慎重に / ETF検討」を表示
- **ボラランク**：新規候補にcs_vol20（当日の全スクリーナー通過銘柄中のボラ相対ランク）を表示

### 段階的売りシグナル（tiered sell signal）

モデルの予測ホライズンは63取引日（約3ヶ月）。保有が長期化するほど予測精度が低下するため、保有日数に応じて売り閾値を段階的に引き締める。

| 保有日数 | 売り閾値 | 意図 |
|---|---|---|
| 0〜63日（モデル内） | net < +6% | 買い水準を下回ったら売り |
| 64日〜（モデル外） | net < +6% | 予測期限切れ、弱ければ売り |

---

## データ永続化（SQLite）

`stock_alert.db` に以下のテーブルを保存する（gitignore対象）。

| テーブル | 内容 |
|---|---|
| `daily_ranking` | 毎日のランキングスコア（コード・確率・ネット・推奨）|
| `held_scores` | 保有銘柄の前日スコア（昨日比差分計算用）|
| `earnings_cache` | 決算日キャッシュ（kabutan.jpから取得、日次更新）|
| `sector_cache` | 業種分類キャッシュ（JPX Excelから取得）|
| `price_cache` | 株価履歴キャッシュ（Yahoo Finance、バックテスト高速化用）|

ランキングCSV（`data/rankings/`）と スクリーナーCSV（`data/screeners/`）も日付付きで保存されるがgitignore対象。

---

## セットアップ

### 必要なSecrets（GitHub Settings → Secrets）

| Secret名 | 内容 |
|---|---|
| `GMAIL_ADDRESS` | 送信元Gmailアドレス |
| `GMAIL_APP_PASSWORD` | Gmailアプリパスワード |
| `SPREADSHEET_ID` | GoogleスプレッドシートID |
| `GCP_KEY_JSON` | Google Cloud サービスアカウントJSON |

### 依存パッケージ
```
requests pandas numpy scikit-learn joblib xgboost python-dotenv openpyxl gspread google-auth yfinance
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
python3 alert_email.py           # メール送信

python3 backtest.py              # バックテスト（通常期）→ simulations/backtests/ に保存
python3 backtest.py bear         # 下落相場テスト（2024年8月クラッシュ期）
python3 backtest.py --start 2025-01-01 --end 2025-04-01  # 任意期間指定
python3 backtest.py --start 2025-03-01 --end 2025-06-01 --model-cutoff 2025-03-01  # ウォークフォワード指定

python3 multi_backtest.py        # 33期間一括バックテスト＋フィルター比較（ウォークフォワード）
python3 multi_backtest.py --skip-run  # 既存CSVのみ集計（バックテスト実行なし）

python3 tests/test_screener.py     # スクリーナーユニットテスト
python3 tests/test_alert_email.py  # メールユニットテスト
```

---

## 設計上の注意点

- **モデルの限界**：AUC 0.663（上昇）/ 0.791（下落）はランダム（0.50）よりわずかに良い程度。参考指標として使い、最終判断は自分で行う。
- **2段階構造が必須**：モデル単体を全銘柄に適用しても効果なし。スクリーナー→モデル→ネット範囲フィルターの順で使うことでアルファが出る。
- **下落相場では慎重に**：日経20日 < -5% のとき赤バナー警告。3ヶ月相対強度閾値も自動引き上げ。
- **日経急騰時の限界**：大型株主導の急騰相場（例：2025年7月 日経+21%超）では中小型株主体の選定が相対的に不利。日経60日 ≥ +15% のときオレンジバナーで警告（新規の日経超え率: 7% vs 通常時59%）。
- **季節性**：3〜5月エントリーが最も好成績（avg+8〜10%、勝率75〜82%）。8〜9月は低調（avg−2.6〜+2.7%）。
- **ボラランクの重要性**：モデルの最重要特徴量はcs_vol20（同日内でのボラ相対ランク、寄与19%）。ランクが高いほど上昇確率が高い傾向。
