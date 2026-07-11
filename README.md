# stock-alert

東証上場株式の機械学習スクリーニングシステム。XGBoostで3ヶ月先の株価変動を予測し、毎日ランキングを生成してGmail・LINEで通知する。

## アーキテクチャ

```
┌─────────────────────────────────────────────────────────┐
│  Modal.com（金曜 19:00 JST）                             │
│                                                         │
│  modal_train.py → rf_train_v3.py                        │
│       ↓                                                 │
│  XGBoostモデル学習 → Supabase Storage（models/）に保存  │
└─────────────────────────────────────────────────────────┘
         ↓ モデルをダウンロード
┌─────────────────────────────────────────────────────────┐
│  GitHub Actions（平日 20:00 JST）                        │
│                                                         │
│  screener.py → rank_stocks.py → export_to_web.py        │
│       ↓              ↓               ↓                  │
│  銘柄抽出      XGBoost予測     Supabase書込              │
│                                      ↓                  │
│                            send_user_alerts.py（メール） │
│                            market_timing_alert.py（LINE）│
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Supabase Edge Function                                 │
│                                                         │
│  line-webhook/index.ts                                  │
│  → Claude API でリアルタイム株式相談                      │
│  → ウォッチリスト管理・銘柄検索・需給情報                  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Supabase (Postgres + Storage)                          │
│  全データの一元管理（ランキング・株価・財務・開示等）       │
│  Storage: models/ バケットにモデルファイルを保存          │
└─────────────────────────────────────────────────────────┘
```

## 主な機能

### 1. 日次スクリーニング & ランキング
- 東証全銘柄（約3,500〜4,000銘柄）を毎日スクリーニング
- XGBoostで63日先（約3ヶ月）の上昇/下落確率を予測
- ネットスコア（上昇確率 − 下落確率）でランキング生成
- 品質フィルター + 複数フェーズの追加フィルターで精度向上

### 2. LINE Bot AI相談（Supabase Edge Function）
- LINEで話しかけるだけで株式相談ができるAIアシスタント
- Claude APIによるリアルタイム分析（銘柄データ・決算・需給情報を参照）
- ウォッチリスト管理（自然言語で追加/削除/一覧）
- TDnet適時開示・JPX空売り残高・信用残高の表示
- 会話履歴を記憶（直近3往復）

### 3. 通知
- **Gmail**: 毎日のランキング・推奨銘柄をメール配信
- **LINE Push**: 日経225シグナル（投資継続OK/キャッシュ推奨）+ ウォッチリスト銘柄の状態

---

## ディレクトリ構成

```
stock-alert/
├── core/                     # コアロジック
│   ├── screener.py           # 全銘柄スクリーニング
│   ├── rank_stocks.py        # ランキング生成（8フェーズ）
│   └── rf_train_v3.py        # XGBoostモデル学習（Modalから呼ばれる）
│
├── lib/                      # 共通ライブラリ
│   ├── utils.py              # 特徴量抽出（64次元）・共通関数
│   ├── db.py                 # Supabase永続化層
│   ├── supabase_client.py    # Supabase REST APIクライアント
│   ├── risk_regime.py        # 相場リスク管制官（リスクオン/オフ判定）
│   ├── data_sanity.py        # QA（データ整合性検証）
│   ├── nlp_sentiment.py      # 決算テキスト感情分析（Claude Haiku）
│   ├── kabutan_earnings.py   # kabutan.jp 決算業績取得
│   ├── earnings_quality.py   # 利益の質フィルター
│   ├── edinet.py             # EDINET API連携
│   ├── edinet_financials.py  # EDINET 決算XBRL解析
│   ├── tdnet.py              # TDnet適時開示取得
│   ├── jpx_market_data.py    # JPX空売り・信用残高取得
│   ├── alt_data.py           # kabutan PER/PBR/ROE取得
│   ├── fundamentals.py       # ファンダメンタル計算
│   └── sheets_helper.py      # Googleスプレッドシート連携
│
├── web/                      # データエクスポート・通知
│   ├── export_to_web.py      # Supabaseへランキング・AI解析エクスポート
│   ├── send_user_alerts.py   # ユーザー通知送信
│   ├── market_timing_alert.py # LINE Push通知（日経シグナル）
│   ├── generate_descriptions.py # 会社説明生成（Yahoo特色 + Claude）
│   ├── sync_descriptions.py  # スプシ手動説明同期
│   └── qa_pages.py           # QAサイト巡回検査
│
├── tools/                    # ツール・バックテスト
│   ├── backtest.py           # バックテスト（bearモード対応）
│   ├── multi_backtest.py     # 33期間一括バックテスト
│   ├── simulate_monthly.py   # 月次シミュレーション
│   ├── screen_catalyst_candidates.py # カタリスト候補スクリーン
│   ├── catalyst_backtest.py  # カタリストBT
│   ├── scan_large_holdings.py # EDINET大量保有スキャン
│   ├── fetch_tdnet.py        # TDnet取得
│   ├── fetch_jpx_market.py   # JPX空売り/信用残取得
│   ├── fetch_jquants_fin.py  # J-Quants財務取得
│   ├── fetch_edinet_financials.py # EDINET決算XBRL取得
│   ├── fetch_history.py      # 株価履歴取得
│   ├── backfill_history.py   # 株価履歴バックフィル
│   ├── download_models.py    # Supabase StorageからモデルDL（daily_alertで使用）
│   ├── optimize_net_weights.py # ネットスコア最適化
│   └── export_report_to_sheets.py # レポートスプシ出力
│
├── supabase/functions/
│   └── line-webhook/index.ts # LINE Bot AI相談（Edge Function）
│
├── tests/
│   ├── test_screener.py      # スクリーナーテスト（9件）
│   ├── test_earnings_quality.py # 利益の質テスト（8件）
│   └── test_data_sanity.py   # QAテスト（29件）
│
├── .github/workflows/
│   ├── daily_alert.yml       # 日次パイプライン（平日20:00 JST）
│   ├── ci.yml                # テスト（.py/.yml変更時のみ）
│   ├── data_backfill.yml     # データバックフィル（手動実行）
│   └── keepalive.yml         # Supabase keepalive（月曜のみ）
│
├── modal_train.py            # Modal.com モデル学習（金曜19:00 JST自動実行）
├── config.py                 # 戦略パラメータ一元管理
├── requirements.txt          # Python依存パッケージ
└── data/                     # ランキングCSV等（gitignore対象）
```

---

## 予測モデル（XGBoost）

### 概要
- **目的**: 63日後（約3ヶ月）に日経225を±5%以上上回る/下回る銘柄を予測
- **AUC**: 上昇 0.642 / 下落 0.766（下落予測の精度が高い）
- **学習**: 東証全銘柄×5年、20営業日ごとサンプリング、金曜にModal.comで自動再学習

### 特徴量（64次元）

| カテゴリ | 数 | 内容 |
|---|---|---|
| テクニカル | 10 | ret5/20/60/90, MA乖離, RSI, ボラ, 52週位置 |
| トレンド反転 | 5 | ドローダウン, 52週高値比, 連続下落, モメンタム加速 |
| 出来高 | 3 | 出来高比率（短期/中期/急増） |
| 日経マクロ | 3 | 日経225の5/20/60日リターン |
| 60日系列要約 | 7 | 自己相関, 歪度, 最大/最小リターン等 |
| 日経相対 | 4 | 相対強度, アルファモメンタム |
| ファンダメンタル | 11 | PER, PBR, ROE, 配当, 決算距離, 季節性 |
| マクロ拡張 | 4 | VIX, S&P500, USD/JPY |
| IB特徴量 | 8 | Amihud, FXβ, Piotroski, アクルーアル等 |
| EDINET | 1 | 大量保有報告の保有比率 |
| モメンタム拡張 | 3 | 504日リターン, 60日トレンド傾き/R² |
| クロスセクショナル | 7 | 上記の日次グループ内相対ランク |

### ランキングロジック

**ネットスコア = 上昇確率(%) − 下落確率(%)**

ランキング生成は8フェーズ:
1. スクリーナー通過銘柄に確率予測
2. ハードフィルター（連続下落>3日 / ドローダウン60日<-15% を除外）
3. 品質フィルター（株価≥300円, RSI<75, 出来高比≥1.0, ボラ≤22%等）
4. モデル予測フィルター（17%≤net≤24%, 下落確率<4%でS買い）
5. 株主優待権利落ち21日前の銘柄を降格
6. S買い1日最大3件キャップ
7. 米国セクターETF前日マイナスなら降格
8. 相場リスク管制官（リスクオフ地合いでS買い全件見送り）

**推奨ラベル**: 💎買い / 🔴売り検討 / — (それ以外)

---

## データベース（Supabase / Postgres）

| テーブル | 内容 |
|---|---|
| `gen_rankings` | 毎日のランキングスコア |
| `jpx_stock_list` | 銘柄メタ（業種分類・優待月） |
| `gen_ai_analyses` | AI分析（会社説明・企業インサイト） |
| `gen_risk_regime` | リスクオン/オフ判定 |
| `jquants_fin_summary` | 四半期財務サマリ（J-Quants） |
| `yahoo_price_cache` | 株価履歴キャッシュ |
| `yahoo_market_index` | VIX/S&P500/USDJPY 日次 |
| `edinet_large_holdings` | EDINET大量保有/変更報告書 |
| `ext_tdnet_disclosures` | TDnet適時開示 |
| `jpx_short_selling` | JPX空売り残高報告 |
| `jpx_margin_balance` | JPX信用取引週末残高 |
| `dp_watchlist` | LINE Botウォッチリスト |
| `line_chat_history` | LINE Bot会話履歴 |

---

## 外部API

| API | 用途 |
|---|---|
| Yahoo Finance (非公式) | 株価・日経225・VIX・S&P500・USD/JPY |
| J-Quants v2 (Free) | 財務サマリ（EPS/BPS/ROE/CFO等） |
| kabutan.jp (スクレイピング) | PER/PBR/ROE・優待月・業績テキスト |
| EDINET API v2 | 大量保有報告書・決算XBRL |
| TDnet (やのしんAPI) | 適時開示（業績修正/増配等） |
| JPX (公式Excel/CSV) | 空売り残高・信用残高 |
| Supabase REST API | データ永続化 |
| Claude API (Anthropic) | LINE Bot AI相談・会社説明生成 |
| LINE Messaging API | LINE Bot通知・webhook |

---

## セットアップ

### GitHub Secrets

| Secret | 内容 |
|---|---|
| `SUPABASE_URL` | Supabase プロジェクトURL |
| `SUPABASE_SERVICE_KEY` | Supabase service_role キー |
| `EDINET_API_KEY` | EDINET API v2 キー（任意） |

### Supabase Edge Function Secrets

| Secret | 内容 |
|---|---|
| `LINE_CHANNEL_SECRET` | LINE Messaging API チャネルシークレット |
| `LINE_CHANNEL_ACCESS_TOKEN` | LINE Messaging API アクセストークン |
| `ANTHROPIC_API_KEY` | Claude API キー |

### 依存パッケージ
```
pip install requests pandas numpy scikit-learn joblib xgboost python-dotenv openpyxl yfinance anthropic pytest
```

### Modal.com セットアップ（モデル学習）

モデル学習はGitHub Actionsではなく[Modal.com](https://modal.com)で実行する（無料枠内）。

```bash
pip install modal
modal setup  # ブラウザでログイン

# シークレット登録
modal secret create stock-alert-secrets \
  SUPABASE_URL="..." SUPABASE_SERVICE_KEY="..." \
  JQUANTS_API_KEY="..." EDINET_API_KEY="..." \
  ANTHROPIC_API_KEY="..." GH_TOKEN="..."

# Supabase Storage に「models」バケットを非公開で作成してからデプロイ
modal deploy modal_train.py

# 初回は手動実行（以降は金曜19:00 JSTに自動実行）
modal run modal_train.py::train
```

### コマンド
```bash
# 日次パイプライン
python3 core/screener.py && python3 core/rank_stocks.py

# バックテスト
python3 tools/backtest.py              # 通常期
python3 tools/backtest.py bear         # 暴落耐性チェック（2024年8月）
python3 tools/multi_backtest.py        # 33期間一括

# テスト
python -m pytest tests/ -v
```

