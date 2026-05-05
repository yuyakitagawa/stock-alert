# stock-alert

東証上場株式の機械学習スクリーニング・アラートシステム。

## ファイル構成

| ファイル | 役割 |
|---|---|
| utils.py | 共通関数（get_prices, extract_features, compute_seq_features 等）、28次元特徴量定義 |
| rf_train_v3.py | XGBoostモデル学習（上昇・下落の2モデル）。毎週土曜に自動実行 |
| rf_model.pkl / rf_drop_model.pkl | 学習済みモデル（上昇・下落） |
| screener.py | TSE全銘柄をスクリーニングしてCSV出力 |
| rank_stocks.py | スクリーナー通過銘柄にモデルスコアをつけてランキング |
| alert_email.py | スコア上位銘柄をGmailでメール通知 |
| backtest.py | バックテスト（先読みバイアスなしモード実装済み） |
| .github/workflows/daily_alert.yml | GitHub Actionsで毎朝8時（JST）に自動実行 |
| sheets_helper.py | Googleスプレッドシートへの書き込みユーティリティ |
| generate_post.py | SNS投稿用テキスト生成 |

## モデル仕様（v6 現在）

- 特徴量: 28次元（テクニカル10 + トレンド反転5 + 出来高3 + 日経マクロ3 + 60日系列要約7）
- 予測期間: 63営業日（約3ヶ月）で±15%以上変動するかどうか
- AUC: 上昇 0.657 / 下落 0.790
- ランキング: ネットスコア = 上昇確率(%) − 下落確率(%)

## フィルター（除外条件）

- `down_streak > 0.15`：連続下落日数が3日超
- `drawdown60 < -0.15`：直近60日高値から15%超の下落

## よく使うコマンド

```bash
# スクリーニング
python3 screener.py

# モデル学習（40〜70分）
python3 rf_train_v3.py

# ランキング生成
python3 rank_stocks.py

# メール送信
python3 alert_email.py

# バックテスト（通常・約15分）
caffeinate -i python3 backtest.py

# バックテスト（2024年8月下落相場）
python3 backtest.py bear
```

## 自動実行フロー

毎朝8時（JST）にGitHub Actionsで実行:

```
screener.py → rank_stocks.py → alert_email.py  （毎日）
rf_train_v3.py                                  （土曜のみ再学習）
```

## 必要なSecrets（GitHub Settings → Secrets）

| Secret名 | 内容 |
|---|---|
| GMAIL_ADDRESS | 送信元Gmailアドレス |
| GMAIL_APP_PASSWORD | Gmailアプリパスワード |
| SPREADSHEET_ID | GoogleスプレッドシートID |
| GCP_KEY_JSON | Google Cloud サービスアカウントJSON |

## 設計上の注意点

- AUC 0.657はランダム（0.50）よりわずかに良い程度。参考指標として使い、最終判断は自分で行う。
- 下落フィルターはルールベース。モデルが「買い」と言っても直近で崩れている銘柄を弾く。
- 学習/テスト分割は2025-01-01（ウォークフォワード）。将来データの先読みバイアスを防ぐ。
