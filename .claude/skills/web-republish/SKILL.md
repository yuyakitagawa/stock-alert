---
name: web-republish
description: フィルター条件やランキングロジックを変更した後に、Web出力を再生成して本番（Supabase / stock-alert-web）へ反映する一連のパイプラインを流す。web再出力・フィルター変更の反映・Supabase反映を求められたときに使う。
---

# Web 再出力

フィルターやスコアロジックを変更したら、当日データを作り直して Web（Supabase 経由 stock-alert-web.vercel.app）へ反映する。CI の日次パイプライン（`.github/workflows/daily_alert.yml`）と同じ順序をローカルで再現する。

## 前提

- venv の python を絶対パスで呼ぶ。以下は `PY` に入れて使う想定。
  ```bash
  PY=/Users/kitagawayuuya/stock-alert/venv/bin/python3
  ```
- `.env` に `SUPABASE_URL` / `SUPABASE_SERVICE_KEY`（Web）、`ANTHROPIC_API_KEY`（説明生成）が必要。未設定の step は各スクリプトが自動スキップする。

## パイプライン（この順序を守る）

```bash
cd /Users/kitagawayuuya/stock-alert
$PY core/screener.py          # 1. スクリーニング
$PY core/rank_stocks.py       # 2. ネットスコア計算
$PY web/export_to_web.py      # 3. Supabase へランキング/メタ/決算をエクスポート
$PY web/generate_descriptions.py --limit 400   # 4. 会社説明を生成
$PY web/sync_descriptions.py  # 5. 説明を同期
$PY web/qa_pages.py           # 6. 全Webページのスモーク検査（QA）
$PY web/send_user_alerts.py   # 7. 売りシグナル保有者へ Web Push
```

メールを送りたくない検証時は step3 を飛ばす。Push を試すだけなら `web/send_user_alerts.py --dry-run`。
ランキングを変えずに表示だけ直すなら step4 以降のみでよい。

## フィルター変更時の必須セット（メモリ規律）

条件を変えたら以下を**常にセットで**行う:

1. `lib/` 等の条件変更
2. `SignalLegend` と `README.md` の更新（フィルター値・コマンド・テスト件数も合わせる、CLAUDE.md §7）
3. このパイプラインで web/メールを再出力
4. 効果は bear バックテストで数値確認（→ `bear-backtest` skill）してからコミット

## 検証

- `web/qa_pages.py` がスモーク検査（鮮度・整合性・説明カバレッジ）を行う。エラーが出たら原因を直してから push。
- 反映後 stock-alert-web.vercel.app のトップ/各ページを確認。
