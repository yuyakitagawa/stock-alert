---
name: weekly-review
description: 5ロール（FM/Quant/Consultant/Engineer/Human）の相互評価による週次チームレビューを生成し、logs と Supabase（/review ページ）へ反映する。週次レビュー・チームレビュー・ピアレビュー・来週アクションの生成を求められたときに使う。
---

# 週次チームレビュー

`pdca/weekly_review.py` を実行し、5ロールの相互評価レビューを生成する。通常は毎週月曜 10:00 JST に GitHub Actions（`.github/workflows/weekly_review.yml`）が自動実行する。ローカルで手動生成・再生成したいときに使う。

## ロール

FM（ファンドマネージャー）/ Quant（数量アナリスト）/ Consultant（マーケットコンサル）/ Engineer（エンジニア）/ Human（オーナー・評価を受けるのみ）。

## 実行

```bash
cd /Users/kitagawayuuya/stock-alert
/Users/kitagawayuuya/stock-alert/venv/bin/python3 pdca/weekly_review.py
```

- `ANTHROPIC_API_KEY`（Claude Haiku 呼び出し）が必須。
- `SUPABASE_URL` / `SUPABASE_SERVICE_KEY` があれば `weekly_reviews` テーブルへ保存し /review ページに反映。未設定なら Supabase step はスキップされ logs のみ生成。

## 生成物

- `logs/weekly_review_YYYY-WNN.md` … レビュー本文
- `pdca/feedback.md` … オーナー方針の反映（更新され得る）
- Supabase `weekly_reviews`（設定時）→ frontend `/review` ページ

スクリプトは入力として `pdca/pdca_log.md`・指標トラジェクトリ・モデル設定・アクティビティログを読む。Step1〜7（FM監査→Quant→Consultant→Engineer→Humanへのフィードバック→来週アクション統合→Supabase保存）の順で進む。

## 手順

1. 実行し、Step1〜7 が完走することを確認。
2. 生成された `logs/weekly_review_YYYY-WNN.md` の要点（各ロールの指摘・来週アクション）をユーザーへ要約報告。
3. 方針はオーナーがチャットで述べ、FM が `feedback.md` に整理する運用（メモリ規律）。新方針があればその流れで反映。
