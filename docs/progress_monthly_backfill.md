# /monthly アーカイブ過去記事バックフィル

## 背景
- kujira-watch.com/monthly に 2025-07〜2026-06 の月がほぼ無かった（microCMS記事の日次自動投稿が2026-07開始のため。それ以前は手動の月1〜5件）
- Supabase `edinet_large_holdings` には 2025-06-18〜 全月分の開示データあり（2026-08-10バックフィル済み）
- 対応: 各月の保有比率変化幅上位50件を記事化（`tools/backfill_monthly_articles.py`。
  `build_and_publish()` 直呼びでX投稿なし、過去日付のため株価チャートなし）

## 手順
- [x] データ調査（microCMS: 2026-07以前は月1〜5件のみ / Supabase: 全月あり）
- [x] バックフィルランナー作成・dry-run確認
- [x] 2026-02 で2件テスト投稿 → 本番 /monthly/2026-02 反映確認
- [x] 全対象月を実行 → **150件投稿した時点でAnthropic API月間上限に到達し停止**（2026-08-16）
  - 全13ヶ月に各12〜19件は入り、本番 /monthly は 2025-06〜2026-08 まで欠けなく表示されるようになった
  - 内訳: 投稿150 / 金額概算不可スキップ228 / 記事生成失敗267（大半がAPI上限到達後）
- [ ] 2026-09-01 0:00 UTC のAPI上限リセット後（または上限引き上げ後）に再実行して各月50件まで埋める
  - `venv/bin/python3 tools/backfill_monthly_articles.py`（投稿済みはalready_publishedで自動スキップ、そのまま再実行でよい）
- [ ] 完了後: `tools/backfill_monthly_articles.py` を削除し、本ファイルに完了記録

## 注意（2026-08-16時点）
- **ANTHROPIC_API_KEY が月間使用上限に達している。2026-09-01までClaude呼び出しを伴う処理
  （日次のブログ記事自動生成 edinet_blog.yml、nlp_sentiment 等）は失敗する。**
  早く戻したい場合は console.anthropic.com で月間上限を引き上げる。
