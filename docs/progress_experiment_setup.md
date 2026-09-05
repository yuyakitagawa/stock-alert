# kujira-watch を GEO の A/B 実験台に切り替える（2026-08-30〜）

目的は**コスト削減**と**実験の母集団固定**。記事を大量生産する運用をやめ、
「毎日決まった本数だけ出す」状態にして、GEO（生成エンジン最適化＝AI検索に引用されるための
最適化）のA/Bの観測条件を揃える。A/Bの実装そのものは本作業に含めない（別タスク）。

## 前提（着手時の実測）

- `api_usage` は2行しか無い（2026-08-29導入。以降は平日実行が無い週末のみ経過）。
  うち1行 `job='local' / task='x' / cache_write 1,000,000 / $1.35` は
  **テストが本番へ書いた合成行**。これで当日合計が $1.359 となり
  `DEFAULT_DAILY_BUDGET_USD=1.2` を超えて `check_daily_cap()` が誤発火する状態だった。
  残る1行 `blog_body $0.0092` が記事本文1本ぶんの実測値。
- `edinet_blog.yml` は平日13便（`0 0-12 * * 1-5`）＋backfill 1便。
- X: 2026-08-19〜08-29で34投稿・総インプレッション43・いいね0・リポスト0・
  URLクリック1・フォロワー10日連続0。
- 動画: 公開9本・総再生6,909回・登録者5人・コメント0（`docs/progress_video_post.md`）。

## ステップ

- [x] 1. テストが本番Supabaseへ書き込む事故を止める（最優先）
  - [x] 1-1. 書き込みガードを `lib/supabase_client.py` に入れる
        （テスト実行中＋書き込み先が本番URLのときだけ upsert/insert_ignore/update/delete を
        握りつぶす。読み取りは止めない。環境変数ガードは「テストが毎回付け忘れる」ため不採用、
        モックは atexit の経路を塞げないため不採用）
  - [x] 1-2. 本番の合成行を削除（`api_usage` の `job='local' AND task='x' AND usage_date='2026-08-29'`）
  - [x] 1-3. 同種の事故を検知するテストを追加
        （`tests/test_supabase_client.py` に2件、`tests/test_api_usage.py` に atexit 経路の1件）
- [x] 2. 記事生成を日2本に絞る（`DAILY_MAX_ARTICLES = 2`）
      `publish_blog_articles.daily_quota()` が当日(UTC)の `article_published_at` を数えて
      残り枠を出し、既存の `max_articles` + `SKIP_MAX_ARTICLES` にそのまま渡す（新概念なし）。
      大量保有・自社株買いでそれぞれ日2本。`--max-articles` と `--backfill` は従来どおり優先。
- [x] 3. `edinet_blog.yml` の便数を 13 → 3（`0 0,6,9 * * 1-5`。backfill便 `30 12 * * 1-5` は据え置き）
      EDINET収集（`scan_large_holdings.py`）のステップは残した。
- [x] 4. X投稿を週1本へ削減（`publish_blog_articles.main()` の
      `post_top_articles`/`post_daily_summary` 呼び出し削除＝記事1本ごとの投稿と日次サマリーを停止。
      `x_post.yml` は土曜18:00 JSTの「1週間のまとめ」（`x_weekly_trending.py`＝急増ランキング）
      1枠だけ残し、他5枠は削除して手動実行のみに）
      週1本を残すのはE-E-A-T（Xが `sameAs` / `contactPoint` の唯一の連絡窓口）。
      `/about` の文言も週次まとめの案内へ書き換えた（実態と食い違う記述を残さない）。
      全面再開の条件を `docs/x_operation_rules.md` の0章に追記。アカウント・既存投稿は削除しない。
- [x] 5. `video_post.yml` の schedule を停止（手動実行のみ。コードは残す）
- [x] 6. 日次予算 `DEFAULT_DAILY_BUDGET_USD` を 1.2 → 0.15
- [x] 7. `tools/output_heartbeat.py` からX・動画の監視を外す
      （止めた成果物を数え続けると毎日「X投稿0件」で誤報になる）
- [x] 8. README.md 更新・既存テスト全通過・コミット

## やらないこと（決定済み）

- 既存976記事のリライト（`tools/export_article_fact_cards.py` の運用）は中止。再開しない。
- A/B実験の実装は別タスク。
- 動画・X のコードは消さない（ワークフローと呼び出しだけ止める）。

## 記録

- 2026-08-30: ステップ1〜8を実施。テストは全件（796件）通過。
- 2026-08-31: Xを完全停止から**週1本（土曜の週次まとめ）**へ戻した。E-E-A-Tの連絡窓口が
  Xしか無く、投稿ゼロだと `/about` の記述が実態と食い違うため。コストは月約$0.06。
- 2026-09-03: **ステップ2・3・6を取り消した**（オーナー指示「やっぱり開示案件は記事にしてほしい」）。
  `DAILY_MAX_ARTICLES`・`daily_quota()`・`articles_published_today()` とそのテストを削除し、
  基準を満たす開示は全件記事にする運用へ戻した。便は平日9〜19時JSTの毎時11便
  （`0 0-10 * * 1-5`）、日次予算は $1.2 へ戻した。ステップ1・4・5・7（テストの本番書き込み
  ガード、X週1本、動画停止、ハートビート）はそのまま。**A/Bの母集団固定はこの時点で無くなった**。
  あわせてTOPの「N件の開示」表示を外した（記事になった開示しか数えられず実態より少なく見えるため）。
