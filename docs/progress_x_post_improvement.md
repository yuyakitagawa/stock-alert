# X投稿改善10施策 進捗（2026-08-19）

出典: `docs/x_post_improvement_1000.md` の収束10施策。オーナー指示「全部お願いします」。

- [x] 1. 投稿ログ＋メトリクス収集基盤（`x_posts` / `x_post_metrics` テーブル作成、`post_tweet()`がtweet_idを返す、`web/x_metrics.py` + x_metrics.yml 毎日10時JST）
- [x] 2. 1行目フック化（`build_tweet_text()`刷新。記事タイトル流用をやめ「誰が・銘柄(コード)・どうした」＋「約N億円・保有比率 X%→Y%」）
- [x] 3. リンクを自己リプライへ（`publish()` / `post_tweet(reply_to=)`。`X_LINK_IN_REPLY=0`で本文に戻すA/B、`x_posts.variant`で識別）
- [x] 4. 数字カード画像＋alt（`web/x_card_image.py`。日次サマリー・週次2本・答え合わせにも一覧カードを追加）
- [x] 5. 投稿時刻の再設計（1回1件・JST8〜22時のみ・日次サマリー21時JST、edinet_blog.yml cron 0-12 UTC）
- [x] 6. 答え合わせ投稿（`web/x_followup.py` + x_followup.yml 水21時JST。実データで5/20の54銘柄=平均+5.9%を確認）
- [x] 7. 訂正報告書の独立枠（件数制限撤廃・既報へ自己リプライ）
- [x] 8. タグと銘柄表記の見直し（`#日本株 #大量保有報告書`のみ、銘柄は本文に`社名(コード)`。未使用の`hashtag()`を削除）
- [x] 9. 解釈行（`web/x_insight.py`。filer_win_rateは廃止済みのため開示件数の事実のみ）
- [x] 10. 運用ルール文書化（`docs/x_operation_rules.md`）
- [x] テスト更新（test_x_client.py 49件・週次2本の追随）・README更新・dev_log追記・commit/push

## 次にやること（効果検証）
1. 2週間ためてから `python3 web/x_metrics.py --report` で種別×variantの平均を確認する
2. リンク位置A/B（`link_in_reply` vs `link_in_body`）を各10投稿以上集めて比較
3. 悪化した施策はコードごと戻す。結果は dev_log.md に追記する
