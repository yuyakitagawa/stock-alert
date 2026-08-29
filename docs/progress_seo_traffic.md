# 進捗: SEO集客（検索流入を増やす）

対象: kujira-watch.com（`kujira-watch/`）。関連: `docs/seo_geo_playbook.md`（横展開用の要点集）、
`progress_seo_aio_30day_plan.md`（施策側・完了済み）、`progress_seo_indexing.md`（インデックス問題）、
`progress_growth_top10.md` #1/#4（効果判定がGSC待ちで止まっている）。

## 0. 現状（2026-08-29 実測・GA4 2026-08-01〜08-28）

| 指標 | 実測 | 出典 |
|---|---|---|
| 自然検索セッション | 171件/28日（Google 127 / Yahoo 43 / Bing 1）＝約6件/日 | `tools/ga4_clicks.py --days 28` |
| AI検索経由の訪問 | 6件/28日（ChatGPT 5 / Gemini 1） | 同上・`tools/geo_report.py` |
| 検索の着地ページ | TOP 25件が最大。以下は記事・銘柄ページが1〜4件ずつに分散 | GA4 landingPage × Organic Search |
| ページ数 | sitemap 6,000超（記事・銘柄2,978・投資家2,938ほか） | `progress_seo_aio_30day_plan.md` |
| GSCベースライン | クリック45 / 表示756 / CTR 6% / 平均9.1位（2026-08-15・スクショ共有） | 同上 |

**診断**: ページ数に対して流入が桁で小さく、かつ1ページあたり0〜4件の完全な長尾。
「どのクエリで表示されているのに押されていないか」が分からないまま施策を打ち続けている状態で、
過去の計画（30日プラン・Discover計画）も効果判定が全部「GSC反映待ち」で止まっている。
**先に測れるようにするのがボトルネック**。

## 1. 計測（P0・ここが最優先）

- [x] `tools/gsc_report.py` を追加（Search Analytics API）。全体／ページ種別／上位クエリ／
      **CTR改善候補（10位以内なのにCTRがサイト平均未満＝titleの書き換えで即効く層）**／
      **あと一歩（11〜20位）**／上位ページ・表示のあったURL数 を前期間比つきで出す。
      テスト `tests/test_gsc_report.py` 15件。
- [ ] **ユーザー作業1**: GCPで Search Console API を有効化
      https://console.cloud.google.com/apis/library/searchconsole.googleapis.com?project=stock-alert-493722
- [ ] **ユーザー作業2**: GSC > 設定 > ユーザーと権限 > ユーザーを追加 で
      `stock-alert-bot@stock-alert-493722.iam.gserviceaccount.com` を「制限付き」で追加
- [ ] 上記2つの後に `venv/bin/python3 tools/gsc_report.py --sites` で疎通確認 →
      `venv/bin/python3 tools/gsc_report.py` でベースラインを本ファイルに記録

## 2. 計測結果を見てから決める（着手前に必ず §1 を通す）

- [ ] **CTRの取りこぼし回収**: 1ページ目に居るのに押されていないクエリのtitle/descriptionを書き換える。
      順位が既にあるので反映が最も速い。対象はGSCの「CTR改善候補」から選ぶ（推測で選ばない）。
- [ ] **あと一歩（11〜20位）の押し上げ**: 該当ページへの内部リンクと本文の加筆。
- [ ] **表示のあったURL数 vs sitemap 6,000超**の差を確認。差が大きければ問題はコンテンツではなく
      インデックス（`progress_seo_indexing.md` の続き）。
- [ ] 上記で埋まらない需要があれば、そのクエリに直答するハブページを新設する。

## 3. やらないと決めていること（再提案しない）

- 既存の薄い記事95件の一括リライト（2026-08-14にユーザー判断で放置と決定）
- 英語版 `/en` の復活（2026-08-29に全除却済み）
- 読者向けLINE公式アカウント・ウォッチ銘柄機能（オーナー判断で不採用）
- APIバッチによる本文の一括再生成（2026-08-29に経路ごと廃止。事実カード→Claude Code執筆→PATCH）
