-- 事業内容・投資家プロフィールのネガティブキャッシュ用カラム（2026-08-24 適用済み）
--
-- get_company_description() / get_filer_profile() は「生成できなかった（空文字）」場合を
-- 記録していなかったため、同じ対象へ何度でも課金していた。とくに前者は web_search
-- （$10/1,000検索 ＋ 検索結果が入力トークン、1社あたり約$0.05）を使うため影響が大きく、
-- 2026-08-15〜18 のバックフィル4回で Anthropic の月次利用上限を使い切った。
--
-- 空文字だった場合も checked_at を刻み、RECHECK_DAYS（既定90日）以内は再試行しない。
-- description / profile 本体は値が取れたときだけ書き込む（空文字で既存値を潰さない）。

alter table jpx_stock_list
  add column if not exists description_checked_at timestamptz;

alter table edinet_filer_classification
  add column if not exists profile_checked_at timestamptz;

comment on column jpx_stock_list.description_checked_at is
  'get_company_description が最後に生成を試みた日時。空文字（不明）で返ったケースを再課金しないためのネガティブキャッシュ。descriptionが埋まっている場合は参照されない。';

comment on column edinet_filer_classification.profile_checked_at is
  'get_filer_profile が最後に生成を試みた日時。profileが空のまま再試行し続けるのを防ぐネガティブキャッシュ。';

-- 初期シード: logs/backfill_company_descriptions*.log から「実際に試行して空文字だった」
-- 1,508銘柄を抽出し、現存する444件に最後のバックフィル日(2026-08-18)を刻んだ。
-- ログに証拠が無い銘柄は未試行として残し、通常どおり1回だけ挑戦させる。
--   update jpx_stock_list set description_checked_at = '2026-08-18T00:00:00+00'
--   where description_checked_at is null and coalesce(description,'') = '' and code in (...);
