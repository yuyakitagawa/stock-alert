-- 上場銘柄の会社HP URL と最新決算（tools/fetch_company_info.py が日次で巡回更新）。
--
-- データ源は株探の銘柄ページ（会社サイト）と決算ページ（通期業績・累計決算）。
-- jquants_fin_summary（EDINET XBRL）は有報・半期報しか拾えず、2024年の四半期報告書廃止以降は
-- 1Q/3Q の数字が入らない。決算短信ベースの最新値をここで補う。
-- 金額の単位は百万円（株探の表示単位のまま）。EPS・DPS は円。
-- 適用済み（MCPのapply_migration経由。2026-10-25に employees_* / tanshin_* / 有報の事業情報を追加し、fetched_at を NULL可に）。
create table if not exists company_info (
  code                   text primary key,
  website_url            text,
  website_checked_at     timestamptz,

  -- 最新の累計決算（四半期・上期）。例: latest_period='26.01-06', latest_label='第２四半期累計'
  latest_label           text,
  latest_period          text,
  latest_sales           numeric,
  latest_op              numeric,
  latest_ord             numeric,
  latest_np              numeric,
  latest_eps             numeric,
  latest_progress_pct    numeric,
  latest_announced       date,

  -- 直近の通期実績。例: fy_period='2025.12'
  fy_period              text,
  fy_sales               numeric,
  fy_op                  numeric,
  fy_ord                 numeric,
  fy_np                  numeric,
  fy_eps                 numeric,
  fy_dps                 numeric,
  fy_announced           date,

  -- 今期の会社予想（株探の「予」行）。
  fc_period              text,
  fc_sales               numeric,
  fc_op                  numeric,
  fc_ord                 numeric,
  fc_np                  numeric,
  fc_eps                 numeric,
  fc_dps                 numeric,
  fc_announced           date,

  -- 決算ページを取った時刻。従業員数・短信URLだけの行は NULL（＝決算は未取得として巡回の先頭に来る）
  fetched_at             timestamptz,

  -- 従業員数（連結）。旧 company_employees を統合（2026-10-25）。lib/employees.py が読み書きする。
  -- employees が NULL で employees_fetched_date がある行は「Yahooに値が無かった」の記録。
  employees              integer,
  employees_source       text,          -- 'yahoo' / 'edinet'（有報）/ 'web'（会社HP）
  employees_fetched_date date,

  -- 最新の決算短信（TDnet、ext_tdnet_disclosures から。訂正は除く）。
  -- TDnet の PDF は公開から約1か月で消えるので、古い URL はリンク切れになり得る。
  tanshin_url            text,
  tanshin_title          text,
  tanshin_disclosed_at   timestamptz,
  tanshin_xbrl_url       text,

  -- その短信の XBRL から読んだ会社予想（百万円）。tanshin_fc_fetched_at が NULL＝未読（次回読む）
  tanshin_fc_fy_end      date,
  tanshin_fc_sales       numeric,
  tanshin_fc_op          numeric,
  tanshin_fc_np          numeric,
  tanshin_fc_fetched_at  timestamptz,

  -- 有価証券報告書（EDINET）の事業情報。旧 company_profile を統合（2026-10-25）。lib/company_profile.py が読み書き。
  -- 本文は DB を膨らませないよう先頭だけ（事業800字・課題400字）。profile_doc_id が NULL＝有報未取得
  business               text,          -- 事業の内容
  issues                 text,          -- 対処すべき課題
  profile_employees      integer,       -- 提出会社（単体）の従業員数。employees（連結）とは別
  avg_age                numeric,
  avg_salary             bigint,        -- 平均年間給与（円）
  avg_service            numeric,
  profile_fy_end         date,
  profile_doc_id         text,
  profile_disc_date      date,
  profile_fetched_at     timestamptz
);

create index if not exists company_info_fetched_at_idx on company_info (fetched_at);
create index if not exists company_info_employees_fetched_date_idx on company_info (employees_fetched_date);

alter table company_info enable row level security;

comment on table company_info is
  '上場銘柄の会社HP URL・最新決算（株探）・決算短信URLと短信予想（TDnet）・従業員数・有報の事業情報。金額は百万円';
