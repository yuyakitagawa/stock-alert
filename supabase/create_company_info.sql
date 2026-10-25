-- 上場銘柄の会社HP URL と最新決算（tools/fetch_company_info.py が日次で巡回更新）。
--
-- データ源は株探の銘柄ページ（会社サイト）と決算ページ（通期業績・累計決算）。
-- jquants_fin_summary（EDINET XBRL）は有報・半期報しか拾えず、2024年の四半期報告書廃止以降は
-- 1Q/3Q の数字が入らない。決算短信ベースの最新値をここで補う。
-- 金額の単位は百万円（株探の表示単位のまま）。EPS・DPS は円。
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

  fetched_at             timestamptz not null default now()
);

create index if not exists company_info_fetched_at_idx on company_info (fetched_at);

alter table company_info enable row level security;

comment on table company_info is
  '上場銘柄の会社HP URLと最新決算（株探）。tools/fetch_company_info.py が古い順に日次巡回。金額は百万円';
