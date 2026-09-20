-- 有価証券報告書（EDINET docTypeCode=120）から抜いた事業情報。
-- 押し目買いの通知（web/dip_buy_alert.py）に「どんな会社か」を1行添えるために使う。
--
-- なぜ有報か: 企業HPは社ごとに構造が違って取りこぼすが、有報は全社が同じXBRLタグで
-- 「事業の内容」「対処すべき課題」「従業員の状況」を出しており、EDINET APIで無料・確実に取れる。
-- 本文をそのまま持つとDBが膨らむ（有報1本のXBRLは3MB前後、Free枠は500MB）ので、
-- lib/company_profile.py が先頭800字（課題は400字）に切って保存する。
-- 適用済み（MCPのapply_migration経由）。
create table if not exists company_profile (
  code          text primary key,
  business      text,      -- 事業の内容（先頭800字）
  issues        text,      -- 対処すべき課題（先頭400字）
  employees     integer,   -- 提出会社の従業員数
  avg_age       numeric,   -- 平均年齢
  avg_salary    bigint,    -- 平均年間給与（円）
  avg_service   numeric,   -- 平均勤続年数
  fy_end        date,
  doc_id        text,
  disc_date     date,
  fetched_at    timestamptz not null default now()
);

create index if not exists company_profile_disc_date_idx on company_profile (disc_date);

-- サービスキー（GitHub Actions）だけが読み書きする
alter table company_profile enable row level security;

comment on table company_profile is
  '有価証券報告書（EDINET docTypeCode=120）から抜いた事業情報。tools/fetch_company_profiles.py が更新し、web/dip_buy_alert.py の通知に事業の要点を添える';
