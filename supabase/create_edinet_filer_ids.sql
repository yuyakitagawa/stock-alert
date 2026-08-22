-- 投資家ページのURLを /investors/<日本語の提出者名> から /investors/<連番ID> に変える
-- ためのID表。EDINETの大量保有データには提出者コードが無いため、提出者名ごとに
-- 連番を振る（初回採番は最初の開示日が古い順）。新しい提出者は
-- edinet_large_holdings へのINSERTトリガーで自動採番される。
create table if not exists edinet_filer_ids (
  id serial primary key,
  filer_name text not null unique,
  created_at timestamptz not null default now()
);

insert into edinet_filer_ids (filer_name)
select filer_name
from edinet_large_holdings
where filer_name is not null and filer_name <> ''
group by filer_name
order by min(disc_date), filer_name
on conflict (filer_name) do nothing;

create or replace function assign_edinet_filer_id() returns trigger
language plpgsql as $$
begin
  if new.filer_name is not null and new.filer_name <> '' then
    insert into edinet_filer_ids (filer_name) values (new.filer_name)
    on conflict (filer_name) do nothing;
  end if;
  return new;
end;
$$;

drop trigger if exists trg_assign_edinet_filer_id on edinet_large_holdings;
create trigger trg_assign_edinet_filer_id
  before insert or update of filer_name on edinet_large_holdings
  for each row execute function assign_edinet_filer_id();

-- 一覧・サイトマップ用ビューにIDを付ける（既存列の順序は維持し末尾に追加）。
create or replace view edinet_filer_summary as
select h.filer_name,
       coalesce(c.category, 'その他') as category,
       count(*) as holding_count,
       max(h.disc_date) as latest_disc_date,
       i.id as filer_id
from edinet_large_holdings h
left join edinet_filer_classification c using (filer_name)
left join edinet_filer_ids i using (filer_name)
where h.filer_name is not null and h.filer_name <> ''
group by h.filer_name, c.category, i.id;
