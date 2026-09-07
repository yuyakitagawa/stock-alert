-- Supabase Free枠（DB 500MB / egress 5GB per month）に収めるための一式。
-- 2026-09-07、egress超過で全RESTが402、DBも544MBで500MB超という状態から入れた。
-- 適用済み（MCPのapply_migration経由）。再構築するときの記録として置く。

-- ── 1. クローラーログのUA正規化 ────────────────────────────────────────
-- user_agent の生テキストだけで62MBあった。実際のUAは432,101行に対して598種類。
create table if not exists blog_crawler_ua (
  id serial primary key,
  user_agent text not null unique
);

alter table blog_crawler_log add column if not exists ua_id integer references blog_crawler_ua(id);
create index if not exists blog_crawler_log_ua_id_idx on blog_crawler_log (ua_id);

-- 書き込み側(proxy.ts)は今まで通り user_agent を生で送ってよい。Edge middlewareに
-- 往復を増やしたくないのでRPCではなくトリガーで受ける。user_agent 列は受け口であり常にNULL。
create or replace function blog_crawler_log_normalize_ua()
returns trigger
language plpgsql
as $$
declare
  v_ua_id integer;
begin
  if new.user_agent is not null and new.user_agent <> '' then
    insert into blog_crawler_ua (user_agent) values (new.user_agent)
    on conflict (user_agent) do update set user_agent = excluded.user_agent
    returning id into v_ua_id;
    new.ua_id := v_ua_id;
    new.user_agent := null;
  end if;
  return new;
end;
$$;

drop trigger if exists blog_crawler_log_normalize_ua_trg on blog_crawler_log;
create trigger blog_crawler_log_normalize_ua_trg
  before insert on blog_crawler_log
  for each row execute function blog_crawler_log_normalize_ua();

comment on column blog_crawler_log.user_agent is
  'INSERT時の受け口。トリガーが blog_crawler_ua へ寄せて ua_id に変換するため保存後は常にNULL。読むときは ua_id を使う。';

-- ── 2. 保持期間（tools/purge_supabase.py から日次で呼ぶ）─────────────────
-- AIクローラーの巡回ログはGEO実験の観測データなので長く残す。
create or replace function purge_blog_crawler_log()
returns table(bucket text, deleted bigint)
language plpgsql
as $$
declare
  ai_bots text[] := array['GPTBot','ChatGPT-User','OAI-SearchBot','ClaudeBot','Claude-Web',
                          'anthropic-ai','PerplexityBot','Amazonbot','Applebot',
                          'meta-externalagent','meta-externalfetcher','Bytespider'];
  n bigint;
begin
  delete from blog_crawler_log
   where bot_name = 'Browser' and occurred_at < now() - interval '30 days';
  get diagnostics n = row_count;
  bucket := 'Browser(30日)'; deleted := n; return next;

  delete from blog_crawler_log
   where bot_name <> 'Browser' and not (bot_name = any(ai_bots))
     and occurred_at < now() - interval '60 days';
  get diagnostics n = row_count;
  bucket := 'その他bot(60日)'; deleted := n; return next;

  delete from blog_crawler_log
   where bot_name = any(ai_bots) and occurred_at < now() - interval '400 days';
  get diagnostics n = row_count;
  bucket := 'AIクローラー(400日)'; deleted := n; return next;

  delete from blog_crawler_ua u
   where not exists (select 1 from blog_crawler_log l where l.ua_id = u.id);
  get diagnostics n = row_count;
  bucket := '未参照UA'; deleted := n; return next;
end;
$$;

-- gen_rankings.date は date型、yahoo_price_cache.date は text型。型が違うので注意。
-- 150日 = サイト companyInfo.ts が使う90営業日を確実に賄う長さ。
create or replace function purge_gen_rankings(keep_days integer default 150)
returns bigint
language plpgsql
as $$
declare n bigint;
begin
  delete from gen_rankings where date < current_date - keep_days;
  get diagnostics n = row_count;
  return n;
end;
$$;

-- 620日 = rf_train_v3.HISTORY_DAYS(600) に余裕を持たせた長さ。
create or replace function purge_yahoo_price_cache(keep_days integer default 620)
returns bigint
language plpgsql
as $$
declare n bigint;
begin
  delete from yahoo_price_cache
   where date < to_char(current_date - keep_days, 'YYYY-MM-DD');
  get diagnostics n = row_count;
  return n;
end;
$$;

-- ── 3. 集約をDB側でやるRPC ─────────────────────────────────────────────
-- いずれも array_agg(distinct ...) は元テーブルぶんのソートになって statement timeout(8s)
-- に当たるため、先に group by で畳んでから配列にする。
-- setof を返すとPostgRESTの max-rows(1000) で切られるので必ず1行にまとめて返すこと。
create or replace function price_cache_codes()
returns text[]
language sql
stable
as $$
  select array_agg(c order by c) from (select code as c from yahoo_price_cache group by code) t;
$$;

create or replace function price_cache_coverage()
returns jsonb
language sql
stable
as $$
  select coalesce(jsonb_object_agg(c, jsonb_build_array(mn, mx)), '{}'::jsonb)
  from (select code as c, min(date) as mn, max(date) as mx
        from yahoo_price_cache group by code) t;
$$;

create or replace function gen_ranking_dates()
returns text[]
language sql
stable
as $$
  select array_agg(to_char(d, 'YYYY-MM-DD') order by d desc)
  from (select date as d from gen_rankings group by date) t;
$$;

-- ── 4. 一度きりの回収 ─────────────────────────────────────────────────
-- 削除しただけでは pg_database_size は縮まない。実サイズの回収には VACUUM FULL が要る。
--   vacuum full blog_crawler_log;
--   vacuum full gen_rankings;
--   vacuum full yahoo_price_cache;
