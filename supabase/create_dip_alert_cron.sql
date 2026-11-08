-- 押し目買い判定（.github/workflows/dip_alert.yml）を大引け直後に起動する。
--
-- なぜ Supabase から起動するか: GitHub Actions の schedule はこのリポジトリで毎回約7〜8時間遅れる
-- （2026-11-02〜11-06 実測: ops.yml 06:00 UTC→13:56〜15:37、daily_alert.yml 07:00 UTC→14:50〜16:53）。
-- 判定が日本時間0時を越え、Yahoo から当日の終値が取れずに止まった（11/4・11/5・11/6）。
-- pg_cron は時刻どおりに動くので、ここから GitHub の workflow_dispatch API を叩く。
--
-- 必要な秘密情報（このファイルには書かない）: Vault の 'github_dispatch_token'
--   GitHub の fine-grained personal access token。対象リポジトリ yuyakitagawa/stock-alert、
--   権限 Actions: Read and write のみ。登録は Supabase の SQL Editor で:
--     select vault.create_secret('<token>', 'github_dispatch_token');
--   トークンが無い間は dispatch_dip_alert() が notice を出して何もしない。

create extension if not exists pg_cron;
create extension if not exists pg_net;

create or replace function public.dispatch_dip_alert()
returns bigint
language plpgsql
security definer
set search_path = public, extensions
as $$
declare
  tok text;
  req bigint;
begin
  select decrypted_secret into tok
    from vault.decrypted_secrets where name = 'github_dispatch_token' limit 1;
  if tok is null then
    raise notice 'github_dispatch_token が Vault に無いため dip_alert.yml を起動しない';
    return null;
  end if;
  select net.http_post(
    url := 'https://api.github.com/repos/yuyakitagawa/stock-alert/actions/workflows/dip_alert.yml/dispatches',
    headers := jsonb_build_object(
      'Authorization', 'Bearer ' || tok,
      'Accept', 'application/vnd.github+json',
      'X-GitHub-Api-Version', '2022-11-28',
      'User-Agent', 'stock-alert-pg-cron'),
    body := jsonb_build_object('ref', 'main')
  ) into req;
  return req;
end;
$$;

revoke all on function public.dispatch_dip_alert() from public, anon, authenticated;

-- 平日 06:50 UTC = 15:50 JST（大引け15:30の20分後）。祝日も起動するが、dip_buy_alert.py が休場日と判定して何も送らない。
select cron.schedule('dip_alert_dispatch', '50 6 * * 1-5', $$select public.dispatch_dip_alert()$$);
