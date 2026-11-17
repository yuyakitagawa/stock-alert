-- 日次パイプライン（.github/workflows/daily_alert.yml）を平日 18:30 JST に起動し、LINE を 20:00 JST までに届ける。
--
-- なぜ Supabase から起動するか: GitHub Actions の schedule はこのリポジトリで 7〜10 時間遅れ、
-- 07:13 UTC の便が 0 時 JST を越えて始まり（11/3〜11/11 実測 23:50〜02:28 JST）、失敗した日は
-- ops.yml の watchdog（14:30 UTC → 実測 06:31 JST）の再実行で LINE が翌朝 7 時台に届いていた（11/13 は 07:22 JST 完了）。
-- 所要時間の実測: 51〜83 分（11/10・11/11・11/13 の成功便）。18:30 開始なら 19:21〜19:53 JST に終わる。
--
-- 必要な秘密情報: Vault の 'github_dispatch_token'（create_dip_alert_cron.sql と共用。Actions: Read and write）。
--   トークンが無い間は dispatch_daily_alert() が notice を出して何もしない。

create extension if not exists pg_cron;
create extension if not exists pg_net;

create or replace function public.dispatch_daily_alert()
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
    raise notice 'github_dispatch_token が Vault に無いため daily_alert.yml を起動しない';
    return null;
  end if;
  select net.http_post(
    url := 'https://api.github.com/repos/yuyakitagawa/stock-alert/actions/workflows/daily_alert.yml/dispatches',
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

revoke all on function public.dispatch_daily_alert() from public, anon, authenticated;

-- 平日 09:30 UTC = 18:30 JST。
select cron.schedule('daily_alert_dispatch', '30 9 * * 1-5', $$select public.dispatch_daily_alert()$$);
