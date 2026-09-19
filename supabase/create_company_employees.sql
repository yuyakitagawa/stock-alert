-- 銘柄ごとの従業員数（連結）。web/dip_buy_alert.py の「従業員1000人以上」の判定に使う。
--
-- なぜテーブルに持つか: 通知のたびにYahoo Financeへ候補の数だけ問い合わせると、
-- 急落日（候補が数百）に数分かかり、一部はレート制限で空振りして候補から漏れる。
-- tools/fetch_employees.py が古い順に少しずつ取り直し（daily_alert.yml Step 2h）、
-- 通知側はここを読んで、無い銘柄だけYahooで補って書き戻す。
-- employees が NULL の行は「Yahooに値が無かった」の記録（毎日取り直さないため残す）。
-- 適用済み（MCPのapply_migration経由）。
create table if not exists company_employees (
  code         text primary key,
  employees    integer,
  source       text not null default 'yahoo',
  fetched_date date not null default current_date
);

create index if not exists company_employees_fetched_date_idx on company_employees (fetched_date);

comment on table company_employees is
  '従業員数（Yahoo Finance fullTimeEmployees の現在値）。tools/fetch_employees.py が古い順に更新し、web/dip_buy_alert.py が参照する';

-- サービスキー（GitHub Actions）だけが読み書きする。匿名キーからは見せない
alter table company_employees enable row level security;
