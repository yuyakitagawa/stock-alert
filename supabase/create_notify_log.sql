-- 同じ原因の通知を送りすぎないための送信履歴（lib/notify.push_once が参照）。
--
-- なぜ必要か: 2026-08-24のAnthropic API上限超過では、9:00〜21:00の毎時13便すべてが
-- 同じ理由で失敗する。素直に通知すると1日13通届き、通知疲れで無視されれば
-- 「無言停止に気づく」という目的自体が達成できなくなる。
-- dedupe_key ごとに最終送信時刻を持ち、窓（既定20時間）の内側なら送らず
-- sent_count だけ積む＝何便ぶん抑制したかは後から追える。
create table if not exists notify_log (
  dedupe_key   text primary key,
  last_sent_at timestamptz not null default now(),
  sent_count   integer     not null default 1,
  last_text    text
);

comment on table notify_log is
  '通知の重複抑制用。lib/notify.push_once が dedupe_key ごとの最終送信時刻を見て再送を抑える';
