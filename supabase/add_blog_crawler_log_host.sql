-- blog_crawler_log に host 列を追加する（2026-09-04）。
-- 英語版をサブドメイン（en.kujira-watch.com）で配信し始めたため、path だけでは日英を区別できない
-- （英語版の公開URLは /en を付けない: kujira-watch/src/proxy.ts）。既存行は NULL のままにし、
-- 集計側（tools/en_crawl_report.py / tools/geo_report.py）は NULL を日本語版として扱う。
alter table public.blog_crawler_log add column if not exists host text;
comment on column public.blog_crawler_log.host is
  'リクエストの Host ヘッダー（kujira-watch.com / en.kujira-watch.com）。2026-09-04以前の行は NULL＝日本語版。';
