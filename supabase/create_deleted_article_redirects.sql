-- 削除した記事URLの引き継ぎ先（lib/article_redirects.py が書き、
-- kujira-watch/src/lib/articleRedirects.ts が読む）。
--
-- なぜ必要か: 2026-08-29のGSC実測で、検索結果に出ているURL924件のうち194件が404を返し、
-- そこに28日で25クリック（全クリックの18%）が着地していた。うち124件が削除済みの記事URL。
-- 低価値・重複・誤報の記事を消す運用は続けるが、順位の付いたURLは捨てずに
-- 銘柄ページ（または重複で残した方の記事）へ恒久リダイレクトして引き継ぐ。
create table if not exists deleted_article_redirects (
  article_id  text primary key,
  target_path text        not null,
  reason      text,
  created_at  timestamptz not null default now()
);

comment on table deleted_article_redirects is
  '削除済み記事URL → 引き継ぎ先パス。記事詳細ページが404の代わりに308で返す';

alter table deleted_article_redirects enable row level security;

drop policy if exists service_role_all on deleted_article_redirects;
create policy service_role_all on deleted_article_redirects
  for all to service_role using (true) with check (true);
