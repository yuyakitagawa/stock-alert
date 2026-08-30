-- edinet_large_holdings に大量保有報告書XBRLの本表項目を追加する。
-- これまで保有割合しか取っていなかったが、XBRLには保有目的・取得資金・保有株数が
-- 構造化されて入っている。ここから「平均取得単価（取得資金÷保有株数）」
-- 「借入比率（レバレッジ）」「報告義務発生日→提出日のラグ」が計算できる。
alter table public.edinet_large_holdings
  add column if not exists purpose_of_holding text,   -- 保有目的（自由記述。共同保有は改行区切り）
  add column if not exists important_proposal text,   -- 重要提案行為等の記載
  add column if not exists shares_held bigint,        -- 保有株券等の数（共同保有の合計）
  add column if not exists shares_outstanding bigint, -- 発行済株式等の総数
  add column if not exists funding_total bigint,      -- 取得資金の総額（円）
  add column if not exists funding_own bigint,        -- うち自己資金（円）
  add column if not exists funding_borrowings bigint, -- うち借入金（円）
  add column if not exists obligation_date date;      -- 報告義務発生日（実際に買った日）

-- 「重要提案行為等を目的とする保有」だけを引く用途が多いので部分インデックスを張る。
create index if not exists edinet_large_holdings_purpose_idx
  on public.edinet_large_holdings (disc_date desc)
  where purpose_of_holding is not null;
