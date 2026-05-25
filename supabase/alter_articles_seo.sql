-- articles テーブルに SEO カラムを追加
-- Supabase Dashboard の SQL Editor で実行してください

ALTER TABLE articles ADD COLUMN IF NOT EXISTS target_keyword   TEXT;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS meta_description TEXT;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS seo_score        INTEGER;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS rewrite_count    INTEGER DEFAULT 0;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS last_rewritten_at TIMESTAMPTZ;
