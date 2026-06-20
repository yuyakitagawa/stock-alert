-- articles テーブル（S買いシグナル銘柄のSEO記事）
-- Supabase Dashboard の SQL Editor で実行してください

CREATE TABLE IF NOT EXISTS articles (
  id           BIGSERIAL PRIMARY KEY,
  slug         TEXT UNIQUE NOT NULL,          -- e.g. "7203-2026-05-25"
  code         TEXT NOT NULL,                 -- 証券コード
  name         TEXT NOT NULL,                 -- 銘柄名
  title        TEXT NOT NULL,                 -- 記事タイトル
  body         TEXT NOT NULL,                 -- Markdown本文
  signal_date  DATE NOT NULL,                 -- シグナル発令日
  net_score    FLOAT,                         -- ネットスコア(%)
  published_at TIMESTAMPTZ DEFAULT NOW(),
  created_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_articles_signal_date ON articles(signal_date DESC);
CREATE INDEX IF NOT EXISTS idx_articles_code        ON articles(code);

-- 匿名ユーザーの読み取りを許可（RLS）
ALTER TABLE articles ENABLE ROW LEVEL SECURITY;

CREATE POLICY "articles_public_read" ON articles
  FOR SELECT USING (true);
