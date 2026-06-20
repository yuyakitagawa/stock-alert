-- 週次ピアレビューテーブル
-- Supabase Dashboard の SQL Editor で1回だけ実行してください

CREATE TABLE IF NOT EXISTS weekly_reviews (
  week        TEXT PRIMARY KEY,          -- 例: "2026-W22"
  created_at  TIMESTAMPTZ DEFAULT NOW(),

  -- 今週の指標サマリー
  avg_start   FLOAT,
  avg_end     FLOAT,
  win_start   FLOAT,
  win_end     FLOAT,
  big_start   FLOAT,
  big_end     FLOAT,
  adopted     INT,
  rejected    INT,
  skipped     INT,
  signals     INT,

  -- 各ロールの評価テキスト
  engineer_eval     TEXT,
  quant_eval        TEXT,
  securities_eval   TEXT,
  fm_eval           TEXT,
  human_feedback    TEXT,
  next_actions      TEXT
);

-- 匿名ユーザーの読み取りを許可（RLS）
ALTER TABLE weekly_reviews ENABLE ROW LEVEL SECURITY;

CREATE POLICY "weekly_reviews_public_read" ON weekly_reviews
  FOR SELECT USING (true);
