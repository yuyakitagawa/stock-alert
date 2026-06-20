-- アクティビティログテーブル
-- 各担当者（FM/Quant/Securities/Engineer/System）が実施中・実施済みのアクションを記録する。
-- Supabase Dashboard の SQL Editor で1回だけ実行してください。

CREATE TABLE IF NOT EXISTS activity_log (
  id          BIGSERIAL PRIMARY KEY,
  ts          TIMESTAMPTZ DEFAULT NOW(),   -- 開始時刻
  run_date    DATE,                        -- 実行日
  role        TEXT,                        -- FM / Quant / Securities / Engineer / System
  step        TEXT,                        -- 何をしたか（例: パラメータ改善提案）
  status      TEXT,                        -- running / done / failed / rejected / skip / improve
  summary     TEXT,                        -- 1行サマリー
  detail      TEXT,                        -- 全文（レポート・推奨コメント・変更内容など）
  updated_at  TIMESTAMPTZ DEFAULT NOW()    -- 完了時刻
);

CREATE INDEX IF NOT EXISTS idx_activity_log_ts ON activity_log(ts DESC);

-- 匿名ユーザーの読み取りを許可（RLS）
ALTER TABLE activity_log ENABLE ROW LEVEL SECURITY;

CREATE POLICY "activity_log_public_read" ON activity_log
  FOR SELECT USING (true);
