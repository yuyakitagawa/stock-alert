-- YouTube（Shorts）の成果を記録するテーブル。
--
-- なぜ必要か:
--   動画パイプライン（video_post.yml）は毎営業日アップロードしているのに、成果を
--   どこにも記録していなかった。動画IDすら保存していないため「何本出して何回見られたか」
--   を後から追えず、続けるか止めるかの判断ができない状態だった（2026-08-27）。
--   実際に調べたら総再生4,747回・登録者3人・公開7本で、X（60日で34インプレッション）より
--   桁違いに届いている一方、GA4上のサイト流入は28日で22セッション＝再生の0.46%だった。
--
-- 取得は video/youtube_metrics.py。公開動画の統計はサービスアカウント（gcp_key.json）の
-- トークンで読めるため、アップロード用のOAuthリフレッシュトークン（scopeがyoutube.upload
-- だけで統計を読めない）を取り直す必要はない。
--
-- x_posts / x_post_metrics と同じ2層構成にする。最新値だけだと「伸びている最中か
-- 頭打ちか」が分からず、日次スナップショットだけだと一覧が重くなるため。
create table if not exists youtube_videos (
  video_id text primary key,
  published_at timestamptz not null,
  title text,
  -- 尺（秒）。短いほど再生が伸びる傾向があるかを見るために持つ。
  duration_sec integer,
  views integer,
  likes integer,
  comments integer,
  metrics_updated_at timestamptz
);

create table if not exists youtube_video_metrics (
  video_id text not null,
  measured_on date not null,
  views integer,
  likes integer,
  comments integer,
  primary key (video_id, measured_on)
);

-- チャンネル単位。登録者は投稿単位では動かないので別テーブルにする。
create table if not exists youtube_channel_stats (
  measured_on date primary key,
  subscribers integer,
  total_views bigint,
  video_count integer,
  captured_at timestamptz not null default now()
);

create index if not exists youtube_videos_published_at_idx on youtube_videos (published_at desc);
