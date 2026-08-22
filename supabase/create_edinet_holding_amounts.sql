-- 銘柄ランキング(/trending)の「推定売買金額」の集計元。
-- EDINET大量保有報告書には取引金額が載らないため、開示1件ごとに
--   推定金額 = 保有比率の変化幅(%) ÷ 100 × 発行済株式数 × 開示日の終値
-- を概算する。web/publish_blog_articles.py の estimate_deal_amount_oku() と同じ式で、
-- 記事の「推定取得金額/推定売却金額」と数字が食い違わないようにしてある
-- （違いは株式数の出どころだけ。記事はyfinanceの最新値、ここは開示日時点の決算値）。
--
-- 規律:
--  1. 保有比率の変化幅は開示自身が持つ holding_ratio_prior を最優先で使う。
--     取れない開示だけ、同じ投資家×同じ銘柄の直前の開示から再導出し、それも無ければ
--     「大量保有報告書」（新規提出）のみ前回0%として扱う。変更報告書で前回比率が
--     どうしても取れない行は方向も規模も不明なので金額を出さない（行ごと落とす）。
--     web/publish_blog_articles.py の ratio_change_pct() と同じ規律。
--  2. 訂正報告書は売買を伴わないので除外する（記事側も推定金額0として扱っている）。
--     件数（/trendingのランキング軸）には入るが金額には入らない。
--  3. 株価または発行済株式数が取れない開示は行を作らない。0円として混ぜると
--     「金額が小さい銘柄」と「金額が分からない銘柄」の区別が付かなくなるため。
--  4. 開示日の終値は開示日以前の直近10日以内の終値のみ採用する（上場廃止・コード変更で
--     何ヶ月も前の株価を掴まない）。
--  5. マテリアライズドビューにしているのは、素のビューだと直近60日ぶんの読み取りだけで
--     約2.8秒かかるため（開示×株価のLATERAL結合。2026-08-22実測）。
--
-- 更新: .github/workflows/edinet_blog.yml（毎時・開示スキャンの直後）と
--       .github/workflows/daily_alert.yml Step 0c（株価キャッシュ更新の直後）。
--       どちらも tools/refresh_holding_amounts.py を叩く。

DROP MATERIALIZED VIEW IF EXISTS edinet_holding_amounts;

CREATE MATERIALIZED VIEW edinet_holding_amounts AS
WITH base AS (
  SELECT
    doc_id,
    issuer_code,
    disc_date,
    holding_ratio,
    COALESCE(
      holding_ratio_prior,
      lag(holding_ratio) OVER (PARTITION BY issuer_code, filer_name ORDER BY disc_date, doc_id),
      CASE WHEN doc_description LIKE '大量保有報告書%' THEN 0 END
    ) AS prior_ratio
  FROM edinet_large_holdings
  WHERE issuer_code IS NOT NULL
    AND holding_ratio IS NOT NULL
    AND doc_description NOT LIKE '訂正%'
),
changed AS (
  SELECT doc_id, issuer_code, disc_date, abs(holding_ratio - prior_ratio) AS ratio_change
  FROM base
  WHERE prior_ratio IS NOT NULL
    AND holding_ratio <> prior_ratio
)
SELECT
  c.doc_id,
  c.issuer_code,
  c.disc_date,
  round(c.ratio_change::numeric, 2) AS ratio_change_pt,
  round((c.ratio_change / 100.0 * f.sh_out * p.close / 1e8)::numeric, 1) AS deal_amount_oku
FROM changed c
-- 発行済株式数は開示日時点で最新の決算（point-in-time）を使う。
CROSS JOIN LATERAL (
  SELECT j.sh_out FROM jquants_fin_summary j
  WHERE j.code = c.issuer_code AND j.sh_out > 0 AND j.disc_date <= c.disc_date
  ORDER BY j.disc_date DESC LIMIT 1
) f
CROSS JOIN LATERAL (
  SELECT y.close FROM yahoo_price_cache y
  WHERE y.code = c.issuer_code AND y.close > 0
    AND y.date <= c.disc_date
    AND y.date >= to_char(c.disc_date::date - 10, 'YYYY-MM-DD')
  ORDER BY y.date DESC LIMIT 1
) p;

COMMENT ON MATERIALIZED VIEW edinet_holding_amounts IS
  'EDINET大量保有報告書1件＝1行の推定売買金額（億円）。保有比率の変化幅×発行済株式数×開示日終値の概算。/trending（銘柄ランキング）の金額表示の集計元。';

CREATE UNIQUE INDEX edinet_holding_amounts_doc_idx ON edinet_holding_amounts (doc_id);
CREATE INDEX edinet_holding_amounts_disc_idx ON edinet_holding_amounts (disc_date);

GRANT SELECT ON edinet_holding_amounts TO service_role;

-- tools/refresh_holding_amounts.py から呼ぶリフレッシュ用。
-- investor_returns_3m と同じ理由でCONCURRENTLYは使えない（PostgRESTのRPCは
-- 1トランザクションで実行され、REFRESH ... CONCURRENTLY はトランザクション内で動かない）。
-- 戻り値は行数。0や失敗をバッチ側で検知できるようvoidではなくintを返す。
DROP FUNCTION IF EXISTS refresh_edinet_holding_amounts();

-- statement_timeoutを明示するのは、PostgRESTの接続ロール(authenticator)に8秒の上限が
-- 設定されているため。再作成は13,000行で約9秒かかる（2026-08-22実測）ので、そのままだと
-- 途中で切られる。
CREATE FUNCTION refresh_edinet_holding_amounts() RETURNS int
LANGUAGE plpgsql SECURITY DEFINER
SET search_path = public
SET statement_timeout = '300s'
AS $$
DECLARE
  n int;
BEGIN
  REFRESH MATERIALIZED VIEW edinet_holding_amounts;
  SELECT count(*) INTO n FROM edinet_holding_amounts;
  RETURN n;
END;
$$;

GRANT EXECUTE ON FUNCTION refresh_edinet_holding_amounts() TO service_role;
