-- 投資家リターンランキング（/ranking/returns）と投資家ページ（/investors/[filer]）の集計元。
-- 「買い開示1件＝1ポジション」として、開示日の終値から63営業日後（＝3ヶ月）の終値までの
-- 騰落率を出し、投資家ごとに等ウェイトで平均する。
--
-- 2層構成:
--   investor_return_positions_3m … ポジション（買い開示）1件＝1行。個別の騰落率を持つ。
--                                   /investors/[filer] の開示テーブルの「3ヶ月後」列で使う。
--   investor_returns_3m           … 上を投資家単位に集計したもの。ランキングで使う。
-- 平均値の裏が取れるよう、ランキングの数字と個別ページの数字は必ず同じ行から作る
-- （別々のクエリで別々に計算すると、片方だけ条件がずれても誰も気付けない）。
--
-- 2026-08-18に廃止した旧 filer_win_rate（乗っかり実績）の失敗を繰り返さないための規律:
--  1. holding_ratio_prior がNULLの開示を「新規取得」とみなさない。新規の大量保有報告書
--     （doc_descriptionが「大量保有報告書」で始まる）だけを prior=0 として扱い、
--     変更報告書でpriorが欠けている行（88件）は方向不明として捨てる。
--     旧実装はこれで売り切り（例: ＢＣＰＥのキオクシア 22.53%→0）を巨額の買いに数えていた。
--  2. 指標は金額加重の名目損益ではなく等ウェイトの騰落率。金額加重にすると貸株玉の大きい
--     プライムブローカーが上位を独占して「乗っかる価値」を表さない。
--  3. (filer_name, issuer_code, disc_date) でDISTINCTして二重計上を防ぐ。
--  4. マテリアライズドビューなので毎回作り直され、対象外になった行が残らない
--     （旧実装は on_conflict=filer_name のupsertのみで削除が無く、1,371行中320行が残骸だった）。
--  5. ランキングに載せるのは n>=3 の投資家だけ（旧実装は68%がn=1の「実績」だった）。
--     ポジション側は件数で絞らない（個別ページは「1件しかない」ことも含めて事実を出す）。
--
-- 更新: .github/workflows/daily_alert.yml Step 0b（tools/refresh_investor_returns.py）。

DROP MATERIALIZED VIEW IF EXISTS investor_returns_3m;
DROP MATERIALIZED VIEW IF EXISTS investor_return_positions_3m;

CREATE MATERIALIZED VIEW investor_return_positions_3m AS
WITH buys AS (
  -- 買い方向の開示。同一(投資家×銘柄×開示日)の重複行は1件に畳む。
  -- issuer_nameを含めたDISTINCTでは畳めない。同じ銘柄でも開示ごとに半角/全角がゆれることが
  -- あり（実例: 4189 KHネオケム/ＫＨネオケム）、その1件が二重計上されるため
  -- キー3列のDISTINCT ONにして名前は代表値を1つ採る。ユニークインデックスで再発も検知できる。
  SELECT DISTINCT ON (filer_name, issuer_code, disc_date)
         doc_id, filer_name, issuer_code, issuer_name, disc_date
  FROM edinet_large_holdings
  WHERE filer_name IS NOT NULL
    AND issuer_code IS NOT NULL
    AND holding_ratio IS NOT NULL
    AND doc_description NOT LIKE '訂正%'
    AND holding_ratio > COALESCE(holding_ratio_prior, 0)
    AND (holding_ratio_prior IS NOT NULL OR doc_description LIKE '大量保有報告書%')
  ORDER BY filer_name, issuer_code, disc_date, issuer_name
),
pairs AS (SELECT DISTINCT issuer_code, disc_date FROM buys),
base AS (
  -- 開示日以降で最初に値の付いた終値を基準にする。7日以上空くなら「開示時点の株価」と
  -- 呼べない（上場廃止・コード変更・価格キャッシュ収録前）ので落とす。
  SELECT p.issuer_code, p.disc_date, b.date AS base_date, b.close AS base_close
  FROM pairs p
  CROSS JOIN LATERAL (
    SELECT y.date, y.close FROM yahoo_price_cache y
    WHERE y.code = p.issuer_code AND y.date >= p.disc_date AND y.close IS NOT NULL
    ORDER BY y.date LIMIT 1
  ) b
  WHERE b.date::date <= p.disc_date::date + 7 AND b.close > 0
),
ret AS (
  SELECT base.issuer_code, base.disc_date, base.base_date, base.base_close,
         f.date AS date_3m, f.close AS close_3m,
         (f.close / base.base_close - 1) * 100 AS ret_3m,
         CASE WHEN bench_base.close > 0 AND bench_3m.close IS NOT NULL
              THEN (f.close / base.base_close - bench_3m.close / bench_base.close) * 100
         END AS excess_3m
  FROM base
  -- 63営業日後（基準日の翌営業日から数えて63本目＝OFFSET 62）。ここが取れない
  -- ＝まだ3ヶ月経っていない開示なので、自動的に評価対象から外れる。
  CROSS JOIN LATERAL (
    SELECT y.date, y.close FROM yahoo_price_cache y
    WHERE y.code = base.issuer_code AND y.date > base.base_date AND y.close IS NOT NULL
    ORDER BY y.date OFFSET 62 LIMIT 1
  ) f
  -- 日経平均は暦日で突き合わせる（銘柄側と営業日の本数がずれても同じ期間を比較するため）。
  LEFT JOIN LATERAL (
    SELECT m.close FROM yahoo_market_index m
    WHERE m.ticker = 'N225' AND m.date <= base.base_date AND m.close IS NOT NULL
    ORDER BY m.date DESC LIMIT 1
  ) bench_base ON true
  LEFT JOIN LATERAL (
    SELECT m.close FROM yahoo_market_index m
    WHERE m.ticker = 'N225' AND m.date <= f.date AND m.close IS NOT NULL
    ORDER BY m.date DESC LIMIT 1
  ) bench_3m ON true
)
SELECT
  -- doc_idを持たせるのは、/investors/[filer]の開示テーブルが「このビューが実際に数えた1行」だけに
  -- 数字を出せるようにするため。(銘柄, 開示日)で突き合わせると、同日に変更報告書と訂正報告書が
  -- 並ぶケース（実例: 2026-03-18のアーカス×SHIFT）で訂正報告書の行にも数字が付いてしまう。
  -- doc_idで引けば買い判定のルールをTS側に書き写す必要も無く、二重定義によるズレが起きない。
  b.doc_id,
  b.filer_name,
  b.issuer_code,
  b.issuer_name,
  b.disc_date,
  r.base_date,
  round(r.base_close::numeric, 1) AS base_close,
  r.date_3m,
  round(r.close_3m::numeric, 1) AS close_3m,
  round(r.ret_3m::numeric, 1) AS ret_3m,
  round(r.excess_3m::numeric, 1) AS excess_3m
FROM buys b
JOIN ret r ON r.issuer_code = b.issuer_code AND r.disc_date = b.disc_date;

COMMENT ON MATERIALIZED VIEW investor_return_positions_3m IS
  'EDINET買い開示1件＝1行。開示日の終値から63営業日後の終値までの騰落率と日経平均比。investor_returns_3mの元になる明細で、/investors/[filer]の開示テーブルの「3ヶ月後」列にも使う。';

CREATE UNIQUE INDEX investor_return_positions_3m_key_idx
  ON investor_return_positions_3m (filer_name, issuer_code, disc_date);
CREATE INDEX investor_return_positions_3m_filer_idx
  ON investor_return_positions_3m (filer_name);

GRANT SELECT ON investor_return_positions_3m TO service_role;

CREATE MATERIALIZED VIEW investor_returns_3m AS
WITH ranked AS (
  SELECT p.*,
         row_number() OVER (PARTITION BY filer_name ORDER BY ret_3m DESC) AS rn_best,
         row_number() OVER (PARTITION BY filer_name ORDER BY ret_3m ASC) AS rn_worst
  FROM investor_return_positions_3m p
)
SELECT
  r.filer_name,
  c.category,
  count(*)::int AS position_count,
  round(avg(r.ret_3m), 1) AS avg_return,
  round((percentile_cont(0.5) WITHIN GROUP (ORDER BY r.ret_3m))::numeric, 1) AS median_return,
  round((100.0 * count(*) FILTER (WHERE r.ret_3m > 0) / count(*)), 0)::int AS win_rate,
  round(avg(r.excess_3m), 1) AS avg_excess_return,
  max(CASE WHEN r.rn_best  = 1 THEN r.issuer_code END) AS best_stock_code,
  max(CASE WHEN r.rn_best  = 1 THEN r.issuer_name END) AS best_stock_name,
  max(CASE WHEN r.rn_best  = 1 THEN r.ret_3m END) AS best_return,
  max(CASE WHEN r.rn_worst = 1 THEN r.issuer_code END) AS worst_stock_code,
  max(CASE WHEN r.rn_worst = 1 THEN r.issuer_name END) AS worst_stock_name,
  min(CASE WHEN r.rn_worst = 1 THEN r.ret_3m END) AS worst_return,
  min(r.disc_date) AS first_buy_date,
  max(r.disc_date) AS latest_buy_date
FROM ranked r
LEFT JOIN edinet_filer_classification c ON c.filer_name = r.filer_name
GROUP BY r.filer_name, c.category
HAVING count(*) >= 3;

COMMENT ON MATERIALIZED VIEW investor_returns_3m IS
  'investor_return_positions_3mを投資家単位に等ウェイト集計したもの。/ranking/returns の集計元。開示3件以上の投資家のみ。';

CREATE UNIQUE INDEX investor_returns_3m_filer_idx ON investor_returns_3m (filer_name);
CREATE INDEX investor_returns_3m_avg_idx ON investor_returns_3m (avg_return DESC);

GRANT SELECT ON investor_returns_3m TO service_role;

-- 日次バッチ(tools/refresh_investor_returns.py)から呼ぶリフレッシュ用。
-- PostgRESTのrpc経由で叩けるようにSECURITY DEFINERにする。
-- CONCURRENTLYは使えない（PostgRESTはRPCを1トランザクションで実行するが、
-- REFRESH MATERIALIZED VIEW CONCURRENTLY はトランザクション内では実行できない）。
-- 通常のREFRESHは数秒このビューの読み取りを止めるが、日次1回・Web側は1時間キャッシュなので許容する。
-- 集計ビューは明細ビューを参照しているので、必ず明細→集計の順で作り直す。
-- 戻り値は再計算後の投資家数。0や失敗をバッチ側で検知できるようvoidではなくintを返す。
DROP FUNCTION IF EXISTS refresh_investor_returns_3m();

CREATE FUNCTION refresh_investor_returns_3m() RETURNS int
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE
  n int;
BEGIN
  REFRESH MATERIALIZED VIEW investor_return_positions_3m;
  REFRESH MATERIALIZED VIEW investor_returns_3m;
  SELECT count(*) INTO n FROM investor_returns_3m;
  RETURN n;
END;
$$;

GRANT EXECUTE ON FUNCTION refresh_investor_returns_3m() TO service_role;
