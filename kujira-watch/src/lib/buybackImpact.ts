// 自社株買いの取得枠が「発行済株式の何%か」「EPSをどれだけ押し上げるか」「市場で何日ぶんの
// 出来高か」を計算する。/tools/buyback-impact の計算本体。
//
// 取締役会が決議するのは上限（取得する株式の総数・取得価額の総額）であって実際の取得額では
// ないため、ここで出るのは「枠を使い切った場合の最大値」。/buybacks で毎日集めている
// TDnetの決定開示に載っている数字をそのまま入れれば、記事の一覧では並ばない
// 「自分の持っている銘柄だとどうなるか」が出せる。

export type BuybackImpactInput = {
  /** 取得価額の総額の上限（円）。 */
  maxAmountYen: number;
  /** 取得する株式の総数の上限（株）。 */
  maxShares: number;
  /** 発行済株式総数（自己株式を除く。株）。 */
  sharesOutstanding: number;
  /** 現在の株価（円）。 */
  price: number;
  /** 当期純利益（円）。任意。入れるとEPSの実額が出る。 */
  netIncomeYen?: number;
  /** 1日平均出来高（株）。任意。入れると需給の消化日数が出る。 */
  avgDailyVolume?: number;
};

export type BuybackImpactResult = {
  /** 金額上限と株数上限のうち先に当たる方で決まる、実際に取得しうる株数。 */
  effectiveShares: number;
  /** 上限に先に当たるのはどちらか。 */
  bindingLimit: "amount" | "shares";
  /** 取得株数 ÷ 発行済株式総数（%）。 */
  sharesRatioPct: number;
  /** 取得価額の上限 ÷ 時価総額（%）。 */
  marketCapRatioPct: number;
  marketCapYen: number;
  /** 取得株を消却した場合のEPS押し上げ率（%）。 */
  epsUpliftPct: number;
  epsBefore: number | null;
  epsAfter: number | null;
  /** 1日平均出来高の何日ぶんか。 */
  volumeDays: number | null;
};

/**
 * 入力が全部そろっていれば計算結果を返す。1つでも0以下・NaNなら null。
 * 画面側で「入力途中は何も出さない」判定に使う。
 */
export function calcBuybackImpact(input: BuybackImpactInput): BuybackImpactResult | null {
  const { maxAmountYen, maxShares, sharesOutstanding, price, netIncomeYen, avgDailyVolume } = input;
  const required = [maxAmountYen, maxShares, sharesOutstanding, price];
  if (required.some((v) => !Number.isFinite(v) || v <= 0)) return null;

  // 「10億円まで」かつ「100万株まで」の枠で株価2,000円なら、金額の方が先に尽きて50万株しか
  // 買えない。決定開示は両方の上限を書くので、小さい方を実際の取得株数として扱う。
  const sharesByAmount = maxAmountYen / price;
  const bindingLimit: "amount" | "shares" = sharesByAmount <= maxShares ? "amount" : "shares";
  const effectiveShares = Math.min(sharesByAmount, maxShares);

  const sharesRatio = effectiveShares / sharesOutstanding;
  const marketCapYen = price * sharesOutstanding;

  // 自己株式は分母から除かれるので、株数が r の割合だけ減ればEPSは 1/(1-r) 倍になる。
  // 取得株を消却せず金庫株のまま持つ場合も、1株当たり利益の計算では自己株式を除くため同じ。
  const epsUplift = sharesRatio >= 1 ? Infinity : 1 / (1 - sharesRatio) - 1;

  const hasIncome = Number.isFinite(netIncomeYen) && (netIncomeYen ?? 0) > 0;
  const epsBefore = hasIncome ? (netIncomeYen as number) / sharesOutstanding : null;
  const epsAfter =
    hasIncome && sharesRatio < 1
      ? (netIncomeYen as number) / (sharesOutstanding - effectiveShares)
      : null;

  const hasVolume = Number.isFinite(avgDailyVolume) && (avgDailyVolume ?? 0) > 0;

  return {
    effectiveShares,
    bindingLimit,
    sharesRatioPct: sharesRatio * 100,
    marketCapRatioPct: (maxAmountYen / marketCapYen) * 100,
    marketCapYen,
    epsUpliftPct: epsUplift * 100,
    epsBefore,
    epsAfter,
    volumeDays: hasVolume ? effectiveShares / (avgDailyVolume as number) : null,
  };
}

/** 需給インパクトの読み方を一言で返す（画面の判定コメント用）。 */
export function buybackImpactVerdict(result: BuybackImpactResult): string {
  const ratio = result.sharesRatioPct;
  if (ratio >= 10) return "発行済の1割以上。需給・EPSの両面で影響が大きい規模。";
  if (ratio >= 5) return "発行済の5%以上。株主還元としては大きい部類。";
  if (ratio >= 2) return "発行済の2%以上。一般的な規模の取得枠。";
  return "発行済の2%未満。需給への影響は限定的になりやすい。";
}
