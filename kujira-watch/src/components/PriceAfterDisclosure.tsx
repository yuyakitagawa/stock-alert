import { formatDate } from "@/lib/format";
import { getPriceAfterDisclosure, type PriceSnapshot } from "@/lib/priceReturns";

// 記事ページの「開示後の株価推移」パネル（サーバーコンポーネント）。
// 「クジラが動いた後、株価はどうなったのか」に記事内で直答するための表示で、
// 開示日の終値を基準に+1ヶ月(21営業日)・+3ヶ月(63営業日)・直近の騰落率を出す。
// 株価データが無い銘柄（価格キャッシュ対象外・開示が古い等）では何も描画しない。

function changePct(base: number, point: number): string {
  const pct = ((point - base) / base) * 100;
  const sign = pct >= 0 ? "+" : "";
  return `${sign}${pct.toFixed(1)}%`;
}

function PricePoint({
  label,
  base,
  point,
}: {
  label: string;
  base: PriceSnapshot;
  point: PriceSnapshot;
}) {
  const up = point.close >= base.close;
  return (
    <div>
      <dt className="text-[11px] text-foreground/40">
        {label}（{formatDate(point.date)}）
      </dt>
      <dd className={`m-0 mt-0.5 text-sm font-bold ${up ? "text-emerald-700" : "text-rose-700"}`}>
        {changePct(base.close, point.close)}
        <span className="ml-1 text-xs font-normal text-foreground/50">
          {Math.round(point.close).toLocaleString("ja-JP")}円
        </span>
      </dd>
    </div>
  );
}

export default async function PriceAfterDisclosure({
  stockCode,
  dealDate,
}: {
  stockCode: string;
  dealDate: string;
}) {
  const prices = await getPriceAfterDisclosure(stockCode, dealDate.slice(0, 10));
  if (!prices) return null;

  const { base, oneMonth, threeMonths, latest } = prices;
  const points = [
    oneMonth && { label: "1ヶ月後", point: oneMonth },
    threeMonths && { label: "3ヶ月後", point: threeMonths },
    latest && { label: "直近", point: latest },
  ].filter((p): p is { label: string; point: PriceSnapshot } => Boolean(p));

  return (
    <section className="mb-8 rounded-md border border-rule bg-section-tint p-4">
      <h2 className="text-sm font-bold text-brand-navy">開示後の株価推移</h2>
      <dl className="m-0 mt-3 grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4">
        <div>
          <dt className="text-[11px] text-foreground/40">基準終値（{formatDate(base.date)}）</dt>
          <dd className="m-0 mt-0.5 text-sm font-bold text-foreground/80">
            {Math.round(base.close).toLocaleString("ja-JP")}円
          </dd>
        </div>
        {points.map(({ label, point }) => (
          <PricePoint key={label} label={label} base={base} point={point} />
        ))}
      </dl>
      <p className="mb-0 mt-3 text-[11px] leading-relaxed text-foreground/40">
        開示日（開示日が休場の場合は直後の営業日）の終値を基準にした騰落率です。1ヶ月後=21営業日後・3ヶ月後=63営業日後。終値ベースの参考値であり、将来の値動きを示すものではありません。
      </p>
    </section>
  );
}
