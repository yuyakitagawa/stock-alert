import type { CompanyInfo, PricePoint } from "@/lib/companyInfo";
import { formatDate } from "@/lib/format";

const CHART_WIDTH = 600;
const CHART_HEIGHT = 120;
const CHART_PADDING = 4;

function PriceChart({ history }: { history: PricePoint[] }) {
  if (history.length < 2) return null;

  const closes = history.map((point) => point.close);
  const min = Math.min(...closes);
  const max = Math.max(...closes);
  const range = max - min || 1;

  const points = history
    .map((point, i) => {
      const x =
        (i / (history.length - 1)) * (CHART_WIDTH - CHART_PADDING * 2) + CHART_PADDING;
      const y =
        CHART_HEIGHT -
        CHART_PADDING -
        ((point.close - min) / range) * (CHART_HEIGHT - CHART_PADDING * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  const first = history[0];
  const last = history[history.length - 1];

  return (
    <div className="mb-4">
      <svg
        viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
        preserveAspectRatio="none"
        className="h-24 w-full sm:h-28"
        role="img"
        aria-label={`株価推移（${formatDate(first.date)}〜${formatDate(last.date)}、${first.close.toLocaleString("ja-JP")}円→${last.close.toLocaleString("ja-JP")}円）`}
      >
        <polyline points={points} fill="none" stroke="var(--color-brand-blue)" strokeWidth="2" />
      </svg>
      <div className="mt-1 flex justify-between text-xs text-foreground/40">
        <span>{formatDate(first.date)}（{first.close.toLocaleString("ja-JP")}円）</span>
        <span>{formatDate(last.date)}（{last.close.toLocaleString("ja-JP")}円）</span>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="kicker text-foreground/40">{label}</dt>
      <dd className="mt-0.5 text-sm font-bold text-brand-navy">{value}</dd>
    </div>
  );
}

export default function CompanyInfoCard({ info }: { info: CompanyInfo }) {
  const stats: { label: string; value: string }[] = [];

  if (info.sector) stats.push({ label: "業種", value: info.sector });
  if (info.close !== null) {
    stats.push({ label: "終値", value: `${info.close.toLocaleString("ja-JP")}円` });
  }
  if (info.per !== null) stats.push({ label: "PER", value: `${info.per}倍` });
  if (info.pbr !== null) stats.push({ label: "PBR", value: `${info.pbr}倍` });
  if (info.pos52 !== null) {
    stats.push({ label: "52週レンジ位置", value: `${Math.round(info.pos52 * 100)}%` });
  }
  stats.push({
    label: "株主優待",
    value: info.hasYutai ? `あり${info.yutaiMonth ? `（${info.yutaiMonth}月権利）` : ""}` : "なし",
  });

  if (stats.length === 0 && info.priceHistory.length < 2) return null;

  return (
    <div className="mb-6 border border-rule bg-paper p-4 sm:p-5">
      <PriceChart history={info.priceHistory} />
      {stats.length > 0 && (
        <dl className="grid grid-cols-2 gap-4 sm:grid-cols-3">
          {stats.map((stat) => (
            <Stat key={stat.label} {...stat} />
          ))}
        </dl>
      )}
      {info.closeDate && (
        <p className="mt-3 text-xs text-foreground/40">
          株価・指標は{formatDate(info.closeDate)}時点
        </p>
      )}
    </div>
  );
}
