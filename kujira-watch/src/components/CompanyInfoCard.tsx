import type { CompanyInfo, PricePoint } from "@/lib/companyInfo";
import { formatDate } from "@/lib/format";
import { UI, type Locale } from "@/lib/i18n";

const CHART_WIDTH = 600;
const CHART_HEIGHT = 120;
const CHART_PADDING = 4;

function PriceChart({ history, locale }: { history: PricePoint[]; locale: Locale }) {
  if (history.length < 2) return null;
  const t = UI[locale];

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
  const localeCode = locale === "en" ? "en-US" : "ja-JP";

  return (
    <div className="mb-4">
      <svg
        viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
        preserveAspectRatio="none"
        className="h-24 w-full sm:h-28"
        role="img"
        aria-label={t.priceChartAlt(
          formatDate(first.date, locale),
          formatDate(last.date, locale),
          first.close.toLocaleString(localeCode),
          last.close.toLocaleString(localeCode)
        )}
      >
        <polyline points={points} fill="none" stroke="var(--color-brand-blue)" strokeWidth="2" />
      </svg>
      <div className="mt-1 flex justify-between text-xs text-foreground/40">
        <span>
          {formatDate(first.date, locale)}
          {locale === "en"
            ? ` (¥${first.close.toLocaleString(localeCode)})`
            : `（${first.close.toLocaleString(localeCode)}円）`}
        </span>
        <span>
          {formatDate(last.date, locale)}
          {locale === "en"
            ? ` (¥${last.close.toLocaleString(localeCode)})`
            : `（${last.close.toLocaleString(localeCode)}円）`}
        </span>
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

export default function CompanyInfoCard({ info, locale = "ja" }: { info: CompanyInfo; locale?: Locale }) {
  const t = UI[locale];
  const localeCode = locale === "en" ? "en-US" : "ja-JP";
  const stats: { label: string; value: string }[] = [];

  if (info.sector) stats.push({ label: t.companySectorLabel, value: info.sector });
  if (info.close !== null) {
    const closeValue = info.close.toLocaleString(localeCode);
    stats.push({ label: t.companyCloseLabel, value: locale === "en" ? `¥${closeValue}` : `${closeValue}円` });
  }
  if (info.per !== null) stats.push({ label: t.companyPerLabel, value: `${info.per}x` });
  if (info.pbr !== null) stats.push({ label: t.companyPbrLabel, value: `${info.pbr}x` });
  if (info.pos52 !== null) {
    stats.push({ label: t.companyPos52Label, value: `${Math.round(info.pos52 * 100)}%` });
  }
  stats.push({
    label: t.companyYutaiLabel,
    value: info.hasYutai ? t.companyYutaiYes(info.yutaiMonth) : t.companyYutaiNo,
  });

  if (stats.length === 0 && info.priceHistory.length < 2 && !info.description) return null;

  return (
    <div className="mb-6 border border-rule bg-paper p-4 sm:p-5">
      {info.description && (
        <div className="mb-4">
          <dt className="kicker text-foreground/40">{t.companyDescriptionLabel}</dt>
          <dd className="mt-0.5 text-sm text-foreground/80">{info.description}</dd>
        </div>
      )}
      <PriceChart history={info.priceHistory} locale={locale} />
      {stats.length > 0 && (
        <dl className="grid grid-cols-2 gap-4 sm:grid-cols-3">
          {stats.map((stat) => (
            <Stat key={stat.label} {...stat} />
          ))}
        </dl>
      )}
      {info.closeDate && (
        <p className="mt-3 text-xs text-foreground/40">
          {t.companyAsOf(formatDate(info.closeDate, locale))}
        </p>
      )}
    </div>
  );
}
