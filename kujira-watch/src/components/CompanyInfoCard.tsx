import Card from "@mui/material/Card";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import type { CompanyInfo, PricePoint } from "@/lib/companyInfo";
import { formatDate } from "@/lib/format";
import { UI } from "@/lib/i18n";

const CHART_WIDTH = 600;
const CHART_HEIGHT = 120;
const CHART_PADDING = 4;

function PriceChart({ history }: { history: PricePoint[] }) {
  if (history.length < 2) return null;
  const t = UI;

  const closes = history.map((point) => point.close);
  const min = Math.min(...closes);
  const max = Math.max(...closes);
  const range = max - min || 1;

  const toX = (i: number) =>
    (i / (history.length - 1)) * (CHART_WIDTH - CHART_PADDING * 2) + CHART_PADDING;
  const toY = (close: number) =>
    CHART_HEIGHT - CHART_PADDING - ((close - min) / range) * (CHART_HEIGHT - CHART_PADDING * 2);

  const points = history
    .map((point, i) => `${toX(i).toFixed(1)},${toY(point.close).toFixed(1)}`)
    .join(" ");

  const first = history[0];
  const last = history[history.length - 1];

  return (
    <Box sx={{ mb: 2 }}>
      {/* 線だけだと水準感が読めないため、期間高値・安値をキャプションと点線ガイドで示す。
          文字はSVG内に置くとpreserveAspectRatio="none"の伸縮で歪むため、HTML側に出す。 */}
      <Typography variant="caption" sx={{ display: "block", textAlign: "right", color: "text.disabled" }}>
        {`期間高値 ${max.toLocaleString("ja-JP")}円 ・ 安値 ${min.toLocaleString("ja-JP")}円`}
      </Typography>
      <Box
        component="svg"
        viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
        preserveAspectRatio="none"
        sx={{ height: { xs: 96, sm: 112 }, width: "100%" }}
        role="img"
        aria-label={t.priceChartAlt(
          formatDate(first.date),
          formatDate(last.date),
          first.close.toLocaleString("ja-JP"),
          last.close.toLocaleString("ja-JP")
        )}
      >
        <line
          x1={CHART_PADDING}
          x2={CHART_WIDTH - CHART_PADDING}
          y1={toY(max)}
          y2={toY(max)}
          stroke="var(--rule)"
          strokeWidth="1"
          strokeDasharray="4 4"
        />
        <line
          x1={CHART_PADDING}
          x2={CHART_WIDTH - CHART_PADDING}
          y1={toY(min)}
          y2={toY(min)}
          stroke="var(--rule)"
          strokeWidth="1"
          strokeDasharray="4 4"
        />
        <polyline points={points} fill="none" stroke="var(--color-brand-blue)" strokeWidth="2" />
        <circle cx={toX(history.length - 1)} cy={toY(last.close)} r="3.5" fill="var(--color-brand-blue)" />
      </Box>
      <Box sx={{ mt: 0.5, display: "flex", justifyContent: "space-between" }}>
        <Typography variant="caption" sx={{ color: "text.disabled" }}>
          {formatDate(first.date)}
          {`（${first.close.toLocaleString("ja-JP")}円）`}
        </Typography>
        <Typography variant="caption" sx={{ color: "text.disabled" }}>
          {formatDate(last.date)}
          {`（${last.close.toLocaleString("ja-JP")}円）`}
        </Typography>
      </Box>
    </Box>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <Box>
      <Typography variant="overline" component="dt" sx={{ display: "block", color: "text.disabled" }}>
        {label}
      </Typography>
      <Typography variant="body2" component="dd" sx={{ mt: 0.25, fontWeight: 700, color: "primary.main" }}>
        {value}
      </Typography>
    </Box>
  );
}

export default function CompanyInfoCard({ info }: { info: CompanyInfo }) {
  const t = UI;
  const stats: { label: string; value: string }[] = [];

  if (info.sector) stats.push({ label: t.companySectorLabel, value: info.sector });
  if (info.close !== null) {
    stats.push({ label: t.companyCloseLabel, value: `${info.close.toLocaleString("ja-JP")}円` });
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

  if (stats.length === 0 && info.priceHistory.length < 2) return null;

  return (
    <Card variant="outlined" sx={{ mb: 3, p: { xs: 2, sm: 2.5 }, borderColor: "divider" }}>
      <PriceChart history={info.priceHistory} />
      {stats.length > 0 && (
        <Box
          component="dl"
          sx={{
            m: 0,
            display: "grid",
            gridTemplateColumns: { xs: "repeat(2, 1fr)", sm: "repeat(3, 1fr)" },
            gap: 2,
          }}
        >
          {stats.map((stat) => (
            <Stat key={stat.label} {...stat} />
          ))}
        </Box>
      )}
      {info.closeDate && (
        <Typography variant="caption" sx={{ display: "block", mt: 1.5, color: "text.disabled" }}>
          {t.companyAsOf(formatDate(info.closeDate))}
        </Typography>
      )}
    </Card>
  );
}
