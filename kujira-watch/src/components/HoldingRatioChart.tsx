import { formatDate } from "@/lib/format";
import type { HoldingHistoryPoint } from "@/lib/investors";

// 記事ページの「この投資家の保有比率の推移」。
// 1件の開示は「前回比 ±Xpt」しか語らないが、投資家が段階的に積み増しているのか
// 降りている最中なのかは複数開示を並べて初めて分かる（2026-08-19のユーザーインタビューで
// 最も要望が多かった項目）。外部ライブラリを増やさずインラインSVGで描く。

const WIDTH = 640;
const HEIGHT = 160;
const PADDING = { top: 16, right: 16, bottom: 28, left: 40 };

function toXY(points: HoldingHistoryPoint[]) {
  const values = points.map((p) => p.holdingRatio);
  const max = Math.max(...values);
  const min = Math.min(...values);
  // 全点が同じ値でも線が潰れないよう、上下に最低幅を取る。
  const span = max - min < 0.5 ? 0.5 : max - min;
  const base = max - min < 0.5 ? min - 0.25 : min;
  const innerW = WIDTH - PADDING.left - PADDING.right;
  const innerH = HEIGHT - PADDING.top - PADDING.bottom;
  return points.map((p, i) => ({
    ...p,
    x: PADDING.left + (points.length === 1 ? innerW / 2 : (innerW * i) / (points.length - 1)),
    y: PADDING.top + innerH - ((p.holdingRatio - base) / span) * innerH,
  }));
}

export default function HoldingRatioChart({
  filerName,
  stockName,
  points,
}: {
  filerName: string;
  stockName: string;
  points: HoldingHistoryPoint[];
}) {
  // 1点だけでは「推移」にならないので出さない（初回の大量保有報告書だけの投資家）。
  if (points.length < 2) return null;

  const xy = toXY(points);
  const first = xy[0];
  const last = xy[xy.length - 1];
  const rising = last.holdingRatio >= first.holdingRatio;
  const line = xy.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
  const label = `${filerName}による${stockName}の保有比率の推移`;

  return (
    <section className="mb-8 rounded-md border border-rule bg-section-tint p-4">
      <h2 className="text-sm font-bold text-brand-navy">この投資家の保有比率の推移</h2>
      <p className="mt-1 text-[11px] text-ink-tertiary">
        EDINETに提出された{points.length}件の開示（{formatDate(first.discDate)}〜{formatDate(last.discDate)}）の保有比率です。
      </p>
      <div className="mt-3 overflow-x-auto">
        <svg
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          className="h-40 w-full min-w-[320px]"
          role="img"
          aria-label={label}
        >
          <title>{label}</title>
          <line
            x1={PADDING.left}
            y1={HEIGHT - PADDING.bottom}
            x2={WIDTH - PADDING.right}
            y2={HEIGHT - PADDING.bottom}
            className="stroke-rule"
            strokeWidth="1"
          />
          <polyline
            points={line}
            fill="none"
            className={rising ? "stroke-emerald-600" : "stroke-rose-600"}
            strokeWidth="2"
            strokeLinejoin="round"
            strokeLinecap="round"
          />
          {xy.map((p) => (
            <circle
              key={p.discDate}
              cx={p.x}
              cy={p.y}
              r="3"
              className={rising ? "fill-emerald-600" : "fill-rose-600"}
            />
          ))}
          <text x={first.x} y={first.y - 8} className="fill-current text-[11px] text-ink-tertiary">
            {first.holdingRatio}%
          </text>
          <text
            x={last.x}
            y={last.y - 8}
            textAnchor="end"
            className="fill-current text-[11px] font-bold text-ink-secondary"
          >
            {last.holdingRatio}%
          </text>
          <text x={first.x} y={HEIGHT - 8} className="fill-current text-[10px] text-ink-muted">
            {formatDate(first.discDate)}
          </text>
          <text
            x={last.x}
            y={HEIGHT - 8}
            textAnchor="end"
            className="fill-current text-[10px] text-ink-muted"
          >
            {formatDate(last.discDate)}
          </text>
        </svg>
      </div>
    </section>
  );
}
