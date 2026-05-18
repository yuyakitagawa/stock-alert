interface Props {
  prices: number[];
  color?: string;
  showLabel?: boolean;
}

const DEFAULT_COLOR = "#22c55e";

export default function Sparkline({ prices, color = DEFAULT_COLOR, showLabel = false }: Props) {
  if (prices.length < 2) return null;

  const min = Math.min(...prices);
  const max = Math.max(...prices);
  const range = max - min || 1;
  const W = 200;
  const H = 56;
  const padX = 0;
  const padY = 4;

  const pts = prices.map((p, i) => ({
    x: padX + (i / (prices.length - 1)) * (W - padX * 2),
    y: padY + (1 - (p - min) / range) * (H - padY * 2),
  }));

  const line = pts.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
  const area =
    `${pts[0].x.toFixed(1)},${H} ` +
    pts.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ") +
    ` ${pts[pts.length - 1].x.toFixed(1)},${H}`;

  const pct = ((prices[prices.length - 1] - prices[0]) / prices[0]) * 100;

  // Parse hex to rgba for fill
  const hex = color.replace("#", "");
  const r = parseInt(hex.slice(0, 2), 16);
  const g = parseInt(hex.slice(2, 4), 16);
  const b = parseInt(hex.slice(4, 6), 16);
  const fill = `rgba(${r},${g},${b},0.15)`;

  return (
    <div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        style={{ height: `${H}px`, display: "block" }}
        preserveAspectRatio="none"
      >
        <polygon points={area} fill={fill} />
        <polyline
          points={line}
          fill="none"
          stroke={color}
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <circle
          cx={pts[pts.length - 1].x}
          cy={pts[pts.length - 1].y}
          r="2.5"
          fill={color}
        />
      </svg>
      {showLabel && (
        <div
          className="text-xs font-mono font-bold text-right px-2 -mt-1"
          style={{ color }}
        >
          1M {pct >= 0 ? "+" : ""}{pct.toFixed(1)}%
        </div>
      )}
    </div>
  );
}
