interface Props {
  prices: number[];
  showLabel?: boolean;
}

export default function Sparkline({ prices, showLabel = false }: Props) {
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
  const area = `${pts[0].x.toFixed(1)},${H} ` +
    pts.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ") +
    ` ${pts[pts.length - 1].x.toFixed(1)},${H}`;

  const isUp = prices[prices.length - 1] >= prices[0];
  const stroke = isUp ? "#22c55e" : "#ef4444";
  const fill   = isUp ? "rgba(34,197,94,0.15)" : "rgba(239,68,68,0.15)";
  const pct = ((prices[prices.length - 1] - prices[0]) / prices[0]) * 100;

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
          stroke={stroke}
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <circle
          cx={pts[pts.length - 1].x}
          cy={pts[pts.length - 1].y}
          r="2.5"
          fill={stroke}
        />
      </svg>
      {showLabel && (
        <div className={`text-xs font-mono font-bold text-right px-2 -mt-1 ${isUp ? "text-green-400" : "text-red-400"}`}>
          1M {pct >= 0 ? "+" : ""}{pct.toFixed(1)}%
        </div>
      )}
    </div>
  );
}
