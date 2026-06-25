"use client";
import { useState, useEffect, useRef, useCallback } from "react";
type Period = "1M" | "3M" | "6M" | "1Y";

interface DataPoint {
  date: string;
  close: number;
}

interface Props {
  code: string;
}

interface Tooltip {
  x: number;
  y: number;
  date: string;
  price: number;
  visible: boolean;
}

const W = 700;
const H = 300;
const PAD = { top: 24, right: 20, bottom: 32, left: 66 };
const CW = W - PAD.left - PAD.right;
const CH = H - PAD.top - PAD.bottom;

export default function StockChart({ code }: Props) {
  const chartPeriods: Record<Period, string> = { "1M": "1ヶ月", "3M": "3ヶ月", "6M": "6ヶ月", "1Y": "1年" };
  const [period, setPeriod] = useState<Period>("1M");
  const [data, setData] = useState<DataPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [tooltip, setTooltip] = useState<Tooltip>({ x: 0, y: 0, date: "", price: 0, visible: false });
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    setLoading(true);
    fetch(`/api/chart/${code}?period=${period}`)
      .then(r => r.json())
      .then((d: DataPoint[]) => { setData(Array.isArray(d) ? d : []); setLoading(false); })
      .catch(() => setLoading(false));
  }, [period, code]);

  const valid = data.filter(d => d.close != null);
  const rawMin = valid.length ? Math.min(...valid.map(d => d.close)) : 0;
  const rawMax = valid.length ? Math.max(...valid.map(d => d.close)) : 1;
  const padding = (rawMax - rawMin) * 0.06 || rawMin * 0.02 || 1;
  const min = rawMin - padding;
  const max = rawMax + padding;
  const priceRange = max - min;

  const px = (i: number) => PAD.left + (i / Math.max(valid.length - 1, 1)) * CW;
  const py = (c: number) => PAD.top + (1 - (c - min) / priceRange) * CH;

  const linePoints = valid.map((d, i) => `${px(i).toFixed(1)},${py(d.close).toFixed(1)}`).join(" ");
  const areaPoints = valid.length >= 2
    ? `${px(0)},${PAD.top + CH} ` +
      valid.map((d, i) => `${px(i).toFixed(1)},${py(d.close).toFixed(1)}`).join(" ") +
      ` ${px(valid.length - 1)},${PAD.top + CH}`
    : "";

  const isUp = valid.length >= 2 && valid[valid.length - 1].close >= valid[0].close;
  const stroke = isUp ? "#22c55e" : "#ef4444";
  const pct = valid.length >= 2
    ? ((valid[valid.length - 1].close - valid[0].close) / valid[0].close) * 100
    : 0;

  const yTicks = [0, 0.2, 0.4, 0.6, 0.8, 1].map(r => ({
    y: PAD.top + (1 - r) * CH,
    price: min + r * priceRange,
  }));

  const xLabelCount = 5;
  const xLabels = valid.length >= 2
    ? Array.from({ length: xLabelCount }, (_, k) => {
        const i = Math.round((k / (xLabelCount - 1)) * (valid.length - 1));
        return { x: px(i), label: valid[i]?.date.slice(5) ?? "" };
      })
    : [];

  const handleMouseMove = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (!svgRef.current || valid.length < 2) return;
    const rect = svgRef.current.getBoundingClientRect();
    const scaleX = W / rect.width;
    const mx = (e.clientX - rect.left) * scaleX;
    const fraction = (mx - PAD.left) / CW;
    const idx = Math.round(Math.max(0, Math.min(1, fraction)) * (valid.length - 1));
    const point = valid[idx];
    if (!point) return;
    setTooltip({ x: px(idx), y: py(point.close), date: point.date, price: point.close, visible: true });
  }, [valid, px, py]);

  const handleMouseLeave = useCallback(() => {
    setTooltip(t => ({ ...t, visible: false }));
  }, []);

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden">
      {/* Header bar */}
      <div className="flex items-center justify-between px-5 pt-4 pb-2">
        <div className="flex items-center gap-3">
          {!loading && valid.length >= 2 && (
            <>
              <span className="font-mono font-bold text-lg text-white">
                ¥{valid[valid.length - 1].close.toLocaleString("ja-JP", { maximumFractionDigits: 0 })}
              </span>
              <span className={`font-mono text-sm font-bold px-2 py-0.5 rounded ${
                isUp ? "bg-green-900/60 text-green-400" : "bg-red-900/60 text-red-400"
              }`}>
                {pct >= 0 ? "+" : ""}{pct.toFixed(2)}%
              </span>
            </>
          )}
        </div>
        {/* Period buttons */}
        <div className="flex gap-1">
          {(["1M", "3M", "6M", "1Y"] as Period[]).map(p => (
            <button
              key={p}
              onClick={() => setPeriod(p)}
              className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all ${
                period === p
                  ? "bg-green-700 text-white shadow-sm"
                  : "bg-gray-800/80 text-gray-400 hover:bg-gray-700 hover:text-white"
              }`}
              suppressHydrationWarning
            >
              {chartPeriods[p]}
            </button>
          ))}
        </div>
      </div>

      {/* SVG */}
      <div className="relative">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-900/70 z-10">
            <span className="text-gray-500 text-sm">読み込み中...</span>
          </div>
        )}
        {!loading && valid.length < 2 ? (
          <div className="flex items-center justify-center text-gray-600 text-sm" style={{ height: `${H}px` }}>
            データなし
          </div>
        ) : (
          <svg
            ref={svgRef}
            viewBox={`0 0 ${W} ${H}`}
            className="w-full cursor-crosshair"
            style={{ height: `${H}px` }}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
          >
            <defs>
              <linearGradient id={`g-${code}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={stroke} stopOpacity="0.3" />
                <stop offset="80%" stopColor={stroke} stopOpacity="0.04" />
              </linearGradient>
            </defs>

            {/* Grid lines */}
            {yTicks.map(({ y, price }) => (
              <g key={price}>
                <line x1={PAD.left} y1={y} x2={W - PAD.right} y2={y}
                  stroke="#1f2937" strokeWidth="1" />
                <text x={PAD.left - 8} y={y + 4} textAnchor="end" fontSize="10" fill="#6b7280"
                  fontFamily="ui-monospace, SFMono-Regular, monospace">
                  ¥{price >= 1000
                    ? price.toLocaleString("ja-JP", { maximumFractionDigits: 0 })
                    : price.toFixed(price < 10 ? 2 : 1)}
                </text>
              </g>
            ))}

            {/* Area fill */}
            {areaPoints && <polygon points={areaPoints} fill={`url(#g-${code})`} />}

            {/* Price line */}
            {linePoints && (
              <polyline points={linePoints} fill="none" stroke={stroke} strokeWidth="2"
                strokeLinecap="round" strokeLinejoin="round" />
            )}

            {/* End dot */}
            {valid.length >= 2 && (
              <circle cx={px(valid.length - 1)} cy={py(valid[valid.length - 1].close)}
                r="4" fill={stroke} stroke="#0a0f1a" strokeWidth="2" />
            )}

            {/* X-axis labels */}
            {xLabels.map(({ x, label }, i) => (
              <text key={i} x={x} y={H - 8} textAnchor="middle" fontSize="10" fill="#6b7280"
                fontFamily="ui-monospace, SFMono-Regular, monospace">
                {label}
              </text>
            ))}

            {/* Hover crosshair + tooltip */}
            {tooltip.visible && (
              <g>
                <line x1={tooltip.x} y1={PAD.top} x2={tooltip.x} y2={PAD.top + CH}
                  stroke="#4b5563" strokeWidth="1" strokeDasharray="4,3" />
                <circle cx={tooltip.x} cy={tooltip.y} r="5"
                  fill={stroke} stroke="#0a0f1a" strokeWidth="2" />
                {/* Tooltip box */}
                {(() => {
                  const boxW = 134;
                  const boxH = 44;
                  const boxX = tooltip.x + 12 + boxW > W - PAD.right
                    ? tooltip.x - boxW - 12
                    : tooltip.x + 12;
                  const boxY = Math.max(PAD.top + 4, Math.min(tooltip.y - boxH / 2, PAD.top + CH - boxH));
                  return (
                    <g>
                      <rect x={boxX} y={boxY} width={boxW} height={boxH} rx="6"
                        fill="#1f2937" stroke="#374151" strokeWidth="1" />
                      <text x={boxX + 10} y={boxY + 16} fontSize="10" fill="#9ca3af"
                        fontFamily="ui-monospace, SFMono-Regular, monospace">
                        {tooltip.date}
                      </text>
                      <text x={boxX + 10} y={boxY + 33} fontSize="13" fontWeight="bold" fill="#f9fafb"
                        fontFamily="ui-monospace, SFMono-Regular, monospace">
                        ¥{tooltip.price.toLocaleString("ja-JP", { maximumFractionDigits: 0 })}
                      </text>
                    </g>
                  );
                })()}
              </g>
            )}
          </svg>
        )}
      </div>
    </div>
  );
}
