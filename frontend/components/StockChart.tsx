"use client";
import { useState, useEffect, useRef, useCallback } from "react";
import { useLang } from "@/contexts/LanguageContext";
import { UI } from "@/lib/i18n";

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

const W = 600;
const H = 220;
const PAD = { top: 16, right: 16, bottom: 28, left: 60 };
const CW = W - PAD.left - PAD.right;
const CH = H - PAD.top - PAD.bottom;

export default function StockChart({ code }: Props) {
  const { lang } = useLang();
  const ui = UI[lang];
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
  const min = valid.length ? Math.min(...valid.map(d => d.close)) * 0.998 : 0;
  const max = valid.length ? Math.max(...valid.map(d => d.close)) * 1.002 : 1;
  const priceRange = max - min || 1;

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

  const yTicks = [0, 0.25, 0.5, 0.75, 1].map(r => ({
    y: PAD.top + (1 - r) * CH,
    price: min + r * priceRange,
  }));

  const xLabels = valid.length >= 2
    ? [0, Math.floor(valid.length / 2), valid.length - 1].map(i => ({
        x: px(i),
        label: valid[i]?.date.slice(5) ?? "",
      }))
    : [];

  const handleMouseMove = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (!svgRef.current || valid.length < 2) return;
    const rect = svgRef.current.getBoundingClientRect();
    const scaleX = W / rect.width;
    const mx = (e.clientX - rect.left) * scaleX;
    const fraction = (mx - PAD.left) / CW;
    const idx = Math.round(fraction * (valid.length - 1));
    const clamped = Math.max(0, Math.min(valid.length - 1, idx));
    const point = valid[clamped];
    if (!point) return;
    setTooltip({
      x: px(clamped),
      y: py(point.close),
      date: point.date,
      price: point.close,
      visible: true,
    });
  }, [valid, px, py]);

  const handleMouseLeave = useCallback(() => {
    setTooltip(t => ({ ...t, visible: false }));
  }, []);

  return (
    <div className="space-y-3">
      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="flex gap-1">
          {(["1M", "3M", "6M", "1Y"] as Period[]).map(p => (
            <button
              key={p}
              onClick={() => setPeriod(p)}
              className={`px-3 py-1 rounded text-xs font-semibold transition-colors ${
                period === p
                  ? "bg-green-700 text-white"
                  : "bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-white"
              }`}
              suppressHydrationWarning
            >
              {ui.chartPeriods[p]}
            </button>
          ))}
        </div>
        {!loading && valid.length >= 2 && (
          <span className={`text-sm font-mono font-bold ${isUp ? "text-green-400" : "text-red-400"}`}>
            {pct >= 0 ? "+" : ""}{pct.toFixed(2)}%
          </span>
        )}
      </div>

      {/* Chart */}
      <div className="relative bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80 z-10">
            <span className="text-gray-500 text-sm">{ui.loading}</span>
          </div>
        )}
        {!loading && valid.length < 2 ? (
          <div className="h-56 flex items-center justify-center text-gray-600 text-sm">
            データなし
          </div>
        ) : (
          <svg
            ref={svgRef}
            viewBox={`0 0 ${W} ${H}`}
            className="w-full cursor-crosshair"
            style={{ height: "220px" }}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
          >
            <defs>
              <linearGradient id={`g-${code}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={stroke} stopOpacity="0.22" />
                <stop offset="100%" stopColor={stroke} stopOpacity="0.02" />
              </linearGradient>
            </defs>

            {/* Grid */}
            {yTicks.map(({ y, price }) => (
              <g key={price}>
                <line x1={PAD.left} y1={y} x2={W - PAD.right} y2={y} stroke="#1f2937" strokeWidth="1" />
                <text x={PAD.left - 6} y={y + 4} textAnchor="end" fontSize="9" fill="#6b7280">
                  ¥{price >= 1000
                    ? price.toLocaleString("ja-JP", { maximumFractionDigits: 0 })
                    : price.toFixed(price < 10 ? 2 : 1)}
                </text>
              </g>
            ))}

            {/* Area */}
            {areaPoints && <polygon points={areaPoints} fill={`url(#g-${code})`} />}

            {/* Line */}
            {linePoints && (
              <polyline points={linePoints} fill="none" stroke={stroke} strokeWidth="1.5"
                strokeLinecap="round" strokeLinejoin="round" />
            )}

            {/* End dot */}
            {valid.length >= 2 && (
              <circle cx={px(valid.length - 1)} cy={py(valid[valid.length - 1].close)} r="3" fill={stroke} />
            )}

            {/* X labels */}
            {xLabels.map(({ x, label }) => (
              <text key={label} x={x} y={H - 6} textAnchor="middle" fontSize="9" fill="#6b7280">
                {label}
              </text>
            ))}

            {/* Tooltip crosshair */}
            {tooltip.visible && (
              <g>
                <line x1={tooltip.x} y1={PAD.top} x2={tooltip.x} y2={PAD.top + CH}
                  stroke="#6b7280" strokeWidth="1" strokeDasharray="3,3" />
                <circle cx={tooltip.x} cy={tooltip.y} r="4" fill={stroke} stroke="#0a0f1a" strokeWidth="1.5" />
                {/* Tooltip box */}
                {(() => {
                  const boxW = 120;
                  const boxH = 38;
                  const boxX = tooltip.x + 8 + boxW > W - PAD.right
                    ? tooltip.x - boxW - 8
                    : tooltip.x + 8;
                  const boxY = Math.max(PAD.top, tooltip.y - boxH / 2);
                  return (
                    <g>
                      <rect x={boxX} y={boxY} width={boxW} height={boxH} rx="4"
                        fill="#1f2937" stroke="#374151" strokeWidth="1" />
                      <text x={boxX + 8} y={boxY + 14} fontSize="10" fill="#9ca3af">
                        {tooltip.date}
                      </text>
                      <text x={boxX + 8} y={boxY + 28} fontSize="11" fontWeight="bold" fill="#f9fafb">
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
