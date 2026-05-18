"use client";
import { useState, useEffect } from "react";
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

const W = 600;
const H = 200;
const PAD = { top: 12, right: 12, bottom: 28, left: 56 };
const CW = W - PAD.left - PAD.right;
const CH = H - PAD.top - PAD.bottom;

export default function StockChart({ code }: Props) {
  const { lang } = useLang();
  const ui = UI[lang];
  const [period, setPeriod] = useState<Period>("1M");
  const [data, setData] = useState<DataPoint[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetch(`/api/chart/${code}?period=${period}`)
      .then(r => r.json())
      .then((d: DataPoint[]) => { setData(Array.isArray(d) ? d : []); setLoading(false); })
      .catch(() => setLoading(false));
  }, [period, code]);

  const valid = data.filter(d => d.close != null);

  const min = valid.length ? Math.min(...valid.map(d => d.close)) : 0;
  const max = valid.length ? Math.max(...valid.map(d => d.close)) : 1;
  const priceRange = max - min || 1;

  const px = (i: number) => PAD.left + (i / Math.max(valid.length - 1, 1)) * CW;
  const py = (c: number) => PAD.top + (1 - (c - min) / priceRange) * CH;

  const linePoints = valid.map((d, i) => `${px(i).toFixed(1)},${py(d.close).toFixed(1)}`).join(" ");

  const areaPoints = valid.length >= 2
    ? `${px(0).toFixed(1)},${(PAD.top + CH).toFixed(1)} ` +
      valid.map((d, i) => `${px(i).toFixed(1)},${py(d.close).toFixed(1)}`).join(" ") +
      ` ${px(valid.length - 1).toFixed(1)},${(PAD.top + CH).toFixed(1)}`
    : "";

  const isUp = valid.length >= 2 && valid[valid.length - 1].close >= valid[0].close;
  const stroke = isUp ? "#22c55e" : "#ef4444";
  const pct = valid.length >= 2
    ? ((valid[valid.length - 1].close - valid[0].close) / valid[0].close) * 100
    : 0;

  const xLabels = valid.length >= 2
    ? [0, Math.floor(valid.length / 2), valid.length - 1].map(i => ({
        x: px(i),
        label: valid[i].date.slice(5),
      }))
    : [];

  const yTicks = [0, 0.25, 0.5, 0.75, 1].map(r => ({
    y: PAD.top + (1 - r) * CH,
    price: min + r * priceRange,
  }));

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
          <div className="h-48 flex items-center justify-center text-gray-600 text-sm">
            データなし
          </div>
        ) : (
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height: "200px" }}>
            <defs>
              <linearGradient id={`grad-${code}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={stroke} stopOpacity="0.18" />
                <stop offset="100%" stopColor={stroke} stopOpacity="0.02" />
              </linearGradient>
            </defs>

            {/* Grid lines */}
            {yTicks.map(({ y, price }) => (
              <g key={price}>
                <line
                  x1={PAD.left} y1={y}
                  x2={W - PAD.right} y2={y}
                  stroke="#1f2937" strokeWidth="1"
                />
                <text
                  x={PAD.left - 4} y={y + 4}
                  textAnchor="end" fontSize="9" fill="#6b7280"
                >
                  ¥{price.toLocaleString("ja-JP", { maximumFractionDigits: 0 })}
                </text>
              </g>
            ))}

            {/* Area fill */}
            {areaPoints && (
              <polygon
                points={areaPoints}
                fill={`url(#grad-${code})`}
              />
            )}

            {/* Price line */}
            {linePoints && (
              <polyline
                points={linePoints}
                fill="none"
                stroke={stroke}
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            )}

            {/* End dot */}
            {valid.length >= 2 && (
              <circle
                cx={px(valid.length - 1)}
                cy={py(valid[valid.length - 1].close)}
                r="3"
                fill={stroke}
              />
            )}

            {/* X-axis labels */}
            {xLabels.map(({ x, label }) => (
              <text
                key={label}
                x={x} y={H - 6}
                textAnchor="middle" fontSize="9" fill="#6b7280"
              >
                {label}
              </text>
            ))}
          </svg>
        )}
      </div>
    </div>
  );
}
