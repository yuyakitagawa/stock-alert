"use client";
import { useState } from "react";
import type { Ranking } from "@/lib/types";
import RecommendBadge from "./RecommendBadge";

const FILTERS = ["すべて", "S買い", "A買い", "買い継続", "買い増し", "売り検討"] as const;
type Filter = (typeof FILTERS)[number];

function fmt(n: number | null, digits = 1) {
  if (n == null) return "—";
  return n.toFixed(digits);
}

function signFmt(n: number | null) {
  if (n == null) return "—";
  return (n >= 0 ? "+" : "") + n.toFixed(1) + "%";
}

export default function RankingsTable({ rows }: { rows: Ranking[] }) {
  const [filter, setFilter] = useState<Filter>("すべて");

  const filtered = filter === "すべて" ? rows : rows.filter((r) => r.recommend === filter);

  return (
    <div>
      {/* フィルタータブ */}
      <div className="flex gap-2 mb-4 flex-wrap">
        {FILTERS.map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-3 py-1 rounded text-sm transition ${
              filter === f
                ? "bg-green-700 text-white"
                : "bg-gray-700 text-gray-300 hover:bg-gray-600"
            }`}
          >
            {f}
            {f !== "すべて" && (
              <span className="ml-1 text-xs opacity-70">
                ({rows.filter((r) => r.recommend === f).length})
              </span>
            )}
          </button>
        ))}
      </div>

      {/* テーブル */}
      <div className="overflow-x-auto rounded-lg border border-gray-700">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-gray-800 text-gray-400 text-left">
              <th className="px-3 py-2">#</th>
              <th className="px-3 py-2">銘柄</th>
              <th className="px-3 py-2">シグナル</th>
              <th className="px-3 py-2 text-right">株価</th>
              <th className="px-3 py-2 text-right">ネット</th>
              <th className="px-3 py-2 text-right">上昇%</th>
              <th className="px-3 py-2 text-right">下落%</th>
              <th className="px-3 py-2 text-right hidden md:table-cell">日経比20日</th>
              <th className="px-3 py-2 text-right hidden lg:table-cell">PER</th>
              <th className="px-3 py-2 text-right hidden lg:table-cell">PBR</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r) => (
              <tr
                key={r.code}
                className="border-t border-gray-700 hover:bg-gray-800 transition"
              >
                <td className="px-3 py-2 text-gray-500">{r.rank}</td>
                <td className="px-3 py-2">
                  <div className="font-medium">{r.name}</div>
                  <div className="text-xs text-gray-500">{r.code}</div>
                </td>
                <td className="px-3 py-2">
                  <RecommendBadge value={r.recommend} />
                </td>
                <td className="px-3 py-2 text-right font-mono">
                  ¥{r.close?.toLocaleString() ?? "—"}
                </td>
                <td
                  className={`px-3 py-2 text-right font-mono font-bold ${
                    r.net >= 0 ? "text-green-400" : "text-red-400"
                  }`}
                >
                  {signFmt(r.net)}
                </td>
                <td className="px-3 py-2 text-right font-mono text-green-400">
                  {fmt(r.rise_prob)}%
                </td>
                <td className="px-3 py-2 text-right font-mono text-red-400">
                  {fmt(r.drop_prob)}%
                </td>
                <td className="px-3 py-2 text-right font-mono hidden md:table-cell">
                  {signFmt(r.rel20)}
                </td>
                <td className="px-3 py-2 text-right font-mono text-gray-400 hidden lg:table-cell">
                  {r.per != null ? `${fmt(r.per)}x` : "—"}
                </td>
                <td className="px-3 py-2 text-right font-mono text-gray-400 hidden lg:table-cell">
                  {r.pbr != null ? `${fmt(r.pbr)}x` : "—"}
                </td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={10} className="px-3 py-8 text-center text-gray-500">
                  該当銘柄なし
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-xs text-gray-600 text-right">
        {filtered.length} 銘柄表示
      </p>
    </div>
  );
}
