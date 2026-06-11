"use client";
import { useState, useMemo } from "react";
import Link from "next/link";
import type { Ranking } from "@/lib/types";
import { useLang } from "@/contexts/LanguageContext";
import { UI } from "@/lib/i18n";

type SortKey = "rank" | "net" | "rise_prob" | "drop_prob" | "rel20" | "close";

function fmt(n: number | null, digits = 1) {
  if (n == null) return "—";
  return n.toFixed(digits);
}
function signFmt(n: number | null) {
  if (n == null) return "—";
  return (n >= 0 ? "+" : "") + n.toFixed(1) + "%";
}

function SortIcon({ active, dir }: { active: boolean; dir: "asc" | "desc" }) {
  if (!active) return <span className="ml-1 opacity-20">⇅</span>;
  return <span className="ml-1 text-green-400">{dir === "asc" ? "↑" : "↓"}</span>;
}

interface Props {
  rows: Ranking[];
  sectorMap?: Record<string, string>;
}

export default function RankingsTable({ rows, sectorMap }: Props) {
  const { lang } = useLang();
  const ui = UI[lang];

  const [search,  setSearch]  = useState("");
  const [sector,  setSector]  = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("rank");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");

  function handleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDir(d => d === "asc" ? "desc" : "asc");
    } else {
      setSortKey(key);
      setSortDir(key === "rank" ? "asc" : "desc");
    }
  }

  const sectors = useMemo(() => {
    if (!sectorMap) return [];
    const set = new Set<string>();
    rows.forEach(r => { const s = sectorMap[r.code]; if (s) set.add(s); });
    return Array.from(set).sort();
  }, [rows, sectorMap]);

  const filtered = useMemo(() => {
    let r = rows;
    if (sector) r = r.filter(x => sectorMap?.[x.code] === sector);
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      r = r.filter(x =>
        x.name.toLowerCase().includes(q) || x.code.toLowerCase().includes(q)
      );
    }
    return [...r].sort((a, b) => {
      const av = a[sortKey] ?? (sortDir === "asc" ? Infinity : -Infinity);
      const bv = b[sortKey] ?? (sortDir === "asc" ? Infinity : -Infinity);
      return sortDir === "asc" ? (av as number) - (bv as number) : (bv as number) - (av as number);
    });
  }, [rows, sector, search, sortKey, sortDir, sectorMap]);

  const ThSort = ({ col, label, className = "" }: { col: SortKey; label: string; className?: string }) => (
    <th
      className={`px-3 py-3 text-xs font-semibold uppercase tracking-wide cursor-pointer select-none hover:text-white transition-colors whitespace-nowrap ${className}`}
      onClick={() => handleSort(col)}
    >
      {label}
      <SortIcon active={sortKey === col} dir={sortDir} />
    </th>
  );

  return (
    <div className="space-y-4">
      {/* Sector + Search */}
      <div className="flex flex-col sm:flex-row gap-2">
        {sectors.length > 0 && (
          <select
            value={sector}
            onChange={e => setSector(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-200 focus:outline-none focus:border-green-600 focus:ring-1 focus:ring-green-600/30 sm:w-52"
            suppressHydrationWarning
          >
            <option value="" suppressHydrationWarning>{ui.allSectors}</option>
            {sectors.map(s => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        )}
        <input
          type="search"
          placeholder={ui.searchPlaceholder}
          value={search}
          onChange={e => setSearch(e.target.value)}
          className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-green-600 focus:ring-1 focus:ring-green-600/30"
          suppressHydrationWarning
        />
      </div>

      {/* Desktop table */}
      <div className="hidden md:block overflow-x-auto rounded-xl border border-gray-800">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-gray-900/80 text-gray-500 text-left border-b border-gray-800">
              <ThSort col="rank"      label="#"         className="w-10 text-center" />
              <th className="px-3 py-3 text-xs font-semibold uppercase tracking-wide">銘柄</th>
              <ThSort col="close"     label="株価"      className="text-right" />
              <ThSort col="net"       label="ネット"    className="text-right" />
              <ThSort col="rise_prob" label="上昇確率"   className="text-right" />
              <ThSort col="drop_prob" label="下落確率"   className="text-right" />
              <ThSort col="rel20"     label="日経比20日" className="text-right hidden lg:table-cell" />
              <th className="px-3 py-3 text-xs font-semibold uppercase tracking-wide text-right hidden xl:table-cell">PER</th>
              <th className="px-3 py-3 text-xs font-semibold uppercase tracking-wide text-right hidden xl:table-cell">PBR</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800/60">
            {filtered.map(r => (
              <tr key={r.code} className="hover:bg-gray-800/50 transition-colors group">
                <td className="px-3 py-2.5 text-center text-gray-600 font-mono text-xs">{r.rank}</td>
                <td className="px-3 py-2.5">
                  <Link href={`/stocks/${r.code}`} className="group-hover:text-green-400 transition-colors">
                    <div className="font-semibold text-white">{r.name}</div>
                    <div className="text-xs text-gray-500 font-mono">
                      {r.code}
                      {sectorMap?.[r.code] && (
                        <span className="ml-2 text-gray-600">{sectorMap[r.code]}</span>
                      )}
                    </div>
                  </Link>
                </td>
                <td className="px-3 py-2.5 text-right font-mono text-gray-200">
                  ¥{r.close?.toLocaleString() ?? "—"}
                </td>
                <td className={`px-3 py-2.5 text-right font-mono font-bold ${r.net >= 0 ? "text-green-400" : "text-red-400"}`}>
                  {signFmt(r.net)}
                </td>
                <td className="px-3 py-2.5 text-right font-mono text-green-400">
                  {fmt(r.rise_prob)}%
                </td>
                <td className="px-3 py-2.5 text-right font-mono text-red-400">
                  {fmt(r.drop_prob)}%
                </td>
                <td className="px-3 py-2.5 text-right font-mono text-gray-400 hidden lg:table-cell">
                  {signFmt(r.rel20)}
                </td>
                <td className="px-3 py-2.5 text-right font-mono text-gray-500 hidden xl:table-cell">
                  {r.per != null ? `${fmt(r.per)}x` : "—"}
                </td>
                <td className="px-3 py-2.5 text-right font-mono text-gray-500 hidden xl:table-cell">
                  {r.pbr != null ? `${fmt(r.pbr)}x` : "—"}
                </td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={9} className="px-4 py-12 text-center text-gray-600">
                  該当銘柄なし
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Mobile card list */}
      <div className="md:hidden space-y-2">
        {filtered.map(r => (
          <Link
            key={r.code}
            href={`/stocks/${r.code}`}
            className="flex items-center gap-3 bg-gray-900 border border-gray-800 rounded-xl px-4 py-3 hover:bg-gray-800 transition-colors"
          >
            <span className="text-gray-700 font-mono text-xs w-6 text-center">{r.rank}</span>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="font-semibold text-sm text-white">{r.name}</span>
              </div>
              <div className="flex gap-2 text-xs font-mono mt-0.5 text-gray-500 flex-wrap">
                <span>{r.code}</span>
                <span className="text-gray-400">¥{r.close?.toLocaleString()}</span>
                {sectorMap?.[r.code] && (
                  <span className="text-gray-600">{sectorMap[r.code]}</span>
                )}
              </div>
            </div>
            <div className="text-right shrink-0">
              <div className={`font-mono font-bold text-sm ${r.net >= 0 ? "text-green-400" : "text-red-400"}`}>
                <span className="text-[10px] text-gray-500 font-sans font-normal mr-1">ネット</span>
                {signFmt(r.net)}
              </div>
              <div className="text-xs text-gray-600 font-mono">
                上昇{fmt(r.rise_prob)}% 下落{fmt(r.drop_prob)}%
              </div>
            </div>
          </Link>
        ))}
        {filtered.length === 0 && (
          <div className="py-12 text-center text-gray-600">該当銘柄なし</div>
        )}
      </div>

      <p className="text-xs text-gray-700 text-right">{filtered.length} 銘柄</p>
    </div>
  );
}
