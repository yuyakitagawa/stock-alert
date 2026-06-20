"use client";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import type { Ranking, WatchMetrics } from "@/lib/types";
import { useBookmarks } from "@/lib/bookmarks";
import { OWNER_HOLDINGS } from "@/lib/owner-holdings";
import { signFmtArrow } from "@/lib/signals";
import { PRICING_POWER_META } from "@/lib/watchlist";
import Sparkline from "@/components/Sparkline";
import BookmarkButton from "@/components/BookmarkButton";

interface Props {
  rankMap: Record<string, Ranking>;
  sectorMap: Record<string, string>;
  asOf: string;
}

function NetBadge({ net }: { net: number | null }) {
  if (net == null) return <span className="text-gray-600 text-sm">—</span>;
  const up = net >= 0;
  return (
    <span className={`font-mono text-sm font-semibold ${up ? "text-green-400" : "text-red-400"}`}>
      {signFmtArrow(net)}
    </span>
  );
}

function ProbBreakdown({ r }: { r: Ranking | undefined }) {
  if (!r) return null;
  return (
    <div className="flex items-center justify-end gap-2 text-xs font-mono mt-0.5">
      <span className="text-green-400">↑{r.rise_prob.toFixed(0)}%</span>
      <span className="text-red-400">↓{r.drop_prob.toFixed(0)}%</span>
    </div>
  );
}

function OverseasBar({ ratio }: { ratio: number | undefined }) {
  if (ratio == null) return <span className="text-gray-600 text-sm">—</span>;
  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-1.5 rounded-full bg-gray-800 overflow-hidden">
        <div className="h-full bg-blue-400" style={{ width: `${Math.min(ratio, 100)}%` }} />
      </div>
      <span className="font-mono text-xs text-gray-400 tabular-nums">{ratio}%</span>
    </div>
  );
}

// 52週高値からの下落率を「お得度」として評価
function bargain(drawdownPct: number | null) {
  if (drawdownPct == null) return null;
  const d = drawdownPct; // 負の値ほど安い
  if (d <= -30) return { label: "🔥 大お得", cls: "bg-emerald-900/50 text-emerald-300 border-emerald-700" };
  if (d <= -20) return { label: "お得",      cls: "bg-green-900/40 text-green-300 border-green-800" };
  if (d <= -10) return { label: "やや安",    cls: "bg-yellow-900/30 text-yellow-300 border-yellow-800" };
  return { label: "高値圏", cls: "bg-gray-800 text-gray-400 border-gray-700" };
}

function DrawdownCell({ m }: { m: WatchMetrics | undefined }) {
  if (!m || m.drawdownPct == null) return <span className="text-gray-600 text-sm">—</span>;
  const b = bargain(m.drawdownPct)!;
  return (
    <div className="flex items-center gap-2">
      <span className={`shrink-0 text-xs font-bold px-2 py-0.5 rounded border ${b.cls}`}>{b.label}</span>
      <span className="font-mono text-sm text-gray-300 tabular-nums">{m.drawdownPct.toFixed(0)}%</span>
    </div>
  );
}

function fmt(v: number | null | undefined, suffix = "") {
  return v == null ? "—" : `${v.toFixed(1)}${suffix}`;
}

// PER/PBR は自前データ(gen_rankings)を優先し、欠損時のみYahooで補完
function valuation(r: Ranking | undefined, m: WatchMetrics | undefined) {
  return { per: r?.per ?? m?.per ?? null, pbr: r?.pbr ?? m?.pbr ?? null };
}

// 直近1ヶ月のミニチャート（上昇=緑/下落=赤）
function MiniChart({ spark }: { spark: number[] | undefined }) {
  if (!spark || spark.length < 2) return null;
  const up = spark[spark.length - 1] >= spark[0];
  return (
    <div className="w-28">
      <Sparkline prices={spark} color={up ? "#22c55e" : "#ef4444"} showLabel />
    </div>
  );
}

function OwnerImportButton({ codes, addMany }: { codes: string[]; addMany: (c: string[]) => void }) {
  const missing = OWNER_HOLDINGS.filter((c) => !codes.includes(c));
  if (missing.length === 0) return null;
  return (
    <button
      onClick={() => addMany(OWNER_HOLDINGS)}
      className="text-sm font-medium px-4 py-2 rounded-lg bg-blue-500/10 border border-blue-700 text-blue-300 hover:bg-blue-500/20 transition-colors"
    >
      保有株を取り込む（{missing.length}銘柄）
    </button>
  );
}

export default function WatchlistClient({ rankMap, sectorMap, asOf }: Props) {
  const { codes, mounted, addMany } = useBookmarks();
  const [metrics, setMetrics] = useState<Record<string, WatchMetrics>>({});

  // ブックマーク銘柄の お得度/ミニチャート/PER-PBR を API 経由で取得
  useEffect(() => {
    if (codes.length === 0) {
      setMetrics({});
      return;
    }
    let aborted = false;
    fetch(`/api/watch-metrics?codes=${encodeURIComponent(codes.join(","))}`)
      .then((res) => (res.ok ? res.json() : {}))
      .then((data) => { if (!aborted) setMetrics(data as Record<string, WatchMetrics>); })
      .catch(() => { if (!aborted) setMetrics({}); });
    return () => { aborted = true; };
  }, [codes]);

  const items = useMemo(
    () => codes.map((code) => ({
      code,
      r: rankMap[code],
      sector: sectorMap[code],
      m: metrics[code],
      meta: PRICING_POWER_META[code],
    })),
    [codes, rankMap, sectorMap, metrics],
  );

  if (!mounted) {
    return <div className="h-40 rounded-xl bg-gray-900/40 border border-gray-800 animate-pulse" />;
  }

  if (codes.length === 0) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-xl px-6 py-12 text-center">
        <div className="text-4xl mb-3">🔖</div>
        <p className="text-gray-300 font-medium">ブックマークはまだありません</p>
        <p className="text-sm text-gray-500 mt-2 leading-relaxed">
          気になる銘柄ページや
          <Link href="/rankings" className="text-green-400 hover:underline mx-1">ランキング</Link>
          のしおりアイコン <span className="inline-block align-middle text-gray-400">🔖</span> を押すと、ここに保存されます。
        </p>
        <div className="mt-5 flex flex-wrap items-center justify-center gap-3">
          <Link
            href="/rankings"
            className="text-sm font-medium px-4 py-2 rounded-lg bg-green-500/10 border border-green-700 text-green-400 hover:bg-green-500/20 transition-colors"
          >
            ランキングから探す →
          </Link>
          <OwnerImportButton codes={codes} addMany={addMany} />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-end">
        <OwnerImportButton codes={codes} addMany={addMany} />
      </div>
      {/* デスクトップ: テーブル */}
      <div className="hidden md:block bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-xs text-gray-500 border-b border-gray-800">
              <th className="text-left font-medium px-4 py-3">銘柄</th>
              <th className="text-left font-medium px-4 py-3">独占商品</th>
              <th className="text-left font-medium px-4 py-3">独占率</th>
              <th className="text-left font-medium px-4 py-3">海外比率</th>
              <th className="text-left font-medium px-4 py-3">高値から（お得度）</th>
              <th className="text-right font-medium px-4 py-3">PER / PBR</th>
              <th className="text-right font-medium px-4 py-3">netスコア（上昇↑/下落↓）</th>
              <th className="w-10 px-2 py-3"><span className="sr-only">削除</span></th>
            </tr>
          </thead>
          <tbody>
            {items.map(({ code, r, sector, m, meta }) => (
              <tr key={code} className="border-b border-gray-800 last:border-0 hover:bg-gray-800/40 transition-colors">
                <td className="px-4 py-3 align-top">
                  <Link href={`/stocks/${code}`} className="group">
                    <div className="font-medium text-white group-hover:text-green-400 transition-colors">{r?.name ?? code}</div>
                    <div className="text-xs text-gray-600 font-mono mb-1">
                      {code}{sector && ` · ${sector}`}
                    </div>
                    <MiniChart spark={m?.spark} />
                  </Link>
                </td>
                <td className="px-4 py-3 align-top">
                  {meta ? (
                    <>
                      <div className="text-gray-300">{meta.product}</div>
                      <div className="text-xs text-gray-600 mt-0.5">{meta.note}</div>
                    </>
                  ) : (
                    <span className="text-gray-600 text-sm">—</span>
                  )}
                </td>
                <td className="px-4 py-3 align-top text-gray-300">{meta?.domesticShare ?? <span className="text-gray-600 text-sm">—</span>}</td>
                <td className="px-4 py-3 align-top"><OverseasBar ratio={meta?.overseasRatio} /></td>
                <td className="px-4 py-3 align-top"><DrawdownCell m={m} /></td>
                <td className="px-4 py-3 align-top text-right font-mono text-gray-300 tabular-nums">
                  {(() => { const v = valuation(r, m); return `${fmt(v.per, "x")} / ${fmt(v.pbr, "x")}`; })()}
                </td>
                <td className="px-4 py-3 align-top text-right">
                  {r ? (
                    <>
                      <NetBadge net={r.net} />
                      <ProbBreakdown r={r} />
                      {r.recommend && <div className="text-xs text-gray-600 mt-0.5">{r.recommend}</div>}
                    </>
                  ) : (
                    <span className="text-xs text-gray-600">データ未取得</span>
                  )}
                </td>
                <td className="px-2 py-3 align-top text-center"><BookmarkButton code={code} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* モバイル: カード */}
      <div className="md:hidden space-y-3">
        {items.map(({ code, r, sector, m, meta }) => (
          <Link
            key={code}
            href={`/stocks/${code}`}
            className="block bg-gray-900 border border-gray-800 rounded-xl p-4 hover:bg-gray-800/40 transition-colors"
          >
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="font-medium text-white truncate">{r?.name ?? code}</div>
                <div className="text-xs text-gray-600 font-mono">
                  {code}{sector && ` · ${sector}`}
                </div>
              </div>
              <div className="flex items-start gap-2 shrink-0">
                <DrawdownCell m={m} />
                <BookmarkButton code={code} className="-mr-1 -mt-1" />
              </div>
            </div>
            {m?.spark && m.spark.length >= 2 && (
              <div className="mt-2"><MiniChart spark={m.spark} /></div>
            )}
            {meta && (
              <>
                <div className="text-sm text-gray-300 mt-2">{meta.product}</div>
                <div className="text-xs text-gray-400 mt-0.5">独占率: <span className="text-gray-300">{meta.domesticShare}</span></div>
                <div className="text-xs text-gray-600 mt-0.5">{meta.note}</div>
              </>
            )}
            <div className="flex items-center justify-between mt-3 text-xs">
              <span className="font-mono text-gray-400">
                {(() => { const v = valuation(r, m); return `PER ${fmt(v.per, "x")} / PBR ${fmt(v.pbr, "x")}`; })()}
              </span>
              {meta && <OverseasBar ratio={meta.overseasRatio} />}
            </div>
            <div className="flex items-center justify-between mt-2 text-xs">
              <span className="text-gray-600">netスコア</span>
              {r ? (
                <div className="text-right">
                  <NetBadge net={r.net} />
                  <ProbBreakdown r={r} />
                </div>
              ) : (
                <span className="text-gray-600">データ未取得</span>
              )}
            </div>
          </Link>
        ))}
      </div>

      <p className="text-xs text-gray-600 leading-relaxed">
        ※ ブックマークはこの端末/ブラウザにのみ保存されます（ログイン不要・しおりアイコンで追加/削除）。
        チャートは直近1ヶ月の終値（緑=上昇 / 赤=下落）。「お得度」は52週高値からの下落率で判定
        （−30%↓=🔥大お得 / −20%↓=お得 / −10%↓=やや安 / それ以外=高値圏）。株価・チャート・52週高値は Yahoo Finance、
        PER は当日ランキングのファンダ値（一部はYahoo補完）。netスコア（上昇確率−下落確率）は {asOf || "—"} 時点のAIモデル値。
        当日の市場価格データが取得できない銘柄（上場廃止・売買停止など）は「データ未取得」と表示します。
      </p>
    </div>
  );
}
