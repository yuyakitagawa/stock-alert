import type { Metadata } from "next";
import Link from "next/link";
import { fetchRankings, fetchWatchMetricsMap } from "@/lib/data";
import { PRICING_POWER_WATCHLIST } from "@/lib/watchlist";
import type { Ranking, WatchMetrics } from "@/lib/types";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import Sparkline from "@/components/Sparkline";

export const revalidate = 3600;

export const metadata: Metadata = {
  title: "ウォッチリスト — StockSignal",
  description: "シェアを独占しインフレ下で値上げを通せる、toC独占ブランド銘柄。長期の押し目買い候補をお得度つきで監視。",
};

function NetBadge({ net }: { net: number | null }) {
  if (net == null) return <span className="text-gray-600 text-sm">—</span>;
  const up = net >= 0;
  return (
    <span className={`font-mono text-sm font-semibold ${up ? "text-green-400" : "text-red-400"}`}>
      {up ? "+" : ""}{net.toFixed(1)}%
    </span>
  );
}

// 上昇確率・下落確率の内訳（net = 上昇 − 下落）
function ProbBreakdown({ r }: { r: Ranking | undefined }) {
  if (!r) return null;
  return (
    <div className="flex items-center justify-end gap-2 text-xs font-mono mt-0.5">
      <span className="text-green-400">↑{r.rise_prob.toFixed(0)}%</span>
      <span className="text-red-400">↓{r.drop_prob.toFixed(0)}%</span>
    </div>
  );
}

function OverseasBar({ ratio }: { ratio: number }) {
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

function fmt(v: number | null, suffix = "") {
  return v == null ? "—" : `${v.toFixed(1)}${suffix}`;
}

// PER/PBR は自前データ(web_rankings)を優先し、欠損時のみYahooで補完
function valuation(r: Ranking | undefined, m: WatchMetrics | undefined) {
  return {
    per: r?.per ?? m?.per ?? null,
    pbr: r?.pbr ?? m?.pbr ?? null,
  };
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

export default async function WatchlistPage() {
  let rankMap: Record<string, Ranking> = {};
  let asOf = "";
  try {
    const { date, rows } = await fetchRankings();
    asOf = date;
    rankMap = Object.fromEntries(rows.map(r => [String(r.code), r]));
  } catch {
    // ランキング取得失敗時も静的リストは表示する
  }

  // 各銘柄のお得度（高値からの下落率）＋ PER/PBR を並列取得（認証は1回）
  const metricsMap = await fetchWatchMetricsMap(PRICING_POWER_WATCHLIST.map(s => s.code));

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />

      <main className="flex-1 max-w-6xl mx-auto w-full px-4 sm:px-6 py-8 space-y-6">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-white">ウォッチリスト</h1>
          <p className="text-sm text-gray-600 mt-1">
            シェアを独占し、インフレ下でも値上げを通せる toC ブランド銘柄。長期で安く仕込むための「お得度」つき監視リストです。
          </p>
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
              </tr>
            </thead>
            <tbody>
              {PRICING_POWER_WATCHLIST.map((s) => {
                const r = rankMap[s.code];
                const m = metricsMap[s.code];
                return (
                  <tr key={s.code} className="border-b border-gray-800 last:border-0 hover:bg-gray-800/40 transition-colors">
                    <td className="px-4 py-3 align-top">
                      <Link href={`/stocks/${s.code}`} className="group">
                        <div className="font-medium text-white group-hover:text-green-400 transition-colors">{s.name}</div>
                        <div className="text-xs text-gray-600 font-mono mb-1">{s.code} · {s.category}</div>
                        <MiniChart spark={m?.spark} />
                      </Link>
                    </td>
                    <td className="px-4 py-3 align-top">
                      <div className="text-gray-300">{s.product}</div>
                      <div className="text-xs text-gray-600 mt-0.5">{s.note}</div>
                    </td>
                    <td className="px-4 py-3 align-top text-gray-300">{s.domesticShare}</td>
                    <td className="px-4 py-3 align-top"><OverseasBar ratio={s.overseasRatio} /></td>
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
                        <span className="text-xs text-gray-600">スクリーン対象外</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* モバイル: カード */}
        <div className="md:hidden space-y-3">
          {PRICING_POWER_WATCHLIST.map((s) => {
            const r = rankMap[s.code];
            const m = metricsMap[s.code];
            return (
              <Link
                key={s.code}
                href={`/stocks/${s.code}`}
                className="block bg-gray-900 border border-gray-800 rounded-xl p-4 hover:bg-gray-800/40 transition-colors"
              >
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="font-medium text-white">{s.name}</div>
                    <div className="text-xs text-gray-600 font-mono">{s.code} · {s.category}</div>
                  </div>
                  <DrawdownCell m={m} />
                </div>
                {m?.spark && m.spark.length >= 2 && (
                  <div className="mt-2"><MiniChart spark={m.spark} /></div>
                )}
                <div className="text-sm text-gray-300 mt-2">{s.product}</div>
                <div className="text-xs text-gray-400 mt-0.5">独占率: <span className="text-gray-300">{s.domesticShare}</span></div>
                <div className="text-xs text-gray-600 mt-0.5">{s.note}</div>
                <div className="flex items-center justify-between mt-3 text-xs">
                  <span className="font-mono text-gray-400">
                    {(() => { const v = valuation(r, m); return `PER ${fmt(v.per, "x")} / PBR ${fmt(v.pbr, "x")}`; })()}
                  </span>
                  <OverseasBar ratio={s.overseasRatio} />
                </div>
                <div className="flex items-center justify-between mt-2 text-xs">
                  <span className="text-gray-600">netスコア</span>
                  {r ? (
                    <div className="text-right">
                      <NetBadge net={r.net} />
                      <ProbBreakdown r={r} />
                    </div>
                  ) : (
                    <span className="text-gray-600">スクリーン対象外</span>
                  )}
                </div>
              </Link>
            );
          })}
        </div>

        <p className="text-xs text-gray-700 leading-relaxed">
          ※ チャートは直近1ヶ月の終値（緑=上昇 / 赤=下落）。「お得度」は52週高値からの下落率で判定
          （−30%↓=🔥大お得 / −20%↓=お得 / −10%↓=やや安 / それ以外=高値圏）。株価・チャート・52週高値は Yahoo Finance、
          PER は当日ランキングのファンダ値（一部はYahoo補完）。PBR は現状データ未整備の銘柄が多く「—」表示になります。
          シェア率・海外売上比率は公知ベースの概算で、投資判断の際は各社IRで最新値をご確認ください。
          netスコア（上昇確率−下落確率）は {asOf || "—"} 時点のAIモデル値。AIスコアはモメンタム・スクリーンを通過した銘柄のみで、
          通過しない銘柄（例: 久光製薬）は「スクリーン対象外」と表示します。
        </p>
      </main>

      <Footer />
    </div>
  );
}
