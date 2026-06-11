"use client";
import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import type { Ranking, QvPosition } from "@/lib/types";
import { netStyle, signFmtArrow } from "@/lib/signals";

const LS_KEY = "qv_portfolio_v1";

function loadPortfolio(): QvPosition[] {
  if (typeof window === "undefined") return [];
  try { return JSON.parse(localStorage.getItem(LS_KEY) || "[]"); } catch { return []; }
}
function savePortfolio(p: QvPosition[]) {
  localStorage.setItem(LS_KEY, JSON.stringify(p));
}

function fmt(n: number | null | undefined, digits = 1) {
  return n == null ? "—" : n.toFixed(digits);
}
function pioLabel(v: number | null | undefined) {
  if (v == null) return "—";
  const score = Math.round(v * 9);
  const color = score >= 7 ? "text-green-400" : score >= 5 ? "text-yellow-400" : "text-red-400";
  return <span className={`font-bold ${color}`}>{score}/9</span>;
}
function pos52Bar(v: number | null | undefined) {
  if (v == null) return null;
  const pct = Math.round(v * 100);
  const color = pct < 30 ? "bg-green-500" : pct < 50 ? "bg-yellow-500" : "bg-gray-600";
  return (
    <div className="flex items-center gap-1.5">
      <div className="w-16 h-1.5 bg-gray-700 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-gray-500">{pct}%</span>
    </div>
  );
}

interface Props {
  candidates: Ranking[];
  allRows: Ranking[];
  date: string;
  sellNetMin: number;
  sellDropMin: number;
}

export default function QvPanel({ candidates, allRows, date, sellNetMin, sellDropMin }: Props) {
  const [portfolio, setPortfolio] = useState<QvPosition[]>([]);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setPortfolio(loadPortfolio());
    setMounted(true);
  }, []);

  const addPosition = useCallback((r: Ranking) => {
    const pos: QvPosition = {
      code: r.code, name: r.name,
      buyDate: date, buyPrice: r.close, buyNet: r.net,
    };
    setPortfolio(prev => {
      if (prev.some(p => p.code === r.code)) return prev;
      const next = [...prev, pos];
      savePortfolio(next);
      return next;
    });
  }, [date]);

  const removePosition = useCallback((code: string) => {
    setPortfolio(prev => {
      const next = prev.filter(p => p.code !== code);
      savePortfolio(next);
      return next;
    });
  }, []);

  // 保有ポジションの現在スコアを allRows から引く
  const currentMap = Object.fromEntries(allRows.map(r => [r.code, r]));

  const heldPositions = portfolio.map(pos => {
    const current = currentMap[pos.code] ?? null;
    const currentNet = current?.net ?? null;
    const currentDrop = current?.drop_prob ?? null;
    const currentPrice = current?.close ?? null;
    const pnlPct = currentPrice != null ? (currentPrice - pos.buyPrice) / pos.buyPrice * 100 : null;
    const sellSignal =
      (currentNet != null && currentNet < sellNetMin) ||
      (currentDrop != null && currentDrop >= sellDropMin);
    return { pos, current, currentNet, currentDrop, currentPrice, pnlPct, sellSignal };
  });

  const sellCount = heldPositions.filter(h => h.sellSignal).length;

  return (
    <div className="space-y-10">

      {/* ── ポートフォリオ ── */}
      {mounted && (
        <section className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-bold text-white">
              保有中
              <span className="ml-2 text-sm font-normal text-gray-500">{portfolio.length} 銘柄</span>
              {sellCount > 0 && (
                <span className="ml-2 text-xs font-bold bg-red-900/60 text-red-300 border border-red-700 px-2 py-0.5 rounded-full">
                  ⚠️ 売りシグナル {sellCount}件
                </span>
              )}
            </h2>
          </div>

          {portfolio.length === 0 ? (
            <div className="bg-gray-900/60 border border-gray-800 rounded-xl px-4 py-8 text-center text-gray-600 text-sm">
              下のQV候補リストから「買い記録」を押すと追加されます
            </div>
          ) : (
            <div className="overflow-x-auto rounded-xl border border-gray-800">
              <table className="w-full text-xs font-mono">
                <thead>
                  <tr className="border-b border-gray-800 text-gray-600 text-left">
                    <th className="px-3 py-2">銘柄</th>
                    <th className="px-3 py-2 text-right">買値</th>
                    <th className="px-3 py-2 text-right">現在値</th>
                    <th className="px-3 py-2 text-right">損益</th>
                    <th className="px-3 py-2 text-right">現net</th>
                    <th className="px-3 py-2 text-right">drop%</th>
                    <th className="px-3 py-2 text-center">判定</th>
                    <th className="px-3 py-2"></th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-800/50">
                  {heldPositions.map(({ pos, currentNet, currentDrop, currentPrice, pnlPct, sellSignal }) => {
                    const up = pnlPct != null && pnlPct >= 0;
                    return (
                      <tr key={pos.code} className={sellSignal ? "bg-red-950/20" : "hover:bg-gray-800/30"}>
                        <td className="px-3 py-2">
                          <Link href={`/stocks/${pos.code}`} className="hover:text-green-400">
                            <span className="text-white font-semibold">{pos.name}</span>
                            <span className="text-gray-600 ml-1">{pos.code}</span>
                          </Link>
                          <div className="text-gray-600 text-[10px]">{pos.buyDate} 買付</div>
                        </td>
                        <td className="px-3 py-2 text-right text-gray-400">
                          ¥{pos.buyPrice.toLocaleString()}
                        </td>
                        <td className="px-3 py-2 text-right text-gray-300">
                          {currentPrice != null ? `¥${currentPrice.toLocaleString()}` : "—"}
                        </td>
                        <td className={`px-3 py-2 text-right font-bold ${up ? "text-green-400" : "text-red-400"}`}>
                          {pnlPct != null ? `${pnlPct >= 0 ? "+" : ""}${pnlPct.toFixed(1)}%` : "—"}
                        </td>
                        <td className={`px-3 py-2 text-right font-bold ${currentNet != null && currentNet >= 0 ? "text-green-400" : "text-red-400"}`}>
                          {currentNet != null ? `${currentNet >= 0 ? "+" : ""}${currentNet.toFixed(1)}%` : "—"}
                        </td>
                        <td className="px-3 py-2 text-right text-gray-400">
                          {currentDrop != null ? `${currentDrop.toFixed(1)}%` : "—"}
                        </td>
                        <td className="px-3 py-2 text-center">
                          {sellSignal ? (
                            <span className="text-xs font-bold text-red-400 bg-red-900/40 px-2 py-0.5 rounded-full">
                              ⚠️ 売り検討
                            </span>
                          ) : (
                            <span className="text-xs text-green-600 bg-green-900/20 px-2 py-0.5 rounded-full">
                              保有継続
                            </span>
                          )}
                        </td>
                        <td className="px-3 py-2 text-right">
                          <button
                            onClick={() => removePosition(pos.code)}
                            className="text-gray-600 hover:text-red-400 text-xs transition-colors"
                          >
                            削除
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          <p className="text-xs text-gray-700">
            売り目安: ネットスコア {sellNetMin}%未満 または 下落確率 {sellDropMin}%以上。手数料・税金は含みません。
          </p>
        </section>
      )}

      {/* ── QV候補リスト ── */}
      <section className="space-y-4">
        <div className="space-y-1">
          <h2 className="text-lg font-bold text-white">
            今日の QV 候補
            <span className="ml-2 text-sm font-normal text-gray-500">{candidates.length} 銘柄</span>
          </h2>
          <p className="text-xs text-gray-600">
            Piotroski ≥ 6/9（財務健全）× 52週安値圏 45%以内（株価低迷）× 業績改善シグナルあり。ネットスコア降順。
          </p>
        </div>

        {candidates.length === 0 ? (
          <div className="bg-gray-900/60 border border-gray-800 rounded-xl px-4 py-10 text-center space-y-2">
            <p className="text-gray-500 text-sm font-medium">本日は条件を満たす銘柄がありません</p>
            <p className="text-gray-700 text-xs">
              強い上昇相場では質株が安値圏に留まりにくいため候補数は減ります。週次で確認してください。
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto rounded-xl border border-gray-800">
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="border-b border-gray-800 text-gray-600 text-left">
                  <th className="px-3 py-2">銘柄</th>
                  <th className="px-3 py-2 text-right">株価</th>
                  <th className="px-3 py-2 text-right">ネット</th>
                  <th className="px-3 py-2 text-right">Piotroski</th>
                  <th className="px-3 py-2 text-right">52週位置</th>
                  <th className="px-3 py-2 text-right">BPS成長</th>
                  <th className="px-3 py-2 text-right">EPSサプライズ</th>
                  <th className="px-3 py-2 text-right">drop%</th>
                  <th className="px-3 py-2 text-right">PBR</th>
                  <th className="px-3 py-2"></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800/50">
                {candidates.map(r => {
                  const s = netStyle(r.net);
                  const alreadyHeld = mounted && portfolio.some(p => p.code === r.code);
                  return (
                    <tr key={r.code} className="hover:bg-gray-800/30 transition-colors">
                      <td className="px-3 py-2">
                        <Link href={`/stocks/${r.code}`} className="hover:text-green-400">
                          <span className="text-white font-semibold">{r.name}</span>
                          <span className="text-gray-600 ml-1">{r.code}</span>
                        </Link>
                      </td>
                      <td className="px-3 py-2 text-right text-gray-300">
                        ¥{r.close.toLocaleString()}
                      </td>
                      <td className={`px-3 py-2 text-right font-bold ${s.text}`}>
                        {signFmtArrow(r.net)}
                      </td>
                      <td className="px-3 py-2 text-right">
                        {pioLabel(r.piotroski)}
                      </td>
                      <td className="px-3 py-2">
                        {pos52Bar(r.pos52)}
                      </td>
                      <td className={`px-3 py-2 text-right ${r.bps_growth != null && r.bps_growth > 0 ? "text-green-400" : "text-gray-600"}`}>
                        {r.bps_growth != null ? `${r.bps_growth > 0 ? "+" : ""}${(r.bps_growth * 100).toFixed(1)}%` : "—"}
                      </td>
                      <td className={`px-3 py-2 text-right ${r.eps_surprise != null && r.eps_surprise > 0 ? "text-green-400" : "text-gray-600"}`}>
                        {r.eps_surprise != null ? `${r.eps_surprise > 0 ? "+" : ""}${r.eps_surprise.toFixed(2)}` : "—"}
                      </td>
                      <td className={`px-3 py-2 text-right ${r.drop_prob != null && r.drop_prob >= 5 ? "text-orange-400" : "text-gray-500"}`}>
                        {fmt(r.drop_prob)}%
                      </td>
                      <td className="px-3 py-2 text-right text-gray-500">
                        {r.pbr != null ? `${r.pbr.toFixed(2)}x` : "—"}
                      </td>
                      <td className="px-3 py-2 text-right">
                        {mounted && (
                          alreadyHeld ? (
                            <span className="text-xs text-green-600 font-medium">✓ 保有中</span>
                          ) : (
                            <button
                              onClick={() => addPosition(r)}
                              className="text-xs bg-green-900/40 text-green-400 border border-green-700 px-2 py-0.5 rounded hover:bg-green-800/50 transition-colors"
                            >
                              買い記録
                            </button>
                          )
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* 戦略メモ */}
      <section className="bg-gray-900/40 border border-gray-800/60 rounded-xl p-5 space-y-3">
        <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wide">QV戦略メモ</h3>
        <div className="grid sm:grid-cols-3 gap-4 text-xs text-gray-500">
          <div>
            <p className="text-gray-400 font-semibold mb-1">📥 買い判断</p>
            <p>上表に候補があれば週1回確認。ネットスコア高い順に検討。5銘柄まで分散推奨。</p>
          </div>
          <div>
            <p className="text-gray-400 font-semibold mb-1">📤 売り目安</p>
            <p>ネット &lt; {sellNetMin}% または drop% ≥ {sellDropMin}% で ⚠️ 表示。または90日経過。</p>
          </div>
          <div>
            <p className="text-gray-400 font-semibold mb-1">📊 バックテスト</p>
            <p>強気17ヶ月: +72.7%（日経+61%超）。暴落期: +2.7%（日経−2.5%）。サンプル小（21件）で参考値。</p>
          </div>
        </div>
      </section>
    </div>
  );
}
